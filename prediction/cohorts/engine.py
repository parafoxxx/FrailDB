"""
Cohort engine: the SHARE pipeline (Steps 2, 2b, 2c, 3 and the predictor bundle), made cohort-agnostic.

Input: .cache/prediction/<cohort>_prepared.pkl, written by prepare_<cohort>.py, holding
  P      one row per person-wave with a frailty index: pid, wave, date (decimal year), age, female, fi, d_* items,
         the cohort's fixed characteristics and changeable levers (standard names), and optionally a region code
  death  Series pid -> first wave by which the person is known to have died
  meta   cohort, years {wave: year}, windows [(b, f)], temporal [(b, f)], triplets [(p, b, f)], fixed, levers,
         region (column or None), region_names, t1_health, notes

Run:  python run_cohort.py <COHORT> [models] [explain] [region] [levers] [bundle]   (no step names = all)
Outputs: prediction/output/<COHORT>/  (aggregates only)   models: .cache/prediction/<cohort>/
Methods are those of the SHARE analysis (see prediction/step2_models.py and step3_levers.py).
"""
import os, json, time, pickle, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.dirname(HERE)
CACHE = os.path.join(PRED, "..", ".cache", "prediction")
SEED = 20261109
FRAIL = 0.25
MAJOR = ["d_cancer", "d_stroke", "d_heart", "d_lung", "d_dementia"]
T0 = time.time()


def step(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def paths(cohort):
    out = os.path.join(PRED, "output", cohort); os.makedirs(out, exist_ok=True)
    cache = os.path.join(CACHE, cohort.lower()); os.makedirs(cache, exist_ok=True)
    return out, cache


def load_prepared(cohort):
    return pickle.load(open(os.path.join(CACHE, f"{cohort.lower()}_prepared.pkl"), "rb"))


# ------------------------------------------------------------------ windows
def outcome(P, death, b, f):
    """Baseline rows at wave b with their status at wave f: 'fi' (re-assessed), 'dead' (died after b, by f) or 'lost'."""
    B = P[P.wave == b].copy()
    F = P[P.wave == f][["pid", "fi", "date"]].rename(columns={"fi": "fi_f", "date": "date_f"})
    B = B.merge(F, on="pid", how="left")
    dw = B.pid.map(death)
    dead = (dw > b) & (dw <= f) & B.fi_f.isna()
    B["status"] = np.where(B.fi_f.notna(), "fi", np.where(dead, "dead", "lost"))
    B["years_planned"] = B.date_f.median() - B.date
    B["years_planned"] = B.years_planned.fillna(B.years_planned.median())
    return B


def build_windows(prep):
    P, death, M = prep["P"], prep["death"], prep["meta"]
    parts = []
    for b, f in M["windows"] + M["temporal"]:
        B = outcome(P, death, b, f)
        B["window"], B["wave_b"], B["wave_f"] = f"w{b}->w{f}", b, f
        B["is_temporal"] = (b, f) in [tuple(x) for x in M["temporal"]]
        parts.append(B)
    A = pd.concat(parts, ignore_index=True)
    A["frail_or_dead"] = ((A.status == "dead") | (A.fi_f >= FRAIL)).astype(int)
    A["at_risk"] = (A.fi < FRAIL).astype(int)
    return A


def features(A, M):
    feats = [c for c in M["fixed"] + M["levers"] if c in A.columns] + ["years_planned"]
    if M.get("region"):
        for r in sorted(A[M["region"]].dropna().unique()):
            A[f"reg_{int(r)}"] = (A[M["region"]] == r).astype(float)
            feats.append(f"reg_{int(r)}")
    A[feats] = A[feats].apply(pd.to_numeric, errors="coerce").astype(float)
    return feats


def split(A, dcw=None):
    rng = np.random.default_rng(SEED)
    people = np.sort(A.pid.unique())
    u = pd.Series(rng.random(len(people)), index=people)
    A["is_train_person"] = A.pid.map(u < 0.7)
    A["set"] = np.where(A.is_temporal, np.where(A.is_train_person, "unused", "temporal"),
                        np.where(A.is_train_person, "train", "internal"))
    A["set_cls"] = A["set"]
    if dcw is not None:   # deaths only recorded up to wave dcw: frail-or-dead uses windows ending by dcw, the last of them as the later-period test
        tmp = A.wave_f == dcw
        A["set_cls"] = np.where(A.wave_f > dcw, "excluded", np.where(tmp, np.where(A.is_train_person, "unused", "temporal"),
                                                                       np.where(A.is_train_person, "train", "internal")))
    return A


def ipcw(A, feats, observed_col, mask_fn):
    w = pd.Series(np.nan, index=A.index)
    for win, g in A.groupby("window"):
        g = g[mask_fn(g)]
        obs = g[observed_col].astype(int)
        tr = g[g.is_train_person]
        if obs[tr.index].nunique() < 2:
            w[g.index] = np.where(obs == 1, 1.0, 0.0); continue
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        m.fit(tr[feats], obs[tr.index])
        p = np.clip(m.predict_proba(g[feats])[:, 1], 0.05, 1)
        sw = obs.mean() / p
        lo, hi = np.percentile(sw[obs == 1], [1, 99])
        w[g.index] = np.where(obs == 1, np.clip(sw, lo, hi), 0.0)
    return w


def fit_weighted(m, X, y, w):
    key = f"{m.steps[-1][0]}__sample_weight" if hasattr(m, "steps") else "sample_weight"
    try:
        return m.fit(X, y, **{key: w})
    except (TypeError, ValueError):
        return m.fit(X, y)


def group_cv_score(make_model, X, y, wts, groups, kind, k=3):
    s = []
    for a, b in GroupKFold(k).split(X, y, groups):
        m = make_model(); fit_weighted(m, X.iloc[a], y.iloc[a], wts.iloc[a])
        s.append(roc_auc_score(y.iloc[b], m.predict_proba(X.iloc[b])[:, 1]) if kind == "cls"
                 else -mean_absolute_error(y.iloc[b], m.predict(X.iloc[b])))
    return float(np.mean(s))


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(y.mean() - p.mean()), float(m.coef_[0, 0])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def net_benefit(y, p, t):
    n = len(y); tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


def prepare_analysis(cohort):
    prep = load_prepared(cohort)
    A = build_windows(prep)
    feats = features(A, prep["meta"])
    A = split(A, prep["meta"].get("death_complete_wave"))
    A["obs_cls"] = A.status.isin(["fi", "dead"]); A["obs_reg"] = A.status == "fi"
    return prep, A, feats


# ------------------------------------------------------------------ Step 2: models
def run_models(cohort, kind, A, feats, out, cache):
    rng = np.random.default_rng(SEED)
    if kind == "cls":
        mask, obs_col, ycol = (lambda g: (g.at_risk == 1) & (g.set_cls != "excluded")), "obs_cls", "frail_or_dead"
    else:
        mask, obs_col, ycol = (lambda g: g.status != "dead"), "obs_reg", "fi_f"
    sc = "set_cls" if kind == "cls" else "set"
    step(f"[{kind}] censoring weights")
    A["w_" + kind] = ipcw(A, feats, obs_col, mask)
    D = A[mask(A) & A[obs_col]].copy()
    tr, it, tt = D[D[sc] == "train"], D[D[sc] == "internal"], D[D[sc] == "temporal"]
    X, y, w, gr = tr[feats], tr[ycol], tr["w_" + kind], tr.pid
    step(f"[{kind}] train {len(tr):,} rows ({tr.pid.nunique():,} people), internal test {len(it):,}, later-period test {len(tt):,}; "
         f"outcome mean {y.mean():.3f}")
    models = {}
    base = ["age", "female", "fi", "years_planned"]
    if kind == "cls":
        models["M0 reference (age, sex, FI)"] = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[base], y, logisticregression__sample_weight=w)
    else:
        models["M0 reference (age, sex, FI)"] = make_pipeline(SimpleImputer(), LinearRegression()).fit(tr[base], y, linearregression__sample_weight=w)
    models["M0 reference (age, sex, FI)"].cols = base
    step(f"[{kind}] M1 elastic net: tuning")
    if kind == "cls":
        best = None
        for l1 in (0.0, 0.5, 1.0):
            for C in (0.01, 0.1, 1.0):
                mk = lambda l1=l1, C=C: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                                      LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1, C=C, max_iter=300, tol=1e-3))
                sc = group_cv_score(mk, X, y, w, gr, kind)
                step(f"    l1_ratio {l1}, C {C}: CV AUC {sc:.4f}")
                if best is None or sc > best[0]: best = (sc, l1, C)
        m1 = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                           LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=best[1], C=best[2], max_iter=500, tol=1e-4))
        m1.fit(X, y, logisticregression__sample_weight=w)
    else:
        m1 = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                           ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=list(GroupKFold(3).split(X, y, gr)), n_jobs=-1, max_iter=3000))
        m1.fit(X, y, elasticnetcv__sample_weight=w)
        best = ("cv", float(m1[-1].l1_ratio_), float(m1[-1].alpha_))
    m1.cols = feats; models["M1 elastic net"] = m1
    step(f"[{kind}] M2 LightGBM: random search, 3-fold grouped CV")
    grid = dict(num_leaves=[15, 31, 63], learning_rate=[0.02, 0.05], min_child_samples=[50, 100, 200],
                colsample_bytree=[0.5, 0.8], subsample=[0.7, 0.9], reg_lambda=[0, 1, 5], n_estimators=[300, 600, 1000])
    Cl = lgb.LGBMClassifier if kind == "cls" else lgb.LGBMRegressor
    best_g = None
    for i, p in enumerate(ParameterSampler(grid, 16, random_state=SEED)):
        mk = lambda p=p: Cl(subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=-1, **p)
        sc = group_cv_score(mk, X, y, w, gr, kind)
        step(f"    config {i + 1}/16: CV {'AUC' if kind == 'cls' else '-MAE'} {sc:.4f}")
        if best_g is None or sc > best_g[0]: best_g = (sc, p)
    m2 = Cl(subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=-1, **best_g[1]).fit(X, y, sample_weight=w)
    m2.cols = feats; models["M2 LightGBM"] = m2
    step(f"[{kind}] M3 neural network: tuning")
    NN = MLPClassifier if kind == "cls" else MLPRegressor
    best_n = None
    for hl in ((64, 32), (128, 64)):
        for alpha in (1e-3, 1e-2):
            mk = lambda hl=hl, alpha=alpha: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                                          NN(hidden_layer_sizes=hl, alpha=alpha, early_stopping=True, max_iter=200, random_state=SEED))
            sc = group_cv_score(mk, X, y, w, gr, kind)
            step(f"    layers {hl}, alpha {alpha}: CV {sc:.4f}")
            if best_n is None or sc > best_n[0]: best_n = (sc, hl, alpha)
    m3 = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       NN(hidden_layer_sizes=best_n[1], alpha=best_n[2], early_stopping=True, max_iter=300, random_state=SEED))
    fit_weighted(m3, X, y, w); m3.cols = feats; models["M3 neural network"] = m3

    step(f"[{kind}] evaluating")
    rows, preds = [], {}
    for sname, S in [("internal", it), ("temporal", tt)]:
        if len(S) == 0: continue
        yt = S[ycol].values; ws = S["w_" + kind].values
        for name, m in models.items():
            p = m.predict_proba(S[m.cols])[:, 1] if kind == "cls" else m.predict(S[m.cols])
            preds[(sname, name)] = p
            if kind == "cls":
                ci, cs = calib(yt, p)
                r = dict(test=sname, model=name, n=len(S), events=int(yt.sum()), AUC=roc_auc_score(yt, p),
                         AUC_ipcw=roc_auc_score(yt, p, sample_weight=ws), Brier=brier_score_loss(yt, p), calib_in_large=ci,
                         calib_slope=cs, **{f"NB_{int(t * 100)}pct": net_benefit(yt, p, t) for t in (0.1, 0.2, 0.3)})
            else:
                r = dict(test=sname, model=name, n=len(S), MAE=mean_absolute_error(yt, p), RMSE=mean_squared_error(yt, p) ** 0.5,
                         R2=r2_score(yt, p), R2_ipcw=r2_score(yt, p, sample_weight=ws))
            rows.append(r)
        if kind == "cls":
            rows.append(dict(test=sname, model="Treat everyone as high risk", **{f"NB_{int(t * 100)}pct": net_benefit(yt, np.ones(len(yt)), t) for t in (0.1, 0.2, 0.3)}))
        else:
            pn = S["fi"].values
            rows.append(dict(test=sname, model="No change (FI stays as now)", n=len(S), MAE=mean_absolute_error(yt, pn),
                             RMSE=mean_squared_error(yt, pn) ** 0.5, R2=r2_score(yt, pn)))
        ref = preds[(sname, "M0 reference (age, sex, FI)")]
        for name in list(models)[1:]:
            p = preds[(sname, name)]; d = []
            for _ in range(300):
                i = rng.integers(0, len(yt), len(yt))
                if kind == "cls":
                    if len(np.unique(yt[i])) < 2: continue
                    d.append(roc_auc_score(yt[i], p[i]) - roc_auc_score(yt[i], ref[i]))
                else:
                    d.append(mean_absolute_error(yt[i], ref[i]) - mean_absolute_error(yt[i], p[i]))
            for r in rows:
                if r["test"] == sname and r["model"] == name:
                    r["gain_vs_ref"] = (r["AUC"] - roc_auc_score(yt, ref)) if kind == "cls" else (mean_absolute_error(yt, ref) - r["MAE"])
                    r["gain_lo"], r["gain_hi"] = np.percentile(d, [2.5, 97.5])
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(out, f"step2_{kind}_performance.csv"), index=False)
    pickle.dump(dict(models=models, feats=feats, tuning=dict(enet=best, lgbm=best_g, mlp=best_n)), open(os.path.join(cache, f"step2_{kind}_models.pkl"), "wb"))
    print(R.round(4).to_string(), flush=True)
    return R


# ------------------------------------------------------------------ Step 2b: recalibration, accuracy, SHAP
def explain(cohort, prep, A, feats, out, cache):
    M = prep["meta"]; res = {}
    gbm = pickle.load(open(os.path.join(cache, "step2_cls_models.pkl"), "rb"))["models"]["M2 LightGBM"]
    D = A[(A.at_risk == 1) & A.obs_cls & (A.set_cls != "excluded")]
    rec, tt, it = D[D.set_cls == "unused"], D[D.set_cls == "temporal"], D[D.set_cls == "internal"]
    a, b = 0.0, 1.0
    if len(rec) > 200 and rec.frail_or_dead.nunique() == 2:
        cal = LogisticRegression(C=1e6).fit(logit(gbm.predict_proba(rec[feats])[:, 1]).reshape(-1, 1), rec.frail_or_dead)
        a, b = float(cal.intercept_[0]), float(cal.coef_[0, 0])
    recal = lambda p: 1 / (1 + np.exp(-(a + b * logit(p))))
    res["recalibration"] = dict(intercept=round(a, 4), slope=round(b, 4), n=len(rec))
    step(f"recalibration on {len(rec):,} rows: intercept {a:.3f}, slope {b:.3f}")
    rows, acc = [], []
    for name, Sx in [("internal", it), ("temporal", tt)]:
        if len(Sx) == 0: continue
        p0 = gbm.predict_proba(Sx[feats])[:, 1]
        y = Sx.frail_or_dead.values
        for lab, q in [("as trained", p0), ("recalibrated", recal(p0))]:
            ci, cs = calib(y, q)
            rows.append(dict(test=name, version=lab, n=len(y), observed=y.mean(), predicted=q.mean(), calib_in_large=ci, calib_slope=cs,
                             AUC=roc_auc_score(y, q), Brier=brier_score_loss(y, q)))
        q = recal(p0) if name == "temporal" else p0
        for t in (0.5, 0.3, 0.2):
            yh = q >= t; tp = (yh & (y == 1)).sum(); tn = (~yh & (y == 0)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
            acc.append(dict(test=name, cutoff=t, n=len(y), event_rate=y.mean(), accuracy=(tp + tn) / len(y), always_no=1 - y.mean(),
                            sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1), PPV=tp / max(tp + fp, 1), NPV=tn / max(tn + fn, 1)))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "step2b_recalibration.csv"), index=False)
    pd.DataFrame(acc).round(4).to_csv(os.path.join(out, "step2b_accuracy.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)
    print(pd.DataFrame(acc).round(3).to_string(), flush=True)
    groups = {f: "changeable" for f in M["levers"]}
    groups.update({f: "fixed" for f in M["fixed"]})
    groups["years_planned"] = "follow-up length"
    for kind, path in [("cls", "step2_cls_models.pkl"), ("reg", "step2_reg_models.pkl")]:
        f = os.path.join(cache, path)
        if not os.path.exists(f): continue
        m = pickle.load(open(f, "rb"))["models"]["M2 LightGBM"]
        X = (it if kind == "cls" else A[(A.status == "fi") & (A.set == "internal")])[feats]
        X = X.sample(min(6000, len(X)), random_state=SEED)
        sh = m.booster_.predict(X, pred_contrib=True)[:, :-1]
        imp = np.abs(sh).mean(axis=0)
        dirn = [np.sign(np.corrcoef(X.iloc[:, j].fillna(X.iloc[:, j].median()), sh[:, j])[0, 1]) if X.iloc[:, j].nunique() > 1 else 0 for j in range(len(feats))]
        T = pd.DataFrame({"variable": feats, "group": [groups.get(v, "fixed") for v in feats], "mean_abs_shap": imp,
                          "share_pct": 100 * imp / imp.sum(), "direction": dirn})
        rg = T[T.variable.str.startswith("reg_")]
        T = pd.concat([T[~T.variable.str.startswith("reg_")]] + ([pd.DataFrame([dict(variable="region (all)", group="fixed", mean_abs_shap=rg.mean_abs_shap.sum(),
                                                                                      share_pct=rg.share_pct.sum(), direction=0)])] if len(rg) else []))
        T = T.sort_values("share_pct", ascending=False)
        T.round(4).to_csv(os.path.join(out, f"step2b_{kind}_shap.csv"), index=False)
        res[kind + "_group_share"] = T.groupby("group").share_pct.sum().round(1).to_dict()
        step(f"{kind}: share of importance by group {res[kind + '_group_share']}")
        print(T.head(15).round(3).to_string(), flush=True)
    json.dump(res, open(os.path.join(out, "step2b_summary.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------ Step 2c: leave one region out
def region_out(cohort, prep, A, feats, out, cache):
    M = prep["meta"]; reg = M.get("region")
    if not reg:
        step("no region variable for this cohort; leave-one-region-out skipped"); return None
    A["is_train_person"] = True
    rows = []
    for kind in ["cls", "reg"]:
        tun = pickle.load(open(os.path.join(cache, f"step2_{kind}_models.pkl"), "rb"))["tuning"]["lgbm"][1]
        mask = (lambda g: (g.at_risk == 1) & (g.set_cls != "excluded")) if kind == "cls" else (lambda g: g.status != "dead")
        obs, ycol = ("obs_cls", "frail_or_dead") if kind == "cls" else ("obs_reg", "fi_f")
        A["w"] = ipcw(A, feats, obs, mask)
        D = A[mask(A) & A[obs]]
        regions = D[reg].value_counts()
        for i, (r, n) in enumerate(regions.items()):
            if n < 300: continue
            tr, te = D[D[reg] != r], D[D[reg] == r]
            Cl = lgb.LGBMClassifier if kind == "cls" else lgb.LGBMRegressor
            m = Cl(subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=-1, **tun).fit(tr[feats], tr[ycol], sample_weight=tr.w)
            name = M.get("region_names", {}).get(int(r), str(int(r)))
            if kind == "cls":
                p = m.predict_proba(te[feats])[:, 1]; ci, cs = calib(te[ycol].values, p)
                row = dict(kind=kind, region=name, n=len(te), observed=te[ycol].mean(), predicted=p.mean(), AUC=roc_auc_score(te[ycol], p),
                           calib_in_large=ci, calib_slope=cs)
            else:
                p = m.predict(te[feats])
                row = dict(kind=kind, region=name, n=len(te), MAE=mean_absolute_error(te[ycol], p), R2=r2_score(te[ycol], p))
            rows.append(row)
            step(f"[{kind}] {i + 1}/{len(regions)} {name} (n={len(te):,}): " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k in ("AUC", "calib_in_large", "MAE", "R2")))
    R = pd.DataFrame(rows); R.round(4).to_csv(os.path.join(out, "step2c_region.csv"), index=False)
    return R


# ------------------------------------------------------------------ Step 3: lever effects (AIPW)
CONTRASTS = [
    ("quit_smoking", "Stopped smoking (vs kept smoking)", ["smoke_now"], lambda d: d.smoke_now_0 == 1, lambda d: d.smoke_now_1 == 0),
    ("start_vigorous", "Took up vigorous activity at least weekly", ["vigorous_weekly"], lambda d: d.vigorous_weekly_0 == 0, lambda d: d.vigorous_weekly_1 == 1),
    ("leave_inactivity", "Inactive -> became at least moderately active", ["inactive"], lambda d: d.inactive_0 == 1, lambda d: d.inactive_1 == 0),
    ("stay_active", "Active -> stayed active (vs became inactive)", ["inactive"], lambda d: d.inactive_0 == 0, lambda d: d.inactive_1 == 0),
    ("cut_heavy_drinking", "Heavy drinker (>14 drinks/wk) -> 14 or fewer", ["heavy_drink"], lambda d: d.heavy_drink_0 == 1, lambda d: d.heavy_drink_1 == 0),
    ("obese_lost_weight", "Obese -> lost 5% or more of BMI", ["bmi"], lambda d: d.bmi_0 >= 30, lambda d: d.bmi_1 <= 0.95 * d.bmi_0),
    ("daily_fruit", "Fruit less than daily -> daily", ["fruit_daily"], lambda d: d.fruit_daily_0 == 0, lambda d: d.fruit_daily_1 == 1),
    ("daily_veg", "Vegetables less than daily -> daily", ["veg_daily"], lambda d: d.veg_daily_0 == 0, lambda d: d.veg_daily_1 == 1),
    ("join_social", "No social activities -> took part", ["social_any"], lambda d: d.social_any_0 == 0, lambda d: d.social_any_1 == 1),
    ("less_lonely", "Lonely -> not lonely", ["lonely"], lambda d: d.lonely_0 == 1, lambda d: d.lonely_1 == 0),
    ("sleep_resolved", "Sleep trouble -> resolved", ["sleep_trouble"], lambda d: d.sleep_trouble_0 == 1, lambda d: d.sleep_trouble_1 == 0),
    ("got_hearing_aid", "Poor hearing, no aid -> started using a hearing aid", ["hearing_aid", "d_hearing"],
     lambda d: (d.d_hearing_0 == 1) & (d.hearing_aid_0 == 0), lambda d: d.hearing_aid_1 == 1),
]


def _learners():
    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.6, verbose=-1, random_state=SEED, n_jobs=-1)
    l = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    return g, l


def _fit_pred(X, y, Xnew):
    if y.nunique() < 2: return np.full(len(Xnew), float(y.mean()))
    g, l = _learners()
    return 0.5 * g.fit(X, y).predict_proba(Xnew)[:, 1] + 0.5 * l.fit(X, y).predict_proba(Xnew)[:, 1]


def _aipw(d, X, ycols):
    n = len(d); e = np.zeros(n); c = np.zeros(n); m = {y: np.zeros((n, 2)) for y in ycols}
    Xa = d[X].copy(); Xa["A"] = d.A.values
    for tr, te in GroupKFold(5).split(d, groups=d.pid):
        dt = d.iloc[tr]
        e[te] = _fit_pred(d[X].iloc[tr], dt.A, d[X].iloc[te])
        c[te] = _fit_pred(Xa.iloc[tr], dt.R, Xa.iloc[te])
        for y in ycols:
            for a in (0, 1):
                s = dt[(dt.A == a) & (dt.R == 1) & dt[y].notna()]
                m[y][te, a] = _fit_pred(s[X], s[y], d[X].iloc[te]) if len(s) > 20 else np.full(len(te), dt[y].mean())
    e = np.clip(e, 0.01, 0.99); c = np.clip(c, 0.05, 1)
    A, R = d.A.values, d.R.values
    out = {}
    for y in ycols:
        Y = np.nan_to_num(d[y].values)
        out[y] = (m[y][:, 1] + A * R / (e * c) * (Y - m[y][:, 1]), m[y][:, 0] + (1 - A) * R / ((1 - e) * c) * (Y - m[y][:, 0]))
    return out, e


def _summ(phi1, phi0, groups):
    n = len(phi1); r1, r0 = phi1.mean(), phi0.mean()
    def se(v):
        s = pd.Series(v - v.mean()).groupby(groups.values).sum(); return float(np.sqrt((s ** 2).sum()) / n)
    se_rd = se(phi1 - phi0); rd = r1 - r0
    lrr, se_lrr = (np.log(r1 / r0), se(phi1 / r1 - phi0 / r0)) if r1 > 0 and r0 > 0 else (np.nan, np.nan)
    return dict(risk_changed=r1, risk_not_changed=r0, RD=rd, RD_lo=rd - 1.96 * se_rd, RD_hi=rd + 1.96 * se_rd,
                RR=np.exp(lrr), RR_lo=np.exp(lrr - 1.96 * se_lrr), RR_hi=np.exp(lrr + 1.96 * se_lrr),
                p=float(2 * (1 - norm.cdf(abs(rd / se_rd)))) if se_rd > 0 else np.nan)


def _evalue(rr, lo, hi):
    f = lambda r: r + np.sqrt(r * (r - 1)) if r >= 1 else np.nan
    if np.isnan(rr): return np.nan, np.nan
    if rr < 1: rr, lim = 1 / rr, (1 / hi if hi < 1 else 1.0)
    else: lim = lo if lo > 1 else 1.0
    return f(rr), f(lim)


def levers(cohort, prep, out, cache):
    P, death, M = prep["P"], prep["death"], prep["meta"]
    base = [c for c in M["fixed"] + M["levers"] if c in P.columns]
    keep = sorted(set(base + ["pid", "date"] + ([M["region"]] if M.get("region") else []) + [c for c in P.columns if c.startswith("d_")]))
    parts = []
    for p, b, f in M["triplets"]:
        a = P[P.wave == p][keep].copy(); a.columns = [c if c == "pid" else c + "_0" for c in a.columns]
        o = outcome(P, death, b, f)[keep + ["status", "fi_f"]]; o.columns = [c if c == "pid" else c + "_1" for c in o.columns]
        d = a.merge(o, on="pid"); d["triplet"] = f"w{p}->w{b}->w{f}"; parts.append(d)
    if not parts:
        step("no wave triplets for this cohort; lever analysis skipped"); return None
    D = pd.concat(parts, ignore_index=True)
    D["gap_years"] = D.date_1 - D.date_0
    D["R"] = (D.status_1 != "lost").astype(int)
    D["Y"] = np.where(D.status_1 == "dead", 1.0, np.where(D.status_1 == "fi", (D.fi_f_1 >= FRAIL).astype(float), np.nan))
    D["Y_dead"] = np.where(D.status_1 == "lost", np.nan, (D.status_1 == "dead").astype(float))
    ill = np.zeros(len(D), bool)
    for c in MAJOR:
        if c + "_0" in D: ill |= ((D[c + "_0"] == 0) & (D[c + "_1"] == 1)).values
    D["very_ill"] = ill | (D.fi_1 >= FRAIL).values
    for t in D.triplet.unique(): D[f"trip_{t}"] = (D.triplet == t).astype(float)
    if M.get("region"):
        for r in sorted(D[M["region"] + "_0"].dropna().unique()): D[f"reg_{int(r)}"] = (D[M["region"] + "_0"] == r).astype(float)
    X0 = [c + "_0" for c in base] + ["age_1", "gap_years"] + [c for c in D.columns if c.startswith(("trip_", "reg_"))]
    XT1 = [c + "_1" for c in M.get("t1_health", ["fi", "d_srh"]) if c + "_1" in D]
    elig_all = D.fi_0 < FRAIL
    step(f"Built {len(D):,} person-triplets ({D.pid.nunique():,} people) {D.triplet.value_counts().to_dict()}; "
         f"not frail at T0 {elig_all.sum():,}; left out as very ill {(elig_all & D.very_ill).sum():,}")
    rows, sub = [], []
    for key, label, need, el, trt in CONTRASTS:
        if not all(c + "_0" in D and c + "_1" in D for c in need):
            continue
        d = D[elig_all & ~D.very_ill].copy()
        d = d[el(d).fillna(False).astype(bool)]
        d = d[d[[c + "_1" for c in need]].notna().all(axis=1)].reset_index(drop=True)
        d["A"] = trt(d).astype(float)
        nA = int(d.A.sum())
        step(f"[{key}] {label}: {len(d):,} eligible, {nA:,} changed, lost {100 * (1 - d.R.mean()) if len(d) else 0:.1f}%")
        if nA < 100 or len(d) - nA < 100:
            step("    too few in one arm; skipped"); continue
        obs = d[d.R == 1]
        for spec, X in [("primary (T0 adjustment)", X0), ("sensitivity (+ health at T1)", X0 + XT1)]:
            res, e = _aipw(d, X, ["Y", "Y_dead"])
            for y, ylab in [("Y", "frail or dead"), ("Y_dead", "dead")]:
                s = _summ(*res[y], d.pid)
                ev, evl = _evalue(s["RR"], s["RR_lo"], s["RR_hi"])
                rows.append(dict(lever=key, label=label, outcome=ylab, spec=spec, n=len(d), people=d.pid.nunique(), n_changed=nA,
                                 events=int(obs[y].sum()), crude_changed=obs[obs.A == 1][y].mean(), crude_not=obs[obs.A == 0][y].mean(),
                                 **s, E_value=ev, E_value_CI=evl, extreme_propensity_pct=100 * np.mean((e <= 0.01) | (e >= 0.99))))
                if spec.startswith("primary"):
                    for gname, gm in [("age < 70", d.age_1 < 70), ("age 70+", d.age_1 >= 70), ("men", d.female_0 == 0), ("women", d.female_0 == 1)]:
                        gm = gm.values
                        if gm.sum() > 200:
                            sub.append(dict(lever=key, outcome=ylab, subgroup=gname, n=int(gm.sum()), **_summ(res[y][0][gm], res[y][1][gm], d.pid[gm])))
            r = [x for x in rows if x["lever"] == key and x["spec"] == spec and x["outcome"] == "frail or dead"][0]
            step(f"    {spec}: {100 * r['risk_changed']:.1f}% if changed vs {100 * r['risk_not_changed']:.1f}% "
                 f"(crude {100 * r['crude_changed']:.1f}% vs {100 * r['crude_not']:.1f}%); RD {100 * r['RD']:+.1f} "
                 f"[{100 * r['RD_lo']:+.1f}, {100 * r['RD_hi']:+.1f}], RR {r['RR']:.2f} [{r['RR_lo']:.2f}, {r['RR_hi']:.2f}]")
        pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "step3_lever_effects.csv"), index=False)
        pd.DataFrame(sub).round(4).to_csv(os.path.join(out, "step3_lever_subgroups.csv"), index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ predictor bundle
def _grade(E):
    """clear = benefit significant in both analyses after Holm correction across all habits tested;
    suggestive = nominally significant (p < 0.05) benefit in either analysis; else no evidence."""
    import numpy as np
    out = {}
    for spec, g in E.groupby(E.spec.str.split(" ").str[0]):
        g = g.sort_values("p"); k = len(g)
        holm = np.maximum.accumulate(np.minimum(1, g.p.values * (k - np.arange(k))))
        for lever, rr, p, h in zip(g.lever, g.RR, g.p, holm):
            out.setdefault(lever, []).append((rr < 1 and h < 0.05, rr < 1 and p < 0.05))
    return {k: "clear" if all(a for a, _ in v) else ("suggestive" if any(b for _, b in v) else "no evidence") for k, v in out.items()}


def bundle(cohort, prep, A, feats, out, cache):
    M = prep["meta"]
    cls = pickle.load(open(os.path.join(cache, "step2_cls_models.pkl"), "rb"))
    reg = pickle.load(open(os.path.join(cache, "step2_reg_models.pkl"), "rb"))
    rec = json.load(open(os.path.join(out, "step2b_summary.json")))["recalibration"]
    if M.get("recalibrate", True) is False: rec = dict(intercept=0.0, slope=1.0)   # model already calibrated: use as trained
    effects = {}
    ef = os.path.join(out, "step3_lever_effects.csv")
    if os.path.exists(ef):
        E = pd.read_csv(ef); E = E[E.outcome == "frail or dead"]
        pr = E[E.spec.str.startswith("primary")].set_index("lever"); se = E[E.spec.str.startswith("sensitivity")].set_index("lever")
        grades = _grade(E)
        for k in pr.index:
            a, b = pr.loc[k], se.loc[k]
            grade = grades[k]
            cons = a if abs(np.log(a.RR)) <= abs(np.log(b.RR)) else b
            effects[k] = dict(label=a.label, grade=grade, RR=float(cons.RR), RR_lo=float(cons.RR_lo), RR_hi=float(cons.RR_hi),
                              RR_primary=float(a.RR), RR_sensitivity=float(b.RR), n=int(a.n))
    T = A[~A.is_temporal].copy()
    T["band"] = (T.age.clip(50, 95) // 5 * 5).astype(int)
    typical = T.groupby(["female", "band"])[[f for f in feats if not f.startswith("reg_") and f != "years_planned"]].median()
    B = dict(cohort=cohort, cls=cls["models"]["M2 LightGBM"], reg=reg["models"]["M2 LightGBM"], feats=feats, recal=(rec["intercept"], rec["slope"]),
             fixed=M["fixed"], levers=M["levers"], effects=effects, typical=typical, region=M.get("region"), notes=M.get("notes", []))
    f = os.path.join(CACHE, f"frailty_predictor_{cohort.lower()}.pkl")
    pickle.dump(B, open(f, "wb"))
    step(f"bundle saved: {f} ({os.path.getsize(f) / 1e6:.1f} MB); lever effects: " + ", ".join(f"{k} {v['grade']} RR {v['RR']:.2f}" for k, v in effects.items()))
