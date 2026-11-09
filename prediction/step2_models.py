"""
Step 2 (Part A): 5-year frailty prediction models for SHARE.

Data: windows w4->w6 and w5->w8 for training; w6->w8 as a later-period test (built by prepare.py).
Split: every person is assigned once (seed fixed) to train (70%) or test (30%); a person's rows never cross.
  internal test  = test people, windows w4->w6 and w5->w8
  temporal test  = w6->w8 rows of everyone not in the training set
Outcomes:
  CLS  frail (FI >= 0.25) or dead by follow-up, among people not frail at baseline
  REG  FI at follow-up, among people alive and re-assessed
Dropout: inverse-probability-of-censoring weights from a logistic model of 'outcome observed' on baseline
characteristics, learned on the training people only (stabilised, truncated at the 1st/99th percentiles).
Follow-up interval: the planned interval (follow-up wave's median interview date - baseline date) is an input;
predictions for new people are made at 5.0 years.
Models: M0 reference (age, sex, FI), M1 elastic net, M2 LightGBM, M3 neural network (MLP).
"""
import os, json, time, warnings, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from prepare import DEFICITS, OUT as CACHE

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "output")
SEED = 20261109
rng = np.random.default_rng(SEED)
T0 = time.time()
def step(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

FIXED = ["age", "female", "education_isced", "partner", "hhsize", "income_quintile", "wealth_quintile", "fi"] + DEFICITS + \
        ["eurod", "recall_delayed", "orienti", "fluency", "numeracy", "maxgrip", "pain", "pain_level", "gali", "casp", "life_satisfaction"]
LEVERS = ["smoke_now", "smoke_ever", "alcohol_units_wk", "vigorous_freq", "moderate_freq", "inactive", "bmi",
          "fruitveg_freq", "protein_freq", "dairy_freq", "legumes_eggs_freq", "social_activities", "loneliness",
          "sleep_trouble", "hearing_aid", "glasses"]
OTHER = ["years_planned"]


def load():
    A = pd.read_pickle(os.path.join(CACHE, "share_windows.pkl"))
    A = A[A.window != "w2->w4"].copy()
    A["female"] = (A.sex == 2).astype(float)
    A["years_planned"] = A.groupby("window").apply(lambda g: g.t1.median() - g.t0, include_groups=False).reset_index(level=0, drop=True)
    A["years_planned"] = A.years_planned.fillna(A.groupby("window").years.transform("median"))
    countries = sorted(A.country.dropna().unique())
    for c in countries:
        A[f"ctry_{int(c)}"] = (A.country == c).astype(float)
    feats = FIXED + LEVERS + OTHER + [f"ctry_{int(c)}" for c in countries]
    A[feats] = A[feats].apply(pd.to_numeric, errors="coerce").astype(float)
    # person split
    people = A.mergeid.unique()
    u = pd.Series(rng.random(len(people)), index=people)
    train_ids = set(u[u < 0.7].index)
    A["is_train_person"] = A.mergeid.isin(train_ids)
    A["set"] = np.where(A.window == "w6->w8", np.where(A.is_train_person, "unused", "temporal"),
                        np.where(A.is_train_person, "train", "internal"))
    return A, feats


def ipcw(A, feats, observed_col, mask_fn):
    """Stabilised inverse-probability-of-censoring weights, fitted on training people per window."""
    w = pd.Series(np.nan, index=A.index)
    for win, g in A.groupby("window"):
        g = g[mask_fn(g)]
        obs = g[observed_col].astype(int)
        tr = g[g.is_train_person]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        m.fit(tr[feats], obs[tr.index])
        p = np.clip(m.predict_proba(g[feats])[:, 1], 0.05, 1)
        sw = obs.mean() / p
        lo, hi = np.percentile(sw[obs == 1], [1, 99])
        w[g.index] = np.where(obs == 1, np.clip(sw, lo, hi), 0.0)
    return w


def fit_weighted(m, X, y, w):
    """Fit with observation weights; pipelines take them through their last step; fall back to unweighted."""
    key = f"{m.steps[-1][0]}__sample_weight" if hasattr(m, "steps") else "sample_weight"
    try:
        return m.fit(X, y, **{key: w})
    except (TypeError, ValueError):
        return m.fit(X, y)


def group_cv_score(make_model, X, y, wts, groups, kind, k=3):
    gkf = GroupKFold(k); s = []
    for a, b in gkf.split(X, y, groups):
        m = make_model()
        fit_weighted(m, X.iloc[a], y.iloc[a], wts.iloc[a])
        if kind == "cls": s.append(roc_auc_score(y.iloc[b], m.predict_proba(X.iloc[b])[:, 1]))
        else: s.append(-mean_absolute_error(y.iloc[b], m.predict(X.iloc[b])))
    return float(np.mean(s))


def calib(y, p):
    """Calibration-in-the-large and slope (logistic recalibration on the logit of p)."""
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(y.mean() - p.mean()), float(m.coef_[0, 0])


def net_benefit(y, p, t):
    n = len(y); tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


def run(kind, A, feats):
    if kind == "cls":
        mask = lambda g: g.at_risk == 1
        obs_col = "obs_cls"; A["obs_cls"] = A.status.isin(["fi", "dead"])
        ycol = "frail_or_dead"
    else:
        mask = lambda g: g.status != "dead"
        obs_col = "obs_reg"; A["obs_reg"] = A.status == "fi"
        ycol = "fi_f"
    step(f"[{kind}] inverse-probability-of-censoring weights")
    A["w_" + kind] = ipcw(A, feats, obs_col, mask)
    D = A[mask(A) & A[obs_col]].copy()
    tr, it, tt = D[D.set == "train"], D[D.set == "internal"], D[D.set == "temporal"]
    X, y, w, gr = tr[feats], tr[ycol], tr["w_" + kind], tr.mergeid
    step(f"[{kind}] train {len(tr):,} rows ({tr.mergeid.nunique():,} people), internal test {len(it):,}, temporal test {len(tt):,}; "
         f"outcome mean train {y.mean():.3f}")
    models = {}
    base = ["age", "female", "fi", "years_planned"]
    if kind == "cls":
        models["M0 reference (age, sex, FI)"] = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[base], y, logisticregression__sample_weight=w)
    else:
        models["M0 reference (age, sex, FI)"] = make_pipeline(SimpleImputer(), LinearRegression()).fit(tr[base], y, linearregression__sample_weight=w)
    models["M0 reference (age, sex, FI)"].cols = base
    # M1 elastic net
    step(f"[{kind}] M1 elastic net: tuning")
    if kind == "cls":
        best = None
        for l1 in (0.0, 0.5, 1.0):
            for C in (0.01, 0.1, 1.0):
                mk = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
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
    step(f"[{kind}] M1 chosen: {best}")
    # M2 LightGBM
    step(f"[{kind}] M2 LightGBM: random search, 3-fold grouped CV")
    grid = dict(num_leaves=[15, 31, 63], learning_rate=[0.02, 0.05], min_child_samples=[50, 100, 200],
                colsample_bytree=[0.5, 0.8], subsample=[0.7, 0.9], reg_lambda=[0, 1, 5], n_estimators=[300, 600, 1000])
    best_g = None
    for i, p in enumerate(ParameterSampler(grid, 16, random_state=SEED)):
        Cl = lgb.LGBMClassifier if kind == "cls" else lgb.LGBMRegressor
        mk = lambda p=p: Cl(subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=-1, **p)
        sc = group_cv_score(mk, X, y, w, gr, kind)
        step(f"    config {i + 1}/16 {p}: CV {'AUC' if kind == 'cls' else '-MAE'} {sc:.4f}")
        if best_g is None or sc > best_g[0]: best_g = (sc, p)
    Cl = lgb.LGBMClassifier if kind == "cls" else lgb.LGBMRegressor
    m2 = Cl(subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=-1, **best_g[1]).fit(X, y, sample_weight=w)
    m2.cols = feats; models["M2 LightGBM"] = m2
    # M3 neural network
    step(f"[{kind}] M3 neural network: tuning")
    best_n = None
    for hl in ((64, 32), (128, 64)):
        for alpha in (1e-3, 1e-2):
            NN = MLPClassifier if kind == "cls" else MLPRegressor
            mk = lambda hl=hl, alpha=alpha: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                                          NN(hidden_layer_sizes=hl, alpha=alpha, early_stopping=True, max_iter=200, random_state=SEED))
            sc = group_cv_score(mk, X, y, w, gr, kind)
            step(f"    layers {hl}, alpha {alpha}: CV {sc:.4f}")
            if best_n is None or sc > best_n[0]: best_n = (sc, hl, alpha)
    NN = MLPClassifier if kind == "cls" else MLPRegressor
    m3 = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       NN(hidden_layer_sizes=best_n[1], alpha=best_n[2], early_stopping=True, max_iter=300, random_state=SEED))
    fit_weighted(m3, X, y, w)
    m3.cols = feats; models["M3 neural network"] = m3

    # ---------------- evaluation
    step(f"[{kind}] evaluating on the internal and temporal test sets (with bootstrap)")
    rows, preds = [], {}
    for sname, S in [("internal", it), ("temporal", tt)]:
        yt = S[ycol].values; ws = S["w_" + kind].values
        for name, m in models.items():
            p = m.predict_proba(S[m.cols])[:, 1] if kind == "cls" else m.predict(S[m.cols])
            preds[(sname, name)] = p
            if kind == "cls":
                ci, cs = calib(yt, p)
                r = dict(test=sname, model=name, n=len(S), events=int(yt.sum()), AUC=roc_auc_score(yt, p),
                         AUC_ipcw=roc_auc_score(yt, p, sample_weight=ws), Brier=brier_score_loss(yt, p),
                         calib_in_large=ci, calib_slope=cs, **{f"NB_{int(t * 100)}pct": net_benefit(yt, p, t) for t in (0.1, 0.2, 0.3)})
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
        # paired bootstrap vs reference
        ref = preds[(sname, "M0 reference (age, sex, FI)")]
        for name in list(models)[1:]:
            p = preds[(sname, name)]; d = []
            for _ in range(300):
                i = rng.integers(0, len(yt), len(yt))
                if kind == "cls": d.append(roc_auc_score(yt[i], p[i]) - roc_auc_score(yt[i], ref[i]))
                else: d.append(mean_absolute_error(yt[i], ref[i]) - mean_absolute_error(yt[i], p[i]))
            for r in rows:
                if r["test"] == sname and r["model"] == name:
                    r["gain_vs_ref"] = (r["AUC"] - roc_auc_score(yt, ref)) if kind == "cls" else (mean_absolute_error(yt, ref) - r["MAE"])
                    r["gain_lo"], r["gain_hi"] = np.percentile(d, [2.5, 97.5])
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(RES, f"step2_{kind}_performance.csv"), index=False)
    # calibration by decile (internal test, best model)
    pickle.dump(dict(models=models, feats=feats, tuning=dict(enet=best, lgbm=best_g, mlp=best_n)), open(os.path.join(CACHE, f"step2_{kind}_models.pkl"), "wb"))
    step(f"[{kind}] done")
    print(R.round(4).to_string(), flush=True)
    return R, models, preds, it, tt


if __name__ == "__main__":
    A, feats = load()
    step(f"Loaded {len(A):,} rows; sets: {A.set.value_counts().to_dict()}; {len(feats)} input variables")
    out = {}
    import sys
    kinds = [k for k in sys.argv[1:] if k in ("cls", "reg")] or ["cls", "reg"]
    for kind in kinds:
        R, models, preds, it, tt = run(kind, A, feats)
        out[kind] = R
    json.dump(dict(seed=SEED, n_features=len(feats), fixed=FIXED, levers=LEVERS), open(os.path.join(RES, "step2_info.json"), "w"), indent=1)
    step("All done")
