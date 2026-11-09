"""
Stage 2: do learned deficit weights predict mortality better than Rockwood's equal weights, in people the
weights were not learned on? And how close does a weighted sum come to the best that ML / deep models can do
with the same 32 items?

SHARE is split 70/30 at random (by person, stratified on death). Everything is learned on the 70% and
evaluated once on the 30%. Weight schemes (all turned into a 0-1 index: sum(w*d) / sum(w)):
  S0  equal weights (Rockwood FI)
  S1  Cox weights: ridge-penalised Cox model of all 32 deficits + age + age^2 + sex; weights = deficit coefficients
  S1+ non-negative Cox weights: deficits with a negative coefficient are dropped and the model refitted, until all
      weights are positive (keeps the index monotone: having a deficit never lowers frailty)
  S2  cluster-balanced weights (outcome-free): deficits grouped by correlation (stage 1 clusters, re-derived on the
      training set); every cluster gets the same total weight, shared equally by its items
Ceiling models (age, sex and the 32 items, any non-linearity and interaction):
  M1  LightGBM (gradient-boosted trees), 5-year mortality
  M2  neural network classifier (MLP), 5-year mortality
  M3  DeepSurv: a neural network trained on the Cox partial likelihood (Katzman et al. 2018), full follow-up

Metrics on the 30% test set: Harrell's C for survival (score alone, and Cox model of age + sex + score);
AUC for 5-year death (score alone, and logistic model of age + sex + score); Brier score; paired bootstrap
(500 resamples) for differences against S0.
"""
import os, json, warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import lightgbm as lgb
import torch, torch.nn as nn
from stage1_associations import load, LABEL, DOMAIN
from prepare import DEFICITS

warnings.filterwarnings("ignore")
SEED = 20261026
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "output")
BASE = ["agec", "agec2", "female"]
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
import time as _t
_T0 = _t.time()
def step(msg): print(f"[{(_t.time() - _T0) / 60:5.1f} min] {msg}", flush=True)


def cox(df, cols, pen=0.0):
    m = CoxPHFitter(penalizer=pen)
    m.fit(df[cols + ["time", "died"]], duration_col="time", event_col="died", fit_options={"step_size": 0.5})
    return m


def cox_cv_penalty(tr, pens=(0.0, 0.001, 0.01, 0.05)):
    """5-fold CV concordance on the training set for the ridge penalty of the 32-deficit Cox model."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {}
    for p in pens:
        cs = []
        for f, (a, b) in enumerate(skf.split(tr, tr.died), 1):
            step(f"  Cox penalty {p}: fold {f}/5")
            m = cox(tr.iloc[a], DEFICITS + BASE, p)
            cs.append(concordance_index(tr.iloc[b].time, -m.predict_partial_hazard(tr.iloc[b]), tr.iloc[b].died))
        out[p] = float(np.mean(cs)); step(f"  Cox penalty {p}: CV concordance {out[p]:.4f}")
    return max(out, key=out.get), out


def index_from(w, X):
    w = pd.Series(w).reindex(DEFICITS).fillna(0.0)
    return (X[DEFICITS].values @ w.values) / w.clip(lower=0).sum() if w.clip(lower=0).sum() else X[DEFICITS].values @ w.values


# ------------------------------------------------------------------ DeepSurv
class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x): return self.f(x).squeeze(-1)


def cox_loss(risk, t, e):
    o = torch.argsort(t, descending=True)
    r, e = risk[o], e[o]
    lse = torch.logcumsumexp(r, 0)
    return -torch.sum((r - lse) * e) / torch.clamp(e.sum(), min=1)


def deepsurv(tr, cols):
    a, b = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=tr.died)
    X = lambda d: torch.tensor(d[cols].values, dtype=torch.float32)
    xa, ta, ea = X(a), torch.tensor(a.time.values, dtype=torch.float32), torch.tensor(a.died.values, dtype=torch.float32)
    xb = X(b)
    net = Net(len(cols)); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1, None, 0
    for ep in range(60):
        net.train(); perm = torch.randperm(len(xa))
        for i in range(0, len(perm), 4096):
            j = perm[i:i + 4096]; opt.zero_grad(); loss = cox_loss(net(xa[j]), ta[j], ea[j]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): c = concordance_index(b.time, -net(xb).numpy(), b.died)
        step(f"  DeepSurv epoch {ep + 1}: validation C {c:.4f}")
        if c > best + 1e-4: best, best_state, bad = c, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6: break
    net.load_state_dict(best_state); net.eval()
    return lambda d: net(X(d)).detach().numpy(), ep + 1, best


# ------------------------------------------------------------------ evaluation helpers
def boot_diff(fn, n, B=500):
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(fn(i))
    v = np.array(vals)
    return np.percentile(v, [2.5, 97.5], axis=0)


if __name__ == "__main__":
    step("Loading SHARE cohort"); df = load()
    tr, te = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df.died)
    tr, te = tr.copy(), te.copy()
    info = dict(n_train=len(tr), n_test=len(te), deaths_train=int(tr.died.sum()), deaths_test=int(te.died.sum()))

    # ---------------- weight schemes (learned on training set only)
    step(f"Split: {len(tr):,} train / {len(te):,} test. Learning weights")
    W = {"S0 equal (Rockwood)": pd.Series(1.0, index=DEFICITS)}
    pen, cvres = cox_cv_penalty(tr)
    info["cox_penalty_cv"] = {str(k): round(v, 4) for k, v in cvres.items()}; info["cox_penalty"] = pen
    m1 = cox(tr, DEFICITS + BASE, pen)
    W["S1 Cox weights"] = m1.params_[DEFICITS]
    step(f"S1 Cox weights fitted (penalty {pen}). Fitting non-negative version")
    keep = list(DEFICITS)
    while True:
        m = cox(tr, keep + BASE, pen)
        neg = [d for d in keep if m.params_[d] <= 0]
        if not neg: break
        keep = [d for d in keep if d not in neg]
    W["S1+ non-negative Cox weights"] = m.params_[keep].reindex(DEFICITS).fillna(0.0)
    C = tr[DEFICITS].corr().values
    Z = linkage(squareform(1 - np.abs(C), checks=False), method="average")
    cl = pd.Series(fcluster(Z, t=0.7, criterion="distance"), index=DEFICITS)
    W["S2 cluster-balanced"] = cl.map(1 / cl.value_counts())
    for k in W:
        tr[k], te[k] = index_from(W[k], tr), index_from(W[k], te)

    # ---------------- ceiling models
    step("Weights done. Training LightGBM")
    feats = BASE + DEFICITS
    k5tr, k5te = tr.dead5.notna(), te.dead5.notna()
    a, b = train_test_split(tr[k5tr], test_size=0.15, random_state=SEED, stratify=tr[k5tr].dead5)
    gbm = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED)
    gbm.fit(a[feats], a.dead5, eval_set=[(b[feats], b.dead5)], callbacks=[lgb.early_stopping(100, verbose=False)])
    step(f"LightGBM done ({gbm.best_iteration_} trees). Training neural network (MLP)")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, validation_fraction=0.15,
                        max_iter=300, random_state=SEED).fit(tr[k5tr][feats], tr[k5tr].dead5)
    step("MLP done. Training DeepSurv"); ds_pred, ds_epochs, ds_val = deepsurv(tr, feats)
    info.update(gbm_trees=int(gbm.best_iteration_ or 0), deepsurv_epochs=ds_epochs, deepsurv_val_C=round(float(ds_val), 4))

    # ---------------- evaluation on the test set
    step("Evaluating all models on the 30% test set")
    schemes = list(W)
    rows, preds_c, preds_a = [], {}, {}
    T5 = te[k5te]
    for k in schemes:
        mc = cox(tr, [k] + BASE)                                     # age + sex + score, survival
        lr = LogisticRegression(max_iter=2000).fit(tr[k5tr][[k] + BASE], tr[k5tr].dead5)
        preds_c[k] = mc.predict_partial_hazard(te).values             # higher = riskier
        preds_a[k] = lr.predict_proba(T5[[k] + BASE])[:, 1]
        rows.append(dict(model=k, kind="weighted index",
                         C_score_alone=concordance_index(te.time, -te[k], te.died),
                         C_with_age_sex=concordance_index(te.time, -preds_c[k], te.died),
                         AUC5_score_alone=roc_auc_score(T5.dead5, T5[k]),
                         AUC5_with_age_sex=roc_auc_score(T5.dead5, preds_a[k]),
                         Brier5_with_age_sex=brier_score_loss(T5.dead5, preds_a[k])))
    mfree = cox(tr, DEFICITS + BASE, pen)
    preds_c["Cox, 32 free coefficients"] = mfree.predict_partial_hazard(te).values
    lr32 = LogisticRegression(max_iter=3000, C=1.0).fit(tr[k5tr][feats], tr[k5tr].dead5)
    preds_a["Cox, 32 free coefficients"] = lr32.predict_proba(T5[feats])[:, 1]
    preds_a["M1 LightGBM"] = gbm.predict_proba(T5[feats])[:, 1]
    preds_a["M2 neural network (MLP)"] = mlp.predict_proba(T5[feats])[:, 1]
    preds_c["M3 DeepSurv"] = np.exp(ds_pred(te))
    preds_a["M3 DeepSurv"] = ds_pred(T5)
    for k, kind in [("Cox, 32 free coefficients", "linear, all items"), ("M1 LightGBM", "ML ceiling"),
                    ("M2 neural network (MLP)", "deep ceiling"), ("M3 DeepSurv", "deep ceiling")]:
        rows.append(dict(model=k, kind=kind,
                         C_with_age_sex=concordance_index(te.time, -preds_c[k], te.died) if k in preds_c else None,
                         AUC5_with_age_sex=roc_auc_score(T5.dead5, preds_a[k]),
                         Brier5_with_age_sex=brier_score_loss(T5.dead5, preds_a[k]) if k != "M3 DeepSurv" else None))
    age_only = cox(tr, BASE)
    lr_age = LogisticRegression(max_iter=2000).fit(tr[k5tr][BASE], tr[k5tr].dead5)
    rows.append(dict(model="Age and sex only", kind="reference",
                     C_with_age_sex=concordance_index(te.time, -age_only.predict_partial_hazard(te), te.died),
                     AUC5_with_age_sex=roc_auc_score(T5.dead5, lr_age.predict_proba(T5[BASE])[:, 1]),
                     Brier5_with_age_sex=brier_score_loss(T5.dead5, lr_age.predict_proba(T5[BASE])[:, 1])))
    R = pd.DataFrame(rows)

    # paired bootstrap of differences vs S0 on the test set
    step("Bootstrap confidence intervals (300 resamples per comparison)")
    s0 = "S0 equal (Rockwood)"
    tt, te_d, tev = te.time.values, te.died.values, None
    ci = {}
    for k in list(preds_c):
        if k == s0: continue
        f = lambda i, k=k: concordance_index(tt[i], -preds_c[k][i], te_d[i]) - concordance_index(tt[i], -preds_c[s0][i], te_d[i])
        ci[(k, "dC_with_age_sex")] = boot_diff(f, len(te), 300)
    y5 = T5.dead5.values
    for k in list(preds_a):
        if k == s0: continue
        f = lambda i, k=k: roc_auc_score(y5[i], preds_a[k][i]) - roc_auc_score(y5[i], preds_a[s0][i])
        ci[(k, "dAUC5_with_age_sex")] = boot_diff(f, len(T5), 300)
    for k in schemes[1:]:
        f = lambda i, k=k: concordance_index(tt[i], -te[k].values[i], te_d[i]) - concordance_index(tt[i], -te[s0].values[i], te_d[i])
        ci[(k, "dC_score_alone")] = boot_diff(f, len(te), 300)
    base_row = R.set_index("model").loc[s0]
    for i, r in R.iterrows():
        for met, col in [("dC_with_age_sex", "C_with_age_sex"), ("dAUC5_with_age_sex", "AUC5_with_age_sex"), ("dC_score_alone", "C_score_alone")]:
            if pd.notna(r.get(col)) and r.model != s0:
                R.loc[i, met] = r[col] - base_row[col]
                if (r.model, met) in ci:
                    R.loc[i, met + "_lo"], R.loc[i, met + "_hi"] = ci[(r.model, met)]

    # weights table (scaled so the mean weight of S1/S1+ is 1, comparable with Rockwood's 1)
    WT = pd.DataFrame({k: v / v[v > 0].mean() if (v > 0).any() else v for k, v in W.items()})
    WT.insert(0, "label", [LABEL[d] for d in WT.index]); WT.insert(1, "domain", [DOMAIN[d] for d in WT.index])
    WT["cluster"] = cl
    WT["share_of_total_S0_pct"] = 100 / 32
    for k in ["S1+ non-negative Cox weights", "S2 cluster-balanced"]:
        WT[k.split()[0] + "_share_pct"] = 100 * W[k].clip(lower=0) / W[k].clip(lower=0).sum()
    WT.round(4).to_csv(os.path.join(RES, "stage2_weights.csv"))
    R.round(4).to_csv(os.path.join(RES, "stage2_performance.csv"), index=False)
    # domain share of the total weight under each scheme
    dom = pd.DataFrame({k: (W[k].clip(lower=0) / W[k].clip(lower=0).sum()).groupby(pd.Series(DOMAIN)).sum() * 100 for k in W}).round(1)
    dom.to_csv(os.path.join(RES, "stage2_domain_share.csv"))
    json.dump(info, open(os.path.join(RES, "stage2_info.json"), "w"), indent=1); step("Done. Results written")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(json.dumps(info, indent=1))
    print(R.round(4).to_string())
    print(dom.to_string())
    print(WT.sort_values("S1 Cox weights", ascending=False).round(2).to_string())
    # keep test-set predictions for stage 3 (cache, not published)
    pd.to_pickle(dict(W=W, cluster=cl, pen=pen, gbm=gbm, mlp=mlp, feats=feats), os.path.join(os.path.dirname(RES), "..", ".cache", "weighting", "stage2_models.pkl"))
