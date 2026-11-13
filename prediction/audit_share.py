"""Pre-publication audit of the SHARE models: independent checks of the split, outcome, and reported performance."""
import pickle, json, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import step2_models as S
warnings.filterwarnings("ignore")
A, feats = S.load()
M = pickle.load(open("../.cache/prediction/step2_cls_models.pkl", "rb"))
gbm, ref = M["models"]["M2 LightGBM"], M["models"]["M0 reference (age, sex, FI)"]
print("1. SPLIT")
tr_p, te_p = set(A[A.set.isin(["train", "unused"])].mergeid), set(A[A.set.isin(["internal", "temporal"])].mergeid)
print("   people in both train and test:", len(tr_p & te_p), "| duplicate person-window rows:", int(A.duplicated(["mergeid", "window"]).sum()))
print("   countries in data:", A.country.nunique(), "| inputs that are not baseline measures:", [f for f in feats if f in ("fi_f", "t1", "years", "status", "frail_f", "death_date")])
A["obs"] = A.status.isin(["fi", "dead"])
D = A[(A.at_risk == 1) & A.obs].copy()
D["p"] = gbm.predict_proba(D[feats])[:, 1]; D["p0"] = ref.predict_proba(D[ref.cols])[:, 1]
it, tt = D[D.set == "internal"], D[D.set == "temporal"]
print("   at-risk rows with baseline FI >= 0.25:", int((D.fi >= 0.25).sum()), "| countries with outcomes:", D.country.nunique())
print("2. OUTCOME MAKE-UP (at risk, observed)")
for w, g in D.groupby("window"):
    print(f"   {w}: n {len(g):,}, frail-or-dead {g.frail_or_dead.mean():.1%} (dead {(g.status=='dead').mean():.1%}, frail {((g.status=='fi')&(g.frail_or_dead==1)).mean():.1%}), years {g.years_planned.median():.1f}")
AA = A[A.at_risk == 1]
print("   lost to follow-up among at-risk:", {w: f"{(g.status=='lost').mean():.0%}" for w, g in AA.groupby("window")})
print("3. DISCRIMINATION, SPLIT UP (internal test)")
auc = lambda g, c="p": roc_auc_score(g.frail_or_dead, g[c])
print(f"   pooled {auc(it):.3f} (ref {auc(it,'p0'):.3f})")
for w, g in it.groupby("window"): print(f"   {w}: model {auc(g):.3f}, reference {auc(g,'p0'):.3f}, gain {auc(g)-auc(g,'p0'):+.3f}")
s = it[it.status == "fi"]; print(f"   frailty only (survivors): model {auc(s):.3f}, reference {auc(s,'p0'):.3f}")
d = it.assign(frail_or_dead=(it.status == "dead").astype(int)); print(f"   death only: model {auc(d):.3f}, reference {auc(d,'p0'):.3f}")
for lab, m in [("age <65", it.age < 65), ("65-74", it.age.between(65, 74.99)), ("75+", it.age >= 75), ("men", it.female == 0), ("women", it.female == 1)]:
    g = it[m]; print(f"   {lab}: n {len(g):,}, events {g.frail_or_dead.mean():.1%}, AUC {auc(g):.3f} (ref {auc(g,'p0'):.3f}), pred {g.p.mean():.3f} vs obs {g.frail_or_dead.mean():.3f}")
print("4. PERSON-CLUSTERED BOOTSTRAP (500 resamples of people)")
rng = np.random.default_rng(1)
for name, T in [("internal", it), ("temporal", tt)]:
    idx = {k: v.values for k, v in T.groupby("mergeid").indices.items()} if False else T.groupby("mergeid").indices
    ids = np.array(list(idx)); y, p, p0 = T.frail_or_dead.values, T.p.values, T.p0.values; a, g = [], []
    for _ in range(500):
        rows = np.concatenate([idx[i] for i in rng.choice(ids, len(ids))])
        a.append(roc_auc_score(y[rows], p[rows])); g.append(a[-1] - roc_auc_score(y[rows], p0[rows]))
    print(f"   {name}: AUC {roc_auc_score(y,p):.3f} [{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]; gain {roc_auc_score(y,p)-roc_auc_score(y,p0):+.3f} [{np.percentile(g,2.5):+.3f}, {np.percentile(g,97.5):+.3f}]; people {len(ids):,}, rows {len(T):,}")
print("5. SHUFFLED-LABEL TEST (should be about 0.50)")
tr = D[D.set == "train"]; ys = tr.frail_or_dead.sample(frac=1, random_state=3).values
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=1).fit(tr[feats], ys)
print(f"   AUC on internal test after shuffling training labels: {roc_auc_score(it.frail_or_dead, m.predict_proba(it[feats])[:,1]):.3f}")
print("6. WITHOUT THE FRAILTY INDEX AND ITS 32 ITEMS (how much rests on current frailty)")
f2 = [f for f in feats if f != "fi" and not f.startswith("d_")]
m2 = lgb.LGBMClassifier(subsample_freq=1, verbose=-1, random_state=S.SEED, **M["tuning"]["lgbm"][1]).fit(tr[f2], tr.frail_or_dead)
print(f"   AUC without FI and deficits: {roc_auc_score(it.frail_or_dead, m2.predict_proba(it[f2])[:,1]):.3f}")
print("7. REGRESSION MODEL: survivors only")
R = pickle.load(open("../.cache/prediction/step2_reg_models.pkl", "rb"))["models"]["M2 LightGBM"]
G = A[(A.status == "fi") & (A.set == "internal")].copy(); G["pr"] = R.predict(G[feats])
print(f"   mean predicted {G.pr.mean():.4f} vs observed {G.fi_f.mean():.4f}; error by baseline FI third:", {str(k): round(float((g.pr - g.fi_f).mean()), 4) for k, g in G.groupby(pd.qcut(G.fi, 3))})
print(f"   predicting CHANGE in FI: R2 {1 - ((G.fi_f - G.pr)**2).sum() / (((G.fi_f - G.fi) - (G.fi_f - G.fi).mean())**2).sum():.3f}")
