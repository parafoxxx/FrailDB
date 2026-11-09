"""
Stage 3: is any gain from reweighting real and transportable?

  A. External validation in ELSA (England): weights learned on the SHARE training set are applied unchanged to ELSA
     wave 1 (2002-03). Outcomes: death before wave 3 (~4 years) and before wave 6 sampling (~10 years).
     AUC of each index alone, and of age + sex + index (logistic, 5-fold cross-validated within ELSA);
     SHARE-trained LightGBM and neural network transported as they are. Paired bootstrap against equal weights.
  B. Stability: Cox weights re-learned leaving out one SHARE country at a time; spread of each weight, rank
     agreement between the runs, and the gain over equal weights in each left-out country.
  C. Age and sex (secondary check of 'one set of weights'): weights learned separately for women and men, and for
     ages < 70 and 70+; likelihood-ratio tests of deficit x sex and deficit x age interactions.
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from stage1_associations import load, LABEL
from stage2_weights import cox, index_from, BASE, SEED, RES
from prepare import DEFICITS, OUT as CACHE

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
import time as _t
_T0 = _t.time()
def step(m): print(f"[{(_t.time() - _T0) / 60:5.1f} min] {m}", flush=True)
step("Loading models and ELSA")
M = pd.read_pickle(os.path.join(CACHE, "stage2_models.pkl"))
W, pen = M["W"], M["pen"]
S0 = "S0 equal (Rockwood)"
out = {}

# ------------------------------------------------------------------ A. ELSA
el = pd.read_pickle(os.path.join(CACHE, "elsa_cohort.pkl")).dropna(subset=["age", "sex"] + DEFICITS).copy()
el["agec"] = (el.age - 65) / 10; el["agec2"] = el.agec ** 2; el["female"] = (el.sex == 2).astype(int)
el[DEFICITS] = el[DEFICITS].astype(float)
for k in W:
    el[k] = index_from(W[k], el)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for y in ["dead4", "dead10"]:
    step(f"ELSA: scoring and bootstrapping outcome {y}")
    P = {}
    for k in W:
        P[k + " | alone"] = el[k].values
        P[k + " | age+sex"] = cross_val_predict(LogisticRegression(max_iter=2000), el[[k] + BASE], el[y], cv=skf, method="predict_proba")[:, 1]
    P["M1 LightGBM (SHARE-trained) | age+sex"] = M["gbm"].predict_proba(el[M["feats"]])[:, 1]
    P["M2 neural network (SHARE-trained) | age+sex"] = M["mlp"].predict_proba(el[M["feats"]])[:, 1]
    P["Age and sex only | age+sex"] = cross_val_predict(LogisticRegression(max_iter=2000), el[BASE], el[y], cv=skf, method="predict_proba")[:, 1]
    yy = el[y].values
    for name, p in P.items():
        model, how = name.split(" | ")
        ref = P[S0 + " | " + how] if (S0 + " | " + how) in P else P[S0 + " | age+sex"]
        auc = roc_auc_score(yy, p)
        boots = []
        for _ in range(500):
            i = rng.integers(0, len(yy), len(yy))
            boots.append(roc_auc_score(yy[i], p[i]) - roc_auc_score(yy[i], ref[i]))
        rows.append(dict(outcome=y, model=model, predictors=how, AUC=auc, dAUC_vs_equal=auc - roc_auc_score(yy, ref),
                         lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5)))
A = pd.DataFrame(rows)
step("ELSA done: " + "; ".join(f"{r.outcome} {r.model.split()[0]} {r.predictors} AUC {r.AUC:.4f}" for r in A.itertuples() if r.model.startswith("S0") or r.model.startswith("S1+")))
A.round(4).to_csv(os.path.join(RES, "stage3_elsa_external.csv"), index=False)
out["elsa"] = dict(n=int(len(el)), deaths4=int(el.dead4.sum()), deaths10=int(el.dead10.sum()))

# ------------------------------------------------------------------ B. leave-one-country-out stability
df = load()
countries = df.country.value_counts()
countries = countries[countries >= 1000].index.tolist()
lw, gains = [], []
for n_, c in enumerate(countries, 1):
    tr, te = df[df.country != c], df[df.country == c]
    m = cox(tr, DEFICITS + BASE, pen)
    w = m.params_[DEFICITS]
    lw.append(w.rename(c))
    te = te.copy()
    s_eq, s_wt = index_from(W[S0], te), index_from(w, te)
    ce, cw = concordance_index(te.time, -s_eq, te.died), concordance_index(te.time, -s_wt, te.died)
    gains.append(dict(country=int(c), n=len(te), deaths=int(te.died.sum()), C_equal=ce, C_weighted=cw, dC=cw - ce))
    step(f"Leave-out country {n_}/{len(countries)} (code {int(c)}): C equal {ce:.4f} -> weighted {cw:.4f}")
LW = pd.concat(lw, axis=1)
stab = pd.DataFrame({"label": [LABEL[d] for d in DEFICITS], "median_weight": LW.median(axis=1), "min": LW.min(axis=1),
                     "max": LW.max(axis=1), "positive_in_pct_of_runs": 100 * (LW > 0).mean(axis=1)}, index=DEFICITS)
rank_r = LW.corr(method="spearman").values[np.triu_indices(len(countries), 1)]
G = pd.DataFrame(gains)
stab.round(4).to_csv(os.path.join(RES, "stage3_weight_stability.csv"))
G.round(4).to_csv(os.path.join(RES, "stage3_country_gain.csv"), index=False)
out["loco"] = dict(countries=len(countries), weight_rank_corr_median=round(float(np.median(rank_r)), 3),
                   weight_rank_corr_min=round(float(rank_r.min()), 3), dC_median=round(float(G.dC.median()), 4),
                   dC_min=round(float(G.dC.min()), 4), dC_max=round(float(G.dC.max()), 4),
                   countries_improved=int((G.dC > 0).sum()))

# ------------------------------------------------------------------ C. age and sex
step("Fitting weights by sex and age group"); full = cox(df, DEFICITS + BASE, pen)
groups = {"women": df.female == 1, "men": df.female == 0, "age <70": df.age < 70, "age 70+": df.age >= 70}
GW = {g: cox(df[mask], DEFICITS + [c for c in BASE if df.loc[mask, c].nunique() > 1], pen).params_[DEFICITS] for g, mask in groups.items()}
GW = pd.DataFrame(GW); GW["overall"] = full.params_[DEFICITS]
lr_tests = {}
for name, var in [("sex", "female"), ("age 70+", "old")]:
    d = df.copy(); d["old"] = (d.age >= 70).astype(int)
    inter = []
    for x in DEFICITS:
        d[x + "_x"] = d[x] * d[var]; inter.append(x + "_x")
    step(f"Interaction test: deficit x {name}")
    m0 = cox(d, DEFICITS + BASE + (["old"] if var == "old" else []), pen)
    m1 = cox(d, DEFICITS + BASE + (["old"] if var == "old" else []) + inter, pen)
    lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    p1 = m1.predict_partial_hazard(d); p0 = m0.predict_partial_hazard(d)
    lr_tests[name] = dict(LR_chi2=round(float(lr), 1), df=len(inter), p=float(stats.chi2.sf(lr, len(inter))),
                          C_without=round(float(concordance_index(d.time, -p0, d.died)), 4),
                          C_with_interactions=round(float(concordance_index(d.time, -p1, d.died)), 4))
GW.insert(0, "label", [LABEL[d] for d in GW.index])
GW.round(4).to_csv(os.path.join(RES, "stage3_weights_by_group.csv"))
out["groups"] = dict(weight_corr_with_overall={g: round(float(np.corrcoef(GW[g], GW["overall"])[0, 1]), 3) for g in groups},
                     women_vs_men=round(float(np.corrcoef(GW["women"], GW["men"])[0, 1]), 3),
                     young_vs_old=round(float(np.corrcoef(GW["age <70"], GW["age 70+"])[0, 1]), 3),
                     interaction_tests=lr_tests)
json.dump(out, open(os.path.join(RES, "stage3_summary.json"), "w"), indent=1); step("Done. Results written")
pd.set_option("display.width", 250)
print(json.dumps(out, indent=1))
print(A.round(4).to_string())
print(G.round(4).to_string())
print(stab.sort_values("median_weight", ascending=False).round(3).to_string())
