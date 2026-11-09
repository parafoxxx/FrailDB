"""
Stage 1: how each of the 32 deficits relates to mortality, and how the deficits relate to each other (SHARE).

  - marginal HR: Cox model with one deficit + age + age^2 + sex (what each deficit says on its own)
  - adjusted HR: one Cox model with all 32 deficits + age + age^2 + sex (what each adds beyond the others)
  - redundancy: share of a deficit's marginal log-HR that disappears once the other deficits are in the model
  - correlation structure: phi coefficients, hierarchical clusters, effective number of independent deficits (Li-Ji)

Outputs (aggregates only) go to weighting/output/.
"""
import os, json, warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from prepare import DEFICITS, OUT as CACHE

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "output")
LABEL = {"d_srh": "Self-rated health fair/poor", "d_hypertension": "Hypertension", "d_diabetes": "Diabetes",
         "d_cancer": "Cancer", "d_lung": "Chronic lung disease", "d_heart": "Heart disease", "d_stroke": "Stroke",
         "d_arthritis": "Arthritis", "d_adl_dress": "ADL: dressing", "d_adl_bath": "ADL: bathing",
         "d_adl_eat": "ADL: eating", "d_adl_toilet": "ADL: toilet", "d_iadl_meal": "IADL: hot meal",
         "d_iadl_shop": "IADL: shopping", "d_hearing": "Hearing", "d_depression": "Depressive symptoms",
         "d_mob_walk": "Walking 100 m", "d_mob_chair": "Getting up from chair", "d_mob_stoop": "Stooping/kneeling",
         "d_mob_lift": "Lifting 5 kg", "d_adl_bed": "ADL: in/out of bed", "d_vision": "Vision",
         "d_adl_walkroom": "ADL: walking across room", "d_mob_sit2h": "Sitting 2 hours",
         "d_mob_stairs_several": "Several flights of stairs", "d_mob_stairs_one": "One flight of stairs",
         "d_mob_arms": "Reaching above shoulder", "d_mob_push": "Pushing/pulling objects", "d_mob_coin": "Picking up coin",
         "d_iadl_meds": "IADL: medications", "d_iadl_money": "IADL: money", "d_dementia": "Dementia"}
DOMAIN = {"d_srh": "General health", "d_hypertension": "Disease", "d_diabetes": "Disease", "d_cancer": "Disease",
          "d_lung": "Disease", "d_heart": "Disease", "d_stroke": "Disease", "d_arthritis": "Disease", "d_dementia": "Disease",
          "d_hearing": "Senses", "d_vision": "Senses", "d_depression": "Mood"}
for d in DEFICITS:
    DOMAIN.setdefault(d, "ADL" if d.startswith("d_adl") else "IADL" if d.startswith("d_iadl") else "Mobility")


def load():
    df = pd.read_pickle(os.path.join(CACHE, "share_cohort.pkl"))
    df = df.dropna(subset=["age", "sex"] + DEFICITS).copy()
    df["agec"] = (df.age - 65) / 10
    df["agec2"] = df.agec ** 2
    df["female"] = (df.sex == 2).astype(int)
    df[DEFICITS] = df[DEFICITS].astype(float)
    return df


def cox(df, cols, pen=0.0):
    m = CoxPHFitter(penalizer=pen)
    m.fit(df[cols + ["time", "died"]], duration_col="time", event_col="died", fit_options={"step_size": 0.5})
    return m


if __name__ == "__main__":
    df = load()
    base = ["agec", "agec2", "female"]
    rows = []
    # marginal (age-sex adjusted)
    for d in DEFICITS:
        m = cox(df, [d] + base)
        s = m.summary.loc[d]
        rows.append(dict(deficit=d, label=LABEL[d], domain=DOMAIN[d], prevalence_pct=round(100 * df[d].mean(), 2),
                         marginal_logHR=s["coef"], marginal_HR=np.exp(s["coef"]),
                         marginal_lo=np.exp(s["coef lower 95%"]), marginal_hi=np.exp(s["coef upper 95%"])))
    R = pd.DataFrame(rows).set_index("deficit")
    # mutually adjusted
    m = cox(df, DEFICITS + base)
    s = m.summary
    R["adjusted_logHR"] = s.loc[DEFICITS, "coef"]
    R["adjusted_HR"] = np.exp(s.loc[DEFICITS, "coef"])
    R["adjusted_lo"] = np.exp(s.loc[DEFICITS, "coef lower 95%"])
    R["adjusted_hi"] = np.exp(s.loc[DEFICITS, "coef upper 95%"])
    R["adjusted_p"] = s.loc[DEFICITS, "p"]
    # redundancy: how much of the marginal effect is shared with other deficits
    R["retained_pct"] = 100 * R.adjusted_logHR / R.marginal_logHR
    # equal-weight FI model for reference (per 0.1 FI)
    df["fi10"] = df[DEFICITS].mean(axis=1) * 10
    mfi = cox(df, ["fi10"] + base)
    # correlation structure
    C = df[DEFICITS].corr().values
    np.fill_diagonal(C, 1)
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    meff_liji = float(np.sum((eig >= 1) + (eig - np.floor(eig))))            # Li & Ji 2005
    Z = linkage(squareform(1 - np.abs(C), checks=False), method="average")
    clusters = fcluster(Z, t=0.7, criterion="distance")                         # |r| >= ~0.3 on average
    R["cluster"] = clusters
    order = [DEFICITS[i] for i in leaves_list(Z)]
    # each deficit's strongest correlates
    Cdf = pd.DataFrame(C, index=DEFICITS, columns=DEFICITS)
    R["mean_abs_r_others"] = [(Cdf.loc[d].drop(d).abs()).mean() for d in DEFICITS]
    R["top_correlates"] = ["; ".join(f"{LABEL[k]} ({v:.2f})" for k, v in Cdf.loc[d].drop(d).sort_values(ascending=False).head(3).items()) for d in DEFICITS]
    R = R.sort_values("marginal_HR", ascending=False)
    R.round(4).to_csv(os.path.join(RES, "stage1_deficit_mortality.csv"))
    Cdf.loc[order, order].round(3).to_csv(os.path.join(RES, "stage1_correlation_matrix_ordered.csv"))
    summ = dict(n=int(len(df)), deaths=int(df.died.sum()), person_years=round(float(df.time.sum())),
                fi_HR_per_0_1=round(float(np.exp(mfi.summary.loc["fi10", "coef"])), 3),
                fi_C=round(float(mfi.concordance_index_), 4), full32_C=round(float(m.concordance_index_), 4),
                eigen_top5=[round(float(e), 2) for e in eig[:5]], var_explained_pc1=round(float(eig[0] / len(eig)), 3),
                effective_independent_deficits_LiJi=round(meff_liji, 1), n_clusters=int(clusters.max()),
                domain_items={k: int(v) for k, v in pd.Series(DOMAIN).value_counts().items()})
    json.dump(summ, open(os.path.join(RES, "stage1_summary.json"), "w"), indent=1)
    pd.set_option("display.width", 220)
    print(json.dumps(summ, indent=1))
    print(R[["label", "domain", "prevalence_pct", "marginal_HR", "adjusted_HR", "adjusted_lo", "adjusted_hi", "retained_pct", "cluster", "mean_abs_r_others"]].round(2).to_string())
    for c in sorted(set(clusters)):
        print("cluster", c, [LABEL[d] for d in DEFICITS if R.loc[d, "cluster"] == c])
