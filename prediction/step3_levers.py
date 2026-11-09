"""
Step 3 (Part B): what happens to 4-6-year risk when a person CHANGES a modifiable factor.

Design: a target-trial emulation per lever (Hernan & Robins 2016) on three consecutive SHARE waves.
  T0 (baseline: confounders)  ->  T1 (did the person change the lever?)  ->  follow-up wave (outcome)
  wave triplets: w2 -> w4 -> w6,  w4 -> w5 -> w8,  w5 -> w6 -> w8   (a person can enter more than one triplet)
Eligible: not frail at T0 (FI < 0.25) and in the 'unhealthy' state of the lever at T0 (e.g. a current smoker).
Treatment A = 1 if the person moved to the healthy state by T1 (e.g. quit), 0 if not.
Outcome Y = frail (FI >= 0.25) or dead at the follow-up wave; secondary outcome: death.
Reverse-causation guard: people who became frail, or got a new cancer, stroke, heart disease, lung disease or
dementia diagnosis between T0 and T1, are excluded (illness makes people change behaviour, e.g. 'sick quitters').
Answers carried forward from an earlier wave (question not asked at T1) never count as a change.

Estimator: augmented inverse-probability weighting (AIPW, doubly robust), with 5-fold cross-fitting grouped by person;
nuisance models (treatment, loss to follow-up, outcome per arm) are an average of LightGBM and a ridge logistic model.
People lost to follow-up are handled by inverse-probability-of-censoring weighting inside AIPW.
95% CIs from the influence function, clustered by person. E-value for the risk ratio (VanderWeele & Ding 2017).
Sensitivity: also adjusting for health measured at T1. Subgroups: age < 70 / 70+, men / women.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import step2_models as S
from prepare import OUT as CACHE

warnings.filterwarnings("ignore")
T0 = time.time()
def step(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)
RES = S.RES
TRIPLETS = [(2, 4), (4, 5), (5, 6)]
MAJOR = ["d_cancer", "d_stroke", "d_heart", "d_lung", "d_dementia"]

# lever contrasts: (key, label, eligible at T0, treated at T1, triplets allowed, carried-flag column at T1)
LEVERS = [
    ("quit_smoking", "Stopped smoking (vs kept smoking)", lambda d: d.smoke_now_0 == 1, lambda d: d.smoke_now_1 == 0, TRIPLETS, "smoke_now_carried_1"),
    ("start_vigorous", "Took up vigorous activity at least weekly (vs stayed below weekly)", lambda d: d.vigorous_freq_0 >= 3, lambda d: d.vigorous_freq_1 <= 2, TRIPLETS, None),
    ("leave_inactivity", "Inactive -> became at least moderately active", lambda d: d.inactive_0 == 1, lambda d: d.inactive_1 == 0, TRIPLETS, None),
    ("stay_active", "Active -> stayed active (vs became inactive)", lambda d: d.inactive_0 == 0, lambda d: d.inactive_1 == 0, TRIPLETS, None),
    ("cut_heavy_drinking", "Heavy drinker (>14 drinks/wk) -> cut to 14 or fewer", lambda d: d.alcohol_units_wk_0 > 14, lambda d: d.alcohol_units_wk_1 <= 14, [(2, 4), (4, 5)], None),
    ("obese_lost_weight", "Obese -> lost 5% or more of BMI", lambda d: d.bmi_0 >= 30, lambda d: d.bmi_1 <= 0.95 * d.bmi_0, TRIPLETS, None),
    ("daily_fruitveg", "Fruit/veg less than daily -> daily", lambda d: d.fruitveg_freq_0 >= 2, lambda d: d.fruitveg_freq_1 == 1, [(4, 5), (5, 6)], "fruitveg_freq_carried_1"),
    ("more_protein", "Meat/fish/poultry < 3 days/wk -> 3+ days/wk", lambda d: d.protein_freq_0 >= 3, lambda d: d.protein_freq_1 <= 2, [(4, 5), (5, 6)], "protein_freq_carried_1"),
    ("join_social", "No social activities -> took part in at least one", lambda d: d.social_activities_0 == 0, lambda d: d.social_activities_1 > 0, TRIPLETS, None),
    ("less_lonely", "Lonely (score 6+) -> not lonely (5 or less)", lambda d: d.loneliness_0 >= 6, lambda d: d.loneliness_1 <= 5, [(5, 6)], None),
    ("got_hearing_aid", "Poor hearing, no aid -> started using a hearing aid", lambda d: (d.d_hearing_0 == 1) & (d.hearing_aid_0 == 0), lambda d: d.hearing_aid_1 == 1, TRIPLETS, None),
    ("sleep_resolved", "Sleep trouble -> sleep trouble resolved", lambda d: d.sleep_trouble_0 == 1, lambda d: d.sleep_trouble_1 == 0, TRIPLETS, None),
]


def build():
    A = pd.read_pickle(os.path.join(CACHE, "share_windows.pkl"))
    A["female"] = (A.sex == 2).astype(float)
    base = S.FIXED + S.LEVERS
    keep0 = sorted(set(base + ["mergeid", "country"]) | {c for c in A.columns if c.endswith("_carried")})
    out = []
    for t0, t1 in TRIPLETS:
        a = A[A.wave_b == t0][keep0].add_suffix("_0").rename(columns={"mergeid_0": "mergeid"})
        b = A[A.wave_b == t1][keep0 + ["status", "fi_f", "years", "t0", "t1"]].add_suffix("_1").rename(columns={"mergeid_1": "mergeid"})
        d = a.merge(b, on="mergeid")
        d["triplet"] = f"w{t0}->w{t1}"; d["t0w"], d["t1w"] = t0, t1
        out.append(d)
    D = pd.concat(out, ignore_index=True)
    D = D.apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name not in ("mergeid", "triplet", "status_1") else s)
    D["gap_years"] = D.age_1 - D.age_0
    D["R"] = (D.status_1 != "lost").astype(int)
    D["Y"] = np.where(D.status_1 == "dead", 1.0, np.where(D.status_1 == "fi", (D.fi_f_1 >= 0.25).astype(float), np.nan))
    D["Y_dead"] = np.where(D.status_1 == "lost", np.nan, (D.status_1 == "dead").astype(float))
    new_major = np.zeros(len(D), bool)
    for c in MAJOR:
        new_major |= ((D[c + "_0"] == 0) & (D[c + "_1"] == 1)).values
    D["very_ill"] = new_major | (D.fi_1 >= 0.25).values
    for c in sorted(D.country_0.dropna().unique()):
        D[f"ctry_{int(c)}"] = (D.country_0 == c).astype(float)
    for t in D.triplet.unique():
        D[f"trip_{t}"] = (D.triplet == t).astype(float)
    X0 = [c + "_0" for c in base] + ["age_1", "gap_years"] + [c for c in D.columns if c.startswith(("ctry_", "trip_"))]
    XT1 = ["fi_1", "d_srh_1", "gali_1", "eurod_1", "maxgrip_1"]
    return D, X0, XT1


def learner():
    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.6, verbose=-1, random_state=S.SEED, n_jobs=4)
    l = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    return g, l


def fit_pred(X, y, Xnew):
    """Average of LightGBM and ridge-logistic predicted probabilities (falls back to the mean if y is constant)."""
    if y.nunique() < 2: return np.full(len(Xnew), float(y.mean()))
    g, l = learner()
    return 0.5 * g.fit(X, y).predict_proba(Xnew)[:, 1] + 0.5 * l.fit(X, y).predict_proba(Xnew)[:, 1]


def aipw(d, X, ycols):
    """Cross-fitted AIPW for risks under A=1 and A=0 with outcome missing at random given (A, X).
    Returns per-row influence-function terms phi1, phi0 for every outcome in ycols."""
    n = len(d); e = np.zeros(n); c = np.zeros(n); m = {y: np.zeros((n, 2)) for y in ycols}
    Xa = d[X].copy(); Xa["A"] = d.A.values
    for tr, te in GroupKFold(5).split(d, groups=d.mergeid):
        dt = d.iloc[tr]
        e[te] = fit_pred(d[X].iloc[tr], dt.A, d[X].iloc[te])
        c[te] = fit_pred(Xa.iloc[tr], dt.R, Xa.iloc[te])
        for y in ycols:
            for a in (0, 1):
                s = dt[(dt.A == a) & (dt.R == 1) & dt[y].notna()]
                m[y][te, a] = fit_pred(s[X], s[y], d[X].iloc[te])
    e = np.clip(e, 0.01, 0.99); c = np.clip(c, 0.05, 1)
    A, R = d.A.values, d.R.values
    out = {}
    for y in ycols:
        Y = np.nan_to_num(d[y].values)
        phi1 = m[y][:, 1] + A * R / (e * c) * (Y - m[y][:, 1])
        phi0 = m[y][:, 0] + (1 - A) * R / ((1 - e) * c) * (Y - m[y][:, 0])
        out[y] = (phi1, phi0)
    return out, e


def summarise(phi1, phi0, groups):
    n = len(phi1); r1, r0 = phi1.mean(), phi0.mean()
    def se(v):
        s = pd.Series(v - v.mean()).groupby(groups.values).sum()
        return float(np.sqrt((s ** 2).sum()) / n)
    se_rd = se(phi1 - phi0)
    if r1 > 0 and r0 > 0:
        lrr = np.log(r1 / r0); se_lrr = se(phi1 / r1 - phi0 / r0)
    else:
        lrr, se_lrr = np.nan, np.nan
    rd = r1 - r0
    return dict(risk_changed=r1, risk_not_changed=r0, RD=rd, RD_lo=rd - 1.96 * se_rd, RD_hi=rd + 1.96 * se_rd,
                RR=np.exp(lrr), RR_lo=np.exp(lrr - 1.96 * se_lrr), RR_hi=np.exp(lrr + 1.96 * se_lrr),
                p=float(2 * (1 - __import__("scipy.stats").stats.norm.cdf(abs(rd / se_rd)))) if se_rd > 0 else np.nan)


def evalue(rr, lo, hi):
    """E-value for the point estimate and for the confidence limit closest to 1 (VanderWeele & Ding 2017)."""
    f = lambda r: r + np.sqrt(r * (r - 1)) if r >= 1 else np.nan
    if np.isnan(rr): return np.nan, np.nan
    if rr < 1: rr, lim = 1 / rr, (1 / hi if hi < 1 else 1.0)
    else: lim = lo if lo > 1 else 1.0
    return f(rr), f(lim)


if __name__ == "__main__":
    D, X0, XT1 = build()
    step(f"Built {len(D):,} person-triplets ({D.mergeid.nunique():,} people): {D.triplet.value_counts().to_dict()}")
    elig_all = (D.fi_0 < 0.25)
    step(f"Not frail at T0: {elig_all.sum():,}; excluded as very ill between T0 and T1: {(elig_all & D.very_ill).sum():,}")
    rows, sub = [], []
    for k, (key, label, el, tr, trips, carried) in enumerate(LEVERS):
        m = elig_all & ~D.very_ill & el(D) & D.triplet.isin([f"w{a}->w{b}" for a, b in trips])
        d = D[m].copy()
        if carried: d = d[d[carried].fillna(0) == 0]
        lever_cols = {"quit_smoking": "smoke_now_1", "start_vigorous": "vigorous_freq_1", "leave_inactivity": "inactive_1", "stay_active": "inactive_1",
                      "cut_heavy_drinking": "alcohol_units_wk_1", "obese_lost_weight": "bmi_1", "daily_fruitveg": "fruitveg_freq_1",
                      "more_protein": "protein_freq_1", "join_social": "social_activities_1", "less_lonely": "loneliness_1",
                      "got_hearing_aid": "hearing_aid_1", "sleep_resolved": "sleep_trouble_1"}
        d = d[d[lever_cols[key]].notna()].reset_index(drop=True)
        d["A"] = tr(d).astype(float)
        nA = int(d.A.sum())
        step(f"[{k + 1}/{len(LEVERS)}] {label}: {len(d):,} eligible ({d.mergeid.nunique():,} people), {nA:,} changed "
             f"({100 * d.A.mean():.1f}%), lost to follow-up {100 * (1 - d.R.mean()):.1f}%")
        if nA < 100 or len(d) - nA < 100:
            step("    too few in one arm; skipped"); continue
        obs = d[d.R == 1]
        crude1, crude0 = obs[obs.A == 1].Y.mean(), obs[obs.A == 0].Y.mean()
        for spec, X in [("primary (T0 adjustment)", X0), ("sensitivity (+ health at T1)", X0 + XT1)]:
            res, e = aipw(d, X, ["Y", "Y_dead"])
            for y, ylab in [("Y", "frail or dead"), ("Y_dead", "dead")]:
                s = summarise(*res[y], d.mergeid)
                ev, evl = evalue(s["RR"], s["RR_lo"], s["RR_hi"])
                rows.append(dict(lever=key, label=label, outcome=ylab, spec=spec, n=len(d), people=d.mergeid.nunique(), n_changed=nA,
                                 events=int(obs[y].sum()), crude_changed=obs[obs.A == 1][y].mean(), crude_not=obs[obs.A == 0][y].mean(),
                                 **s, E_value=ev, E_value_CI=evl, extreme_propensity_pct=100 * np.mean((e <= 0.01) | (e >= 0.99))))
                if spec.startswith("primary"):
                    for gname, gm in [("age < 70", d.age_1 < 70), ("age 70+", d.age_1 >= 70), ("men", d.female_0 == 0), ("women", d.female_0 == 1)]:
                        gm = gm.values
                        if gm.sum() > 200:
                            sub.append(dict(lever=key, outcome=ylab, subgroup=gname, n=int(gm.sum()), **summarise(res[y][0][gm], res[y][1][gm], d.mergeid[gm])))
            r = [x for x in rows if x["lever"] == key and x["spec"] == spec and x["outcome"] == "frail or dead"][0]
            step(f"    {spec}: risk {100 * r['risk_changed']:.1f}% if changed vs {100 * r['risk_not_changed']:.1f}% if not "
                 f"(crude {100 * crude1:.1f}% vs {100 * crude0:.1f}%); difference {100 * r['RD']:+.1f} points "
                 f"[{100 * r['RD_lo']:+.1f}, {100 * r['RD_hi']:+.1f}], RR {r['RR']:.2f} [{r['RR_lo']:.2f}, {r['RR_hi']:.2f}], E-value {r['E_value']:.2f}")
        pd.DataFrame(rows).round(4).to_csv(os.path.join(RES, "step3_lever_effects.csv"), index=False)
        pd.DataFrame(sub).round(4).to_csv(os.path.join(RES, "step3_lever_subgroups.csv"), index=False)
    step("Done")
