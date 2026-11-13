"""
ELSA: person-wave table for the cohort engine.

Frailty index and its 32 deficits: FrailDB's per-wave Rockwood files (waves 1, 3-10; wave 2 and 11 files are not on disk).
Other characteristics and levers: Harmonized ELSA (gh_elsa_h.dta, Gateway to Global Aging), waves 1-10.
Deaths: harmonised interview status (5 = died this wave, 6 = died in an earlier wave).
Windows: every pair of frailty waves 3.5 to 7.5 years apart; pairs ending at wave 10 (2021-23) form the later-period test.
Triplets for lever effects: consecutive frailty waves (about 2 years apart), then the outcome about 4 years later.
"""
import os, sys, pickle
import numpy as np, pandas as pd, pyreadr, pyreadstat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import CACHE, step

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Data Sources", "4. ELSA_England")
FI_DIR = os.path.join(ROOT, "ELSA Analysis")
H = os.path.join(ROOT, "ELSA data", "UKDA-5050-stata", "stata13_se", "gh_elsa_h.dta")
YEARS = {1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010, 6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2022}
RVARS = ["agey", "iwstat", "iwy", "iwm", "smoken", "smokev", "drinkd_e", "drinkwn_e", "vgactx_e", "mdactx_e", "ltactx_e",
         "mbmi", "lonely", "sleepr", "socyr", "mstat", "cesd", "painfr", "tr20", "orient", "ser7", "gripsum", "region_e"]
HVARS = ["hhres", "itot", "atotb"]
REGIONS = {1: "North East", 2: "North West", 3: "Yorkshire and The Humber", 4: "East Midlands", 5: "West Midlands",
           6: "East of England", 7: "London", 8: "South East", 9: "South West"}


def read_fi(w):
    b = os.path.join(FI_DIR, f"Wave {w}", f"ELSA_w{w}_Rockwood")
    if os.path.exists(b + "_summary.csv"): x = pd.read_csv(b + "_summary.csv", low_memory=False)
    elif os.path.exists(b + ".rds"): x = list(pyreadr.read_r(b + ".rds").values())[0]
    else: return None
    x = x.rename(columns={"idauniq": "pid", "frailty_index_score": "fi"})
    x["pid"] = pd.to_numeric(x.pid, errors="coerce").astype("int64")
    keep = ["pid", "age", "sex", "fi"] + [c for c in x.columns if c.startswith("d_")]
    x = x[keep].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    x["wave"] = w
    return x


def freq(v):
    """Harmonised activity frequency 2 (>1/week), 3 (weekly), 4 (1-3/month), 5 (hardly ever) -> 1..4 as in SHARE."""
    return (v - 1).where(v.between(2, 5))


if __name__ == "__main__":
    _, meta = pyreadstat.read_dta(H, metadataonly=True)
    have = set(meta.column_names)
    cols = ["idauniq", "raeducl"] + [f"r{w}{v}" for w in YEARS for v in RVARS if f"r{w}{v}" in have] + \
           [f"h{w}{v}" for w in YEARS for v in HVARS if f"h{w}{v}" in have]
    step(f"reading {len(cols)} columns from the Harmonized ELSA file")
    h, _ = pyreadstat.read_dta(H, usecols=cols)
    h = h.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    h = h.mask(h < 0)                                   # negative codes are missing
    h["pid"] = h.idauniq.astype("int64")
    # deaths: first wave whose status says the person has died
    st = pd.DataFrame({w: h[f"r{w}iwstat"] for w in YEARS if f"r{w}iwstat" in h})
    dead_at = st.isin([5, 6])
    death = pd.Series(np.where(dead_at.any(axis=1), dead_at.idxmax(axis=1), np.nan), index=h.pid).dropna()
    step(f"deaths recorded: {len(death):,}")
    parts = []
    for w in YEARS:
        fi = read_fi(w)
        if fi is None:
            step(f"wave {w}: no frailty file, skipped"); continue
        g = lambda v: h[f"r{w}{v}"] if f"r{w}{v}" in h else pd.Series(np.nan, index=h.index)
        gh = lambda v: h[f"h{w}{v}"] if f"h{w}{v}" in h else pd.Series(np.nan, index=h.index)
        x = pd.DataFrame({"pid": h.pid, "iwy": g("iwy"), "iwm": g("iwm"),
                          "education": h.raeducl, "partner": g("mstat").isin([1, 2, 3]).astype(float).where(g("mstat").notna()),
                          "hhsize": gh("hhres"), "income": gh("itot"), "wealth": gh("atotb"),
                          "cesd": g("cesd"), "recall": g("tr20"), "orient": g("orient"), "serial7": g("ser7"), "grip": g("gripsum"),
                          "pain": g("painfr"), "region": g("region_e"),
                          "smoke_now": g("smoken"), "smoke_ever": g("smokev"), "drinks_week": g("drinkwn_e"), "drink_days": g("drinkd_e"),
                          "vigorous_freq": freq(g("vgactx_e")), "moderate_freq": freq(g("mdactx_e")), "light_freq": freq(g("ltactx_e")),
                          "bmi": g("mbmi").where(g("mbmi").between(13, 70)), "loneliness": g("lonely"), "sleep_trouble": g("sleepr"),
                          "social_any": g("socyr")})
        x["heavy_drink"] = (x.drinks_week > 14).astype(float).where(x.drinks_week.notna())
        x["vigorous_weekly"] = (x.vigorous_freq <= 2).astype(float).where(x.vigorous_freq.notna())
        x["inactive"] = ((x.vigorous_freq >= 3) & (x.moderate_freq >= 3)).astype(float).where(x.vigorous_freq.notna() & x.moderate_freq.notna())
        x["lonely"] = (x.loneliness >= 2).astype(float).where(x.loneliness.notna())
        for c in ["income", "wealth"]:
            x[c + "_quintile"] = pd.qcut(x[c].rank(method="first"), 5, labels=False) + 1
        x = fi.merge(x.drop(columns=["income", "wealth"]), on="pid", how="left")
        x["female"] = (x.sex == 2).astype(float)
        x["date"] = np.where(x.iwy.notna(), x.iwy + (x.iwm.fillna(6) - 0.5) / 12, YEARS[w] + 0.5)
        parts.append(x.drop(columns=["iwy", "iwm", "sex"]))
        step(f"wave {w}: {len(x):,} people with a frailty index; harmonised record found for {x.education.notna().mean():.1%}")
    P = pd.concat(parts, ignore_index=True)
    waves = sorted(P.wave.unique())
    yr = P.groupby("wave").date.median().to_dict()
    pairs = [(b, f) for b in waves for f in waves if f > b and 3.5 <= yr[f] - yr[b] <= 7.5]
    last = max(waves)
    temporal = [p for p in pairs if p[1] == last]
    windows = [p for p in pairs if p[1] != last]
    # deaths are recorded only up to wave 6 (2012) in the files available (later waves code 9: 'dk if alive or died'),
    # so lever triplets must end by wave 6; the starting gap may be up to 4.5 years
    DCW = 6
    triplets = []
    for i in range(len(waves) - 1):
        p, b = waves[i], waves[i + 1]
        if yr[b] - yr[p] > 4.5: continue
        fs = [f for f in waves if f > b and f <= DCW and 3.5 <= yr[f] - yr[b] <= 6.5]
        if fs: triplets.append((p, b, min(fs, key=lambda f: (abs(yr[f] - yr[b] - 5), f))))
    deficits = [c for c in P.columns if c.startswith("d_")]
    fixed = ["age", "female", "education", "partner", "hhsize", "income_quintile", "wealth_quintile", "fi"] + deficits + \
            ["cesd", "recall", "orient", "serial7", "grip", "pain"]
    lev = ["smoke_now", "smoke_ever", "drinks_week", "drink_days", "heavy_drink", "vigorous_freq", "moderate_freq", "light_freq",
           "vigorous_weekly", "inactive", "bmi", "loneliness", "lonely", "sleep_trouble", "social_any"]
    meta = dict(cohort="ELSA", years={int(k): float(v) for k, v in yr.items()}, windows=windows, temporal=temporal, triplets=triplets,
                fixed=fixed, levers=lev, region="region", death_complete_wave=DCW, recalibrate=False, region_names=REGIONS, t1_health=["fi", "d_srh", "cesd", "pain"],
                notes=["Deaths are recorded only up to wave 6 (2012) in the End User Licence files: the frail-or-dead model and the lever "
                       "analysis use windows ending by 2012 (later period = windows ending at wave 6); the frailty-score model uses all windows.",
                       "Frailty files for waves 2 and 11 are not on disk, so those waves are not used.",
                       "BMI is measured by nurses at waves 2, 4, 6, 8 and 9 only.",
                       "Wave 10 (2021-23) follow-up spans the COVID-19 pandemic."])
    pickle.dump(dict(P=P, death=death, meta=meta), open(os.path.join(CACHE, "elsa_prepared.pkl"), "wb"))
    step(f"saved: {len(P):,} person-waves, {P.pid.nunique():,} people; windows {windows}; later-period {temporal}; triplets {triplets}")
    step("missing share of levers: " + ", ".join(f"{c} {P[c].isna().mean():.0%}" for c in lev))
