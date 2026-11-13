"""
MHAS: person-wave table for the cohort engine.

Frailty index: rebuilt from the Harmonized MHAS file (H_MHAS_d) with FrailDB's 32-item recipe (wave notes in
"MHAS Analysis/Wave N/Steps followed..."), because the per-person files for 2012 and 2021 are not on disk. The rebuilt
index is checked against FrailDB's per-person files for 2001, 2003, 2015 and 2018 (printed below). In 2012 hearing was
asked of about 1% of people, so the 2012 index uses the other 31 items (FrailDB does not publish a 2012 index).
Characteristics and levers: Harmonized MHAS. Deaths: harmonised interview status (5, 6).
Windows: pairs of waves 2.5 to 7.5 years apart (2012->2015, 2012->2018, 2015->2018); pairs ending in 2021 form the
later-period test (2015->2021, 2018->2021). Predictions are made at 5 years.
"""
import os, sys, pickle
import numpy as np, pandas as pd, pyreadr, pyreadstat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import CACHE, step

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Data Sources", "6. MHAS_Mexico")
H = os.path.join(ROOT, "MHAS data", "H_MHAS_d.dta")
YEARS = {1: 2001, 2: 2003, 3: 2012, 4: 2015, 5: 2018, 6: 2021}
ITEMS = {"d_srh": ("shlt", "fairpoor"), "d_hypertension": ("hibpe", "yes"), "d_diabetes": ("diabe", "yes"), "d_cancer": ("cancre", "yes"),
         "d_lung": ("respe", "yes"), "d_heart": ("hrtatte", "yes"), "d_stroke": ("stroke", "yes"), "d_arthritis": ("arthre", "yes"),
         "d_adl_dress": ("dressa", "yes"), "d_adl_bath": ("batha", "yes"), "d_adl_eat": ("eata", "yes"), "d_adl_toilet": ("toilta", "yes"),
         "d_iadl_meal": ("mealsa", "yes"), "d_iadl_shop": ("shopa", "yes"), "d_hearing": ("hearing", "fairpoor"),
         "d_depression": ("cesd_m", "cesd"), "d_mob_walk": ("walksa", "yes"), "d_mob_chair": ("chaira", "yes"), "d_mob_stoop": ("stoopa", "yes"),
         "d_mob_lift": ("lifta", "yes"), "d_adl_bed": ("beda", "yes"), "d_vision": ("sight", "fairpoor"), "d_adl_walkroom": ("walkra", "yes"),
         "d_mob_sit2h": ("sita", "yes"), "d_mob_stairs_several": ("climsa", "yes"), "d_mob_stairs_one": ("clim1a", "yes"),
         "d_mob_arms": ("armsa", "yes"), "d_mob_push": ("pusha", "yes"), "d_mob_coin": ("dimea", "yes"), "d_iadl_meds": ("medsa", "yes"),
         "d_iadl_money": ("moneya", "yes"), "d_equip_walkroom": ("walkre", "equip")}
RVARS = ["agey", "iwstat", "iwy", "iwm", "smoken", "smokev", "drink", "drinkd", "drinkn", "vigact", "bmi", "sleepr", "flone", "socwk",
         "hearaid", "glasses", "mstat", "cesd_m", "painfr"] + sorted({v for v, _ in ITEMS.values()})
HVARS = ["hhres", "itot", "atotb", "rural"]


def score(raw, rule, walkra=None):
    v = raw
    if rule == "yes": return v.where(v.isin([0, 1]))
    if rule == "fairpoor": return v.isin([4, 5, 6]).astype(float).where(v.between(1, 6))
    if rule == "cesd": return (v >= 5).astype(float).where(v.notna())
    if rule == "equip":                                   # asked only of people with difficulty walking across a room
        return v.where(v.isin([0, 1])).fillna((walkra == 0).astype(float).where(walkra == 0) * 0)


def old_file(w):
    b = os.path.join(ROOT, "MHAS Analysis", f"Wave {w}", f"MHAS_w{w}_Rockwood")
    if os.path.exists(b + "_summary.csv"): return pd.read_csv(b + "_summary.csv")
    if os.path.exists(b + ".rds"): return list(pyreadr.read_r(b + ".rds").values())[0]
    return None


if __name__ == "__main__":
    _, meta = pyreadstat.read_dta(H, metadataonly=True)
    have = set(meta.column_names)
    cols = ["unhhidnp", "ragender", "raeducl"] + [f"r{w}{v}" for w in YEARS for v in RVARS if f"r{w}{v}" in have] + \
           [f"h{w}{v}" for w in YEARS for v in HVARS if f"h{w}{v}" in have]
    step(f"reading {len(cols)} columns from Harmonized MHAS")
    h, _ = pyreadstat.read_dta(H, usecols=cols)
    h = h.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    h["pid"] = h.unhhidnp.astype("int64")
    st = pd.DataFrame({w: h[f"r{w}iwstat"] for w in YEARS})
    dead_at = st.isin([5, 6])
    death = pd.Series(np.where(dead_at.any(axis=1), dead_at.idxmax(axis=1), np.nan), index=h.pid).dropna()
    step(f"deaths recorded: {len(death):,}")
    parts = []
    for w in YEARS:
        g = lambda v: h[f"r{w}{v}"] if f"r{w}{v}" in h else pd.Series(np.nan, index=h.index)
        gh = lambda v: h[f"h{w}{v}"] if f"h{w}{v}" in h else pd.Series(np.nan, index=h.index)
        x = pd.DataFrame({"pid": h.pid, "age": g("agey"), "female": (h.ragender == 2).astype(float).where(h.ragender.notna())})
        for d, (v, rule) in ITEMS.items():
            x[d] = score(g(v), rule, g("walkra"))
        items = list(ITEMS)
        if w == 3:
            items = [i for i in items if i != "d_hearing"]; x["d_hearing"] = np.nan
        x = x[(x.age >= 50) & (g("iwstat") == 1)]
        x = x.dropna(subset=items)
        x["fi"] = x[items].mean(axis=1)
        o = old_file(w)
        if o is not None:
            m = o.merge(x[["pid", "fi"]], left_on="unhhidnp", right_on="pid")
            step(f"wave {w}: check against FrailDB file: {len(m):,} of {len(o):,} matched, identical index for "
                 f"{(abs(m.fi - m.frailty_index_score) < 1e-6).mean():.1%}, mean {m.fi.mean():.4f} vs {m.frailty_index_score.mean():.4f}")
        idx = x.index
        x["date"] = (g("iwy") + (g("iwm").fillna(6) - 0.5) / 12)[idx].fillna(YEARS[w] + 0.5)
        x["education"] = h.raeducl[idx]
        x["partner"] = g("mstat")[idx].isin([1, 3]).astype(float).where(g("mstat")[idx].notna())
        x["hhsize"] = gh("hhres")[idx]; x["rural"] = gh("rural")[idx]
        for c, v in [("income_quintile", gh("itot")), ("wealth_quintile", gh("atotb"))]:
            vv = v[idx]; x[c] = (pd.qcut(vv.rank(method="first"), 5, labels=False) + 1) if vv.notna().sum() > 100 else np.nan
        x["cesd"] = g("cesd_m")[idx]; x["pain"] = g("painfr")[idx]
        x["smoke_now"] = g("smoken")[idx]; x["smoke_ever"] = g("smokev")[idx]
        dd, dn = g("drinkd")[idx], g("drinkn")[idx]
        x["drinks_week"] = (dd * dn).where(g("drink")[idx] == 1, 0.0).where(g("drink")[idx].notna()).clip(upper=140)
        x["heavy_drink"] = (x.drinks_week > 14).astype(float).where(x.drinks_week.notna())
        x["vigorous_weekly"] = g("vigact")[idx]          # MHAS: vigorous activity 3 or more times a week
        x["bmi"] = g("bmi")[idx].where(g("bmi")[idx].between(13, 70))
        x["sleep_trouble"] = g("sleepr")[idx]; x["lonely"] = g("flone")[idx]; x["social_any"] = g("socwk")[idx]
        x["hearing_aid"] = g("hearaid")[idx]; x["glasses"] = g("glasses")[idx]
        x["wave"] = w
        parts.append(x)
        step(f"wave {w}: {len(x):,} people aged 50+ with a frailty index (mean {x.fi.mean():.3f}, frail {(x.fi >= 0.25).mean():.1%})")
    P = pd.concat(parts, ignore_index=True)
    waves = sorted(P.wave.unique()); yr = P.groupby("wave").date.median().to_dict()
    pairs = [(b, f) for b in waves for f in waves if f > b and 2.5 <= yr[f] - yr[b] <= 7.5]
    temporal = [p for p in pairs if p[1] == max(waves)]
    windows = [p for p in pairs if p[1] != max(waves)]
    triplets = []
    for i in range(len(waves) - 1):
        p, b = waves[i], waves[i + 1]
        if yr[b] - yr[p] > 3.5: continue
        fs = [f for f in waves if f > b and 2.5 <= yr[f] - yr[b] <= 6.5]
        if fs: triplets.append((p, b, min(fs, key=lambda f: (abs(yr[f] - yr[b] - 5), f))))
    deficits = list(ITEMS)
    fixed = ["age", "female", "education", "partner", "hhsize", "income_quintile", "wealth_quintile", "rural", "fi"] + deficits + ["cesd", "pain"]
    lev = ["smoke_now", "smoke_ever", "drinks_week", "heavy_drink", "vigorous_weekly", "bmi", "sleep_trouble", "lonely", "social_any",
           "hearing_aid", "glasses"]
    meta = dict(cohort="MHAS", years={int(k): float(v) for k, v in yr.items()}, windows=windows, temporal=temporal, triplets=triplets,
                fixed=fixed, levers=lev, region=None, t1_health=["fi", "d_srh", "cesd", "pain"],
                notes=["Frailty index rebuilt from Harmonized MHAS with FrailDB's 32-item recipe; 2012 uses 31 items (hearing not asked).",
                       "Follow-up gaps are 3 or 6 years (waves in 2012, 2015, 2018, 2021); predictions are made at 5 years.",
                       "Vigorous activity in MHAS means 3 or more times a week.",
                       "The 2018->2021 and 2015->2021 windows span the COVID-19 pandemic, which raised mortality in Mexico."])
    pickle.dump(dict(P=P, death=death, meta=meta), open(os.path.join(CACHE, "mhas_prepared.pkl"), "wb"))
    step(f"saved: {len(P):,} person-waves, {P.pid.nunique():,} people; windows {windows}; later-period {temporal}; triplets {triplets}")
    step("missing share of levers: " + ", ".join(f"{c} {P[c].isna().mean():.0%}" for c in lev))
