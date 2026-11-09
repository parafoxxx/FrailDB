"""
Step 1 of the SHARE frailty prediction model: build the longitudinal analysis file.

One row per person per window (baseline wave b -> follow-up wave f, about 4-6 years later):
  windows: w2->w4 (2006/07->2011), w4->w6 (2011->2015), w5->w8 (2013->2019/20), w6->w8 (2015->2019/20)
Baseline: the person's Rockwood FI assessment in wave b (complete case, from the project's FI files), plus fixed and
changeable characteristics from the SHARE modules of wave b.
Outcome status at follow-up:
  'fi'   alive with an FI score at wave f          -> fi_f, frail_f (FI >= 0.25)
  'dead' died between the baseline and follow-up interviews (end-of-life interview or coverscreen flag)
  'lost' alive (not known dead) but no FI at wave f -> handled later with inverse-probability-of-censoring weights
Participant-level output stays in FrailDB/.cache/prediction/ (git-ignored). Run: python FrailDB/prediction/prepare.py
"""
import os
import numpy as np, pandas as pd, pyreadstat

HERE = os.path.dirname(os.path.abspath(__file__))
FDB = os.path.dirname(HERE)
FI_DIR = os.path.join(FDB, "..", "Data Sources", "2. SHARE data set_wave 1-9", "SHARE Analysis")
MODS = os.path.join(FDB, ".cache", "share_mods")
MORT = os.path.join(FDB, ".cache", "share_mort")
OUT = os.path.join(FDB, ".cache", "prediction")
os.makedirs(OUT, exist_ok=True)
WINDOWS = [(2, 4), (4, 6), (5, 8), (6, 8)]
DEFICITS = ["d_srh", "d_hypertension", "d_diabetes", "d_cancer", "d_lung", "d_heart", "d_stroke", "d_arthritis",
            "d_adl_dress", "d_adl_bath", "d_adl_eat", "d_adl_toilet", "d_iadl_meal", "d_iadl_shop", "d_hearing",
            "d_depression", "d_mob_walk", "d_mob_chair", "d_mob_stoop", "d_mob_lift", "d_adl_bed", "d_vision",
            "d_adl_walkroom", "d_mob_sit2h", "d_mob_stairs_several", "d_mob_stairs_one", "d_mob_arms", "d_mob_push",
            "d_mob_coin", "d_iadl_meds", "d_iadl_money", "d_dementia"]
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)


def read(w, m, cols=None):
    """Read a SHARE module for wave w (Stata or SPSS cache); only the columns that exist; negative codes -> NaN."""
    for p, f in ((os.path.join(MODS, f"w{w}_{m}.dta"), pyreadstat.read_dta), (os.path.join(MODS, f"w{w}_{m}.sav"), pyreadstat.read_sav),
                 (os.path.join(MORT, f"w{w}_{m}.dta"), pyreadstat.read_dta), (os.path.join(MORT, f"w{w}_{m}.sav"), pyreadstat.read_sav)):
        if os.path.exists(p):
            have = f(p, metadataonly=True)[1].column_names
            use = ["mergeid"] + [c for c in (cols or have) if c in have and c != "mergeid"]
            d = f(p, usecols=use)[0]
            for c in d.columns:                     # some answer columns arrive as text: make them numeric
                if c != "mergeid" and not pd.api.types.is_numeric_dtype(d[c]):
                    conv = pd.to_numeric(d[c], errors="coerce")
                    if conv.notna().sum() >= 0.9 * d[c].notna().sum():
                        d[c] = conv
            num = [c for c in d.columns if c != "mergeid" and pd.api.types.is_numeric_dtype(d[c])]
            d[num] = d[num].where(d[num] >= 0)
            return d.drop_duplicates("mergeid")
    return pd.DataFrame(columns=["mergeid"])


def ym(y, m):
    y = pd.to_numeric(y, errors="coerce"); m = pd.to_numeric(m, errors="coerce")
    return y.where(y > 1900) + (m.where((m >= 1) & (m <= 12)).fillna(6.5) - 0.5) / 12


# ------------------------------------------------------------------ answers carried forward from earlier waves
def history(w):
    """Most recent earlier answer (waves before w) for questions not asked every wave."""
    rows = []
    for v in [x for x in (1, 2, 4, 5, 6, 7, 8, 9) if x < w]:
        br = read(v, "br", ["br001_", "br002_", "br026_", "br027_", "br028_", "br029_"])
        br["wave"] = v; rows.append(br)
    if not rows:
        return pd.DataFrame(columns=["mergeid", "h_ever", "h_now"])
    H = pd.concat(rows).sort_values("wave")
    out = pd.DataFrame({"mergeid": H.mergeid.unique()}).set_index("mergeid")
    out["h_ever"] = H.assign(e=(H.br001_ == 1).astype(float).where(H.br001_.notna())).groupby("mergeid").e.max()
    out["h_now"] = H.dropna(subset=["br002_"]).groupby("mergeid").br002_.last().map({1: 1.0, 5: 0.0})
    for c, n in [("br029_", "fruitveg_freq"), ("br028_", "protein_freq"), ("br026_", "dairy_freq"), ("br027_", "legumes_eggs_freq")]:
        if c in H: out["h_" + n] = H.dropna(subset=[c]).groupby("mergeid")[c].last()
    return out.reset_index()


# ------------------------------------------------------------------ baseline characteristics of one wave
DAYS = {1: 6.5, 2: 5.5, 3: 3.5, 4: 1.5, 5: 0.35, 6: 0.1, 7: 0.0}   # br010 -> drinking days per week


def features(w):
    f = read(w, "cv_r", ["int_year", "int_month", "partnerinhh", "hhsize"]).rename(columns={"partnerinhh": "partner"})
    f["t0"] = ym(f.int_year, f.int_month); f["partner"] = (f.partner == 1).astype(float).where(f.partner.notna())
    f = f.drop(columns=["int_year", "int_month"])
    # behaviour (changeable)
    br = read(w, "br", ["br001_", "br002_", "br010_", "br019_", "br039_", "br040_", "br015_", "br016_", "br026_", "br027_", "br028_", "br029_"])
    b = pd.DataFrame({"mergeid": br.mergeid})
    b["smoke_now"] = (br.br002_ == 1).astype(float).where(br.br002_.notna() | (br.br001_ == 5))
    b.loc[br.br001_ == 5, "smoke_now"] = 0.0
    b["smoke_ever"] = (br.br001_ == 1).astype(float).where(br.br001_.notna())
    if "br040_" in br:      # waves 6+: units in the last 7 days
        b["alcohol_units_wk"] = np.where(br.br039_ == 5, 0.0, br.br040_)
    else:                    # waves 2-5: days per week x drinks per day
        days = br.br010_.map(DAYS)
        b["alcohol_units_wk"] = np.where(days == 0, 0.0, days * br.get("br019_"))
    b["alcohol_units_wk"] = pd.Series(b["alcohol_units_wk"]).clip(upper=140)
    b["vigorous_freq"] = br.br015_; b["moderate_freq"] = br.br016_          # 1 = >1/week ... 4 = hardly ever
    b["inactive"] = ((br.br015_ == 4) & (br.br016_ == 4)).astype(float).where(br.br015_.notna() & br.br016_.notna())
    for c, n in [("br029_", "fruitveg_freq"), ("br028_", "protein_freq"), ("br026_", "dairy_freq"), ("br027_", "legumes_eggs_freq")]:
        if c in br: b[n] = br[c]                                            # 1 = every day ... 5 = < once a week
    # Some questions are asked only once per person ('ever smoked') or were skipped for panel members (wave 6 smoking
    # and diet): fill from the person's most recent earlier answer and flag it as carried forward.
    hist = history(w)
    b = b.merge(hist, on="mergeid", how="left")
    ever_yes = (b.smoke_ever == 1) | (b.h_ever == 1)
    b["smoke_ever"] = np.where(ever_yes, 1.0, np.where(b.smoke_ever.notna(), b.smoke_ever, pd.to_numeric(b.h_ever, errors="coerce"))).astype(float)
    b.loc[(b.smoke_ever == 0) & b.smoke_now.isna(), "smoke_now"] = 0.0
    b["smoke_now_carried"] = (b.smoke_now.isna() & b.h_now.notna()).astype(int)
    b["smoke_now"] = b.smoke_now.fillna(b.h_now)
    for n in ["fruitveg_freq", "protein_freq", "dairy_freq", "legumes_eggs_freq"]:
        if "h_" + n in b:
            if n not in b: b[n] = np.nan
            b[n + "_carried"] = (b[n].isna() & b["h_" + n].notna()).astype(int)
            b[n] = b[n].fillna(b["h_" + n])
    b = b.drop(columns=[c for c in b.columns if c.startswith("h_")])
    f = f.merge(b, on="mergeid", how="outer")
    # health summaries (mostly fixed; BMI, loneliness changeable)
    gh = read(w, "gv_health", ["bmi", "eurod", "casp", "loneliness", "maxgrip", "orienti", "gali", "cf016tot", "numeracy"])
    gh = gh.rename(columns={"cf016tot": "recall_delayed"})
    if "bmi" in gh: gh["bmi"] = gh.bmi.where(gh.bmi.between(13, 70))
    f = f.merge(gh, on="mergeid", how="left")
    ph = read(w, "ph", ["ph041_", "ph045_", "ph084_", "ph085_"])
    p = pd.DataFrame({"mergeid": ph.mergeid, "glasses": (ph.ph041_ == 1).astype(float).where(ph.ph041_.notna()),
                      "hearing_aid": (ph.ph045_ == 1).astype(float).where(ph.ph045_.notna())})
    if "ph084_" in ph:
        p["pain"] = (ph.ph084_ == 1).astype(float).where(ph.ph084_.notna()); p["pain_level"] = ph.get("ph085_")
    f = f.merge(p, on="mergeid", how="left")
    mh = read(w, "mh", ["mh007_"])
    f = f.merge(pd.DataFrame({"mergeid": mh.mergeid, "sleep_trouble": (mh.mh007_ == 1).astype(float).where(mh.mh007_.notna())}), on="mergeid", how="left")
    ac = read(w, "ac")
    items = [c for c in ac.columns if c.startswith("ac035d") and c[6:].isdigit()] or [c for c in ac.columns if c.startswith("ac002d") and c[6:].isdigit()]
    a = pd.DataFrame({"mergeid": ac.mergeid, "social_activities": ac[items].sum(axis=1, min_count=1) if items else np.nan,
                      "life_satisfaction": ac.get("ac012_")})
    f = f.merge(a, on="mergeid", how="left")
    cf = read(w, "cf", ["cf010_"]).rename(columns={"cf010_": "fluency"})
    f = f.merge(cf, on="mergeid", how="left")
    nw = read(w, "gv_networks")
    sz = [c for c in nw.columns if c == f"sn_size_w{w}" or c == "sizeofsocialnetwork"]
    if sz:
        f = f.merge(nw[["mergeid", sz[0]]].rename(columns={sz[0]: "network_size"}), on="mergeid", how="left")
    ed = read(w, "gv_isced", ["isced1997_r"])
    ed["education_isced"] = ed.isced1997_r.where(ed.isced1997_r <= 6)
    f = f.merge(ed[["mergeid", "education_isced"]], on="mergeid", how="left")
    return f, None


def money_quintiles(im, w):
    """Household income and net worth as within-country quintiles (first imputation)."""
    p = os.path.join(MODS, f"w{w}_gv_imputations.dta") if os.path.exists(os.path.join(MODS, f"w{w}_gv_imputations.dta")) else os.path.join(MODS, f"w{w}_gv_imputations.sav")
    f = pyreadstat.read_dta if p.endswith(".dta") else pyreadstat.read_sav
    d = f(p, usecols=["mergeid", "implicat", "thinc", "hnetw", "country"])[0]
    d = d[d.implicat == 1].drop_duplicates("mergeid")
    for c, n in [("thinc", "income_quintile"), ("hnetw", "wealth_quintile")]:
        d[n] = d.groupby("country")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    return d[["mergeid", "income_quintile", "wealth_quintile"]]


# ------------------------------------------------------------------ deaths
def deaths():
    rec = []
    for w in (2, 4, 5, 6, 7, 8, 9):
        x = read(w, "xt", ["xt008_", "xt009_"])
        if len(x): rec.append(pd.DataFrame({"mergeid": x.mergeid, "wave": w, "death": ym(x.xt009_, x.xt008_)}))
        c = read(w, "cv_r", ["deceased"])
        if len(c): rec.append(pd.DataFrame({"mergeid": c.mergeid[c.deceased == 1], "wave": w, "death": np.nan}))
    d = pd.concat(rec)
    return d.groupby("mergeid").agg(death_wave=("wave", "min"), death_date=("death", "min"))


def fi_file(w):
    d = pd.read_csv(os.path.join(FI_DIR, f"Wave {w}", f"SHARE_w{w}_Rockwood_summary.csv"))
    return d.rename(columns={"frailty_index_score": "fi"})


if __name__ == "__main__":
    D = deaths()
    say(f"Deaths known: {len(D):,} people")
    FI = {w: fi_file(w) for w in {w for p in WINDOWS for w in p}}
    med_date = {}
    out = []
    for b, fw in WINDOWS:
        base = FI[b][["mergeid", "country", "age", "sex", "fi"] + DEFICITS].copy()
        feats, _ = features(b)
        money = money_quintiles(None, b)
        base = base.merge(feats, on="mergeid", how="left").merge(money, on="mergeid", how="left")
        fol = FI[fw][["mergeid", "fi"]].rename(columns={"fi": "fi_f"})
        base = base.merge(fol, on="mergeid", how="left")
        cvf = read(fw, "cv_r", ["int_year", "int_month"]); cvf["t1"] = ym(cvf.int_year, cvf.int_month)
        med_date[fw] = cvf.t1.median()
        base = base.merge(cvf[["mergeid", "t1"]], on="mergeid", how="left")
        dd = D.reindex(base.mergeid)
        died = (dd.death_wave.values > b) & (dd.death_wave.values <= fw)
        base["status"] = np.where(died, "dead", np.where(base.fi_f.notna(), "fi", "lost"))
        base["death_date"] = np.where(died, dd.death_date.values, np.nan)
        base["years"] = np.where(base.status == "fi", base.t1 - base.t0, med_date[fw] - base.t0)
        base["frail_f"] = (base.fi_f >= 0.25).astype(float).where(base.fi_f.notna())
        # composite for people not frail at baseline: frail at follow-up, or died before it
        base["frail_or_dead"] = np.where(base.status == "dead", 1.0, base.frail_f)
        base["at_risk"] = (base.fi < 0.25).astype(int)
        base["window"] = f"w{b}->w{fw}"; base["wave_b"] = b; base["wave_f"] = fw
        out.append(base)
        s = base.status.value_counts()
        say(f"w{b}->w{fw}: {len(base):,} baseline; FI at follow-up {s.get('fi', 0):,}; died {s.get('dead', 0):,}; "
            f"lost {s.get('lost', 0):,}; median {np.nanmedian(base.years):.1f} y; not frail at baseline {base.at_risk.sum():,}; "
            f"of whom frail-or-dead {int(base.loc[base.at_risk == 1, 'frail_or_dead'].sum()):,}")
    A = pd.concat(out, ignore_index=True)
    A.to_pickle(os.path.join(OUT, "share_windows.pkl"))
    say(f"All windows: {len(A):,} rows, {A.mergeid.nunique():,} people")
    miss = A.drop(columns=DEFICITS).groupby("window").apply(lambda g: g.isna().mean().round(3), include_groups=False).T
    miss.to_csv(os.path.join(HERE, "output", "step1_missingness_by_window.csv"))
    open(os.path.join(HERE, "output", "step1_log.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
