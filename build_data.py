"""
FrailDB data build.

Reads the per-wave participant-level outputs in "Data Sources/*/<cohort> Analysis/Wave N/",
and writes AGGREGATES ONLY to FrailDB/site/frailtydb.json. No participant identifier, single
record, or cell describing fewer than MIN_CELL people leaves this script.

Run:  python FrailDB/build_data.py
"""
import json, re, glob, os, math, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import pyreadr

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data Sources")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "frailtydb.json")
MIN_CELL = 10          # counts below this are suppressed
MIN_PCTL = 30          # percentiles only for groups at least this large

AGE_BANDS = [(50, 59, "50–59"), (60, 69, "60–69"), (70, 79, "70–79"), (80, 89, "80–89"), (90, 200, "90+")]
AGE5 = list(range(50, 100, 5))   # 5-year bands for age curves; 95 = 95+

# --------------------------------------------------------------------------- cohorts
COHORTS = [
    dict(key="SHARE", id="FDBC01", folder="2. SHARE data set_wave 1-9/SHARE Analysis", prefix="SHARE",
         name="Survey of Health, Ageing and Retirement in Europe", region="Europe and Israel", flag="EU",
         countries=28, period="2004–2022", design="Biennial panel, community-dwelling adults 50+ and partners",
         source="SHARE release 9.0.0, per-wave Stata modules (ph, mh, br, gs, gv_health, cv_r)",
         access="SHARE-ERIC research data centre, free registration", url="https://share-eric.eu",
         mortality="End-of-life interviews with proxy respondents",
         fried_ref=["FDBR03", "FDBR04"], rock_ref=["FDBR02", "FDBR05"],
         notes=["Wave 3 (2008–09) was SHARELIFE, a retrospective life-history wave with no health module, so it has no frailty data.",
                "Rockwood FI not computed for wave 1 (dementia not asked) or wave 7 (depression, vision and hearing asked of about 18% only).",
                "Fried slowness is self-reported (no timed walk in SHARE), following Romero-Ortuno et al. 2010."]),
    dict(key="HRS", id="FDBC02", folder="3. HRS_United States/HRS Analysis", prefix="HRS",
         name="Health and Retirement Study", region="United States", flag="US", countries=1, period="1998–2022",
         design="Biennial panel of adults 50+ with steady-state refresh cohorts",
         source="RAND HRS Longitudinal File 1992–2022 v1; Harmonized HRS (H_HRS_d) for measured grip",
         access="HRS online registration (public files)", url="https://hrs.isr.umich.edu",
         mortality="Exit interviews and National Death Index linkage",
         fried_ref=["FDBR03"], rock_ref=["FDBR02", "FDBR06"],
         notes=["Physical measures (grip) are collected from a random half of the sample each wave, so Fried samples are about half the Rockwood samples.",
                "Fried phenotype available for waves 7–14 (2004–2018), the waves with measured grip in the harmonised file."]),
    dict(key="ELSA", id="FDBC03", folder="4. ELSA_England/ELSA Analysis", prefix="ELSA",
         name="English Longitudinal Study of Ageing", region="England", flag="GB", countries=1, period="2002–2023",
         design="Biennial panel of adults 50+ living in private households; nurse visits every other wave",
         source="ELSA core and nurse-visit files, UK Data Service SN 5050",
         access="UK Data Service, End User Licence", url="https://www.elsa-project.ac.uk",
         mortality="Exit interviews and NHS Central Register linkage",
         fried_ref=["FDBR07"], rock_ref=["FDBR02", "FDBR05"],
         notes=["Fried phenotype uses measured grip and timed walk from the nurse visits (waves 2, 4, 6, 8, 9) and is restricted to age 60+, following Gale et al. 2015.",
                "Wave 10 asks the chronic conditions as single 'ever diagnosed' items."]),
    dict(key="MHAS", id="FDBC04", folder="6. MHAS_Mexico/MHAS Analysis", prefix="MHAS",
         name="Mexican Health and Aging Study", region="Mexico", flag="MX", countries=1, period="2001–2021",
         design="Panel of adults 50+ with refresher samples in 2012 and 2018",
         source="Gateway Harmonized MHAS (H_MHAS_d, 2001–2022) and constructed-variable files",
         access="mhasweb.org, public download after registration", url="https://www.mhasweb.org",
         mortality="Next-of-kin exit interviews",
         fried_ref=["FDBR08"], rock_ref=["FDBR02", "FDBR05"],
         notes=["Fried weakness and slowness are self-reported (no grip or walk test).",
                "Fried weight loss is operationalised as BMI below 22 and low activity as not exercising three times a week; both are lenient and raise pre-frail prevalence.",
                "Item 32 of the FI is 'uses equipment to walk across a room' in place of dementia, which is not asked consistently."]),
    dict(key="CLHLS", id="FDBC05", folder="6. CLHLS_China/CLHLS Analysis", prefix="CLHLS",
         name="Chinese Longitudinal Healthy Longevity Survey", region="China", flag="CN", countries=1, period="2002–2018",
         design="Panel oversampling the oldest old (65+, many centenarians) in 22 provinces",
         source="ICPSR 38899, CLHLS 1998–2018 cross-sections (DS0003–DS0008)",
         access="ICPSR, registered users", url="https://www.icpsr.umich.edu/web/ICPSR/studies/38899",
         mortality="Interviews with close relatives of deceased respondents",
         fried_ref=[], rock_ref=["FDBR09", "FDBR05"],
         notes=["CLHLS interviews people aged 65 and over, so the mean age is about 84. Compare with other cohorts by age band, not overall.",
                "Uses its own 32-item list: 13 diseases, CLHLS ADL/IADL items, a chair stand, and four CLHLS-specific graded items.",
                "Fried phenotype not computed (no grip strength or gait speed in comparable form)."]),
    dict(key="LASI", id="FDBC06", folder=None, prefix="LASI",
         name="Longitudinal Ageing Study in India", region="India", flag="IN", countries=1, period="2017–2019",
         design="Nationally representative survey of adults 45+ and spouses, all states and union territories",
         source="LASI Wave 1, analytic file of respondents 60+ (n = 26,275)",
         access="International Institute for Population Sciences, data request", url="https://www.iipsindia.ac.in/lasi",
         mortality="Not yet available (single wave)",
         fried_ref=["FDBR01"], rock_ref=[],
         notes=["Reference implementation of the Fried phenotype for this project, with measured grip and timed gait.",
                "Only wave 1 is released, so LASI contributes cross-sectional data only."]),
    dict(key="KLoSA", id="FDBC07", folder=None, prefix="KLoSA",
         name="Korean Longitudinal Study of Aging", region="South Korea", flag="KR", countries=1, period="2006–2022",
         design="Biennial panel of adults 45+",
         source="KLoSA waves 1–9, Korea Employment Information Service",
         access="Korea Employment Information Service, free registration", url="https://survey.keis.or.kr",
         mortality="Exit interviews",
         fried_ref=["FDBR10"], rock_ref=[],
         notes=["Raw data for waves 1–9 acquired. Frailty scoring is in progress and will appear in a later release."]),
]
# Cohorts withheld from this release (LASI held back on 2026-10-13; set to set() to restore it)
EXCLUDE = {"LASI"}
COHORTS = [c for c in COHORTS if c["key"] not in EXCLUDE]
CK = {c["key"]: c for c in COHORTS}

SHARE_COUNTRY = {11: "Austria", 12: "Germany", 13: "Sweden", 14: "Netherlands", 15: "Spain", 16: "Italy", 17: "France",
                 18: "Denmark", 19: "Greece", 20: "Switzerland", 23: "Belgium", 25: "Israel", 28: "Czech Republic",
                 29: "Poland", 30: "Ireland", 31: "Luxembourg", 32: "Hungary", 33: "Portugal", 34: "Slovenia",
                 35: "Estonia", 47: "Croatia", 48: "Lithuania", 51: "Bulgaria", 53: "Cyprus", 55: "Finland",
                 57: "Latvia", 59: "Malta", 61: "Romania", 63: "Slovakia"}

# --------------------------------------------------------------------------- deficits
DOMAINS = ["General health", "Chronic disease", "Mental health", "Sensory", "ADL", "IADL", "Mobility", "Social participation"]
# key, label, domain, scoring rule (core 32 set)
CORE = [
    ("d_srh", "Self-rated health", "General health", "Fair or poor (CLHLS: bad or very bad)"),
    ("d_hypertension", "Hypertension", "Chronic disease", "Ever diagnosed"),
    ("d_diabetes", "Diabetes", "Chronic disease", "Ever diagnosed"),
    ("d_cancer", "Cancer", "Chronic disease", "Ever diagnosed"),
    ("d_lung", "Chronic lung disease", "Chronic disease", "Ever diagnosed"),
    ("d_heart", "Heart disease", "Chronic disease", "Ever diagnosed"),
    ("d_stroke", "Stroke", "Chronic disease", "Ever diagnosed"),
    ("d_arthritis", "Arthritis", "Chronic disease", "Ever diagnosed"),
    ("d_adl_dress", "ADL: dressing", "ADL", "Any difficulty"),
    ("d_adl_bath", "ADL: bathing", "ADL", "Any difficulty"),
    ("d_adl_eat", "ADL: eating", "ADL", "Any difficulty"),
    ("d_adl_toilet", "ADL: using the toilet", "ADL", "Any difficulty"),
    ("d_iadl_meal", "IADL: preparing a hot meal", "IADL", "Any difficulty"),
    ("d_iadl_shop", "IADL: shopping", "IADL", "Any difficulty"),
    ("d_hearing", "Hearing", "Sensory", "Fair or poor, or deaf"),
    ("d_depression", "Depressive symptoms", "Mental health", "Above the scale cut-off"),
    ("d_mob_walk", "Walking a set distance", "Mobility", "Any difficulty"),
    ("d_mob_chair", "Getting up from a chair", "Mobility", "Any difficulty"),
    ("d_mob_stoop", "Stooping, kneeling or crouching", "Mobility", "Any difficulty"),
    ("d_mob_lift", "Lifting or carrying 5 kg", "Mobility", "Any difficulty"),
    ("d_adl_bed", "ADL: getting in or out of bed", "ADL", "Any difficulty"),
    ("d_vision", "Vision", "Sensory", "Fair or poor, or blind"),
    ("d_adl_walkroom", "ADL: walking across a room", "ADL", "Any difficulty"),
    ("d_mob_sit2h", "Sitting for two hours", "Mobility", "Any difficulty"),
    ("d_mob_stairs_several", "Climbing several flights of stairs", "Mobility", "Any difficulty"),
    ("d_mob_stairs_one", "Climbing one flight of stairs", "Mobility", "Any difficulty"),
    ("d_mob_arms", "Reaching arms above shoulder", "Mobility", "Any difficulty"),
    ("d_mob_push", "Pulling or pushing large objects", "Mobility", "Any difficulty"),
    ("d_mob_coin", "Picking up a small coin", "Mobility", "Any difficulty"),
    ("d_iadl_meds", "IADL: taking medications", "IADL", "Any difficulty"),
    ("d_iadl_money", "IADL: managing money", "IADL", "Any difficulty"),
    ("d_dementia", "Dementia or memory disease", "Chronic disease", "Ever diagnosed"),
    ("d_equip_walkroom", "Uses equipment to walk across a room", "Mobility", "Uses equipment"),
    # CLHLS-only
    ("d_tuberculosis", "Tuberculosis", "Chronic disease", "Currently suffering from it"),
    ("d_cataract", "Cataract", "Chronic disease", "Currently suffering from it"),
    ("d_glaucoma", "Glaucoma", "Chronic disease", "Currently suffering from it"),
    ("d_ulcer", "Gastric or duodenal ulcer", "Chronic disease", "Currently suffering from it"),
    ("d_parkinson", "Parkinson's disease", "Chronic disease", "Currently suffering from it"),
    ("d_continence", "Continence", "ADL", "Needs any assistance"),
    ("d_visit_neighbours", "Able to visit neighbours", "Social participation", "Any difficulty, or unable"),
    ("d_wash_clothes", "Able to wash clothes", "IADL", "Any difficulty, or unable"),
    ("d_public_transport", "Able to take public transport", "IADL", "Any difficulty, or unable"),
    ("d_fearful", "Feeling fearful or anxious", "Mental health", "Graded 0–1 by frequency"),
    ("d_useless_with_age", "Feeling useless with age", "Mental health", "Graded 0–1 by frequency"),
    ("d_housework", "Does housework at present", "Social participation", "Graded 0–1 by frequency (never = 1)"),
    ("d_hand_behind_neck", "Putting a hand behind the neck", "Mobility", "Graded: both 0, one 0.5, neither 1"),
]
# CLHLS columns that measure the same concept as a core item
CLHLS_ALIAS = {"d_iadl_cook": "d_iadl_meal", "d_walk_1km": "d_mob_walk", "d_carry_5kg": "d_mob_lift",
               "d_chair_stand": "d_mob_chair", "d_crouch_stand": "d_mob_stoop"}

# Source variables (from the analysis scripts and per-wave "Steps followed" files)
SHARE_VARS = {"d_srh": "ph003_", "d_hypertension": "ph006d2", "d_diabetes": "ph006d5", "d_cancer": "ph006d10",
              "d_lung": "ph006d6", "d_heart": "ph006d1", "d_stroke": "ph006d4",
              "d_arthritis": "ph006d8 (w2, w4); ph006d19 or ph006d20 (w5+)", "d_dementia": "ph006d16",
              "d_adl_dress": "ph049d1", "d_adl_bath": "ph049d3", "d_adl_eat": "ph049d4", "d_adl_toilet": "ph049d6",
              "d_iadl_meal": "ph049d8", "d_iadl_shop": "ph049d9", "d_adl_bed": "ph049d5", "d_adl_walkroom": "ph049d2",
              "d_iadl_meds": "ph049d11", "d_iadl_money": "ph049d13", "d_mob_walk": "ph048d1 (100 m)",
              "d_mob_sit2h": "ph048d2", "d_mob_chair": "ph048d3", "d_mob_stairs_several": "ph048d4",
              "d_mob_stairs_one": "ph048d5", "d_mob_stoop": "ph048d6", "d_mob_arms": "ph048d7", "d_mob_push": "ph048d8",
              "d_mob_lift": "ph048d9", "d_mob_coin": "ph048d10", "d_hearing": "ph046_", "d_vision": "ph043_ (6 = registered blind)",
              "d_depression": "EURO-D ≥ 4 (gv_health.eurod)"}


def parse_steps(path):
    """Parse a 'Steps followed' markdown into item rows and Fried criteria."""
    t = open(path, encoding="utf-8").read()
    items = []
    for m in re.finditer(r"^\s+(\d+)\.\s(.+?)\s{2,}(.+?)\s{2,}(.+?)\s*$", t, re.M):
        items.append(dict(n=int(m.group(1)), label=m.group(2).strip(), var=m.group(3).strip(), rule=m.group(4).strip()))
    fried = {}
    for m in re.finditer(r"^\s+FQ(\d)\s+([A-Za-z ]+?)\s*:\s*(.+)$", t, re.M):
        fried[int(m.group(1))] = m.group(3).strip()
    meth = re.search(r"^Method:\s*(.+)$", t, re.M)
    src = re.search(r"^Source:\s*(.+)$", t, re.M)
    return dict(items=items, fried=fried, method=meth.group(1).strip() if meth else None,
                source=src.group(1).strip() if src else None, text=t)


def steps_file(cohort, wave):
    f = glob.glob(os.path.join(ROOT, CK[cohort]["folder"], f"Wave {wave}", "Steps followed*.md"))
    return f[0] if f else None


# --------------------------------------------------------------------------- helpers
def sup(n):
    """Suppress small counts."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return None
    n = int(n)
    return n if n >= MIN_CELL else None


def pct(a, b):
    return None if not b or b < MIN_CELL or a is None else round(100 * a / b, 2)


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def band(age):
    for lo, hi, lab in AGE_BANDS:
        if lo <= age <= hi:
            return lab
    return None


def read_any(folder, prefix, wave, method):
    base = os.path.join(ROOT, folder, f"Wave {wave}", f"{prefix}_w{wave}_{method}")
    if os.path.exists(base + "_summary.csv"):
        return pd.read_csv(base + "_summary.csv", low_memory=False), "csv"
    if os.path.exists(base + ".rds"):
        return list(pyreadr.read_r(base + ".rds").values())[0], "rds"
    return None, None


FRIED_NORM = {"f_exhaustion": "FQ1", "f_weight_loss": "FQ2", "f_weakness": "FQ3", "f_slowness": "FQ4", "f_low_activity": "FQ5",
              "FQ1_exhaustion": "FQ1", "FQ2_weightloss": "FQ2", "FQ3_weakness": "FQ3", "FQ4_slowness": "FQ4",
              "FQ5_lowactivity": "FQ5", "fq1": "FQ1", "fq2": "FQ2", "fq3": "FQ3", "fq4": "FQ4", "fq5": "FQ5"}


def norm_fried(df):
    df = df.rename(columns=FRIED_NORM).copy()
    df["id"] = df.iloc[:, 0].astype(str)
    df["score"] = pd.to_numeric(df["frailty_score_fried"], errors="coerce")
    # three-level status derived from the 0–5 score (frail ≥3, pre-frail 1–2, non-frail 0)
    df["status"] = np.select([df.score >= 3, df.score >= 1], ["frail", "pre-frail"], "non-frail")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    return df


def norm_rock(df):
    df = df.rename(columns=CLHLS_ALIAS).copy()
    df["id"] = df.iloc[:, 0].astype(str)
    df["fi"] = pd.to_numeric(df["frailty_index_score"], errors="coerce")
    df["frail"] = (df.fi >= 0.25).astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    return df


def group_table(df, flag_fn):
    """Age band × sex table with suppression."""
    rows = []
    df = df.assign(band=df.age.map(lambda a: band(a) if pd.notna(a) else None))
    for lab in [b[2] for b in AGE_BANDS]:
        for sx, sl in [(1, "Male"), (2, "Female"), (None, "All")]:
            g = df[(df.band == lab) & ((df.sex == sx) if sx else True)]
            if len(g) == 0:
                continue
            rows.append(dict(band=lab, sex=sl, **flag_fn(g)))
    return rows


def rock_stats(g):
    n = len(g)
    out = dict(n=sup(n), frail_pct=pct(g.frail.sum(), n), mean_fi=r4(g.fi.mean()) if n >= MIN_CELL else None)
    if n >= MIN_PCTL:
        q = g.fi.quantile([.1, .25, .5, .75, .9]).round(4).tolist()
        out["p"] = q
    return out


def fried_stats(g):
    n = len(g)
    return dict(n=sup(n), frail_pct=pct((g.status == "frail").sum(), n), prefrail_pct=pct((g.status == "pre-frail").sum(), n),
                mean_score=r4(g.score.mean()) if n >= MIN_CELL else None)


# --------------------------------------------------------------------------- per-wave summaries from by_wave / sample flow files
def wave_table(cohort):
    folder = os.path.join(ROOT, CK[cohort]["folder"])
    pre = CK[cohort]["prefix"]
    W = {}
    sf = os.path.join(folder, f"{pre}_sample_flow.csv")
    if os.path.exists(sf):
        for r in pd.read_csv(sf).to_dict("records"):
            W[int(r["wave"])] = dict(year=int(r["year"]), raw=r["raw"], age50=r["age50"],
                                     rock_n=r.get("rock_n"), fried_n=r.get("fried_n"),
                                     fried_frail=r.get("fried_frail"))
    rb = os.path.join(folder, f"{pre}_Rockwood_by_wave.csv")
    if os.path.exists(rb):
        for r in pd.read_csv(rb).to_dict("records"):
            w = W.setdefault(int(r["wave"]), dict(year=int(r["year"])))
            w.update(raw=r["raw"], age50=r["age50"], rock_n=r["N"], rock_mean=r["mean_FI"], rock_sd=r["sd_FI"],
                     rock_frail_pct=r["pct_frail"], rock_retention=r["retention_pct"], rock_age=r["mean_age"])
    fb = os.path.join(folder, f"{pre}_Fried_by_wave.csv")
    if os.path.exists(fb):
        for r in pd.read_csv(fb).to_dict("records"):
            w = W.setdefault(int(r["wave"]), dict(year=int(r["year"])))
            w.update(fried_n=r["N"], fried_frail_pct=r["pct_frail"], fried_prefrail_pct=r["pct_prefrail"],
                     fried_mean=r["mean_score"], fried_age=r["mean_age"])
    # waves that only exist as folders
    for d in glob.glob(os.path.join(folder, "Wave *")):
        W.setdefault(int(d.rsplit(" ", 1)[1]), {})
    return W


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


# --------------------------------------------------------------------------- main build
db = dict(meta={}, cohorts=[], waves=[], deficits=[], criteria=[], instruments=[], references=[], countries=[],
          agecurves={}, longitudinal={}, cutoffs=[], concordance=[])
deficit_prev = defaultdict(lambda: defaultdict(dict))      # key -> cohort -> wave -> pct
deficit_bystatus = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))   # key -> cohort -> [sum_frail, n_frail, sum_non, n_non]
deficit_age = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0])))  # key -> cohort -> band -> [sum, n]
crit_prev = defaultdict(lambda: defaultdict(dict))
crit_age = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0])))
agecurve_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # cohort -> method -> (age5,sex) -> values
country_acc = defaultdict(lambda: defaultdict(lambda: dict(rock=[], fried=[])))  # country -> wave
long_frames = defaultdict(dict)  # (cohort, method) -> wave -> df[id, value, status]
cohort_items = defaultdict(set)
deficit_vars = defaultdict(dict)    # key -> cohort -> {var, rule, note}

wave_seq = 0
for c in COHORTS:
    key = c["key"]
    entry = {k: v for k, v in c.items() if k not in ("folder",)}
    entry["waves"] = []
    entry["status"] = "Analysed"
    if key == "KLoSA":
        entry["status"] = "Data acquired, analysis in progress"
        db["cohorts"].append(entry)
        continue
    if key == "LASI":
        # reference implementation: one cross-section, Fried 60+
        # local cache of the few LASI columns used (the workbook has ~2,900 columns); never published
        cache = os.path.join(os.path.dirname(OUT), "..", ".cache", "lasi_fried.pkl")
        if os.path.exists(cache):
            x = pd.read_pickle(cache)
        else:
            x = pd.read_excel(os.path.join(ROOT, "1. LASI/Important Documents/LASI_Analytic_60plus.xlsx (n=26,275).xlsx"),
                              usecols=["prim_key", "dm003", "dm005", "fq1", "fq2", "fq3", "fq4", "fq5", "frailty score",
                                       "frailty status", "average grip strength", "average walk time"])
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            x.to_pickle(cache)
        lasi_label_frail = (x["frailty status"].astype(str).str.lower() == "frail").sum()
        x = x.rename(columns={"dm003": "sex", "dm005": "age", "frailty score": "frailty_score_fried"})
        f = norm_fried(x)
        wave_seq += 1
        wid = f"FDBW{wave_seq:04d}"
        n = len(f)
        w = dict(id=wid, cohort=key, wave=1, year=2018, raw=None, age_n=n,
                 fried=dict(n=n, frail_pct=pct((f.status == "frail").sum(), n), prefrail_pct=pct((f.status == "pre-frail").sum(), n),
                            nonfrail_pct=pct((f.status == "non-frail").sum(), n), mean_score=r4(f.score.mean()),
                            mean_age=r4(f.age.mean()), female_pct=pct((f.sex == 2).sum(), n), age_rule="60+",
                            score_dist=[sup((f.score == k).sum()) for k in range(6)],
                            criteria={q: pct(f[q].sum(), n) for q in ["FQ1", "FQ2", "FQ3", "FQ4", "FQ5"]},
                            bands=group_table(f, fried_stats), has_detail=True),
                 rock=None, flags=["Reference implementation. Age 60+ only.",
                                   "Binary status in the source file (Frail / Non-frail); pre-frail is derived here from a score of 1–2."]
                       + ([f"The source file labels {lasi_label_frail:,} people Frail, but {int((f.score >= 3).sum()):,} have a score of 3 or more. "
                           "This page uses the score (frail = 3+)."] if lasi_label_frail != int((f.score >= 3).sum()) else []),
                 method_fried="LASI reference protocol: measured grip (sex × BMI cut-offs) and timed 4 m gait.")
        for q in ["FQ1", "FQ2", "FQ3", "FQ4", "FQ5"]:
            crit_prev[q][key][1] = pct(f[q].sum(), n)
            for b, g in f.groupby(f.age.map(band)):
                crit_age[q][key][b][0] += g[q].sum(); crit_age[q][key][b][1] += len(g)
        for (a5, sx), g in f.groupby([(f.age.clip(upper=95) // 5 * 5), f.sex]):
            agecurve_acc[key]["fried"][(int(a5), int(sx))].append(((g.status == "frail").sum(), len(g), g.score.sum()))
        db["waves"].append(w); entry["waves"].append(wid)
        db["cohorts"].append(entry)
        continue

    W = wave_table(key)
    for wn in sorted(W):
        meta = W[wn]
        rec = dict(cohort=key, wave=wn, year=clean(meta.get("year")), raw=clean(meta.get("raw")), age_n=clean(meta.get("age50")),
                   rock=None, fried=None, flags=[])
        sf = steps_file(key, wn)
        steps = parse_steps(sf) if sf else None
        if steps:
            rec["steps_source"] = steps["source"]
            rec["method_fried"] = steps["method"]
        # ---------------- Rockwood
        rdf, rsrc = read_any(c["folder"], c["prefix"], wn, "Rockwood")
        if rdf is not None:
            r = norm_rock(rdf)
            dcols = [col for col in r.columns if col.startswith("d_")]
            cohort_items[key].update(dcols)
            n = len(r)
            hist = np.histogram(r.fi, bins=np.linspace(0, 1, 33))[0]  # 1/32 bins up to 1
            rec["rock"] = dict(n=n, mean_fi=r4(r.fi.mean()), sd_fi=r4(r.fi.std()), median_fi=r4(r.fi.median()),
                               frail_pct=pct(r.frail.sum(), n), mean_age=r4(r.age.mean()), female_pct=pct((r.sex == 2).sum(), n),
                               items=len(dcols), retention=clean(meta.get("rock_retention")),
                               hist=[sup(h) for h in hist], bands=group_table(r, rock_stats),
                               deficits={d: round(100 * r[d].mean(), 2) for d in dcols}, has_detail=True)
            # domain mean deficit share
            for d in dcols:
                deficit_prev[d][key][wn] = round(100 * r[d].mean(), 2)
                fr, nf = r[r.frail == 1], r[r.frail == 0]
                acc = deficit_bystatus[d][key]
                acc[0] += fr[d].sum(); acc[1] += len(fr); acc[2] += nf[d].sum(); acc[3] += len(nf)
                for b, g in r.groupby(r.age.map(band)):
                    deficit_age[d][key][b][0] += g[d].sum(); deficit_age[d][key][b][1] += len(g)
            for (a5, sx), g in r.groupby([(r.age.clip(upper=95) // 5 * 5), r.sex]):
                agecurve_acc[key]["rock"][(int(a5), int(sx))].append((g.frail.sum(), len(g), g.fi.sum()))
            long_frames[(key, "rock")][wn] = r[["id", "fi", "frail", "age", "sex"]]
            if key == "SHARE":
                for cc, g in r.groupby("country"):
                    country_acc[int(cc)][wn]["rock"] = [len(g), int(g.frail.sum()), float(g.fi.sum())]
        elif meta.get("rock_n") and not pd.isna(meta.get("rock_n")):
            rec["rock"] = dict(n=clean(meta.get("rock_n")), mean_fi=clean(meta.get("rock_mean")), sd_fi=clean(meta.get("rock_sd")),
                               frail_pct=clean(meta.get("rock_frail_pct")), mean_age=clean(meta.get("rock_age")),
                               retention=clean(meta.get("rock_retention")), items=32, has_detail=False)
            rec["flags"].append("Rockwood: wave-level results only. The participant-level output file is missing from the analysis folder.")
        # ---------------- Fried
        fdf, fsrc = read_any(c["folder"], c["prefix"], wn, "Fried")
        if fdf is not None:
            f = norm_fried(fdf)
            n = len(f)
            rec["fried"] = dict(n=n, frail_pct=pct((f.status == "frail").sum(), n), prefrail_pct=pct((f.status == "pre-frail").sum(), n),
                                nonfrail_pct=pct((f.status == "non-frail").sum(), n), mean_score=r4(f.score.mean()),
                                mean_age=r4(f.age.mean()), female_pct=pct((f.sex == 2).sum(), n),
                                score_dist=[sup((f.score == k).sum()) for k in range(6)],
                                criteria={q: pct(f[q].sum(), n) for q in ["FQ1", "FQ2", "FQ3", "FQ4", "FQ5"] if q in f},
                                bands=group_table(f, fried_stats), has_detail=True,
                                age_rule="60+" if key == "ELSA" else "50+")
            for q in ["FQ1", "FQ2", "FQ3", "FQ4", "FQ5"]:
                if q in f:
                    crit_prev[q][key][wn] = pct(f[q].sum(), n)
                    for b, g in f.groupby(f.age.map(band)):
                        crit_age[q][key][b][0] += g[q].sum(); crit_age[q][key][b][1] += len(g)
            for (a5, sx), g in f.groupby([(f.age.clip(upper=95) // 5 * 5), f.sex]):
                agecurve_acc[key]["fried"][(int(a5), int(sx))].append(((g.status == "frail").sum(), len(g), g.score.sum()))
            long_frames[(key, "fried")][wn] = f[["id", "score", "status", "age"]]
            if key == "SHARE":
                for cc, g in f.groupby("country"):
                    country_acc[int(cc)][wn]["fried"] = [len(g), int((g.status == "frail").sum()), int((g.status == "pre-frail").sum())]
            # status label check against source file
            orig = fdf["frailty_status_fried"].astype(str).str.lower().value_counts().to_dict()
            if "pre-frail" not in orig and key != "LASI":
                rec["flags"].append("Fried: the source file has no 'pre-frail' labels although scores of 1–2 exist. "
                                    "Pre-frail shown here is re-derived from the 0–5 score.")
        elif meta.get("fried_n") and not pd.isna(meta.get("fried_n")):
            rec["fried"] = dict(n=clean(meta.get("fried_n")), frail_pct=clean(meta.get("fried_frail_pct")) or pct(clean(meta.get("fried_frail")), clean(meta.get("fried_n"))),
                                prefrail_pct=clean(meta.get("fried_prefrail_pct")), mean_score=clean(meta.get("fried_mean")),
                                mean_age=clean(meta.get("fried_age")), has_detail=False,
                                age_rule="60+" if key == "ELSA" else "50+")
            rec["flags"].append("Fried: wave-level results only. The participant-level output file is missing from the analysis folder.")
        # ---------------- cut-off tables
        for kind in ("grip", "walk"):
            p = os.path.join(ROOT, c["folder"], f"Wave {wn}", f"{c['prefix']}_w{wn}_{kind}_cutoff_table.csv")
            if os.path.exists(p):
                t = pd.read_csv(p)
                db["cutoffs"].append(dict(cohort=key, wave=wn, year=rec["year"], kind=kind,
                                          rows=[{k: clean(v) for k, v in row.items()} for row in t.to_dict("records")]))
        # ---------------- quality flags
        if rec["rock"] and rec["rock"].get("retention") and rec["rock"]["retention"] < 70:
            rec["flags"].append(f"Rockwood: complete-case retention is {rec['rock']['retention']}% of the age-eligible sample.")
        if rec["fried"] and rec["age_n"] and rec["fried"]["n"] and rec["fried"]["n"] / rec["age_n"] < 0.25 and key == "SHARE":
            rec["flags"].append(f"Fried: only {rec['fried']['n']:,} of {int(rec['age_n']):,} age-eligible respondents were scored. Check the merge for this wave.")
        if not rec["rock"] and not rec["fried"]:
            continue
        if key == "SHARE" and wn in (1, 7) and not rec["rock"]:
            rec["flags"].append("Rockwood not computed: " + ("dementia was not asked in wave 1." if wn == 1 else
                                "depression, vision and hearing were asked of about 18% of respondents in wave 7."))
        wave_seq += 1
        rec["id"] = f"FDBW{wave_seq:04d}"
        db["waves"].append(rec)
        entry["waves"].append(rec["id"])
    db["cohorts"].append(entry)

    # ------------- crosswalk for this cohort
    if key == "SHARE":
        for d, v in SHARE_VARS.items():
            deficit_vars[d][key] = dict(var=v)
    else:
        sfs = sorted(glob.glob(os.path.join(ROOT, c["folder"], "Wave *", "Steps followed*.md")))
        if sfs:
            parsed = parse_steps(sfs[-1])
            cols = sorted(cohort_items[key], key=lambda d: 0)
            # order of d_ columns in the participant files matches the numbered list
            sample = None
            for wn in sorted(W):
                rdf, _ = read_any(c["folder"], c["prefix"], wn, "Rockwood")
                if rdf is not None:
                    sample = [CLHLS_ALIAS.get(x, x) for x in rdf.columns if x.startswith("d_")]
                    break
            for it in parsed["items"]:
                if sample and it["n"] <= len(sample):
                    deficit_vars[sample[it["n"] - 1]][key] = dict(var=it["var"], rule=it["rule"], label=it["label"])

# --------------------------------------------------------------------------- deficits
did = 0
for k, label, dom, rule in CORE:
    cohorts_with = [c for c in deficit_prev[k]]
    if not cohorts_with:
        continue
    did += 1
    prevpool = {}
    for co in cohorts_with:
        a = deficit_bystatus[k][co]
        prevpool[co] = dict(frail=round(100 * a[0] / a[1], 2) if a[1] >= MIN_CELL else None,
                            nonfrail=round(100 * a[2] / a[3], 2) if a[3] >= MIN_CELL else None)
    agep = {co: {b: (round(100 * v[0] / v[1], 2) if v[1] >= MIN_CELL else None) for b, v in deficit_age[k][co].items() if b}
            for co in cohorts_with}
    db["deficits"].append(dict(id=f"FDBD{did:03d}", key=k, label=label, domain=dom, rule=rule,
                               sets=sorted(cohorts_with, key=lambda x: [c["key"] for c in COHORTS].index(x)),
                               core=k in [x[0] for x in CORE[:33]] and k != "d_equip_walkroom",
                               vars={co: deficit_vars[k].get(co, {}) for co in cohorts_with},
                               prev={co: deficit_prev[k][co] for co in cohorts_with},
                               by_status=prevpool, by_age=agep))

# --------------------------------------------------------------------------- Fried criteria
CRIT = [("FQ1", "Exhaustion", "Self-reported exhaustion or fatigue"),
        ("FQ2", "Weight loss", "Unintentional weight loss or poor appetite"),
        ("FQ3", "Weakness", "Low grip strength, or self-reported weakness where grip was not measured"),
        ("FQ4", "Slowness", "Slow gait, or self-reported walking difficulty where gait was not timed"),
        ("FQ5", "Low physical activity", "Low self-reported physical activity")]
fried_ops = {}
for co in ["SHARE", "HRS", "ELSA", "MHAS"]:
    sfs = sorted(glob.glob(os.path.join(ROOT, CK[co]["folder"], "Wave *", "Steps followed*.md")))
    for s in sfs:
        p = parse_steps(s)
        if p["fried"]:
            fried_ops[co] = dict(ops=p["fried"], method=p["method"])
            break
if "LASI" not in EXCLUDE:
  fried_ops["LASI"] = dict(method="LASI reference protocol (Fried 2001 adapted), age 60+",
                           ops={1: "CES-D items: everything was an effort, or could not get going",
                                2: "Low BMI or self-reported weight loss (bmi_score)",
                                3: "Measured grip: mean of readings, bottom 20% within sex × BMI group",
                                4: "Timed 4 m walk: slowest 20% within sex × height group",
                                5: "Low self-reported physical activity"})
MEASURE = {"FQ3": {"SHARE": "measured", "HRS": "measured", "ELSA": "measured", "MHAS": "self-report", "LASI": "measured"},
           "FQ4": {"SHARE": "self-report", "HRS": "self-report", "ELSA": "measured", "MHAS": "self-report", "LASI": "measured"}}
for i, (q, name, desc) in enumerate(CRIT, 1):
    db["criteria"].append(dict(id=f"FDBF{i:02d}", key=q, name=name, desc=desc,
                               ops={co: v["ops"].get(i) for co, v in fried_ops.items()},
                               measure={k: v for k, v in MEASURE.get(q, {}).items() if k in CK},
                               prev={co: dict(v) for co, v in crit_prev[q].items()},
                               by_age={co: {b: (round(100 * v[0] / v[1], 2) if v[1] >= MIN_CELL else None) for b, v in crit_age[q][co].items() if b}
                                       for co in crit_age[q]}))
db["fried_methods"] = {co: v["method"] for co, v in fried_ops.items()}

# --------------------------------------------------------------------------- age curves (pooled over waves)
for co, meths in agecurve_acc.items():
    db["agecurves"][co] = {}
    for m, cells in meths.items():
        rows = []
        for a5 in AGE5:
            for sx, sl in [(1, "Male"), (2, "Female"), (0, "All")]:
                parts = [v for (aa, s), vs in cells.items() if aa == a5 and (s == sx or sx == 0) for v in vs]
                n = sum(p[1] for p in parts)
                if n < MIN_CELL:
                    continue
                rows.append(dict(age=a5, sex=sl, n=n, frail_pct=round(100 * sum(p[0] for p in parts) / n, 2),
                                 mean=round(sum(p[2] for p in parts) / n, 4)))
        db["agecurves"][co][m] = rows

# --------------------------------------------------------------------------- FI norms (pooled person-waves)
QS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
db["norms"] = dict(q=QS, cohorts={})
for (co, m), frames in long_frames.items():
    if m != "rock":
        continue
    allp = pd.concat(frames.values())
    allp = allp[allp.age >= 50]
    allp["a5"] = (allp.age.clip(upper=95) // 5 * 5).astype(int)
    rows = []
    for a5 in AGE5:
        for sx, sl in [(1, "Male"), (2, "Female"), (0, "All")]:
            g = allp[(allp.a5 == a5) & ((allp.sex == sx) if sx else True)]
            if len(g) >= MIN_PCTL:
                rows.append(dict(age=a5, sex=sl, n=int(len(g)), v=g.fi.quantile(QS).round(4).tolist()))
    db["norms"]["cohorts"][co] = rows

# --------------------------------------------------------------------------- SHARE countries
nid = 0
for cc in sorted(country_acc, key=lambda x: SHARE_COUNTRY.get(x, str(x))):
    if cc not in SHARE_COUNTRY:
        continue
    nid += 1
    waves = []
    for wn in sorted(country_acc[cc]):
        v = country_acc[cc][wn]
        row = dict(wave=wn)
        if v["rock"]:
            n, fr, s = v["rock"]
            row.update(rock_n=sup(n), rock_frail_pct=pct(fr, n), rock_mean=round(s / n, 4) if n >= MIN_CELL else None)
        if v["fried"]:
            n, fr, pf = v["fried"]
            row.update(fried_n=sup(n), fried_frail_pct=pct(fr, n), fried_prefrail_pct=pct(pf, n))
        waves.append(row)
    db["countries"].append(dict(id=f"FDBN{nid:02d}", code=cc, name=SHARE_COUNTRY[cc], cohort="SHARE", waves=waves))

# --------------------------------------------------------------------------- longitudinal: transitions, volatility, concordance
ROCK_STATES = ["FI < 0.10", "FI 0.10–0.24", "FI ≥ 0.25"]
FRIED_STATES = ["non-frail", "pre-frail", "frail"]


def rock_state(fi):
    return np.select([fi < 0.10, fi < 0.25], ROCK_STATES[:2], ROCK_STATES[2])


for (co, m), frames in long_frames.items():
    waves = sorted(frames)
    out = dict(waves=waves, transitions=[])
    states = ROCK_STATES if m == "rock" else FRIED_STATES
    pooled = np.zeros((3, 4), dtype=int)
    # Destination = the person's status at their next scored assessment within the next two analysed waves.
    # (HRS measures grip on a rotating half-sample, so Fried assessments recur every second wave.)
    smap = {w: dict(zip(frames[w].id, rock_state(frames[w].fi) if m == "rock" else frames[w].status.values)) for w in waves}
    for i, a in enumerate(waves[:-1]):
        nxt = waves[i + 1:i + 3]
        mat = np.zeros((3, 4), dtype=int)
        for i_, s in smap[a].items():
            t = next((smap[b][i_] for b in nxt if i_ in smap[b]), None)
            mat[states.index(s), states.index(t) if t is not None else 3] += 1
        pooled += mat
        out["transitions"].append(dict(frm=a, to=nxt, matrix=[[sup(x) for x in row] for row in mat.tolist()]))
    out["pooled"] = [[sup(x) for x in row] for row in pooled.tolist()]
    out["states"] = states + ["not scored at next assessment"]
    # trajectories of the panel (persons observed in every wave would be tiny; use all observed person-waves)
    if m == "rock":
        long = pd.concat([f.assign(wave=w) for w, f in frames.items()])
        per = long.groupby("id").agg(k=("fi", "size"), mean=("fi", "mean"), sd=("fi", "std"))
        # RMSSD in wave order
        def rmssd(s):
            d = np.diff(s.values)
            return float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan
        ls = long.sort_values(["id", "wave"])
        per["rmssd"] = ls.groupby("id")["fi"].apply(rmssd)
        v = per[per.k >= 3]
        if len(v) >= MIN_CELL:
            h = np.histogram(v.sd.clip(upper=0.299), bins=np.linspace(0, 0.30, 31))[0]
            bands_ = [(0, .1, "Mean FI < 0.10"), (.1, .25, "Mean FI 0.10–0.24"), (.25, 1.01, "Mean FI ≥ 0.25")]
            byb = []
            for lo, hi, lab in bands_:
                g = v[(v["mean"] >= lo) & (v["mean"] < hi)]
                if len(g) >= MIN_PCTL:
                    byb.append(dict(label=lab, n=len(g), sd_p=g.sd.quantile([.1, .25, .5, .75, .9]).round(4).tolist(),
                                    rmssd_med=round(float(g.rmssd.median()), 4)))
            kdist = per.k.value_counts().sort_index()
            out["volatility"] = dict(n=int(len(v)), min_waves=3, sd_hist=[sup(x) for x in h], sd_bin=0.01,
                                     sd_p=v.sd.quantile([.1, .25, .5, .75, .9]).round(4).tolist(),
                                     rmssd_p=v.rmssd.quantile([.1, .25, .5, .75, .9]).round(4).tolist(),
                                     corr_mean_sd=round(float(v[["mean", "sd"]].corr().iloc[0, 1]), 3),
                                     by_level=byb,
                                     waves_per_person={int(k): sup(n) for k, n in kdist.items()},
                                     persons=int(len(per)))
        traj = []
        for w in waves:
            fi = frames[w].fi
            traj.append(dict(wave=w, n=len(fi), p=fi.quantile([.1, .25, .5, .75, .9]).round(4).tolist(), mean=round(float(fi.mean()), 4)))
        out["trajectory"] = traj
    db["longitudinal"].setdefault(co, {})[m] = out

# Fried × Rockwood concordance within the same wave
for co in set(k[0] for k in long_frames):
    if (co, "rock") not in long_frames or (co, "fried") not in long_frames:
        continue
    R, F = long_frames[(co, "rock")], long_frames[(co, "fried")]
    pooled = np.zeros((3, 2), dtype=int); fi_by = defaultdict(list); per_wave = []
    for w in sorted(set(R) & set(F)):
        mm = F[w].merge(R[w], on="id", suffixes=("_f", "_r"))
        if len(mm) < MIN_CELL:
            continue
        mat = np.zeros((3, 2), dtype=int)
        for s, fr in zip(mm.status, mm.frail):
            mat[FRIED_STATES.index(s), int(fr)] += 1
        pooled += mat
        for s, g in mm.groupby("status"):
            fi_by[s].append((g.fi.sum(), len(g)))
        # Cohen's kappa on frail (Fried) vs frail (FI ≥ 0.25)
        a = (mm.status == "frail").astype(int); b = mm.frail
        po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        per_wave.append(dict(wave=w, n=len(mm), kappa=round(float((po - pe) / (1 - pe)), 3) if pe < 1 else None,
                             matrix=[[sup(x) for x in row] for row in mat.tolist()]))
    if per_wave:
        db["concordance"].append(dict(cohort=co, matrix=[[sup(x) for x in row] for row in pooled.tolist()], waves=per_wave,
                                      mean_fi_by_fried={s: round(sum(x[0] for x in v) / sum(x[1] for x in v), 4) for s, v in fi_by.items()}))

# --------------------------------------------------------------------------- instruments & references
db["instruments"] = [
    dict(id="FDBI01", key="fried", name="Fried frailty phenotype",
         summary="Five criteria scored 0/1: exhaustion, weight loss, weakness, slowness and low physical activity. "
                 "Score 0 = non-frail, 1–2 = pre-frail, 3–5 = frail.",
         scoring="frailty_score_fried = FQ1 + FQ2 + FQ3 + FQ4 + FQ5 (0–5)", cutoff="≥ 3 criteria = frail; 1–2 = pre-frail",
         refs=["FDBR01", "FDBR03", "FDBR04", "FDBR07", "FDBR10"]),
    dict(id="FDBI02", key="rockwood", name="Rockwood frailty index (deficit accumulation)",
         summary="Proportion of 32 health deficits present, covering disease, function, mobility, senses, mood and cognition. "
                 "Each deficit is coded 0–1; missing items are not imputed (complete case).",
         scoring="frailty_index_score = total_deficits / 32", cutoff="FI ≥ 0.25 = frail",
         refs=["FDBR05", "FDBR02", "FDBR06", "FDBR09"]),
]
db["references"] = [
    dict(id="FDBR01", cite="Fried LP, Tangen CM, Walston J, et al. Frailty in older adults: evidence for a phenotype. J Gerontol A Biol Sci Med Sci. 2001;56(3):M146–M156.", doi="10.1093/gerona/56.3.M146", used_for="Fried phenotype definition"),
    dict(id="FDBR02", cite="Wang C, Qian X, Zhao X, Jin M, Jia C, Jia F. Frailty index changes before and after stroke: findings from four cohorts covering 18 countries. Int J Stroke. 2026;21(6):871–878.", doi="10.1177/17474930251396655", used_for="FI item selection and age-50+ justification across CHARLS, ELSA, HRS and SHARE"),
    dict(id="FDBR03", cite="Romero-Ortuno R, Walsh CD, Lawlor BA, Kenny RA. A frailty instrument for primary care: findings from the Survey of Health, Ageing and Retirement in Europe (SHARE). BMC Geriatr. 2010;10:57.", doi="10.1186/1471-2318-10-57", used_for="SHARE Fried operationalisation (also applied to HRS)"),
    dict(id="FDBR04", cite="Marconcin P, Barak S, Ferrari G, et al. Prevalence of frailty and its association with depressive symptoms among European older adults from 17 countries: a 5-year longitudinal study. Int J Environ Res Public Health. 2022;19(21):14055.", doi="10.3390/ijerph192114055", used_for="SHARE Fried prevalence comparison"),
    dict(id="FDBR05", cite="Theou O, Haviva C, Wallace L, Searle SD, Rockwood K. How to construct a frailty index from an existing dataset in 10 steps. Age Ageing. 2023;52(12):afad221.", doi="10.1093/ageing/afad221", used_for="FI construction rules (≥ 30 deficits, multiple domains)"),
    dict(id="FDBR06", cite="Chao YS, Wu HC, Wu CJ, Chen WC. Index or illusion: the case of frailty indices in the Health and Retirement Study. PLoS One. 2018;13(7):e0197859.", doi="10.1371/journal.pone.0197859", used_for="HRS frailty index reference"),
    dict(id="FDBR07", cite="Gale CR, Cooper C, Sayer AA. Prevalence of frailty and disability: findings from the English Longitudinal Study of Ageing. Age Ageing. 2015;44(1):162–165.", doi="10.1093/ageing/afu148", used_for="ELSA Fried operationalisation (measured grip and gait, age 60+)"),
    dict(id="FDBR08", cite="Michaels-Obregon A, Wilkens J, Phillips D, Wong R, Lee J. Gateway Harmonized MHAS, Version D (2001–2022). Gateway to Global Aging Data, 2025.", doi="10.25553/ws6f-3a", used_for="MHAS harmonised variables"),
    dict(id="FDBR09", cite="Wang HY, Zhang M, Sun X. Sex-specific association between socioeconomic status, lifestyle, and the risk of frailty among the elderly in China. Front Med. 2021;8:775518.", doi="10.3389/fmed.2021.775518", used_for="CLHLS frailty index items"),
    dict(id="FDBR10", cite="Bandeen-Roche K, et al. Self-report substitution for grip strength and slow walk in the Fried frailty phenotype. BMC Geriatr. 2023;23:451.", doi="10.1186/s12877-023-04105-8", used_for="Method for self-reported weakness and slowness (KLoSA, MHAS)"),
]

# --------------------------------------------------------------------------- meta
rock_waves = [w for w in db["waves"] if w.get("rock")]
fried_waves = [w for w in db["waves"] if w.get("fried")]
db["meta"] = dict(
    name="FrailDB", version="0.1 (sample build)", built=pd.Timestamp.now().strftime("%Y-%m-%d"),
    min_cell=MIN_CELL,
    counts=dict(cohorts=len([c for c in db["cohorts"]]), cohorts_analysed=len([c for c in db["cohorts"] if c["waves"]]),
                waves=len(db["waves"]), deficits=len(db["deficits"]), criteria=len(db["criteria"]),
                countries=len(db["countries"]) + len([c for c in db["cohorts"] if c["waves"] and c["key"] != "SHARE"]),
                share_countries=len(db["countries"]),
                rock_assessments=int(sum(w["rock"]["n"] for w in rock_waves if w["rock"]["n"])),
                fried_assessments=int(sum(w["fried"]["n"] for w in fried_waves if w["fried"]["n"])),
                references=len(db["references"]), cutoff_tables=len(db["cutoffs"]),
                persons=int(sum(len(set().union(*[set(f.id) for f in fr.values()])) for (co, m), fr in long_frames.items() if m == "rock"))),
)


def jdefault(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def nan_to_none(o):
    if isinstance(o, dict): return {str(k): nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list): return [nan_to_none(v) for v in o]
    if isinstance(o, float) and math.isnan(o): return None
    return o


# age-window analysis (FrailDB/age_window.py) is merged in when present
AW = os.path.join(os.path.dirname(OUT), "age_window.json")
if os.path.exists(AW):
    db["agewindow"] = json.load(open(AW, encoding="utf-8"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nan_to_none(json.loads(json.dumps(db, default=jdefault))), fh, ensure_ascii=False, separators=(",", ":"))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
print(json.dumps(db["meta"], indent=1, default=jdefault))
