"""
Most influential predictors of frailty, per wave and per dataset.

Two independent tracks: Rockwood frailty index (continuous FI score, 0–1) and Fried phenotype
(continuous score, 0–5). For every analysed wave of every cohort:

  1. Outcome   = the person's score from the existing per-wave frailty outputs.
  2. Predictors = every variable in that wave's raw survey files (SHARE, ELSA, CLHLS; HRS 2020–2022),
                 or the full wave slice of the Gateway harmonised file where raw files are not
                 available (HRS 1998–2018, MHAS).
  3. Exclusions (each logged with its reason):
       admin       IDs, weights, interview dates/modes, proxy flags, design variables
       constituent the instrument's own source variables (explicit list per cohort)
       derived     summary scores built from the same items (ADL/IADL/mobility counts, CES-D/EURO-D, grip…)
       near-copy   data-driven: |Spearman rho| >= 0.75 with any constituent item of that instrument,
                   or >= 0.80 with the outcome score itself
       quality     > 95% missing, constant, or free text
  4. Model     = LightGBM regression, 80/20 split; held-out R² reported.
  5. Influence = mean |SHAP| on held-out people (LightGBM pred_contrib). Direction = sign of the
                 correlation between the variable and its SHAP value.
  6. Per wave: top 10 and top 20. Per dataset: mean share of total |SHAP| across waves
     (0 when absent), and how often the variable is in a wave's top 10 / top 20 -> overall top 5.

Results are variable names and importances only (no participant data).
Resumable: each wave's result is saved to output/parts/ and skipped on re-run.

Run:  python FrailDB/predictors/run_predictors.py [COHORT ...]
"""
import os, re, sys, json, glob, zipfile, tempfile, time, gc, warnings
import numpy as np, pandas as pd, pyreadr, pyreadstat
import lightgbm as lgb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "..", "Data Sources")
OUT = os.path.join(HERE, "output"); PARTS = os.path.join(OUT, "parts")
os.makedirs(PARTS, exist_ok=True)
RNG = 20261018
MAX_MISSING = 0.95
RHO_ITEM, RHO_OUTCOME = 0.75, 0.80
SCREEN_N, SHAP_N = 5000, 4000


import hashlib
SCRIPT_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()   # recorded in every result


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ outcomes
ANALYSIS = {
    "SHARE": "2. SHARE data set_wave 1-9/SHARE Analysis",
    "HRS": "3. HRS_United States/HRS Analysis",
    "ELSA": "4. ELSA_England/ELSA Analysis",
    "MHAS": "6. MHAS_Mexico/MHAS Analysis",
    "CLHLS": "6. CLHLS_China/CLHLS Analysis",
}
WAVES = {
    ("SHARE", "Rockwood"): [2, 4, 5, 6, 8, 9], ("SHARE", "Fried"): [1, 2, 4, 5, 6, 7, 8, 9],
    ("HRS", "Rockwood"): list(range(4, 17)), ("HRS", "Fried"): list(range(7, 15)),
    ("ELSA", "Rockwood"): [1, 3, 4, 5, 6, 7, 8, 9, 10], ("ELSA", "Fried"): [4, 6, 9],
    ("MHAS", "Rockwood"): [1, 2, 4, 5], ("MHAS", "Fried"): [1, 2, 5, 6],
    ("CLHLS", "Rockwood"): [3, 6, 7, 8],
}
FRIED_RENAME = {"f_exhaustion": "FQ1", "f_weight_loss": "FQ2", "f_weakness": "FQ3", "f_slowness": "FQ4", "f_low_activity": "FQ5",
                "FQ1_exhaustion": "FQ1", "FQ2_weightloss": "FQ2", "FQ3_weakness": "FQ3", "FQ4_slowness": "FQ4", "FQ5_lowactivity": "FQ5"}


def load_outcome(cohort, method, w):
    base = os.path.join(DS, ANALYSIS[cohort], f"Wave {w}", f"{cohort}_w{w}_{method}")
    if os.path.exists(base + "_summary.csv"):
        df = pd.read_csv(base + "_summary.csv", low_memory=False)
    elif os.path.exists(base + ".rds"):
        df = list(pyreadr.read_r(base + ".rds").values())[0]
        if method == "Fried" and df.shape[1] > 40:      # SHARE Fried .rds carries the whole wave; keep outcome columns only
            df = df[[c for c in df.columns if c in ("mergeid", "age", "sex", "BMI", "grip") or c in FRIED_RENAME or c.startswith("frailty")]]
    else:
        return None
    df = df.rename(columns=FRIED_RENAME)
    out = pd.DataFrame({"id": norm_id(cohort, df.iloc[:, 0])})
    if method == "Rockwood":
        out["y"] = pd.to_numeric(df["frailty_index_score"], errors="coerce").values
        items = [c for c in df.columns if c.startswith("d_")]
    else:
        out["y"] = pd.to_numeric(df["frailty_score_fried"], errors="coerce").values
        items = [c for c in ["FQ1", "FQ2", "FQ3", "FQ4", "FQ5", "grip", "BMI", "walk_time", "activity_score"] if c in df.columns]
    for c in items:
        out["item__" + c] = pd.to_numeric(df[c], errors="coerce").values
    return out.dropna(subset=["y"]).drop_duplicates("id")


def norm_id(cohort, s):
    if cohort in ("SHARE",):
        return s.astype(str).str.strip()
    return pd.to_numeric(s, errors="coerce").round().astype("Int64").astype(str)


# ============================================================================ predictor sources
SHARE_SKIP = {"weights", "exrates", "imputations", "variables", "xt", "dropoff_technical", "technical_variables",
              "interviewer", "iv", "gv_weights", "gv_imputations", "gv_exrates", "gv_linkage", "gv_ssw", "gv_dbs", "gv_big5",
              "gv_children", "gv_networks", "sn", "rp", "re", "rc", "st", "gl", "wq", "ac_", "as_long", "gv_allwaves_cv_r",
              "lifehistory", "gv_housing_long"}


def share_files(w):
    root = os.path.join(DS, "2. SHARE data set_wave 1-9", "SHARE data")
    if w == 9:
        d = os.path.join(root, "SHARE_Europe_wave 9", "SHARE Data wave 9", "sharew9_rel9-0-0_ALL_datasets_spss")
        return [(os.path.basename(f), f) for f in sorted(glob.glob(os.path.join(d, "*.sav")))], None
    z = zipfile.ZipFile(os.path.join(root, f"sharew{w}_rel9-0-0_ALL_datasets_stata.zip"))
    return [(n, n) for n in z.namelist() if n.endswith(".dta")], z


def read_stat(path, usecols=None):
    if path.lower().endswith(".sav"):
        return pyreadstat.read_sav(path, usecols=usecols, apply_value_formats=False)
    return pyreadstat.read_dta(path, usecols=usecols, apply_value_formats=False)


def share_predictors(w, ids):
    files, z = share_files(w)
    frames, labels, tmp = [], {}, tempfile.mkdtemp()
    for name, ref in files:
        mod = re.sub(r"^sharew\d+_rel[\d-]+_", "", os.path.basename(name)).rsplit(".", 1)[0]
        if mod in SHARE_SKIP or any(mod.startswith(s) for s in ("gv_weights", "gv_imput", "gv_exrates", "gv_linkage")):
            continue
        path = z.extract(ref, tmp) if z else ref
        try:
            df, meta = read_stat(path)
        except Exception as e:
            log("   skip", mod, e); continue
        finally:
            if z: os.remove(path)
        if "mergeid" not in df.columns:
            continue
        df = df[df.mergeid.isin(ids)]
        if df.mergeid.duplicated().any():
            continue                                        # long-format module (one row per child, partner…) – not person-level
        df = df.set_index("mergeid")
        df.columns = [f"{c}" for c in df.columns]
        labels.update({c: f"[{mod}] {meta.column_names_to_labels.get(c) or ''}".strip() for c in df.columns})
        frames.append(clean_frame(df, negatives_missing=True))
        del df; gc.collect()
    X = pd.concat(frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, labels


HRS_RAW = {15: "Wave 15 (2020)", 16: "Wave 16 (2022)"}


def hrs_predictors(w, ids):
    if w in HRS_RAW:
        frames, labels = [], {}
        for f in sorted(glob.glob(os.path.join(DS, "3. HRS_United States", "HRS data", "Raw Core Waves", HRS_RAW[w], "stata", "*_R.dta"))):
            df, meta = pyreadstat.read_dta(f, apply_value_formats=False)
            df["id"] = norm_id("HRS", pd.to_numeric(df.HHID, errors="coerce") * 1000 + pd.to_numeric(df.PN, errors="coerce"))
            df = df[df.id.isin(ids)]
            if df.id.duplicated().any():
                continue
            sec = os.path.basename(f).replace(".dta", "")
            labels.update({c: f"[{sec}] {meta.column_names_to_labels.get(c) or ''}" for c in df.columns})
            frames.append(clean_frame(df.set_index("id")))
        X = pd.concat(frames, axis=1)
        return X.loc[:, ~X.columns.duplicated()], labels
    return harmonised("3. HRS_United States/HRS data/H_HRS_d.dta", "hhidpn", w, ids, "HRS")


def harmonised(rel, idcol, w, ids, cohort):
    path = os.path.join(DS, rel)
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    pat = re.compile(rf"^(r{w}|h{w})[a-z]|^ra[a-z]")       # respondent + household for wave w, time-invariant ra*; spouse s* excluded
    cols = [idcol] + [c for c in meta.column_names if pat.match(c)]
    df, meta = pyreadstat.read_dta(path, usecols=cols, apply_value_formats=False)
    df["id"] = norm_id(cohort, df[idcol])
    df = df[df.id.isin(ids)].drop_duplicates("id").set_index("id").drop(columns=[idcol])
    labels = {c: f"[harmonised] {meta.column_names_to_labels.get(c) or ''}" for c in df.columns}
    return clean_frame(df), labels


ELSA_CORE = {1: "wave_1_core_data_v3", 2: "wave_2_core_data_v4", 3: "wave_3_elsa_data_v4", 4: "wave_4_elsa_data_v3",
             5: "wave_5_elsa_data_v4", 6: "wave_6_elsa_data_v2", 7: "wave_7_elsa_data", 8: "wave_8_elsa_data_eul_v2",
             9: "wave_9_elsa_data_eul_v2", 10: "wave_10_elsa_data_eul_v4", 11: "wave_11_elsa_data_eul_v1"}
ELSA_NURSE = {2: "wave_2_nurse_data_v2", 4: "wave_4_nurse_data", 6: "wave_6_elsa_nurse_data_v2"}


def elsa_predictors(w, ids):
    d = os.path.join(DS, "4. ELSA_England", "ELSA data", "UKDA-5050-stata", "stata13_se")
    frames, labels = [], {}
    for tag, fn in [("core", ELSA_CORE[w]), ("nurse", ELSA_NURSE.get(w))]:
        if not fn:
            continue
        df, meta = pyreadstat.read_dta(os.path.join(d, fn + ".dta"), apply_value_formats=False)
        df["id"] = norm_id("ELSA", df.idauniq)
        df = df[df.id.isin(ids)].drop_duplicates("id").set_index("id")
        labels.update({c: f"[{tag}] {meta.column_names_to_labels.get(c) or ''}" for c in df.columns})
        frames.append(clean_frame(df, negatives_missing=True).add_prefix("nurse_" if tag == "nurse" else ""))
        if tag == "nurse":
            labels.update({"nurse_" + c: labels[c] for c in df.columns})
    X = pd.concat(frames, axis=1)
    return X.loc[:, ~X.columns.duplicated()], labels


def mhas_predictors(w, ids):
    return harmonised("6. MHAS_Mexico/MHAS data/H_MHAS_d.dta", "unhhidnp", w, ids, "MHAS")


CLHLS_DS = {3: "DS0003", 6: "DS0006", 7: "DS0007", 8: "DS0008"}


def clhls_predictors(w, ids):
    # .rda parsing in Python takes 10+ min for the 5,500-column 2002 file; a one-off local pickle cache
    # (FrailDB/.cache, git-ignored, never published) is made by cache_clhls.py and used when present.
    cache = os.path.join(HERE, "..", ".cache", f"clhls_{CLHLS_DS[w]}.pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        df = list(pyreadr.read_r(os.path.join(DS, "6. CLHLS_China", CLHLS_DS[w], f"38899-{CLHLS_DS[w][2:]}-Data.rda")).values())[0]
    # DS0003/6/7 are longitudinal: they also carry every later follow-up (suffix _5, _8, _11, _14, _18) and
    # death / loss-to-follow-up status (DTH..). Keep this wave's own baseline variables only, or future health leaks in.
    df = df[[c for c in df.columns if not re.search(r"_\d+$", c) and not c.upper().startswith("DTH")]]
    idc = [c for c in df.columns if c.upper() == "ID"][0]
    df["id"] = norm_id("CLHLS", df[idc])
    df = df[df.id.isin(ids)].drop_duplicates("id").set_index("id")
    for c in df.columns:
        if str(df[c].dtype) == "category":
            df[c] = df[c].cat.codes.replace(-1, np.nan)     # ICPSR factors -> level codes
    lab = clhls_labels(CLHLS_DS[w])
    return clean_frame(df), {c: f"[{CLHLS_DS[w]}] {lab.get(c.upper(), '')}" for c in df.columns}


def clhls_labels(ds):
    """Variable labels from the ICPSR codebook PDF (lines 'NAME: label'), cached."""
    cache = os.path.join(OUT, f"clhls_labels_{ds}.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    import fitz                                          # PyMuPDF: ~100x faster than pypdf on these 500–3,500 page codebooks
    pdf = os.path.join(DS, "6. CLHLS_China", ds, f"38899-{ds[2:]}-Codebook-ICPSR.pdf")
    lab = {}
    text = "\n".join(p.get_text() for p in fitz.open(pdf))
    for m in re.finditer(r"^([A-Z][A-Z0-9_]{0,15}): (.+)$", text, re.M):
        lab.setdefault(m.group(1), m.group(2).strip())
    json.dump(lab, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return lab


SOURCES = {"SHARE": share_predictors, "HRS": hrs_predictors, "ELSA": elsa_predictors, "MHAS": mhas_predictors, "CLHLS": clhls_predictors}


def source_label(cohort, w):
    if cohort == "HRS":
        return "raw core files" if w in HRS_RAW else "harmonised H_HRS (raw core not available)"
    if cohort == "MHAS":
        return "harmonised H_MHAS (raw questionnaire files not available)"
    return {"SHARE": "raw modules (all person-level)", "ELSA": "raw core" + (" + nurse visit" if w in ELSA_NURSE else ""),
            "CLHLS": f"raw ICPSR {CLHLS_DS.get(w, '')}"}[cohort]


def clean_frame(df, negatives_missing=False):
    """Numeric float32 frame; short-category strings become codes, free text dropped."""
    out = {}
    for c in df.columns:
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith("string"):
            s = s.replace("", np.nan)
            if s.nunique(dropna=True) > 30:
                continue
            s = pd.Series(pd.factorize(s)[0], index=s.index).replace(-1, np.nan)
        s = pd.to_numeric(s, errors="coerce")
        if negatives_missing:
            s = s.where(s >= 0)
        if s.notna().mean() < 1 - MAX_MISSING or s.nunique(dropna=True) <= 1:
            continue
        out[c] = s.astype("float32")
    return pd.DataFrame(out, index=df.index)


# ============================================================================ exclusions
ADMIN_NAME = re.compile(
    r"^(id|mergeid.*|coupleid.*|hhid.*|.*hhid|pn|hhidpn|idauniq|idahh.*|perid|unhhid.*|rahhidnp|cunicah|np|codent.*|subhog.*|acthog|"
    r"country.*|language|.*wgt.*|.*wght.*|cciw.*|cchw.*|dw_.*|.*_w\d*_weight|(r|h)\d+wt.*|"
    r"int_?year|int_?month|.*iwy|.*iwm|.*iwmid|.*iwbeg|.*iwend|.*iwstat|.*iwlang|inw\d.*|.*proxy.*|.*psu.*|.*strat.*|.*sec[uw]|"
    r"interviewer.*|.*intdat.*|.*_mode|mode|.*samp.*|.*version|wave|.*finr.*|.*resp(onden)?t.*type.*|residenc|prov|.*province.*)$", re.I)
ADMIN_LABEL = re.compile(r"(identifier|id number|household id|person number|sample weight|cross-sectional weight|longitudinal weight|"
                         r"calibrated weight|design weight|weight variable|interview date|date of interview|month of interview|"
                         r"year of interview|interviewer|proxy|stratum|strata|primary sampling|psu|sampling|interview mode|language of|"
                         r"country identifier|wave status|response status|in wave|"
                         r"death|died|deceased|decedent|exit interview|mortality|vital status|lost to follow|"
                         # interview paradata: section timings, time stamps, interviewer observations
                         r"total time|in seconds|time ?stamp|timing|duration of (the )?interview|interviewer observ|"
                         r"iwer obs|showcard|needed help (to|with) (answer|read)|willingness to answer|"
                         r"effort that r gave|record respondent status|safe to do|position of r for this test|"
                         r"helped (the )?interviewee|who answered|assisted (the )?respondent|country-specific|"
                         r"reason for (not obtaining|non-completion)|not completed successfully|willing (and able )?to (do|complete)|"
                         r"outcome of (first|second)|floor surface|type of floor|compliance during|followed directions|"
                         r"understanding of .{0,20}procedure|screening question|safe to (do|carry out)|feels walk would be safe|"
                         r"complete(d)? at least one trial|completed .{0,20}as directed|comments why|\binterviewee\b|"
                         r"reason (for )?(inability|unable)|unable to (answer|participate|take part)|inability to participate)", re.I)
# Harmonised ra* variables are built across all waves ("last reported", "ever"), so they can carry later-wave
# information. Only characteristics fixed before or at entry are kept: sex, race/ethnicity, birth, education,
# parents' education, childhood circumstances, place of birth, religion, veteran status, language.
RA_FIXED = re.compile(r"^ra(gender|female|racem|race|hispan|ethn|bplace|bplcf|byear|bmonth|bdate|cohbyr|"
                      r"educ|educl|edyrs|edegrm|edegree|dadeduc|momeduc|dadocc|momocc|dadliv|momliv|"
                      r"ch[a-z]*|relig|vetrn|veteran|lang|natv|citizen|cendiv|urbrur)\w*$", re.I)
# Survey modules that record help or care RECEIVED for functional difficulty, matched by name because their
# labels often omit the word "care" (ELSA ca*: care module, CASP scores excepted; SHARE sp002–sp007: help received
# from outside the household — help GIVEN by the respondent, sp008 onward, stays a predictor).
HELP_MODULES = {("ELSA", "Rockwood"): r"^ca(?!sp)[a-z]", ("SHARE", "Rockwood"): r"^sp00[2-7]"}
# Death / exit-interview content, incl. CLHLS interviews with relatives about later deaths ("the elder", "before dying")
DEATH_LABEL = re.compile(r"death|died|deceased|dying|decedent|\bthe elder\b|mortality|vital status|lost to follow", re.I)
FUTURE_NAME = re.compile(r"^ra(d|x)[a-z]*$|^dth.*|^ff[a-z]", re.I)   # death / exit-interview (radyear, raxyear…); ELSA ff* = fed forward from the previous wave

# instrument source variables per cohort (raw or harmonised stems); matched case-insensitively.
# Harmonised names are r{w}<stem>: we match on the stem after the wave prefix.
CONSTITUENTS = {
    # SHARE: whole grids incl. their "none of these" option (…dno), help/limitation follow-ups (ph004/5 GALI,
    # ph050/51 help, sp020–024 personal-care help), the repeated self-rated-health item (ph060/61), aids (ph059).
    ("SHARE", "Rockwood"): [r"ph003_", r"ph006d\w*", r"ph049d\w*", r"ph048d\w*", r"ph046_", r"ph043_", r"ph044_", r"ph045_",
                            r"ph00[45]_", r"ph05[0-9]\w*", r"ph06[01]_", r"sp02[0-4]\w*", r"mh0(0[2-9]|1[0-7])_"],
    # Fried: the grip (gs) and walking-speed (ws) test modules, plus the item variables themselves.
    ("SHARE", "Fried"): [r"mh013_", r"mh011_", r"mh012_", r"gs\d+\w*", r"ws\d+\w*", r"ph048d(1|5)", r"br01[56]_", r"ph012_", r"ph013_"],
    ("HRS", "Rockwood"): [r"shlt", r"hibpe", r"diabe", r"cancre", r"lunge", r"hearte", r"stroke", r"arthre", r"psyche", r"memrye", r"alzhee", r"demene",
                          r"dressa", r"batha", r"eata", r"toilta", r"beda", r"walkra", r"mealsa", r"shopa", r"medsa", r"moneya", r"phonea",
                          r"walk1a", r"walksa", r"chaira", r"climsa", r"clim1a", r"lifta", r"dimea", r"stoopa", r"armsa", r"pusha", r"sita",
                          r"hearing", r"earrt", r"eyesight", r"sight", r"eyert", r"cesd", r"depres", r"effort", r"sleepr", r"whappy", r"flone",
                          r"fsad", r"going", r"enlife", r"[rs]d11[0-8]"],      # [rs]d110–118: raw 2020/2022 CES-D block
    ("HRS", "Fried"): [r"effort", r"going", r"weight", r"bmi", r"height", r".?grip.*", r"walk1a", r"climsa", r"vgactx", r"mdactx", r"ltactx"],
    ("MHAS", "Rockwood"): [r"shlt", r"hibpe", r"diabe", r"cancre", r"respe", r"hrtatte", r"hearte", r"stroke", r"arthre", r"dressa", r"batha",
                           r"eata", r"toilta", r"beda", r"walkra", r"walkre", r"mealsa", r"shopa", r"medsa", r"moneya", r"walk1a", r"walksa", r"chaira",
                           r"climsa", r"clim1a", r"lifta", r"dimea", r"stoopa", r"armsa", r"pusha", r"sita", r"hearing", r"sight", r"cesd",
                           r"depres", r"effort", r"tired", r"sleepr", r"whappy", r"flone", r"fsad", r"enlife"],
    ("MHAS", "Fried"): [r"effort", r"tired", r"bmi", r"weight", r"height", r"lifta", r"walk1a", r"walksa", r"vgact.*", r"exer.*"],
    # ELSA item families. Wave 1 names the ADL/IADL grids heada01–/headb01– (later waves headl–); Stata labels are
    # truncated at 80 characters, so these are matched by name: self-rated health, ADL/IADL/mobility grids, help
    # received and aids used for those difficulties, sight/hearing, CES-D items, and the doctor-diagnosed condition
    # grids with their follow-ups (age first told, medication, limiting).
    ("ELSA", "Rockwood"): [r"hehelf.*", r"hegenh.*", r"headl.*", r"heada\d+", r"headb\d+", r"hemob.*", r"hemda.*", r"hehp.*", r"heaid.*",
                           r"hehear.*", r"hefrnd.*", r"hefla.*", r"heeye.*", r"hesee.*", r"psced.*", r"pscedb.*",
                           r"hedi[a-z]*\d*", r"hedb[a-z]*\d*", r"heds[a-z]*\d*", r"heag[a-z]*\d*", r"hebp[a-z]*\d*", r"hedm[a-z]*\d*",
                           r"hehrt.*", r"hestr.*", r"hecan.*", r"hearth.*", r"helng.*", r"hedem.*", r"helim.*", r"helwk.*", r"heill.*",
                           r"cvd.*", r"totalpsc", r"psc.*", r"cesd.*",
                           r"heavoid.*", r"hefunc.*", r"hetemp.*", r"hehs[a-z]*\d*"],   # disability screener and its follow-ups
    ("ELSA", "Fried"): [r"pscedb", r"pscedh", r"wtval", r"bmival", r"htval", r"(nurse_)?mmgs.*", r"(nurse_)?mmwlk.*", r"heact[a-c]", r"(nurse_)?bmi.*",
                        r"(nurse_)?wtval", r"(nurse_)?htval"],
    ("CLHLS", "Rockwood"): [r"b12.*", r"g15[a-n]\d*", r"e\d+[a-z]?\d*", r"g9.*", r"b2[36].*", r"b3[46].*", r"d11a.*", r"g81.*"],
}
# ---- Generic exclusions (every cohort): summary scores built from instrument items, global activity limitation,
# help / care / aids received for functional difficulty, EQ-5D function dimensions, values fed forward from a past wave.
GENERIC = {
    "Rockwood": re.compile(
        r"(\badls?\b|\biadls?\b|activities of daily living|mobility|\bnagi\b|gross motor|fine motor|large muscle|"
        r"receiv\w* (any )?(formal |informal )?care|"
        r"\bgali\b|limitation|\blimit(s|ed|ing)?\b|long-standing illness|usual activities|health: ?mobility|health: ?self-care|"
        r"eurod|euro-d|ces-?d|frailty|number of (chronic )?(conditions|diseases)|chronic (diseases|conditions)|"
        r"help (with|moving)|received help|help received|help (you )?receive|meets needs|care (patterns|provider|arranged|package)|"
        r"local authority|council care|social services|\bcarer\b|physical functioning diffic|"
        r"\baids?\b ?:|aids used|use of aids|use any of the following|\bcane\b|walking stick|zimmer|wheelchair|crutch|"
        r"feedforward|feed forward|"
        r"health (stops|prevents|limits)|prevents? (you|them|r) from walk|preventing walking|able to walk alone|walk alone \(with aid|"
        r"adaptation|bed (lever|rail)|commode|stair ?lift|chair ?lift|stair glide|bathroom modif|grab rail|"
        r"walk\w* (what )?equipment|aid used|(bathroom|toilet)\W+(\w+\W+)?modif|\bthe lift\b|lift adaptation|"
        r"health: ?to walk|walk about|walking symptom|can.?t walk|cannot walk|help needed: ?medic|person who helps|paying for care)", re.I),
    "Fried": re.compile(
        r"(ces-?d|euro-?d|depress|mobility|\bnagi\b|gross motor|large muscle|frailty)", re.I),
}
GENERIC_NAME = {
    "Rockwood": re.compile(r"(^|_)(adl|iadl|adla|adlw|adlwa|iadla|iadlza|mobila|mobilsev|lgmusa|grossa|finea|gali|eurod|cesd|conde|sphus|spheu|chronic|nagi)", re.I),
    "Fried": re.compile(r"(grip|gait|wspeed|bmi|frail|cesd|eurod|mobil|nagi|gross|lgmus)", re.I),
}

# ---- Item-specific follow-ups: a rule applies only in cohorts whose index contains that item
# (keys are the item columns of the group's per-wave outputs; CLHLS spellings included).
ROCK_CONCEPTS = {
    "d_srh": r"self[- ]?(rated|perceived|reported|assessed) health|health in general|general health|health thermometer|"
             r"\brate (your |r'?s? )?health|spheu|sphus",
    "d_depression": r"depress|\bcidi\b|anxiety/depression|felt (sad|lonely|happy|depressed|unmotivated)|was (r |you )?happy|enjoyed life|"
                    r"sleep (was )?restless|could ?n.?t get going|everything (was )?an effort|activities were efforts?|unmotivated",
    "d_hypertension": r"high blood pressure|hypertens|\bhibp\b|blood pressure (med|pill|drug|treat|diagnos|problem|condition|control)",
    "d_diabetes": r"diabet|insulin|blood sugar|sugar in (the )?(blood|urine)",
    "d_cancer": r"\bcancer\b(?!\s*screen)|tumou?r|chemotherap|radiotherap|radiation therap",
    "d_lung": r"lung (disease|condition|problem|med|respiratory)|respiratory therap|chronic lung|copd|emphysema|chronic bronch|asthma|\boxygen\b|"
              r"(meds?|drugs?|medication) for:? ?(breathing|lung)",
    "d_heart": r"heart (disease|problem|condition|attack|trouble|failure|surgery|med|doctor|better|treat)|heart murmur|heart complaint|"
               r"abnormal heart rhythm|heart rhythm med|arrhythmia|irregular heart|(pain|discomfort) (or (pain|discomfort) )?in (your |the )?chest|across (the )?front of (your |the )?chest|chest pain|cardiac|angina|coronary|myocard|congestive|"
               r"(meds?|drugs?|medication) for:? ?heart",
    "d_stroke": r"\bstroke\b|transient isch",
    "d_arthritis": r"arthritis|osteoarth|rheumat|joint replace|hip replace|knee replace",
    "d_dementia": r"dementia|alzheimer|memory[- ]related|memory difficulty med|senility|(meds?|drugs?|medication) for:? ?memory",
    "d_adl_dress": r"\bdress(ing|ed)?\b",
    "d_adl_bath": r"\bbath(ing|e|ed)?\b|shower(ing)?",
    "d_adl_eat": r"\beating\b|\bfeeding\b|difficult\w*[- ]+eat",
    "d_adl_toilet": r"us(e|ing) (the )?toilet|\btoileting\b|get(ting)? to the toilet|toilet(ing)? (by yourself|without)",
    "d_adl_bed": r"in or out of bed|in(to)?/? ?(and )?out (of )?bed|get(ting)? (in|out of|into) bed|bed equipment",
    "d_adl_walkroom": r"across (a|the) room",
    "d_equip_walkroom": r"across (a|the) room",
    "d_iadl_meal": r"meals[- ]on[- ]wheels|frozen meals|brought (a )?meal|hot meal|prepar\w* (a )?(hot )?meals?|preparing food|meal preparation|cook\w* (a )?(hot )?meal|make food",
    "d_iadl_cook": r"meals[- ]on[- ]wheels|frozen meals|brought (a )?meal|hot meal|prepar\w* (a )?(hot )?meals?|preparing food|meal preparation|cook\w* (a )?(hot )?meal|make food",
    "d_iadl_shop": r"\bshopping\b|groceries",
    "d_iadl_meds": r"(taking|take|manag\w*) (your |own |r'?s? )?(medications?|medicines?|pills)|medication management",
    "d_iadl_money": r"manag\w* (your |own |r'?s? )?(money|finances)|handling money|money management|paying bills",
    "d_mob_walk": r"walk(ing|\(ing\))? (one|1|several|100|a block|blocks|a mile|1 mile|a quarter|quarter|km|kilomet|a short distance|a distance)|"
                  r"quarter of a mile|prevent\w* (you )?(from )?walk|"
                  r"help(s)? (r |you )?(to )?walk|difficult\w*[- ]+walk|walking (aid|stick|frame|equipment)|equipment\W+walk",
    "d_walk_1km": r"walk(ing)? (one|1|a) ?(km|kilomet)",
    "d_mob_chair": r"from (a )?chair|getting up from",
    "d_chair_stand": r"from (a )?chair|getting up from",
    "d_mob_stoop": r"stoop(ing)?|kneeling|crouch(ing)?",
    "d_crouch_stand": r"stoop(ing)?|kneeling|crouch(ing)?",
    "d_mob_lift": r"\blifting\b|\blift(ing)? (or carry|weights|5|10|heavy)|\bcarry(ing)? (5|10|heavy|weights|a heavy)",
    "d_carry_5kg": r"\bcarry(ing)? (5|10|heavy|weights|a heavy)|\blifting\b",
    "d_mob_sit2h": r"sit(ting)? for (about )?(2|two) hours",
    "d_mob_stairs_several": r"\bstairs\b|flights? of",
    "d_mob_stairs_one": r"\bstairs\b|flights? of",
    "d_mob_arms": r"arms above|reach\w* (up|above|or extend)",
    "d_mob_push": r"\bpush(ing)?\b|\bpull(ing)?\b",
    "d_mob_coin": r"\bcoin\b|\bdime\b",
    "d_hearing": r"\bhearing\b|\bdeaf|follow(ing)? a conversation|background noise|hear (well|clearly|a conversation|someone)",
    "d_vision": r"eyesight|vision|\bblind|see (well|clearly|things|print|at a distance)",
    "d_continence": r"continen",
    "d_visit_neighbours": r"visit\w* neighbo",
    "d_wash_clothes": r"wash\w* cloth",
    "d_public_transport": r"public transport",
    "d_housework": r"\bhousework\b",
    "d_hand_behind_neck": r"behind (the|your) neck",
}
FRIED_CONCEPTS = {  # criterion -> (pattern, cohorts where the criterion is measured this way; None = all)
    "FQ1": (r"effort|get going|could ?n.?t get going|tired|fatigue|exhaust|\benergy\b|unmotivated|worn out", None),
    "FQ2": (r"weight|appetite|eat(ing)? (less|more)|\bbmi\b|body mass|height", None),
    "FQ3": (r"grip|dynamometer|hand strength|muscle strength|\bstrength\b|\bweak(ness)?\b", None),
    "FQ3_lift": (r"\blifting\b|\blift(ing)? (or carry|5|10|heavy)", {"MHAS"}),
    "FQ4": (r"\bwalk|gait", None),
    "FQ4_stairs": (r"\bstairs\b|flights? of", {"SHARE", "HRS"}),
    "FQ5": (r"physical(ly)? activ|exercise|vigorous|moderate(ly)? (energetic|activ)|\bsports?\b|energetic", None),
}


def follow_up_rule(cohort, method, item_cols):
    """One regex of the item-specific follow-up rules that apply to this cohort and instrument."""
    if method == "Rockwood":
        pats = [p for k, p in ROCK_CONCEPTS.items() if f"item__{k}" in item_cols]
    else:
        pats = [p for p, cos in FRIED_CONCEPTS.values() if cos is None or cohort in cos]
    return re.compile("(" + "|".join(pats) + ")", re.I) if pats else None


def stem(name):
    return re.sub(r"^(nurse_)?(r|h|ra|s)\d{0,2}(?=[a-z])", "", name.lower()) if re.match(r"^(r|h|s)\d", name.lower()) else name.lower()


def exclusions(cohort, method, X, labels, Y):
    reasons = {}
    cons = [re.compile(rf"^{p}$", re.I) for p in CONSTITUENTS.get((cohort, method), [])]
    fu = follow_up_rule(cohort, method, set(Y.columns))
    for c in X.columns:
        lab = labels.get(c, "")
        lab_text = re.sub(r"^\[[^\]]*\]\s*", "", lab)
        base = c.lower().replace("nurse_", "")
        if FUTURE_NAME.match(base) or DEATH_LABEL.search(lab_text):
            reasons[c] = "admin (death / future information)"
        elif cohort in ("HRS", "MHAS") and re.match(r"^ra[a-z]", base) and not RA_FIXED.match(base):
            reasons[c] = "admin (cross-wave summary, may use later waves)"
        elif ADMIN_NAME.match(base) or ADMIN_LABEL.search(lab_text):
            reasons[c] = "admin"
        elif any(p.match(base) or p.match(stem(c)) for p in cons):
            reasons[c] = "constituent"
        elif cohort == "HRS" and re.match(r"^\[H\d\dPR", lab):
            reasons[c] = "admin (preloaded from the previous wave)"
        elif method == "Rockwood" and cohort == "HRS" and re.match(r"^\[H\d\dM", lab):
            reasons[c] = "derived (disability section: asked only with a work-limiting health problem)"
        elif GENERIC[method].search(lab_text) or GENERIC_NAME[method].search(stem(c)):
            reasons[c] = "derived"
        elif method == "Rockwood" and (cohort, method) in HELP_MODULES and re.match(HELP_MODULES[(cohort, method)], base):
            reasons[c] = "follow-up of a constituent item (help or care received module)"
        elif fu and fu.search(lab_text):
            reasons[c] = "follow-up of a constituent item"
    # data-driven near-copy screen on a subsample (rank correlations)
    keep = [c for c in X.columns if c not in reasons]
    sub = X[keep]
    if len(sub) > SCREEN_N:
        sub = sub.sample(SCREEN_N, random_state=RNG)
    ys = Y.loc[sub.index]
    items = ys[[c for c in ys.columns if c.startswith("item__")]]
    R = sub.rank(pct=True).fillna(0.5)
    Rz = (R - R.mean()) / R.std(ddof=0).replace(0, np.nan)
    I = items.rank(pct=True).fillna(0.5); Iz = (I - I.mean()) / I.std(ddof=0).replace(0, np.nan)
    yz = ys.y.rank(pct=True); yz = (yz - yz.mean()) / yz.std(ddof=0)
    n = len(sub)
    rho_items = np.abs(np.nan_to_num(Rz.values).T @ np.nan_to_num(Iz.values) / n) if items.shape[1] else np.zeros((len(keep), 1))
    rho_y = np.abs(np.nan_to_num(Rz.values).T @ yz.values / n)
    for j, c in enumerate(keep):
        mi = rho_items[j].max() if rho_items.shape[1] else 0
        if mi >= RHO_ITEM:
            k = int(rho_items[j].argmax()); reasons[c] = f"near-copy of {items.columns[k][6:]} (rho {mi:.2f})"
        elif rho_y[j] >= RHO_OUTCOME:
            reasons[c] = f"near-copy of the score (rho {rho_y[j]:.2f})"
    return reasons


# ============================================================================ model
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.3,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=600, verbose=-1, n_jobs=-1, random_state=RNG)


def fit_and_rank(X, y):
    rng = np.random.default_rng(RNG)
    idx = rng.permutation(len(X)); cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[te], y[te])], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = m.predict(X.iloc[te])
    r2 = 1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
    te_s = te if len(te) <= SHAP_N else rng.choice(te, SHAP_N, replace=False)
    Xs = X.iloc[te_s]
    contrib = m.booster_.predict(Xs, pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    share = imp / imp.sum() if imp.sum() > 0 else imp
    direction = []
    for j in range(X.shape[1]):
        v = Xs.iloc[:, j].values; s = contrib[:, j]; ok = ~np.isnan(v)
        if imp[j] == 0 or ok.sum() < 30 or np.nanstd(v[ok]) == 0:
            direction.append(0); continue
        r = np.corrcoef(v[ok], s[ok])[0, 1]
        direction.append(int(np.sign(r)) if np.isfinite(r) else 0)
    return dict(r2=float(r2), n_train=int(len(tr)), n_test=int(len(te)), best_iter=int(m.best_iteration_ or PARAMS["n_estimators"]),
                imp=imp, share=share, direction=np.array(direction))


# ============================================================================ driver
def run_one(cohort, method, w):
    part = os.path.join(PARTS, f"{cohort}_{method}_w{w}.json")
    if os.path.exists(part):
        log("cached", cohort, method, w); return
    t0 = time.time()
    Y = load_outcome(cohort, method, w)
    if Y is None or len(Y) < 200:
        log("no outcome file", cohort, method, w); return
    Y = Y.set_index("id")
    X, labels = SOURCES[cohort](w, set(Y.index))
    common = X.index.intersection(Y.index)
    X, Y = X.loc[common], Y.loc[common]
    n_vars_raw = X.shape[1]
    reasons = exclusions(cohort, method, X, labels, Y)
    Xk = X.drop(columns=list(reasons))
    del X; gc.collect()
    log(f"{cohort} {method} w{w}: n={len(Xk):,} vars={n_vars_raw:,} excluded={len(reasons):,} kept={Xk.shape[1]:,}")
    res = fit_and_rank(Xk, Y.y.values.astype("float32"))
    order = np.argsort(-res["imp"])
    top = [dict(rank=i + 1, variable=Xk.columns[j], key=stem(Xk.columns[j]), label=labels.get(Xk.columns[j], ""),
                mean_abs_shap=float(res["imp"][j]), share=float(res["share"][j]), direction=int(res["direction"][j]),
                missing_pct=float(round(100 * Xk.iloc[:, j].isna().mean(), 1)))
           for i, j in enumerate(order[:20])]
    allimp = {stem(Xk.columns[j]): float(res["share"][j]) for j in order if res["share"][j] > 0}
    json.dump(dict(cohort=cohort, method=method, wave=int(w), source=source_label(cohort, w), n=int(len(Xk)),
                   script_sha256=SCRIPT_SHA, lightgbm=lgb.__version__,
                   n_vars_raw=int(n_vars_raw), n_excluded=len(reasons), n_used=int(Xk.shape[1]),
                   r2=res["r2"], n_train=res["n_train"], n_test=res["n_test"], trees=res["best_iter"], top20=top,
                   share_all=allimp,
                   excluded=[dict(variable=c, label=labels.get(c, ""), reason=r) for c, r in reasons.items()],
                   seconds=round(time.time() - t0)),
              open(part, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"   R²={res['r2']:.3f}  top: {', '.join(t['variable'] for t in top[:5])}  ({time.time() - t0:.0f}s)")
    del Xk; gc.collect()


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for (cohort, method), waves in WAVES.items():
        if only and cohort not in only:
            continue
        for w in waves:
            try:
                run_one(cohort, method, w)
            except Exception as e:
                import traceback; log("FAILED", cohort, method, w, repr(e)); traceback.print_exc()
    log("all done")
