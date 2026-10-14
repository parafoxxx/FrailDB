"""
Age window of pre-frailty.

Question (from Rohit): frailty rises with age, but is there an age band in which being pre-frail
matters most, i.e. after which a pre-frail person is more likely to progress to frailty than to
recover? If so, the ages before it are the window in which intervention has the best chance.

Design
  * Unit: a person-wave in which the person is PRE-FRAIL (Fried score 1–2; or FI 0.10–0.24).
    Non-frail person-waves are analysed alongside for comparison.
  * Outcome: their state at the next assessment 2–4 calendar years later
    (the first later wave they were scored in, within that gap).
  * Age: age at the baseline wave.
  * Estimates: observed proportions by 5-year age band, and logistic curves by single year of age
    (natural cubic spline, df = 4). Crossover age = first age at which P(progress to frail)
    >= P(recover to non-frail). 95% CI from a person-level cluster bootstrap.
  * People not re-assessed (death, drop-out, incomplete data) are counted and reported separately;
    the progression probabilities are among those re-assessed.

Writes FrailDB/site/age_window.json (aggregates only, counts < 10 suppressed).
"""
import os, json, math, warnings
import numpy as np, pandas as pd, pyreadr
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "Data Sources")
OUT = os.path.join(HERE, "site", "age_window.json")
MIN_CELL = 10
N_BOOT = 200
RNG = np.random.default_rng(20261013)

COHORTS = {
    "SHARE": ("2. SHARE data set_wave 1-9/SHARE Analysis", {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020, 9: 2022}),
    "HRS": ("3. HRS_United States/HRS Analysis", {w: 1992 + 2 * (w - 1) for w in range(4, 17)}),
    "ELSA": ("4. ELSA_England/ELSA Analysis", {1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010, 6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2022, 11: 2023}),
    "MHAS": ("6. MHAS_Mexico/MHAS Analysis", {1: 2001, 2: 2003, 3: 2012, 4: 2015, 5: 2018, 6: 2021}),
    "CLHLS": ("6. CLHLS_China/CLHLS Analysis", {3: 2002, 4: 2005, 5: 2008, 6: 2011, 7: 2014, 8: 2018}),
}
BANDS = [(50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 89), (90, 120)]
BL = lambda lo, hi: f"{lo}+" if hi >= 120 else f"{lo}–{hi}"
STATES = ["non-frail", "pre-frail", "frail"]


def read(folder, prefix, w, method):
    base = os.path.join(ROOT, folder, f"Wave {w}", f"{prefix}_w{w}_{method}")
    if os.path.exists(base + "_summary.csv"):
        return pd.read_csv(base + "_summary.csv", low_memory=False)
    if os.path.exists(base + ".rds"):
        return list(pyreadr.read_r(base + ".rds").values())[0]
    return None


def load(cohort, method):
    folder, years = COHORTS[cohort]
    frames = []
    for w, yr in years.items():
        df = read(folder, cohort, w, method)
        if df is None:
            continue
        out = pd.DataFrame({"id": df.iloc[:, 0].astype(str), "age": pd.to_numeric(df["age"], errors="coerce"),
                            "sex": pd.to_numeric(df["sex"], errors="coerce"), "wave": w, "year": yr})
        if method == "Fried":
            s = pd.to_numeric(df["frailty_score_fried"], errors="coerce")
            out["state"] = np.select([s >= 3, s >= 1], ["frail", "pre-frail"], "non-frail")
        else:
            fi = pd.to_numeric(df["frailty_index_score"], errors="coerce")
            out["state"] = np.select([fi >= 0.25, fi >= 0.10], ["frail", "pre-frail"], "non-frail")
        frames.append(out.dropna(subset=["age"]))
    return pd.concat(frames, ignore_index=True) if frames else None


def transitions(P):
    """Each person-wave -> state at the first later assessment 2–4 years on (NaN if none)."""
    P = P.sort_values(["id", "year"])
    nxt = P[["id", "year", "state"]].rename(columns={"year": "year2", "state": "state2"})
    m = P.merge(nxt, on="id")
    m = m[(m.year2 - m.year >= 2) & (m.year2 - m.year <= 4)]
    m = m.sort_values(["id", "year", "year2"]).drop_duplicates(["id", "year"])
    T = P.merge(m[["id", "year", "year2", "state2"]], on=["id", "year"], how="left")
    # a baseline is eligible only if the cohort had a scored wave 2–4 years later
    years = sorted(P.year.unique())
    has_next = {y: any(2 <= y2 - y <= 4 for y2 in years) for y in years}
    T = T[T.year.map(has_next)]
    T = T[T.age >= 50]
    T["band"] = pd.cut(T.age, [b[0] - 0.5 for b in BANDS] + [200], labels=[BL(*b) for b in BANDS])
    return T


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def band_table(T):
    rows = []
    for base in STATES[:2]:
        for b in [BL(*x) for x in BANDS]:
            g = T[(T.state == base) & (T.band == b)]
            n = len(g); re = g.state2.notna().sum()
            if n < MIN_CELL:
                continue
            r = dict(base=base, band=b, n=int(n), reassessed=int(re), lost_pct=round(100 * (n - re) / n, 1))
            if re >= MIN_CELL:
                for s, key in [("frail", "frail"), ("non-frail", "recover"), ("pre-frail", "prefrail")]:
                    k = int((g.state2 == s).sum())
                    r[key] = round(100 * k / re, 1)
                    r[key + "_ci"] = wilson(k, re)
            rows.append(r)
    return rows


AGES = np.arange(50, 91)


def fit_curves(T, by_sex=False):
    """Logistic spline curves for pre-frail baselines: P(frail) and P(recover) by age."""
    d = T[(T.state == "pre-frail") & T.state2.notna() & (T.age <= 94)].copy()
    if len(d) < 200:
        return None
    d["y_frail"] = (d.state2 == "frail").astype(int)
    d["y_rec"] = (d.state2 == "non-frail").astype(int)
    lo, hi = int(d.age.quantile(0.01)), int(d.age.quantile(0.99))
    ages = AGES[(AGES >= max(50, lo)) & (AGES <= min(90, hi))]
    knots = "df=4, lower_bound=49, upper_bound=95"
    out = dict(ages=ages.tolist())
    for y in ("y_frail", "y_rec"):
        mod = smf.glm(f"{y} ~ cr(age, {knots})", d, family=sm.families.Binomial()).fit()
        pr = mod.get_prediction(pd.DataFrame({"age": ages})).summary_frame()
        out[y] = dict(p=(100 * pr["mean"]).round(1).tolist(), lo=(100 * pr["mean_ci_lower"]).round(1).tolist(),
                      hi=(100 * pr["mean_ci_upper"]).round(1).tolist())
    out["crossover"] = crossover(ages, np.array(out["y_frail"]["p"]), np.array(out["y_rec"]["p"]))
    out["n"] = int(len(d))
    return out, d, ages, knots


def crossover(ages, pf, pr):
    diff = pf - pr
    if diff[0] >= 0:
        return dict(age=None, note="Progression already exceeds recovery at the youngest age")
    idx = np.where(diff >= 0)[0]
    if not len(idx):
        return dict(age=None, note="Recovery exceeds progression at every age modelled")
    i = idx[0]
    # linear interpolation between ages i-1 and i
    a0, a1, d0, d1 = ages[i - 1], ages[i], diff[i - 1], diff[i]
    return dict(age=round(float(a0 + (0 - d0) * (a1 - a0) / (d1 - d0)), 1))


def boot_crossover(d, ages, knots):
    """Person-level cluster bootstrap of the crossover age (design matrix built once, rows resampled)."""
    from patsy import dmatrix, build_design_matrices
    X = dmatrix(f"cr(age, {knots})", d, return_type="matrix")
    Xp = np.asarray(build_design_matrices([X.design_info], pd.DataFrame({"age": ages}))[0])
    X = np.asarray(X); yf = d.y_frail.values; yr = d.y_rec.values
    codes, uniq = pd.factorize(d.id)
    order = np.argsort(codes, kind="stable"); starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(codes))
    xs = []
    for b in range(N_BOOT):
        pick = RNG.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[i]:ends[i]] for i in pick])
        try:
            pf = sm.GLM(yf[idx], X[idx], family=sm.families.Binomial()).fit().predict(Xp)
            pr = sm.GLM(yr[idx], X[idx], family=sm.families.Binomial()).fit().predict(Xp)
            c = crossover(ages, 100 * pf, 100 * pr)["age"]
            if c is not None:
                xs.append(c)
        except Exception:
            pass
    if len(xs) < N_BOOT * 0.5:
        return None
    return [round(float(np.percentile(xs, 2.5)), 1), round(float(np.percentile(xs, 97.5)), 1), len(xs)]


def excess_risk(T):
    """Risk difference: P(frail next | pre-frail) − P(frail next | non-frail), by band."""
    rows = []
    for b in [BL(*x) for x in BANDS]:
        g = T[(T.band == b) & T.state2.notna()]
        a, c = g[g.state == "pre-frail"], g[g.state == "non-frail"]
        if len(a) >= MIN_CELL and len(c) >= MIN_CELL:
            pa, pc_ = (a.state2 == "frail").mean(), (c.state2 == "frail").mean()
            rows.append(dict(band=b, pre=round(100 * pa, 1), non=round(100 * pc_, 1), rd=round(100 * (pa - pc_), 1),
                             rr=round(pa / pc_, 2) if pc_ > 0 else None))
    return rows


res = dict(meta=dict(design="Pre-frail person-waves; outcome at first re-assessment 2–4 years later; ages 50+.",
                     fi_states="FI < 0.10 non-frail · 0.10–0.24 pre-frail (intermediate) · ≥ 0.25 frail",
                     fried_states="Fried score 0 non-frail · 1–2 pre-frail · 3–5 frail",
                     n_boot=N_BOOT, min_cell=MIN_CELL), results={})
pooled = {"Fried": [], "Rockwood": []}
for method, cohorts in [("Fried", ["SHARE", "HRS", "ELSA", "MHAS"]), ("Rockwood", ["SHARE", "HRS", "ELSA", "MHAS", "CLHLS"])]:
    for co in cohorts:
        P = load(co, method)
        if P is None:
            continue
        T = transitions(P)
        if T.state2.notna().sum() < 100:
            print(co, method, "too few transitions", T.state2.notna().sum()); continue
        gaps = (T.year2 - T.year).dropna()
        entry = dict(bands=band_table(T), excess=excess_risk(T),
                     pairs=sorted({f"{int(a)}→{int(b)}" for a, b in T[T.state2.notna()][["year", "year2"]].drop_duplicates().values}),
                     median_gap=float(gaps.median()), n_baseline=int(len(T)), n_reassessed=int(T.state2.notna().sum()))
        fc = fit_curves(T)
        if fc:
            curves, d, ages, knots = fc
            curves["crossover_ci"] = boot_crossover(d, ages, knots)
            entry["curves"] = curves
            for sx, lab in [(1, "Male"), (2, "Female")]:
                fs = fit_curves(T[T.sex == sx])
                if fs:
                    entry.setdefault("by_sex", {})[lab] = dict(crossover=fs[0]["crossover"], n=fs[0]["n"],
                                                              ages=fs[0]["ages"], frail=fs[0]["y_frail"]["p"], recover=fs[0]["y_rec"]["p"])
        res["results"].setdefault(method, {})[co] = entry
        pooled[method].append(T.assign(cohort=co))
        c = entry.get("curves", {}).get("crossover", {})
        print(f"{method:8s} {co:6s} baseline={entry['n_baseline']:>7,} reassessed={entry['n_reassessed']:>7,} gap={entry['median_gap']} "
              f"crossover={c.get('age')} CI={entry.get('curves', {}).get('crossover_ci')} {c.get('note', '')}")

# pooled across cohorts: cohort fixed effects, predictions averaged over the cohort mix
for method, parts in pooled.items():
    T = pd.concat(parts, ignore_index=True)
    d = T[(T.state == "pre-frail") & T.state2.notna() & (T.age <= 94)].copy()
    d["y_frail"] = (d.state2 == "frail").astype(int); d["y_rec"] = (d.state2 == "non-frail").astype(int)
    knots = "df=4, lower_bound=49, upper_bound=95"
    ages = AGES
    mix = d.cohort.value_counts(normalize=True)
    curves = {}
    for y in ("y_frail", "y_rec"):
        m = smf.glm(f"{y} ~ cr(age, {knots}) + C(cohort)", d, family=sm.families.Binomial()).fit()
        p = sum(w * m.predict(pd.DataFrame({"age": ages, "cohort": c})).values for c, w in mix.items())
        curves[y] = (100 * p).round(1).tolist()
    res["results"].setdefault(method, {})["Pooled"] = dict(
        bands=band_table(T), excess=excess_risk(T), n_baseline=int(len(T)), n_reassessed=int(T.state2.notna().sum()),
        curves=dict(ages=ages.tolist(), y_frail=dict(p=curves["y_frail"]), y_rec=dict(p=curves["y_rec"]),
                    crossover=crossover(ages, np.array(curves["y_frail"]), np.array(curves["y_rec"])), n=int(len(d))),
        cohorts=list(mix.index))
    print(f"{method:8s} POOLED crossover={res['results'][method]['Pooled']['curves']['crossover']}")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, separators=(",", ":"))
print("wrote", OUT)

# console summary: observed pre-frail outcomes by band, pooled
for method in ("Fried", "Rockwood"):
    print(f"\n{method} — pre-frail at baseline, outcome 2–4 y later (pooled, among re-assessed)")
    print(f"{'band':>7} {'n':>7} {'lost%':>6} {'→frail%':>8} {'recover%':>9} {'stay%':>6}")
    for r in res["results"][method]["Pooled"]["bands"]:
        if r["base"] == "pre-frail":
            print(f"{r['band']:>7} {r['n']:>7,} {r['lost_pct']:>6} {r.get('frail', '—'):>8} {r.get('recover', '—'):>9} {r.get('prefrail', '—'):>6}")
    print("excess risk vs non-frail:", [(r["band"], r["rd"], r["rr"]) for r in res["results"][method]["Pooled"]["excess"]])
