"""
Builds the participant-level analysis files for the deficit-weighting study. Nothing written here is published:
outputs go to FrailDB/.cache/weighting/ (git-ignored).

SHARE (derivation): one baseline per person = their first frailty index assessment in waves 2, 4, 5, 6 or 8
(wave 9 has no follow-up; wave 7 has no FI). Follow-up runs from the baseline interview to death, or to the
person's last interview while alive (any wave 2-9). Death dates come from the end-of-life interviews (xt009_ year,
xt008_ month); a death flagged in the coverscreen with no date is placed midway between the last interview alive
and the median interview date of the wave that recorded it.

ELSA (external check): baseline = wave 1 (2002-03) FI. Outcome from the index file's mortwave (deaths from
national registration up to wave 6 sampling): died before wave 3 (about 4 years) and before wave 6 (about 10 years).

Run: python FrailDB/weighting/prepare.py
"""
import os, glob
import numpy as np, pandas as pd, pyreadstat

HERE = os.path.dirname(os.path.abspath(__file__))
FDB = os.path.dirname(HERE)
SRC = os.path.join(FDB, "..", "Data Sources")
MORT = os.path.join(FDB, ".cache", "share_mort")
OUT = os.path.join(FDB, ".cache", "weighting")
os.makedirs(OUT, exist_ok=True)

DEFICITS = ["d_srh", "d_hypertension", "d_diabetes", "d_cancer", "d_lung", "d_heart", "d_stroke", "d_arthritis",
            "d_adl_dress", "d_adl_bath", "d_adl_eat", "d_adl_toilet", "d_iadl_meal", "d_iadl_shop", "d_hearing",
            "d_depression", "d_mob_walk", "d_mob_chair", "d_mob_stoop", "d_mob_lift", "d_adl_bed", "d_vision",
            "d_adl_walkroom", "d_mob_sit2h", "d_mob_stairs_several", "d_mob_stairs_one", "d_mob_arms", "d_mob_push",
            "d_mob_coin", "d_iadl_meds", "d_iadl_money", "d_dementia"]
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)


def ym(year, month):
    """Decimal date; month unknown -> mid-year."""
    y = pd.to_numeric(year, errors="coerce"); m = pd.to_numeric(month, errors="coerce")
    y = y.where(y > 1900); m = m.where((m >= 1) & (m <= 12))
    return y + (m.fillna(6.5) - 0.5) / 12


# ------------------------------------------------------------------ SHARE
def read_mod(w, m, cols):
    p = os.path.join(MORT, f"w{w}_{m}.dta")
    if os.path.exists(p):
        return pyreadstat.read_dta(p, usecols=cols)[0]
    return pyreadstat.read_sav(os.path.join(MORT, f"w{w}_{m}.sav"), usecols=cols)[0]


def share():
    A = os.path.join(SRC, "2. SHARE data set_wave 1-9", "SHARE Analysis")
    fi = []
    for w in (2, 4, 5, 6, 8, 9):
        d = pd.read_csv(os.path.join(A, f"Wave {w}", f"SHARE_w{w}_Rockwood_summary.csv"))
        fi.append(d.assign(wave=w))
    fi = pd.concat(fi, ignore_index=True)
    say(f"SHARE FI assessments, waves 2/4/5/6/8/9: {len(fi):,} ({fi.mergeid.nunique():,} people)")

    cv = []
    for w in (2, 4, 5, 6, 7, 8, 9):
        c = read_mod(w, "cv_r", ["mergeid", "interview", "deceased", "int_year", "int_month"])
        c["wave"] = w; c["date"] = ym(c.int_year, c.int_month); cv.append(c)
    cv = pd.concat(cv, ignore_index=True)
    alive_iv = cv[(cv.interview == 1) & cv.date.notna()]
    wave_mid = alive_iv.groupby("wave").date.median()

    xt = pd.concat([read_mod(w, "xt", ["mergeid", "xt008_", "xt009_"]).assign(wave=w) for w in (2, 4, 5, 6, 7, 8, 9)])
    xt["death"] = ym(xt.xt009_, xt.xt008_)
    death_date = xt.dropna(subset=["death"]).groupby("mergeid").death.min()
    dead_wave = cv[cv.deceased == 1].groupby("mergeid").wave.min()
    say(f"SHARE deaths: {len(death_date):,} with a date (end-of-life interviews); "
        f"{len(dead_wave):,} people flagged deceased in a coverscreen")

    # baseline: first FI assessment in waves 2-8
    base = fi[fi.wave < 9].sort_values("wave").drop_duplicates("mergeid", keep="first")
    base = base.merge(cv[["mergeid", "wave", "date"]].rename(columns={"date": "t0"}), on=["mergeid", "wave"], how="left")
    base["t0"] = base.t0.fillna(base.wave.map(wave_mid))   # 'interview date missing' -> wave median
    last_alive = alive_iv.groupby("mergeid").date.max()
    base["last_alive"] = base.mergeid.map(last_alive)
    base["death"] = base.mergeid.map(death_date)
    # flagged dead but undated: midway between last interview and the recording wave's median date
    undated = base.death.isna() & base.mergeid.isin(dead_wave.index)
    rec = base.loc[undated, "mergeid"].map(dead_wave).map(wave_mid)
    base.loc[undated, "death"] = (base.loc[undated, "last_alive"].fillna(base.loc[undated, "t0"]) + rec) / 2
    base["died"] = base.death.notna().astype(int)
    # a dated death before the last interview seen alive is a data error -> keep the death, move it to the last
    # interview (counted), never earlier than baseline
    bad = (base.died == 1) & (base.death < base.last_alive)
    say(f"SHARE: {int(undated.sum()):,} deaths without a date imputed; {int(bad.sum()):,} death dates earlier than the "
        f"last interview alive (moved to that interview)")
    base.loc[bad, "death"] = base.loc[bad, "last_alive"]
    base["end"] = np.where(base.died == 1, base.death, base.last_alive)
    base["time"] = base.end - base.t0
    n0 = len(base)
    base = base[(base.time > 0.05) | ((base.died == 1) & (base.time >= 0))].copy()   # need follow-up
    base.loc[base.time < 0.05, "time"] = 0.05
    say(f"SHARE baseline cohort: {n0:,} people; {len(base):,} with follow-up; deaths {base.died.sum():,}; "
        f"person-years {base.time.sum():,.0f}; median follow-up {base.time.median():.1f} y")
    # 5-year status: died within 5 y, or seen alive at >= 5 y; otherwise unknown
    base["dead5"] = np.where((base.died == 1) & (base.time <= 5), 1, np.where(base.time >= 5, 0, np.nan))
    say(f"SHARE 5-year status known: {base.dead5.notna().sum():,} (deaths {int(base.dead5.sum()):,})")
    keep = ["mergeid", "country", "wave", "age", "sex"] + DEFICITS + ["frailty_index_score", "time", "died", "dead5"]
    base = base[keep].rename(columns={"mergeid": "id", "frailty_index_score": "fi"})
    base.to_pickle(os.path.join(OUT, "share_cohort.pkl"))
    return base


# ------------------------------------------------------------------ ELSA
def elsa():
    A = os.path.join(SRC, "4. ELSA_England", "ELSA Analysis")
    fi = pd.read_csv(os.path.join(A, "Wave 1", "ELSA_w1_Rockwood_summary.csv"))
    ix = glob.glob(os.path.join(SRC, "4. ELSA_England", "**", "index_file_wave_0-wave_5_v2.dta"), recursive=True)[0]
    m = pyreadstat.read_dta(ix, usecols=["idauniq", "mortwave"])[0]
    d = fi.merge(m, on="idauniq", how="left")
    say(f"ELSA wave 1 FI: {len(fi):,}; matched to index file: {d.mortwave.notna().sum():,}")
    mw = d.mortwave
    known = mw.isin([0, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53, 61, 63])
    d = d[known].copy()
    d["dead4"] = d.mortwave.isin([21, 22, 23, 31, 32, 33]).astype(int)     # died before wave 3 (~2006)
    d["dead10"] = (d.mortwave >= 21).astype(int)                          # died before wave 6 sampling (~2012)
    say(f"ELSA with mortality status: {len(d):,}; died ~4 y: {d.dead4.sum():,}; died ~10 y: {d.dead10.sum():,}")
    d = d.rename(columns={"idauniq": "id", "frailty_index_score": "fi"})
    d[["id", "age", "sex"] + DEFICITS + ["fi", "dead4", "dead10"]].to_pickle(os.path.join(OUT, "elsa_cohort.pkl"))
    return d


if __name__ == "__main__":
    share(); elsa()
    open(os.path.join(HERE, "output", "cohort_log.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n") \
        if os.path.isdir(os.path.join(HERE, "output")) else None
