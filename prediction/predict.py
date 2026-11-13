"""
FrailDB 5-year frailty predictor (SHARE model).

    python predict.py build          # bundle the models, recalibration and lever effects into one file
    python predict.py demo           # run three example people

    from predict import load, predict
    B = load(); predict(B, {"age": 68, "female": 1, "fi": 0.12, "smoke_now": 1, ...}, country=12)

predict() returns:
  fi_5y            expected Rockwood frailty index about 5 years ahead (if alive)
  risk_5y          probability of becoming frail (FI >= 0.25) or dying within about 5 years (people not frail today;
                   recalibrated to the most recent SHARE period)
  fixed_drivers    the person's characteristics that push their risk up most (age, health, ... : not changeable)
  changeable       the modifiable factors that apply to this person, with the estimated effect of changing each one
                   (from the causal analysis in step3_levers.py, applied to the person's own risk)
Model explanations (drivers) describe the prediction, not causes; lever effects come from the separate causal study.
"""
import os, sys, json, pickle
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", ".cache", "prediction")
BUNDLE = os.path.join(CACHE, "frailty_predictor_share.pkl")
FRAIL = 0.25

# which lever contrasts apply to a person, from their current answers (same definitions as step3_levers.py)
APPLIES = {
    "quit_smoking": lambda p: p.get("smoke_now") == 1,
    "start_vigorous": lambda p: (p.get("vigorous_freq") or 0) >= 3,
    "leave_inactivity": lambda p: p.get("inactive") == 1,
    "stay_active": lambda p: p.get("inactive") == 0,
    "cut_heavy_drinking": lambda p: (p.get("alcohol_units_wk") or 0) > 14,
    "obese_lost_weight": lambda p: (p.get("bmi") or 0) >= 30,
    "daily_fruitveg": lambda p: (p.get("fruitveg_freq") or 0) >= 2,
    "more_protein": lambda p: (p.get("protein_freq") or 0) >= 3,
    "join_social": lambda p: p.get("social_activities") == 0,
    "less_lonely": lambda p: (p.get("loneliness") or 0) >= 6,
    "got_hearing_aid": lambda p: p.get("d_hearing") == 1 and p.get("hearing_aid") == 0,
    "sleep_resolved": lambda p: p.get("sleep_trouble") == 1,
}


def _grade(E):
    """clear = benefit significant in both analyses after Holm correction across all habits tested;
    suggestive = nominally significant (p < 0.05) benefit in either analysis; else no evidence."""
    import numpy as np
    out = {}
    for spec, g in E.groupby(E.spec.str.split(" ").str[0]):
        g = g.sort_values("p"); k = len(g)
        holm = np.maximum.accumulate(np.minimum(1, g.p.values * (k - np.arange(k))))
        for lever, rr, p, h in zip(g.lever, g.RR, g.p, holm):
            out.setdefault(lever, []).append((rr < 1 and h < 0.05, rr < 1 and p < 0.05))
    return {k: "clear" if all(a for a, _ in v) else ("suggestive" if any(b for _, b in v) else "no evidence") for k, v in out.items()}


def build():
    sys.path.insert(0, HERE)
    import step2_models as S
    cls = pickle.load(open(os.path.join(CACHE, "step2_cls_models.pkl"), "rb"))
    reg = pickle.load(open(os.path.join(CACHE, "step2_reg_models.pkl"), "rb"))
    rec = json.load(open(os.path.join(S.RES, "step2b_summary.json")))["recalibration"]
    E = pd.read_csv(os.path.join(S.RES, "step3_lever_effects.csv"))
    E = E[E.outcome == "frail or dead"]
    pr = E[E.spec.str.startswith("primary")].set_index("lever")
    se = E[E.spec.str.startswith("sensitivity")].set_index("lever")
    effects = {}
    grades = _grade(E)
    for k in pr.index:
        a, b = pr.loc[k], se.loc[k]
        grade = grades[k]
        cons = a if abs(np.log(a.RR)) <= abs(np.log(b.RR)) else b          # the more cautious of the two estimates
        effects[k] = dict(label=a.label, grade=grade, RR=float(cons.RR), RR_lo=float(cons.RR_lo), RR_hi=float(cons.RR_hi),
                          RR_primary=float(a.RR), RR_sensitivity=float(b.RR), n=int(a.n))
    # typical values by sex and 5-year age band, for answers a user leaves blank
    A, feats = S.load()
    A = A[A.window != "w6->w8"]
    A["band"] = (A.age.clip(50, 95) // 5 * 5).astype(int)
    typical = A.groupby(["female", "band"])[[f for f in feats if not f.startswith("ctry_") and f != "years_planned"]].median()
    B = dict(cls=cls["models"]["M2 LightGBM"], reg=reg["models"]["M2 LightGBM"], feats=cls["feats"],
             recal=(rec["intercept"], rec["slope"]), fixed=S.FIXED, levers=S.LEVERS, effects=effects, typical=typical)
    pickle.dump(B, open(BUNDLE, "wb"))
    print(f"Saved {BUNDLE} ({os.path.getsize(BUNDLE) / 1e6:.1f} MB)")
    for k, e in effects.items(): print(f"  {k:20s} {e['grade']:12s} RR {e['RR']:.2f} [{e['RR_lo']:.2f}, {e['RR_hi']:.2f}]")


def load():
    return pickle.load(open(BUNDLE, "rb"))


def predict(B, person, country, years=5.0, top=5):
    feats = B["feats"]
    x = pd.DataFrame([{f: np.nan for f in feats}])
    for k, v in person.items():
        if k in x and v is not None: x.loc[0, k] = v
    # blanks -> typical value for the person's sex and age band (a missing answer is not itself a risk signal here)
    band = int(min(max(person.get("age", 70), 50), 95) // 5 * 5)
    typ = B["typical"].loc[(float(person.get("female", 0)), band)]
    assumed = [f for f in typ.index if pd.isna(x.loc[0, f]) and pd.notna(typ[f])]
    for f in assumed: x.loc[0, f] = typ[f]
    x["years_planned"] = years
    for f in feats:
        if f.startswith("ctry_"): x[f] = float(f == f"ctry_{int(country)}")
    x = x.astype(float)
    out = {"fi_5y": round(float(np.clip(B["reg"].predict(x)[0], 0, 1)), 3)}
    age = person.get("age")
    if age is None or age < 50:
        out["age_warning"] = "The model was built on people aged 50 and over; below 50 it treats the person like a 50-year-old, so the result is not valid."
    elif age >= 90:
        out["age_warning"] = "Few people aged 90+ were in the training data (about 1% of rows), so this prediction is less certain."
    frail_now = person.get("fi", 0) >= FRAIL
    if frail_now:
        out["risk_5y"] = None
        out["note"] = "Already frail today (FI >= 0.25): the risk model is for people not yet frail, so only the FI projection is given."
    else:
        p = float(np.clip(B["cls"].predict_proba(x)[0, 1], 1e-6, 1 - 1e-6))
        a, b = B["recal"]
        out["risk_5y"] = round(float(1 / (1 + np.exp(-(a + b * np.log(p / (1 - p)))))), 3)
    out["assumed_typical"] = [f for f in assumed if not f.startswith("d_")] + (["some deficit items"] if any(f.startswith("d_") for f in assumed) else [])
    # what drives this person's prediction (TreeSHAP), among the answers actually given
    m = B["reg"] if frail_now else B["cls"]
    s = pd.Series(m.booster_.predict(x, pred_contrib=True)[0, :-1], index=feats)
    s = s[[f for f in feats if f in person and person[f] is not None and f not in B["levers"]]]
    out["fixed_drivers"] = [dict(factor=f, value=person.get(f), shap=round(float(v), 3))
                            for f, v in s.sort_values(ascending=False).head(top).items() if v > 0]
    # changeable factors that apply to this person, with causal effect estimates (the more cautious of two analyses)
    ch = []
    for key, rule in APPLIES.items():
        if key not in B["effects"] or not rule(person): continue
        e = B["effects"][key]
        item = dict(lever=key, change=e["label"], evidence=e["grade"], relative_risk=round(e["RR"], 2),
                    ci=(round(e["RR_lo"], 2), round(e["RR_hi"], 2)))
        if out["risk_5y"] is not None and e["grade"] != "no evidence":
            if key == "stay_active":
                item["risk_if_became_inactive"] = round(float(min(out["risk_5y"] / e["RR"], 1)), 3)
            else:
                item["risk_if_changed"] = round(float(out["risk_5y"] * e["RR"]), 3)
        if frail_now:
            item["caution"] = "effect estimated in people not yet frail"
        ch.append(item)
    order = {"clear": 0, "suggestive": 1, "no evidence": 2}
    out["changeable"] = sorted(ch, key=lambda d: (order[d["evidence"]], d["relative_risk"]))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    elif sys.argv[1:] == ["demo"]:
        B = load()
        people = {
            "68-year-old woman, smoker, inactive, lonely": (dict(age=68, female=1, fi=0.12, smoke_now=1, smoke_ever=1, inactive=1,
                vigorous_freq=4, moderate_freq=4, bmi=31, social_activities=0, loneliness=7, sleep_trouble=1, d_hearing=0,
                hearing_aid=0, fruitveg_freq=2, protein_freq=2, alcohol_units_wk=2, education_isced=2, partner=0, d_srh=1), 12),
            "74-year-old man, active, non-smoker": (dict(age=74, female=0, fi=0.08, smoke_now=0, smoke_ever=1, inactive=0,
                vigorous_freq=1, moderate_freq=1, bmi=25, social_activities=2, loneliness=3, sleep_trouble=0, d_hearing=1,
                hearing_aid=0, fruitveg_freq=1, protein_freq=2, alcohol_units_wk=6, education_isced=4, partner=1), 17),
            "81-year-old woman, already frail": (dict(age=81, female=1, fi=0.33, smoke_now=0, inactive=1, bmi=27,
                social_activities=0, loneliness=6, sleep_trouble=1), 15),
        }
        for name, (p, c) in people.items():
            print(f"\n=== {name}"); print(json.dumps(predict(B, p, c), indent=1, default=str))
