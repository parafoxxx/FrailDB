"""Audit: variables excluded as derived/follow-up under the pilot rules that the current rules would KEEP.
Each must be a legitimate predictor (not a leak). Output: output/audit_rule_change.csv"""
import os, re, json, glob, importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("R", os.path.join(HERE, "run_predictors.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

# item columns per cohort and instrument, taken from one outcome file each
items = {}
for (co, m), ws in R.WAVES.items():
    Y = R.load_outcome(co, m, ws[-1] if co != "ELSA" or m != "Rockwood" else 9)
    items[(co, m)] = set(Y.columns)

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "output", "pilot_parts", "*.json"))):
    p = json.load(open(f, encoding="utf-8"))
    co, m = p["cohort"], p["method"]
    fu = R.follow_up_rule(co, m, items[(co, m)])
    for e in p["excluded"]:
        if not e["reason"].startswith(("derived", "follow")):
            continue
        lab = re.sub(r"^\[[^\]]*\]\s*", "", e["label"])
        still = bool(R.DEATH_LABEL.search(lab) or R.GENERIC[m].search(lab) or R.GENERIC_NAME[m].search(R.stem(e["variable"])) or (fu and fu.search(lab))
                     or R.ADMIN_LABEL.search(lab) or (co == "HRS" and re.match(r"^\[H\d\d(PR|M)", e["label"])))
        cons = any(re.compile(rf"^{c}$", re.I).match(e["variable"].lower()) or re.compile(rf"^{c}$", re.I).match(R.stem(e["variable"]))
                   for c in R.CONSTITUENTS.get((co, m), []))
        if not still and not cons:
            rows.append(dict(cohort=co, instrument=m, variable=R.stem(e["variable"]), label=lab[:110]))
d = pd.DataFrame(rows).drop_duplicates(["cohort", "instrument", "variable"])
d.to_csv(os.path.join(HERE, "output", "audit_rule_change.csv"), index=False)
print(len(d), "variables newly kept")
for (co, m), g in d.groupby(["cohort", "instrument"]):
    print(f"\n{co} {m} ({len(g)})")
    for r in g.itertuples():
        print(f"   {r.variable:14s} {r.label}")
