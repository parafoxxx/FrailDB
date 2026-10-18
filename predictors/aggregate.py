"""
Aggregate per-wave predictor results (output/parts/*.json) into tables and one Excel workbook.

Per wave     : top 10 and top 20 variables by mean |SHAP| (share of the model's total |SHAP|).
Per dataset  : for each variable, the mean share across ALL analysed waves of that cohort and instrument
               (a wave where the variable was absent or unused counts as 0), plus how many waves had it in
               their top 10 / top 20. Overall top 5 = highest mean share; ties broken by top-10 frequency.
Variables are matched across waves on a wave-free key (harmonised r10shlt and r11shlt -> rshlt... -> shlt).

Writes output/: per_wave_top20.csv, dataset_top5.csv, dataset_ranking_full.csv, exclusions.csv,
models.csv, FrailDB_frailty_predictors.xlsx, predictors.json (for the portal).
"""
import os, re, json, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
parts = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(OUT, "parts", "*.json")))]
if not parts:
    raise SystemExit("No results in output/parts yet.")
shas = {p.get("script_sha256") for p in parts}
if len(shas) != 1 or None in shas:
    raise SystemExit(f"Results come from {len(shas)} different script versions {shas} – re-run all waves with one version.")
SCRIPT_SHA = shas.pop()
COH = ["SHARE", "HRS", "ELSA", "MHAS", "CLHLS"]
parts.sort(key=lambda p: (COH.index(p["cohort"]), p["method"] != "Rockwood", p["wave"]))
def clean_label(s):
    s = s.split("] ", 1)[1] if s.startswith("[") and "] " in s else s
    return re.sub(r"^[a-z]+\d*[a-z_0-9]*:\s*w\d+\s+", "", s)      # drop harmonised "r14paina:w14 " prefix


def merge_key(k, keys, cohort):
    """Gateway harmonised files code each functional item twice: 'difficulty' (x) and 'some difficulty' (xa).
    Treat the pair as one item so its importance is not split across two ranks."""
    if cohort in ("HRS", "MHAS") and k.endswith("a") and k[:-1] in keys:
        return k[:-1]
    return k
src_of = lambda s: s[1:s.index("]")] if s.startswith("[") and "]" in s else ""

# ---------------------------------------------------------------- per wave
rows = []
for p in parts:
    for t in p["top20"]:
        rows.append(dict(cohort=p["cohort"], instrument=p["method"], wave=p["wave"], rank=t["rank"], in_top10=t["rank"] <= 10,
                         variable=t["variable"], label=clean_label(t["label"]), file_or_module=src_of(t["label"]),
                         importance_pct=round(100 * t["share"], 2), mean_abs_shap=t["mean_abs_shap"],
                         direction="higher value → higher score" if t["direction"] > 0 else "higher value → lower score" if t["direction"] < 0 else "mixed",
                         missing_pct=t["missing_pct"]))
per_wave = pd.DataFrame(rows)

models = pd.DataFrame([dict(cohort=p["cohort"], instrument=p["method"], wave=p["wave"], predictor_source=p["source"], n_people=p["n"],
                            variables_in_files=p["n_vars_raw"], excluded=p["n_excluded"], used=p["n_used"], heldout_R2=round(p["r2"], 3),
                            trees=p["trees"], seconds=p["seconds"]) for p in parts])

exc = pd.DataFrame([dict(cohort=p["cohort"], instrument=p["method"], wave=p["wave"], variable=e["variable"], label=clean_label(e["label"]),
                         reason=e["reason"]) for p in parts for e in p["excluded"]])

# ---------------------------------------------------------------- per dataset
agg_rows, top5_rows = [], []
for (co, m), grp in models.groupby(["cohort", "instrument"], sort=False):
    ps = [p for p in parts if p["cohort"] == co and p["method"] == m]
    nw = len(ps)
    universe = {k for p in ps for k in p["share_all"]}
    keys = {}
    for p in ps:                                   # importance shares summed within merged items
        for k, sh in p["share_all"].items():
            mk = merge_key(k, universe, co)
            keys.setdefault(mk, {}); keys[mk][p["wave"]] = keys[mk].get(p["wave"], 0) + sh
    lab, var = {}, {}
    for p in ps:                                   # most recent wave's label wins; base item's label preferred
        for t in sorted(p["top20"], key=lambda t: t["key"] != merge_key(t["key"], universe, co)):
            mk = merge_key(t["key"], universe, co)
            if mk == t["key"] or mk not in lab:
                lab[mk] = clean_label(t["label"]); var[mk] = t["variable"]
    tops = {}
    for p in ps:                                   # best rank of the merged item in each wave
        best = {}
        for t in p["top20"]:
            mk = merge_key(t["key"], universe, co)
            best[mk] = min(best.get(mk, 99), t["rank"])
        for mk, r in best.items():
            tops.setdefault(mk, []).append(r)
    recs = []
    for k, ws in keys.items():
        ranks = tops.get(k, [])
        recs.append(dict(cohort=co, instrument=m, key=k, example_variable=var.get(k, k), label=lab.get(k, ""),
                         mean_importance_pct=round(100 * sum(ws.values()) / nw, 3),
                         waves_present=len(ws), waves_in_top10=sum(r <= 10 for r in ranks), waves_in_top20=len(ranks),
                         waves_analysed=nw, best_rank=min(ranks) if ranks else None,
                         waves=",".join(str(w) for w in sorted(ws))))
    d = pd.DataFrame(recs).sort_values(["mean_importance_pct", "waves_in_top10"], ascending=False).reset_index(drop=True)
    d.insert(2, "overall_rank", range(1, len(d) + 1))
    agg_rows.append(d)
    top5_rows.append(d.head(5))
ranking = pd.concat(agg_rows, ignore_index=True)
top5 = pd.concat(top5_rows, ignore_index=True)

# ---------------------------------------------------------------- sensitivity: pilot rules vs final rules
def mean_shares(ps):
    nw = len(ps); acc = {}
    for p in ps:
        for k, sh in p["share_all"].items():
            acc[k] = acc.get(k, 0) + sh / nw
    return acc


pilot = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(OUT, "pilot_parts", "*.json")))]
sens_rows = []
for (co, m), _ in models.groupby(["cohort", "instrument"], sort=False):
    fin = mean_shares([p for p in parts if p["cohort"] == co and p["method"] == m])
    pil = mean_shares([p for p in pilot if p["cohort"] == co and p["method"] == m])
    if not pil:
        continue
    f5 = sorted(fin, key=fin.get, reverse=True)[:5]; p5 = sorted(pil, key=pil.get, reverse=True)[:5]
    common = set(fin) & set(pil)
    rho = pd.Series({k: fin[k] for k in common}).rank().corr(pd.Series({k: pil[k] for k in common}).rank()) if len(common) > 5 else None
    r2f = models[(models.cohort == co) & (models.instrument == m)].heldout_R2.mean()
    r2p = sum(p["r2"] for p in pilot if p["cohort"] == co and p["method"] == m) / max(1, sum(1 for p in pilot if p["cohort"] == co and p["method"] == m))
    sens_rows.append(dict(cohort=co, instrument=m, top5_final=", ".join(f5), top5_pilot=", ".join(p5),
                          top5_overlap=len(set(f5) & set(p5)), spearman_shared_variables=round(rho, 3) if rho is not None else None,
                          mean_R2_final=round(r2f, 3), mean_R2_pilot=round(r2p, 3)))
sens = pd.DataFrame(sens_rows)

# ---------------------------------------------------------------- write
per_wave.to_csv(os.path.join(OUT, "per_wave_top20.csv"), index=False)
top5.to_csv(os.path.join(OUT, "dataset_top5.csv"), index=False)
ranking[ranking.waves_in_top20 > 0].to_csv(os.path.join(OUT, "dataset_ranking_full.csv"), index=False)
exc.to_csv(os.path.join(OUT, "exclusions.csv"), index=False)
models.to_csv(os.path.join(OUT, "models.csv"), index=False)
sens.to_csv(os.path.join(OUT, "sensitivity_pilot_vs_final.csv"), index=False)

readme = pd.DataFrame({"FrailDB — most influential predictors of frailty": [
    "Two independent tracks: Rockwood frailty index score (0–1) and Fried phenotype score (0–5).",
    "Per wave: LightGBM regression on every variable in that wave's survey files, after exclusions; influence = mean |SHAP| on held-out people, shown as % of the model's total.",
    "Excluded (sheet 'Exclusions'): the instrument's own items (constituent), summary scores built from them (derived), follow-up questions about the same activities, "
    "near-copies found by correlation (|rho| >= 0.75 with an item or >= 0.80 with the score), IDs/weights/interview metadata (admin), and variables > 95% missing.",
    "Per dataset: mean importance % across all analysed waves (0 where absent) -> overall top 5; 'waves_in_top10/20' show consistency.",
    "Direction is the sign of the relation between a variable's value and its SHAP effect. Check each variable's coding (e.g. self-rated health 1 = excellent).",
    "Predictor source per wave is in 'Models' (raw files for SHARE, ELSA, CLHLS and HRS 2020–22; harmonised Gateway files for HRS 1998–2018 and MHAS).",
    "Aggregated results only; no participant data.",
    f"All {len(parts)} wave models were produced by one script version: run_predictors.py sha256 {SCRIPT_SHA} (LightGBM {parts[0].get('lightgbm')})."]})
xl = os.path.join(OUT, "FrailDB_frailty_predictors.xlsx")
with pd.ExcelWriter(xl, engine="openpyxl") as w:
    readme.to_excel(w, "README", index=False)
    top5.to_excel(w, "Dataset top 5", index=False)
    for (co, m), g in per_wave.groupby(["cohort", "instrument"], sort=False):
        g.to_excel(w, f"{co} {m[:4]} waves"[:31], index=False)
    ranking[ranking.waves_in_top20 > 0].to_excel(w, "Dataset ranking (full)", index=False)
    models.to_excel(w, "Models", index=False)
    if len(sens):
        sens.to_excel(w, "Sensitivity pilot vs final", index=False)
    exc.to_excel(w, "Exclusions", index=False)
    for ws in w.book.worksheets:
        for col in ws.columns:
            width = min(70, max(10, *(len(str(c.value or "")) for c in col[:200])))
            ws.column_dimensions[col[0].column_letter].width = width
        ws.freeze_panes = "A2"

json.dump(dict(models=models.to_dict("records"), top5=top5.to_dict("records"),
               per_wave=per_wave.drop(columns=["mean_abs_shap"]).to_dict("records")),
          open(os.path.join(OUT, "predictors.json"), "w", encoding="utf-8"), ensure_ascii=False, default=str)

print(models.to_string(index=False))
print()
for (co, m), g in top5.groupby(["cohort", "instrument"], sort=False):
    print(f"{co} · {m} ({g.waves_analysed.iloc[0]} waves)")
    for r in g.itertuples():
        print(f"   {r.overall_rank}. {r.example_variable:16s} {r.mean_importance_pct:6.2f}%  top10 in {r.waves_in_top10}/{r.waves_analysed}  {r.label[:70]}")
print("\nwrote", xl)
