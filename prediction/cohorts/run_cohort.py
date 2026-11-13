"""
Run the prediction pipeline for one cohort.

    python run_cohort.py ELSA                 # all steps
    python run_cohort.py ELSA models explain  # selected steps: models, explain, region, levers, bundle
The cohort's data must first be prepared with prepare_<cohort>.py.
"""
import sys, json
import engine as E

if __name__ == "__main__":
    cohort = sys.argv[1].upper()
    steps = sys.argv[2:] or ["models", "explain", "region", "levers", "bundle"]
    out, cache = E.paths(cohort)
    prep, A, feats = E.prepare_analysis(cohort)
    M = prep["meta"]
    E.step(f"{cohort}: {len(A):,} rows from {A.pid.nunique():,} people; windows {A.window.value_counts().sort_index().to_dict()}; "
           f"sets {A.set.value_counts().to_dict()}; {len(feats)} inputs")
    E.step("status by window: " + json.dumps({w: g.status.value_counts(normalize=True).round(3).to_dict() for w, g in A.groupby("window")}))
    json.dump(dict(cohort=cohort, n_rows=len(A), n_people=int(A.pid.nunique()), windows=M["windows"], temporal=M["temporal"],
                   triplets=M["triplets"], fixed=M["fixed"], levers=M["levers"], n_features=len(feats), notes=M.get("notes", []),
                   status_by_window={w: g.status.value_counts().to_dict() for w, g in A.groupby("window")}),
              open(f"{out}/step1_info.json", "w"), indent=1, default=str)
    if "models" in steps:
        for kind in ["cls", "reg"]:
            E.run_models(cohort, kind, A, feats, out, cache)
    if "explain" in steps:
        E.explain(cohort, prep, A, feats, out, cache)
    if "region" in steps:
        E.region_out(cohort, prep, A.copy(), feats, out, cache)
    if "levers" in steps:
        E.levers(cohort, prep, out, cache)
    if "bundle" in steps:
        E.bundle(cohort, prep, A, feats, out, cache)
    E.step(f"{cohort}: done")
