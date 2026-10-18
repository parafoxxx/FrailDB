"""One-off: convert the CLHLS ICPSR .rda files to local pickles (FrailDB/.cache, git-ignored).
Keeps only each wave's own baseline variables (drops follow-up suffixes _5.._18 and DTH* survival status)."""
import os, re, sys, time
import pandas as pd, pyreadr

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "..", "Data Sources", "6. CLHLS_China")
CACHE = os.path.join(HERE, "..", ".cache"); os.makedirs(CACHE, exist_ok=True)
for ds in sys.argv[1:] or ["DS0008", "DS0007", "DS0006", "DS0003"]:
    out = os.path.join(CACHE, f"clhls_{ds}.pkl")
    if os.path.exists(out):
        print(ds, "cached", flush=True); continue
    t = time.time()
    df = list(pyreadr.read_r(os.path.join(DS, ds, f"38899-{ds[2:]}-Data.rda")).values())[0]
    df = df[[c for c in df.columns if not re.search(r"_\d+$", c) and not c.upper().startswith("DTH")]]
    df.to_pickle(out)
    print(ds, df.shape, f"{time.time() - t:.0f}s", flush=True)
