"""One command to rebuild FrailDB.

    python build.py              # aggregate data -> page -> dist/
    python build.py --analysis   # also re-run the pre-frailty age-window analysis first (~10 min)

Needs the raw analysis outputs in "../Data Sources" (they never leave this machine).
Only aggregated, small-cell-suppressed results are written to site/ and dist/.
"""
import subprocess, sys, os

here = os.path.dirname(os.path.abspath(__file__))
steps = (["age_window.py"] if "--analysis" in sys.argv else []) + ["build_data.py", "build_site.py"]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
for s in steps:
    print(f"\n== {s}")
    r = subprocess.run([sys.executable, os.path.join(here, s)], env=env)
    if r.returncode:
        sys.exit(f"{s} failed (exit {r.returncode})")
print("\nDone. Deploy the dist/ folder.")
