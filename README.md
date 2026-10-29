# FrailDB — the Multi-Cohort Frailty Database

FrailDB publishes the Fried frailty phenotype and the Rockwood frailty index, scored with one set of
rules, across the harmonised ageing cohorts SHARE, HRS, ELSA, MHAS and CLHLS. It follows the layout of
the Wishart-lab databases (HMDB, MiMeDB): browse tables, record pages with stable IDs, text search,
downloads and statistics, plus tools specific to frailty research.

Frailty_Volatility Project.

## What is in the portal

| Record | ID | Contents |
|---|---|---|
| Cohort | `FDBC01`… | Design, data source and access, mortality ascertainment, results by wave, caveats |
| Wave | `FDBW0001`… | Sample flow, FI distribution, age × sex tables, deficit prevalence, Fried criteria, grip/walk cut-offs, Fried–FI agreement, quality flags |
| Deficit | `FDBD001`… | Source variable and coding in every cohort, prevalence by wave, frail vs non-frail, age gradient |
| Fried criterion | `FDBF01`… | How each cohort measures it (measured vs self-report), prevalence |
| Instrument, country, reference | `FDBI`, `FDBN`, `FDBR` | |

Tools: compare cohorts, harmonisation matrix, FI reference norms, **pre-frailty age window**,
transitions, Fried vs FI agreement, frailty calculator, variable lookup.

## Rebuilding

The build reads the per-wave analysis outputs in `../Data Sources/` (not part of this folder and never
committed) and writes aggregated results only.

```bash
pip install -r requirements.txt
python build.py              # data -> page -> dist/   (~5 min)
python build.py --analysis   # also re-run the pre-frailty age-window analysis (~10 min)
```

| Script | Does |
|---|---|
| `age_window.py` | Pre-frailty age-window analysis → `site/age_window.json` |
| `build_data.py` | Aggregates every cohort/wave → `site/frailtydb.json` (set `EXCLUDE` to withhold a cohort; LASI is withheld in this release) |
| `build_site.py` | Injects the data into `site/template.html` → `site/index.html` and `dist/` |

## Deploying

`dist/` is a complete static website: `index.html` (the whole portal, data inlined, hash-based routing,
so no server rewrites are needed) and `frailtydb.json` (the database as a public file). It needs no
server-side code. Chart.js and Google Fonts load from public CDNs.

- **Netlify (used for this project):** in Netlify choose Add new site → Import an existing project →
  GitHub, and pick this repository. `netlify.toml` already sets `publish = "dist"` with no build command,
  so leave the build settings empty. Every push to `main` redeploys. (Or drag `dist/` onto
  app.netlify.com/drop for a one-off upload.)
- **GitHub Pages:** Settings → Pages → Deploy from a branch is not enough because the site lives in
  `dist/`. Use a Pages workflow that uploads `dist/`.
- **Vercel:** import the repository; `vercel.json` serves `dist/`.
- **Institute web server:** copy the contents of `dist/` into any web directory.
- **Local preview:** `python -m http.server -d dist 8000` and open http://localhost:8000.

After rebuilding, commit the new `dist/` and push; the host redeploys.

## Data policy

Only aggregated results are published. Every count below 10 is suppressed, and percentiles are given only
for groups of 30 or more. No participant identifiers or individual records leave `build_data.py` or
`age_window.py`. The cohorts' data-use agreements do not allow participant-level data to be
redistributed. `.cache/` can hold local extracts and is git-ignored.
