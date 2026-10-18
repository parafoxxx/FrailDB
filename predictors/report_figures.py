"""Data for the report figures (run after aggregate.py). Writes output/report_figures.json.

themes    : share of total |SHAP| (mean over waves) per predictor theme and dataset, from the variables that reach
            any wave's top 20. Themes are assigned from variable labels by the ordered rules below (first match wins);
            the assignment of every variable is written to output/theme_assignment.csv for review.
pipeline  : candidate, excluded-by-layer and kept variable counts over all 59 models.
stability : importance share by wave of the dataset's top-5 variables, for one long series (HRS frailty index,
            harmonised waves 4–14, whose variable names are comparable across waves).
"""
import os, re, json, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
rank = pd.read_csv(os.path.join(OUT, "dataset_ranking_full.csv"))
models = pd.read_csv(os.path.join(OUT, "models.csv"))
exc = pd.read_csv(os.path.join(OUT, "exclusions.csv"))

THEMES = [  # ordered: first match wins
    ("Pain", r"\bpain|ache\b|aching"),
    ("Energy and fatigue", r"energy|fatigue|tired|rested|worn out|sleep"),
    ("Physical activity", r"activit|sport|exercis|inactiv|vigor|moderate|field work|grow vegetables|outdoor|felt active"),
    ("Performance tests", r"chair ?r|chairraise|tandem|semi-tandem|\btest\b|book from the floor|turn around|360|peak flow|\bpf\d|"
                          r"gait|grip|walking speed|lung function|single cs|put paper|draw the figure|balance: outcome|hand behind|raise arms|pulse|blood pressure measure"),
    ("Function and limitation", r"difficult|limit|mobilit|iadl|\badl|aids?\b|help|care|jog|push|bath|eating|toilet|climb|stoop|"
                                r"dress|\bbed\b|walk|can.?t|cannot|stops (them|r|me)|health prevents|holding a pencil|do the things"),
    ("Symptoms and conditions", r"swell|breath|faint|fall|symptom|balance|incontinen|urin|dizz|cholesterol|stomach|disease|"
                                r"condition|claudication|legs|vision|visual|hear|teeth|not-well|unwell|bmi|body mass|obese|weight|serious illness|cataract|joint replaced"),
    ("Medication and health care", r"drug|medic|meds\b|doctor|hospital|prescri|treatment|nurse|prev care|screening|services"),
    ("Self-rated health and outlook", r"self-rated|self-perceived|self-report of health|perceived age|feels they are|how old|satisf|"
                                      r"future looks|age prevents|control|happiness|casp|general health|health change|health in general|"
                                      r"sense of|continue your life|spheu|sphus|thermometer"),
    ("Mood and cognition", r"memory|irritab|interest|sad|lonel|happy|enjoy|depress|anxi|cognit|fluency|animals mentioned|writing|"
                           r"recall|optimis|bright side|fearful|useless|tearful|panas|purpose of life|food named|tv or listen"),
    ("Work, money and social life", r"work|job|retire|pension|money|income|email|web|internet|drive|pets|volunteer|benefit|social|"
                                    r"friend|famil|club|church|claims|labor|labour|unemploy|financ|spending|insurance|home|house"),
    ("Age and background", r"birth|\bage\b|age confirmed|year of|educat|childhood|sex|gender|born|trueage"),
]


def theme(label):
    for t, p in THEMES:
        if re.search(p, str(label), re.I):
            return t
    return "Other"


rank["theme"] = rank.label.map(theme)
rank["dataset"] = rank.cohort + " · " + rank.instrument
rank[["dataset", "key", "label", "theme", "mean_importance_pct"]].to_csv(os.path.join(OUT, "theme_assignment.csv"), index=False)
order = ["SHARE · Rockwood", "HRS · Rockwood", "ELSA · Rockwood", "MHAS · Rockwood", "CLHLS · Rockwood",
         "SHARE · Fried", "HRS · Fried", "ELSA · Fried", "MHAS · Fried"]
piv = rank.pivot_table(index="theme", columns="dataset", values="mean_importance_pct", aggfunc="sum").reindex(columns=order).fillna(0)
themes = [dict(theme=t, dataset=d, pct=round(float(piv.at[t, d]), 1)) for t in piv.index for d in order]

layers = exc.reason.str.replace(r" \(.*", "", regex=True).str.replace(r" of .*", "", regex=True)
pipeline = dict(candidates=int(models.variables_in_files.sum()), kept=int(models.used.sum()), models=int(len(models)),
                excluded=int(len(exc)), by_layer={k: int(v) for k, v in layers.value_counts().items()},
                r2_fi=[float(models[models.instrument == "Rockwood"].heldout_R2.min()), float(models[models.instrument == "Rockwood"].heldout_R2.max())],
                r2_fried=[float(models[models.instrument == "Fried"].heldout_R2.min()), float(models[models.instrument == "Fried"].heldout_R2.max())])

# stability: HRS frailty index, harmonised waves 4–14
parts = [json.load(open(f, encoding="utf-8")) for f in glob.glob(os.path.join(OUT, "parts", "HRS_Rockwood_w*.json"))]
parts = sorted([p for p in parts if p["wave"] <= 14], key=lambda p: p["wave"])
top5 = rank[(rank.cohort == "HRS") & (rank.instrument == "Rockwood")].sort_values("mean_importance_pct", ascending=False).head(5)
year = {w: 1992 + 2 * (w - 1) for w in range(1, 17)}
stab = []
for p in parts:
    ranked = sorted(p["share_all"].items(), key=lambda kv: -kv[1])
    pos = {k: i + 1 for i, (k, _) in enumerate(ranked)}
    for k, lab in zip(top5.key, top5.label):
        stab.append(dict(key=k, label=lab, wave=p["wave"], year=year[p["wave"]], pct=round(100 * p["share_all"].get(k, 0), 2), rank=pos.get(k)))

json.dump(dict(themes=themes, pipeline=pipeline, stability=stab), open(os.path.join(OUT, "report_figures.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(piv.round(1).to_string())
print(json.dumps(pipeline, indent=1))
print(pd.DataFrame(stab).pivot_table(index="label", columns="year", values="rank").to_string())
oth = rank[rank.theme == "Other"].sort_values("mean_importance_pct", ascending=False).head(15)
print("\nlargest 'Other':"); [print(f"  {r.dataset:18s} {r.mean_importance_pct:5.2f} {str(r.label)[:70]}") for r in oth.itertuples()]
