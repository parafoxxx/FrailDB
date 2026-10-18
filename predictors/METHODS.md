# Most influential predictors of frailty: methods

Draft Methods text for the Frailty_Volatility manuscript and its supplement. Numbers are from the final
run (59 wave models, one script version); tables are in `output/`. Code: `run_predictors.py` (models), `aggregate.py` (tables).

## 1. What the analysis was asked to do (study specification)

The research group specified:

1. Two independent tracks, one for the Rockwood frailty index and one for the Fried frailty phenotype.
2. Self-contamination exclusion: when modelling an instrument, its own constituent variables are not
   predictors (the 32 deficits for Rockwood; the five criteria for Fried).
3. For every wave of every dataset, the top 10 and top 20 most influential predictors for each instrument.
4. Aggregation across waves within each dataset to an overall top 5.

The specification did not fix the outcome scale, the model, the measure of influence, or how far
"constituent variables" extends. Those choices, and the reasons for them, are set out below.

## 2. Inputs that were provided and reused unchanged

| Input | Reused as | Reason for reuse |
|---|---|---|
| Per-wave frailty outputs produced by the group (`<COHORT>_w<N>_Rockwood_summary.csv/.rds`, `_Fried_…`) | The outcome (each person's FI score and Fried score) and the item indicators used for leakage screening | Keeps the predictor analysis consistent with the frailty values reported elsewhere in the study and in FrailDB. Frailty was not re-scored. |
| Per-wave "Steps followed" documents and analysis scripts | The list of source variables behind each deficit and criterion | These define what "constituent" means for each cohort. |
| Raw survey files (SHARE release 9.0.0 modules; ELSA UKDS SN 5050 core and nurse files; CLHLS ICPSR 38899; HRS 2020 and 2022 core) | Candidate predictors | The specification asks for the full set of measured variables. |
| Gateway harmonised files (H_HRS version D; H_MHAS version D) | Candidate predictors where raw files were not available | HRS raw core files for 1998–2018 and MHAS raw questionnaire files are not in the project data. |

## 3. Analysis performed

### 3.1 Samples and outcome
Each wave's analytic sample is the group's complete-case sample for that instrument (age ≥ 50; ≥ 60 for the
ELSA Fried phenotype). Outcomes are continuous: frailty index score (0–1) and Fried score (0–5).
*Why continuous:* a 0.25 or ≥ 3 threshold discards within-category variation and makes importance rankings
less stable; the continuous score uses all of the information the instrument records.

### 3.2 Candidate predictors
Every variable in the wave's survey files was a candidate, after removing free-text fields, variables with
> 95% missing values in the analytic sample, and constant variables. Survey non-response codes (negative
codes in SHARE and ELSA) were set to missing. Only same-wave information was used: for CLHLS, whose ICPSR
files also carry later follow-up waves (suffixes _5 to _18) and survival status (DTH…), those columns were
dropped before analysis.

### 3.3 Exclusion of constituent, derived and leaking variables
A literal exclusion of only the instrument's own variables was not sufficient. In pilot runs the rankings
were dominated by other measurements of the same items (for example, in MHAS a NAGI mobility score built
from the same mobility items ranked first with 24.5% of importance, held-out R² 0.92; in MHAS the year of
death entered the top 20). Variables were therefore excluded in four layers, each applied per instrument:

1. **Constituent items:** the instrument's own source variables, listed per cohort (including all options of
   the same question grid, such as SHARE's "none of these" category).
2. **Derived scores and follow-up questions.**
   *Generic rules (all cohorts):* scores computed from instrument items (ADL/IADL/mobility counts, NAGI,
   CES-D/EURO-D totals), global activity limitation (GALI-type items, EQ-5D "usual activities" and
   "self-care"), help, care and aids received for functional difficulty, and values fed forward from the
   previous interview.
   *Item-specific rules:* each rule is tied to one deficit or criterion and applies **only in cohorts whose
   instrument contains that item**. Examples: questions about bathing are excluded only where bathing is a
   deficit; blood-pressure medication only where hypertension is a deficit; stair climbing counts as Fried
   slowness in SHARE and HRS (self-reported slowness) but not in ELSA (timed walk). The rules match the
   specific wording of the item (e.g. "managing money", "getting in or out of bed") rather than single
   words, after pilot runs showed that bare words removed legitimate predictors (diet items matched "eat",
   "address" matched "dress", measured heart rate matched "heart"). Every variable whose classification
   changed between the pilot and final rules was reviewed by hand (`audit_rule_change.py`; 573 variables
   restored as legitimate predictors, e.g. diet, finances, measured blood pressure, cancer screening, gait
   timings). The final rules are checked by 111 regression tests (`test_exclusions.py`) covering known leaks
   that must be excluded and legitimate predictors that must be kept.
3. **Data-driven near-copies:** any remaining variable with |Spearman ρ| ≥ 0.75 with a constituent item, or
   ≥ 0.80 with the outcome score, in a random subsample of up to 5,000 people. This catches copies under
   different names without hand-mapping thousands of variables.
4. **Administrative and future information:** identifiers, survey weights, design variables, interview
   dates and timings, interviewer observations and other paradata, death and exit-interview variables,
   values fed forward from the previous wave, and harmonised cross-wave summaries (only fixed
   characteristics such as sex, ethnicity, birth year, education and childhood circumstances were kept).

Following the specification, the other instrument's items are **not** excluded: Rockwood deficits may
predict the Fried score and vice versa. Every excluded variable is listed with its reason in
`output/exclusions.csv` (supplementary table).

### 3.4 Model
For each cohort × instrument × wave, a gradient-boosted regression tree model (LightGBM 4.7.0; Ke et
al., 2017) was fitted to a random 80% of participants, with early stopping on the remaining 20% (learning
rate 0.05, 31 leaves, minimum 40 per leaf, 30% of features per tree, 80% bagging, L2 penalty 1, up to 600
trees, seed 20261018). Model fit is reported as held-out R².
*Why this model:* survey files contain thousands of mixed-type variables with structured missingness.
Gradient-boosted trees handle missing values natively, without imputation, and capture non-linear and
interaction effects. Penalised linear regression would require imputing thousands of variables and
assumes linear effects.

### 3.5 Measure of influence
Influence is the mean absolute SHAP value (Lundberg & Lee, 2017), computed exactly for tree models
(TreeSHAP) on up to 4,000 held-out participants and expressed as each variable's share of the total.
Direction is the sign of the correlation between a variable's value and its SHAP value.
*Why SHAP:* split-count and impurity importances are biased towards variables with many categories or
values (Strobl et al., 2007); SHAP values are additive, consistent, comparable across variables, and
computed on people the model has not seen.

### 3.6 Per-wave and per-dataset rankings
Per wave, variables are ranked by mean |SHAP|; the top 10 and top 20 are reported. Per dataset, each
variable's importance share is averaged over all analysed waves of that cohort and instrument, counting 0
in waves where it was absent or unused, so that consistently important variables rank above single-wave
peaks. The overall top 5 are the five highest averages; ties are broken by how often the variable appears
in a wave's top 10. Variables are matched across waves on a wave-free name (for harmonised files,
r10shlt and r11shlt both become shlt). The Gateway harmonised files code each functional item twice
("difficulty" and "some difficulty", e.g. jog and joga); each pair is treated as one item and its
importance summed, so that one question does not occupy two ranks.

## 4. Reproducibility
All 59 wave models were produced by a single version of `run_predictors.py`, identified by its SHA-256
fingerprint (`2a7339500a6857fb45dc43c3686bf28aecf0845fe2737305829fab9436076b1f`), which is stored in every result file and verified by `aggregate.py`. Software: Python
3.12.8, LightGBM 4.7.0, pandas 2.3.3, NumPy 1.26.4, scikit-learn 1.9.1, pyreadstat 1.3.6, pyreadr 0.5.7,
PyMuPDF 1.26.7. Random seed 20261018.

## 5. Limitations
- HRS 1998–2018 and all MHAS waves use harmonised rather than raw variables, so fewer candidate predictors
  are available there than in SHARE, ELSA and CLHLS.
- CLHLS 2005 and 2008 data files (ICPSR DS0004, DS0005) were not available.
- Rankings describe associations within a wave, not causes; some predictors may be consequences of frailty
  (for example, receipt of disability benefits).
- The exclusion rules are necessarily a judgement about what counts as the same measurement. They are
  documented variable by variable so that readers can apply a stricter or looser definition.
- Cohorts differ in how the frailty instruments were operationalised (e.g. measured vs self-reported
  weakness and slowness in the Fried phenotype), which limits direct comparison of rankings across cohorts.

## 6. Development of the exclusion rules (for the supplement)
The rules were developed iteratively on pilot runs, by inspecting each cohort's top-ranked variables for
anything measuring the same thing as an instrument item. The pilot results (`output/pilot_parts/`, previous
rule version) are retained, and the change between the previous and final rule versions is reported as a sensitivity
analysis (`output/sensitivity_pilot_vs_final.csv`): Spearman correlation of variable importance 0.81–0.97,
4–5 of the top 5 unchanged in 7 of 9 datasets (ELSA Rockwood 2 of 5, where follow-up questions were most
numerous), and mean held-out R² changed by at most 0.01. Leak types found, with the cohort where each was first seen:

| Leak type | Example (cohort) | Pilot effect |
|---|---|---|
| Summary score of the items | NAGI mobility score (MHAS) | ranked 1st, 24.5% of importance, R² 0.92 |
| Death / survival information | year of death (MHAS); 2005–2018 follow-ups and survival status inside the 2002 file (CLHLS) | in top 20 |
| Item grids under other names | wave-1 ADL grids `heada*`/`headb*` with truncated labels (ELSA); "none of these" options (SHARE) | R² 0.95 (ELSA w1), 0.85 (SHARE) |
| Help, care and aids follow-ups | care module, help received getting in/out of bed (ELSA); bed equipment (HRS) | ranked 1st |
| Interview paradata | section timings in seconds (HRS raw); interviewer observations (SHARE, CLHLS) | in top 10 |
| Previous-wave information | feed-forward variables (ELSA); preload file (HRS raw); harmonised "last reported" summaries | in top 20 |
| Near-copies of Fried criteria | walking-speed test module (SHARE); self-rated hand strength (HRS) | ranked 1st (SHARE w1) |

## 7. Planned additional analyses (before submission)
- Stability: repeat each wave model with 5 seeds and a participant bootstrap; report rank intervals for
  the top 20.
- Sensitivity: frail/not-frail outcome; near-copy thresholds 0.70 and 0.85; literal (constituents-only)
  exclusion, to show how much the extended exclusions change the rankings.

## References
- Ke G, et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 2017.
- Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. NeurIPS 2017.
- Strobl C, Boulesteix A-L, Zeileis A, Hothorn T. Bias in random forest variable importance measures.
  BMC Bioinformatics 2007;8:25.
- Searle SD, Mitnitski A, Gahbauer EA, Gill TM, Rockwood K. A standard procedure for creating a frailty
  index. BMC Geriatr 2008;8:24.
- Fried LP, et al. Frailty in older adults: evidence for a phenotype. J Gerontol A 2001;56:M146–56.
