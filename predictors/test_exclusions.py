"""Regression tests for the exclusion rules: true leaks must be caught, legitimate predictors kept."""
import importlib.util, re
spec = importlib.util.spec_from_file_location("R", r"D:\Database - portal\FrailDB\predictors\run_predictors.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

CORE = {f"item__{k}" for k in R.ROCK_CONCEPTS if not k.startswith(("d_iadl_cook", "d_walk_1km", "d_chair_stand", "d_crouch_stand",
        "d_carry_5kg", "d_continence", "d_visit", "d_wash", "d_public", "d_housework", "d_hand", "d_equip"))}
CLHLS = {f"item__{k}" for k in ["d_srh", "d_hypertension", "d_diabetes", "d_heart", "d_stroke", "d_lung", "d_cancer", "d_arthritis",
         "d_adl_bath", "d_adl_dress", "d_adl_toilet", "d_adl_bed", "d_continence", "d_adl_eat", "d_visit_neighbours", "d_iadl_shop",
         "d_iadl_cook", "d_wash_clothes", "d_walk_1km", "d_carry_5kg", "d_crouch_stand", "d_public_transport", "d_chair_stand",
         "d_housework", "d_hand_behind_neck"]}


def excluded(label, cohort="HRS", method="Rockwood", items=CORE, mode="own"):
    """The pipeline's own rule function (run_predictors.label_reason), applied to a label alone."""
    return R.label_reason(cohort, method, "zz_test", label, frozenset(items), mode=mode) is not None


cases = [
    # (label, cohort, method, items, should_be_excluded)
    ("BED EQUIPMENT", "HRS", "Rockwood", CORE, True),           # follow-up of the in/out-of-bed ADL
    ("whether anyone helps R get into/out of bed", "HRS", "Rockwood", CORE, True),
    ("whether anyone helps R walk", "MHAS", "Rockwood", CORE, True),
    ("Received help in last month: getting in and out of bed", "ELSA", "Rockwood", CORE, True),
    ("M007 LIMIT IN ANYWAY", "HRS", "Rockwood", CORE, True),
    ("FELT UNMOTIVATED W/IN PREV WK", "HRS", "Rockwood", CORE, True),
    ("WAS R HAPPY W/IN PREV WK", "HRS", "Rockwood", CORE, True),
    ("LUNG OXYGEN", "HRS", "Rockwood", CORE, True),
    ("Health thermometer", "SHARE", "Rockwood", CORE, True),
    ("Health: self-care", "SHARE", "Rockwood", CORE, True),
    ("Health: usual activities", "SHARE", "Rockwood", CORE, True),
    ("Drugs for: high blood pressure", "SHARE", "Rockwood", CORE, True),
    ("r takes meds for breathing problem", "HRS", "Rockwood", CORE, True),
    ("Aids: whether cane acquired since last interview", "ELSA", "Rockwood", CORE, True),
    ("Country-specific question", "SHARE", "Rockwood", CORE, True),
    ("SECTION G TOTAL TIME - IN SECONDS", "HRS", "Rockwood", CORE, True),
    ("(H71) who helped interviewee to answer question?", "CLHLS", "Rockwood", CLHLS, True),
    ("w1 R Some difficulty-Lower Body Mobility 0-4", "MHAS", "Rockwood", CORE, True),
    ("w1 R receives any formal care for ADLs/IADLs", "MHAS", "Rockwood", CORE, True),
    ("w2 R receives any informal care for ADLs", "MHAS", "Rockwood", CORE, True),
    ("ADL&IADL: whether said had none of listed difficulties", "ELSA", "Rockwood", CORE, True),
    ("DIFFICULTY GET IN/OUT BED", "HRS", "Rockwood", CORE, True),
    ("DIFFICULTY USING TOILET", "HRS", "Rockwood", CORE, True),
    ("w5 whether anyone helps R use the toilet", "MHAS", "Rockwood", CORE, True),
    ("WALK EQUIPMENT", "HRS", "Rockwood", CORE, True),
    ("HEART MEDICATION", "HRS", "Rockwood", CORE, True),
    ("HAS R SEEN HEART DOCTOR", "HRS", "Rockwood", CORE, True),
    ("LUNG MEDICATION", "HRS", "Rockwood", CORE, True),
    ("Diagnosed heart murmur", "ELSA", "Rockwood", CORE, True),
    ("Do you get it when you walk uphill or hurry? {pain in chest)", "ELSA", "Rockwood", CORE, True),
    ("What symptoms [prevent you/make it difficult to] walk(ing) quarter of a mile? 1st", "ELSA", "Rockwood", CORE, True),
    ("Able to walk alone (with aid)?", "ELSA", "Rockwood", CORE, True),
    ("Whether used meals on wheels in last month", "ELSA", "Rockwood", CORE, True),
    ("Adaptations in property: bed lever or bed rail", "ELSA", "Rockwood", CORE, True),
    ("Reasons for not using public transport: their health prevents them", "ELSA", "Rockwood", CORE, True),
    ("CASP: health stops r from doing what they want", "HRS", "Rockwood", CORE, True),
    ("Weight: reason for not obtaining weight measurement: confined to bed", "ELSA", "Rockwood", CORE, True),
    ("r willing and able to complete walking speed test", "HRS", "Rockwood", CORE, True),
    ("(D5MEDICA) got timely medication before dying", "CLHLS", "Rockwood", CLHLS, True),
    ("(D11G1061) in which ear(s) did the elder have a hearing difficulty?", "CLHLS", "Rockwood", CLHLS, True),
    ("Need help: getting in or out of bed", "SHARE", "Rockwood", CORE, True),
    ("Health: to walk about", "SHARE", "Rockwood", CORE, True),
    ("Special features in accommodation: bathroom/toilet modifications", "SHARE", "Rockwood", CORE, True),
    ("Memory difficulty medication", "SHARE", "Rockwood", CORE, True),
    ("Help needed: medicines", "SHARE", "Rockwood", CORE, True),
    ("Walking symptom: fatigue", "ELSA", "Rockwood", CORE, True),
    ("NHS paid for the lift", "ELSA", "Rockwood", CORE, True),
    ("Lung function: Whether admitted to hospital for heart complaint in last 6 weeks", "ELSA", "Rockwood", CORE, True),
    ("Have you ever had any pain or discomfort in your chest?", "ELSA", "Rockwood", CORE, True),
    ("Do you find it difficult to follow a conversation if there is background noise", "ELSA", "Rockwood", CORE, True),
    ("did anyone help the interviewee to answer question?", "CLHLS", "Rockwood", CLHLS, True),
    ("(H1) was interviewee able to hear?", "CLHLS", "Rockwood", CLHLS, True),
    ("Have you ever had a severe pain across the front of your chest lasting for half an hour", "ELSA", "Rockwood", CORE, True),
    ("Whether person who helps lives in same household as respondent", "ELSA", "Rockwood", CORE, True),
    ("How LA is involved in paying for care: direct payments", "ELSA", "Rockwood", CORE, True),
    ("Whether states cannot walk", "ELSA", "Rockwood", CORE, True),
    ("LUNG RESPIRATORY THERAPY", "HRS", "Rockwood", CORE, True),
    ("LOSE INTEREST- CIDI", "HRS", "Rockwood", CORE, True),
    ("(H22) the reason for inability to participate in physical check", "CLHLS", "Rockwood", CLHLS, True),
    ("(C55) what is the main reason unable to answer questions?", "CLHLS", "Rockwood", CLHLS, True),
    ("In the last two years have you received care at home that was not", "ELSA", "Rockwood", CORE, True),
    # legitimate predictors that must be KEPT
    ("R+S provided personal care to parents 1hr+/wk", "MHAS", "Rockwood", CORE, False),
    ("During the last 12 months, have you lost any amount of urine beyond your control", "ELSA", "Rockwood", CORE, False),
    ("Any other helper from outside the household", "SHARE", "Rockwood", CORE, False),
    ("RELATIVES/FRIENDS HELP W/ FUTURE NEEDS", "HRS", "Rockwood", CORE, False),
    ("Have you heard of telemedicine before?", "SHARE", "Rockwood", CORE, False),
    ("ActiPASS Check: 1+ day with < 30 sec walking", "SHARE", "Rockwood", CORE, False),
    ("Time spent walking or exercising yesterday (HOURS)", "ELSA", "Rockwood", CORE, False),
    ("Q27C. TRY HARD CARRY OUT BELIEFS", "HRS", "Rockwood", CORE, False),
    ("CASP: age prevents r from doing things they like", "HRS", "Rockwood", CORE, False),
    ("CASP: money shortage stops r from doing things they want", "HRS", "Rockwood", CORE, False),
    ("whether r feels safe/afraid walking alone in this area", "HRS", "Rockwood", CORE, False),
    ("People would be afraid to walk alone in this area after dark", "ELSA", "Rockwood", CORE, False),
    ("r time for walking speed test 1(sec)", "HRS", "Rockwood", CORE, False),
    ("Indoor flushing toilet", "SHARE", "Rockwood", CORE, False),
    ("How many meals a day", "SHARE", "Rockwood", CORE, False),
    ("DAYS IN BED", "HRS", "Rockwood", CORE, False),
    ("(G20) heart rhythm", "CLHLS", "Rockwood", CLHLS, False),
    ("Back pain rating walking on flat surface", "ELSA", "Rockwood", CORE, False),
    ("How often do you have problems with keeping your balance when you are walking", "ELSA", "Rockwood", CORE, False),
    ("Screening (women): whether ever had breast cancer screening (mammogram or x-ray)", "ELSA", "Rockwood", CORE, False),
    ("w2 R Doctor visits, previous 12 months", "MHAS", "Rockwood", CORE, False),
    ("w1 R Persistent swelling in feet/ankles", "MHAS", "Rockwood", CORE, False),
    ("(D31) how often eat fresh fruit?", "CLHLS", "Rockwood", CLHLS, False),
    ("how often eat meat at present?", "CLHLS", "Rockwood", CLHLS, False),
    ("was the county (city) you born in the same as current address?", "CLHLS", "Rockwood", CLHLS, False),
    ("(G7) heart rate", "CLHLS", "Rockwood", CLHLS, False),
    ("(F66) often went to bed hungry as a child", "CLHLS", "Rockwood", CLHLS, False),
    ("length from kneel to floor", "CLHLS", "Rockwood", CLHLS, False),
    ("raise arms upright", "CLHLS", "Rockwood", CLHLS, False),           # CLHLS index has no arms item
    ("Do you have enough money to meet your needs", "ELSA", "Rockwood", CORE, False),
    ("Bathroom in the accommodation", "SHARE", "Rockwood", CORE, False),
    ("Times talked to medical doctor/nurse about your health", "SHARE", "Rockwood", CORE, False),
    ("how many meals per day", "CLHLS", "Rockwood", CLHLS, False),
    ("average blood pressure measure (systolic)", "HRS", "Rockwood", CORE, False),
    ("Had mammogram / cancer screening", "HRS", "Rockwood", CORE, False),
    ("DIFFICULTY- JOGGING 1 MILE", "HRS", "Rockwood", CORE, False),
    ("Severity of pain most of the time", "ELSA", "Rockwood", CORE, False),
    ("Feel full of energy", "SHARE", "Rockwood", CORE, False),
    ("RATE MEMORY", "HRS", "Rockwood", CORE, False),
    ("At least taking 5 different drugs a typical day", "SHARE", "Rockwood", CORE, False),
    ("Drugs for: joint pain", "SHARE", "Rockwood", CORE, False),
    ("COMPARE HEALTH TO PREVIOUS WAVE", "HRS", "Rockwood", CORE, False),
    # Fried
    ("r self-rated hand strength", "HRS", "Fried", CORE, True),
    ("Record respondent status", "SHARE", "Fried", CORE, True),
    ("Difficulties: climbing several flights of stairs", "SHARE", "Fried", CORE, True),
    ("Difficulties: climbing several flights of stairs", "ELSA", "Fried", CORE, False),   # ELSA slowness = timed walk
    ("Difficulty-Lifting 5 kilos", "MHAS", "Fried", CORE, True),
    ("Difficulty-Lifting 5 kilos", "SHARE", "Fried", CORE, False),
    ("PANAS-X: degree r felt active", "HRS", "Fried", CORE, False),
    ("Difficulties: pulling or pushing large objects", "SHARE", "Fried", CORE, False),
]
bad = 0
for lab, co, m, items, want in cases:
    got = excluded(lab, co, m, items); bad += got != want
    print(f"{'OK   ' if got == want else 'WRONG'} {m[:4]} {co:5s} excl={got!s:5} want={want!s:5} {lab}")

# Cross-instrument rules: (label, cohort, track, excluded in "own" mode, excluded in "both" mode)
cross = [
    # Fried criteria in the Rockwood track
    ("Sports or activities that are vigorous", "SHARE", "Rockwood", False, True),
    ("Frequency does moderate sports or activities", "ELSA", "Rockwood", False, True),
    ("Physical inactivity", "SHARE", "Rockwood", False, True),
    ("r self-rated hand strength", "HRS", "Rockwood", False, True),
    ("Body mass index", "SHARE", "Rockwood", False, True),
    ("Time taken for first walk (seconds)", "ELSA", "Rockwood", False, True),
    ("Feel full of energy", "SHARE", "Rockwood", False, True),
    ("w1 R Severe fatigue", "MHAS", "Rockwood", False, True),
    ("Difficulty-Lifting 5 kilos", "SHARE", "Rockwood", True, True),          # a Rockwood item anyway
    # Rockwood items in the Fried track
    ("Difficulties: dressing, including putting on shoes and socks", "SHARE", "Fried", False, True),
    ("w1 R Difficulty-Bathing or showering", "MHAS", "Fried", False, True),
    ("Self-reported general health", "ELSA", "Fried", False, True),
    ("Limitations with activities - gali", "SHARE", "Fried", False, True),
    ("r receives any informal care for adls/iadls", "HRS", "Fried", False, True),
    ("Aids used: not use any of listed aids", "ELSA", "Fried", False, True),
    ("Ever diagnosed: high blood pressure or hypertension", "SHARE", "Fried", False, True),
    ("Irritability", "SHARE", "Fried", False, False),                           # EURO-D item, not an FI deficit itself
    ("r whether obese", "HRS", "Rockwood", False, True),
    ("Special features in accommodation: none of these", "SHARE", "Rockwood", True, True),
    ("Received home care: none of these", "SHARE", "Rockwood", True, True),
    ("Received home care: none of these", "SHARE", "Fried", False, True),
    # legitimate predictors kept in both modes and both tracks
    ("PANAS-X: degree r felt active", "HRS", "Rockwood", False, False),
    ("Feels lonely", "SHARE", "Fried", False, False),
    ("R Prev Care: Mammogram last 2y", "MHAS", "Fried", False, False),
    ("frequency r short of breath", "HRS", "Rockwood", False, False),
    ("Severity of pain most of the time", "ELSA", "Rockwood", False, False),
    ("Severity of pain most of the time", "ELSA", "Fried", False, False),
    ("At least taking 5 different drugs a typical day", "SHARE", "Rockwood", False, False),
    ("Have you heard of telemedicine before?", "SHARE", "Fried", False, False),
    ("r full-tandem test result(sec)", "HRS", "Fried", False, False),
    ("whether r regularly uses web for emails", "HRS", "Rockwood", False, False),
]
for lab, co, m, want_own, want_both in cross:
    g1, g2 = excluded(lab, co, m, CLHLS if co == "CLHLS" else CORE, "own"), excluded(lab, co, m, CLHLS if co == "CLHLS" else CORE, "both")
    ok = (g1, g2) == (want_own, want_both); bad += not ok
    print(f"{'OK   ' if ok else 'WRONG'} {m[:4]} {co:5s} own={g1!s:5} both={g2!s:5} (want {want_own}/{want_both}) {lab}")
print("failures:", bad, "of", len(cases) + len(cross))
