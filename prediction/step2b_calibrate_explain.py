"""
Step 2b: recalibrate the chosen models for the later period, and explain what drives their predictions.

Recalibration: the classifiers over-predicted risk in the later window (w6->w8). The w6->w8 rows of TRAINING people
(never used for fitting, never part of any test) form a recalibration set: a logistic recalibration of the model's
logit (intercept and slope) is fitted there, then checked on the temporal test people.
Explanation: TreeSHAP contributions of the LightGBM models on the internal test set; mean |SHAP| per input,
grouped into fixed characteristics and changeable levers. These describe the model, not causal effects.
"""
import os, json, pickle, warnings, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import step2_models as S
from prepare import OUT as CACHE

warnings.filterwarnings("ignore")
RES = S.RES
step = S.step


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


if __name__ == "__main__":
    A, feats = S.load()
    out = {}
    # ---------------- classifier: recalibration
    M = pickle.load(open(os.path.join(CACHE, "step2_cls_models.pkl"), "rb"))
    gbm = M["models"]["M2 LightGBM"]
    A["obs_cls"] = A.status.isin(["fi", "dead"])
    D = A[(A.at_risk == 1) & A.obs_cls]
    rec, tt, it = D[D.set == "unused"], D[D.set == "temporal"], D[D.set == "internal"]
    step(f"Recalibration set (training people, w6->w8): {len(rec):,} rows, {int(rec.frail_or_dead.sum()):,} events")
    p_rec = gbm.predict_proba(rec[feats])[:, 1]
    cal = LogisticRegression(C=1e6).fit(logit(p_rec).reshape(-1, 1), rec.frail_or_dead)
    a, b = float(cal.intercept_[0]), float(cal.coef_[0, 0])
    recal = lambda p: 1 / (1 + np.exp(-(a + b * logit(p))))
    rows = []
    for name, Sx in [("internal (2011-13 baseline)", it), ("temporal (2015 baseline)", tt)]:
        p = gbm.predict_proba(Sx[feats])[:, 1]
        for lab, q in [("as trained", p), ("recalibrated for the later period", recal(p))]:
            ci, cs = S.calib(Sx.frail_or_dead.values, q)
            rows.append(dict(test=name, version=lab, observed=Sx.frail_or_dead.mean(), predicted=q.mean(), calib_in_large=ci,
                             calib_slope=cs, AUC=roc_auc_score(Sx.frail_or_dead, q), Brier=brier_score_loss(Sx.frail_or_dead, q)))
            # calibration by tenth of predicted risk
            dec = pd.qcut(q, 10, labels=False, duplicates="drop")
            pd.DataFrame({"decile": dec, "pred": q, "obs": Sx.frail_or_dead.values}).groupby("decile").mean().assign(test=name, version=lab) \
                .to_csv(os.path.join(RES, "step2b_calibration_deciles.csv"), mode="a", header=not os.path.exists(os.path.join(RES, "step2b_calibration_deciles.csv")))
    C = pd.DataFrame(rows)
    C.round(4).to_csv(os.path.join(RES, "step2b_recalibration.csv"), index=False)
    out["recalibration"] = dict(intercept=round(a, 4), slope=round(b, 4), n=len(rec))
    print(C.round(4).to_string(), flush=True)

    # ---------------- explanation (classifier and regression LightGBM)
    groups = {f: "changeable" for f in S.LEVERS}
    groups.update({f: "fixed" for f in S.FIXED})
    groups.update({f: "fixed" for f in feats if f.startswith("ctry_")})
    groups["years_planned"] = "follow-up length"
    for kind, path, ycol in [("cls", "step2_cls_models.pkl", "frail_or_dead"), ("reg", "step2_reg_models.pkl", "fi_f")]:
        if not os.path.exists(os.path.join(CACHE, path)) or os.path.getsize(os.path.join(CACHE, path)) == 0:
            step(f"{kind}: model file not ready; skipped"); continue
        m = pickle.load(open(os.path.join(CACHE, path), "rb"))["models"]["M2 LightGBM"]
        X = (it if kind == "cls" else A[(A.status == "fi") & (A.set == "internal")])[feats]
        X = X.sample(min(6000, len(X)), random_state=S.SEED)
        sh = m.booster_.predict(X, pred_contrib=True)[:, :-1]
        imp = np.abs(sh).mean(axis=0)
        dirn = [np.sign(np.corrcoef(X.iloc[:, j].fillna(X.iloc[:, j].median()), sh[:, j])[0, 1]) if X.iloc[:, j].nunique() > 1 else 0 for j in range(len(feats))]
        T = pd.DataFrame({"variable": feats, "group": [groups.get(f, "fixed") for f in feats], "mean_abs_shap": imp,
                          "share_pct": 100 * imp / imp.sum(), "direction": dirn}).sort_values("mean_abs_shap", ascending=False)
        # countries pooled into one row
        ct = T[T.variable.str.startswith("ctry_")]
        T = pd.concat([T[~T.variable.str.startswith("ctry_")], pd.DataFrame([dict(variable="country (all)", group="fixed",
                       mean_abs_shap=ct.mean_abs_shap.sum(), share_pct=ct.share_pct.sum(), direction=0)])]).sort_values("share_pct", ascending=False)
        T.round(4).to_csv(os.path.join(RES, f"step2b_{kind}_shap.csv"), index=False)
        out[kind + "_group_share"] = T.groupby("group").share_pct.sum().round(1).to_dict()
        step(f"{kind}: share of importance by group {out[kind + '_group_share']}")
        print(T.head(25).round(3).to_string(), flush=True)
    json.dump(out, open(os.path.join(RES, "step2b_summary.json"), "w"), indent=1)
    step("Done")
