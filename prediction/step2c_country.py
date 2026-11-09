"""
Step 2c: leave-one-country-out validation of the chosen LightGBM models (same tuned settings as Step 2).
For each SHARE country: train on every other country (all windows, censoring-weighted as in Step 2), test on that
country. Countries are disjoint, so no person is in both. Shows whether the model transports to a population it never saw.
"""
import os, sys, json, pickle, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, r2_score
import step2_models as S
from prepare import OUT as CACHE

warnings.filterwarnings("ignore")
step = S.step

if __name__ == "__main__":
    A, feats = S.load()
    A["obs_cls"] = A.status.isin(["fi", "dead"]); A["obs_reg"] = A.status == "fi"
    A["is_train_person"] = True                         # censoring models fitted on everyone here
    rows = []
    for kind in ["cls", "reg"]:
        tun = pickle.load(open(os.path.join(CACHE, f"step2_{kind}_models.pkl"), "rb"))["tuning"]["lgbm"][1]
        mask = (lambda g: g.at_risk == 1) if kind == "cls" else (lambda g: g.status != "dead")
        obs, ycol = ("obs_cls", "frail_or_dead") if kind == "cls" else ("obs_reg", "fi_f")
        A["w"] = S.ipcw(A, feats, obs, mask)
        D = A[mask(A) & A[obs]]
        countries = D.country.value_counts()
        step(f"[{kind}] {len(D):,} rows, {len(countries)} countries")
        for i, (c, n) in enumerate(countries.items()):
            tr, te = D[D.country != c], D[D.country == c]
            Cl = lgb.LGBMClassifier if kind == "cls" else lgb.LGBMRegressor
            m = Cl(subsample_freq=1, verbose=-1, random_state=S.SEED, n_jobs=-1, **tun).fit(tr[feats], tr[ycol], sample_weight=tr.w)
            if kind == "cls":
                p = m.predict_proba(te[feats])[:, 1]; ci, cs = S.calib(te[ycol].values, p)
                r = dict(kind=kind, country=int(c), n=len(te), observed=te[ycol].mean(), predicted=p.mean(),
                         AUC=roc_auc_score(te[ycol], p), Brier=brier_score_loss(te[ycol], p), calib_in_large=ci, calib_slope=cs)
            else:
                p = m.predict(te[feats])
                r = dict(kind=kind, country=int(c), n=len(te), observed=te[ycol].mean(), predicted=p.mean(),
                         MAE=mean_absolute_error(te[ycol], p), R2=r2_score(te[ycol], p))
            rows.append(r)
            step(f"[{kind}] {i + 1}/{len(countries)} country {int(c)} (n={len(te):,}): " +
                 ", ".join(f"{k} {v:.3f}" for k, v in r.items() if k in ("AUC", "calib_in_large", "calib_slope", "MAE", "R2")))
            pd.DataFrame(rows).round(4).to_csv(os.path.join(S.RES, "step2c_country.csv"), index=False)
    R = pd.DataFrame(rows)
    summ = {}
    for kind, g in R.groupby("kind"):
        cols = ["AUC", "calib_in_large", "calib_slope"] if kind == "cls" else ["MAE", "R2"]
        summ[kind] = {c: dict(median=round(g[c].median(), 4), min=round(g[c].min(), 4), max=round(g[c].max(), 4),
                              weighted_mean=round(np.average(g[c], weights=g.n), 4)) for c in cols}
    json.dump(summ, open(os.path.join(S.RES, "step2c_summary.json"), "w"), indent=1)
    step(f"Done: {summ}")
