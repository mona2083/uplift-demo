import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class TLearnerUplift:
    def __init__(self, random_state: int = 42):
        params = dict(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state)
        self.model_t = GradientBoostingClassifier(**params)
        self.model_c = GradientBoostingClassifier(**params)
        self.scaler  = StandardScaler()
        self.feat_cols: list[str] = []

    def fit(self, df: pd.DataFrame, feat_cols: list[str]) -> None:
        self.feat_cols = feat_cols
        X = self.scaler.fit_transform(df[feat_cols])
        mask_t = df["treatment"] == 1
        mask_c = df["treatment"] == 0
        self.model_t.fit(X[mask_t], df.loc[mask_t, "conversion"])
        self.model_c.fit(X[mask_c], df.loc[mask_c, "conversion"])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.scaler.transform(df[self.feat_cols])
        p_treatment = self.model_t.predict_proba(X)[:, 1]
        p_control   = self.model_c.predict_proba(X)[:, 1]
        uplift      = p_treatment - p_control
        return pd.DataFrame({
            "customer_id":   df["customer_id"].values,
            "p_treatment":   p_treatment.round(4),
            "p_control":     p_control.round(4),
            "uplift_score":  uplift.round(4),
        })


def classify_segments(
    scores: pd.DataFrame,
    uplift_threshold: float = 0.0,
    buy_threshold:    float = 0.5,
) -> pd.DataFrame:
    df = scores.copy()
    conditions = [
        (df["uplift_score"] >= uplift_threshold) & (df["p_control"] >= buy_threshold),
        (df["uplift_score"] >= uplift_threshold) & (df["p_control"] <  buy_threshold),
        (df["uplift_score"] <  uplift_threshold) & (df["p_control"] <  buy_threshold),
        (df["uplift_score"] <  uplift_threshold) & (df["p_control"] >= buy_threshold),
    ]
    labels = ["Sure Things", "Persuadables", "Lost Causes", "Sleeping Dogs"]
    df["segment"] = np.select(conditions, labels, default="Unknown")
    return df


def calculate_roi(
    scored: pd.DataFrame,
    coupon_cost:       float = 5.0,
    avg_order_value:   float = 60.0,
    gross_margin:      float = 0.30,
) -> dict:
    segs = scored.groupby("segment")

    n_persuadable  = len(scored[scored["segment"] == "Persuadables"])
    n_sure_things  = len(scored[scored["segment"] == "Sure Things"])
    n_all          = len(scored)

    # 全員にクーポン配布した場合
    baseline_coupon_cost     = n_all * coupon_cost
    baseline_uplift_revenue  = scored["uplift_score"].clip(lower=0).sum() * avg_order_value * gross_margin
    baseline_net             = baseline_uplift_revenue - baseline_coupon_cost

    # 説得可能層のみにクーポン配布した場合
    persuadable_rows         = scored[scored["segment"] == "Persuadables"]
    targeted_coupon_cost     = n_persuadable * coupon_cost
    targeted_uplift_revenue  = persuadable_rows["uplift_score"].clip(lower=0).sum() * avg_order_value * gross_margin
    targeted_net             = targeted_uplift_revenue - targeted_coupon_cost

    saving = baseline_coupon_cost - targeted_coupon_cost
    saving_pct = (saving / baseline_coupon_cost * 100) if baseline_coupon_cost > 0 else 0

    return {
        "n_all":                   n_all,
        "n_persuadable":           n_persuadable,
        "n_sure_things":           n_sure_things,
        "baseline_coupon_cost":    round(baseline_coupon_cost, 0),
        "baseline_uplift_revenue": round(baseline_uplift_revenue, 0),
        "baseline_net":            round(baseline_net, 0),
        "targeted_coupon_cost":    round(targeted_coupon_cost, 0),
        "targeted_uplift_revenue": round(targeted_uplift_revenue, 0),
        "targeted_net":            round(targeted_net, 0),
        "coupon_saving":           round(saving, 0),
        "coupon_saving_pct":       round(saving_pct, 1),
    }
