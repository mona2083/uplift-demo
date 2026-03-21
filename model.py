import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class TLearnerUplift:
    def __init__(self, random_state: int = 42):
        # サンプルサイズの偏りに強いよう、少し深めの木とサブサンプリングを設定
        params = dict(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.05, 
            subsample=0.8,
            random_state=random_state
        )
        self.model_t = GradientBoostingClassifier(**params)
        self.model_c = GradientBoostingClassifier(**params)
        self.scaler  = StandardScaler()
        self.feat_cols: list[str] = []

    def fit(self, df: pd.DataFrame, feat_cols: list[str]) -> None:
        self.feat_cols = feat_cols
        X = self.scaler.fit_transform(df[feat_cols])
        mask_t = df["treatment"] == 1
        mask_c = df["treatment"] == 0
        
        # 処置群・対照群それぞれで学習
        self.model_t.fit(X[mask_t], df.loc[mask_t, "conversion"])
        self.model_c.fit(X[mask_c], df.loc[mask_c, "conversion"])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.scaler.transform(df[self.feat_cols])
        # 確率予測。[:, 1]で購入確率を取得
        p_treatment = self.model_t.predict_proba(X)[:, 1]
        p_control   = self.model_c.predict_proba(X)[:, 1]
        
        # アップリフトスコア = 処置時の確率 - 制御時の確率
        uplift = p_treatment - p_control
        
        return pd.DataFrame({
            "customer_id":   df["customer_id"].values,
            "p_treatment":   p_treatment.round(4),
            "p_control":     p_control.round(4),
            "uplift_score":  np.clip(uplift, -1.0, 1.0).round(4), # 物理的な限界値でクリップ
        })

# classify_segments と calculate_roi は既存のロジックが優秀なため維持
def classify_segments(scores: pd.DataFrame, uplift_threshold: float = 0.0, buy_threshold: float = 0.5) -> pd.DataFrame:
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

def calculate_roi(scored: pd.DataFrame, coupon_cost: float = 5.0, avg_order_value: float = 60.0, gross_margin: float = 0.30) -> dict:
    n_persuadable  = len(scored[scored["segment"] == "Persuadables"])
    n_all          = len(scored)
    
    # Clip(0) により、マイナスの効果（あまのじゃく層）を売上計算から除外
    baseline_uplift_revenue  = scored["uplift_score"].clip(lower=0).sum() * avg_order_value * gross_margin
    baseline_coupon_cost     = n_all * coupon_cost
    
    persuadable_rows         = scored[scored["segment"] == "Persuadables"]
    targeted_uplift_revenue  = persuadable_rows["uplift_score"].clip(lower=0).sum() * avg_order_value * gross_margin
    targeted_coupon_cost     = n_persuadable * coupon_cost

    return {
        "baseline_coupon_cost": round(baseline_coupon_cost, 0),
        "baseline_uplift_revenue": round(baseline_uplift_revenue, 0),
        "baseline_net": round(baseline_uplift_revenue - baseline_coupon_cost, 0),
        "targeted_coupon_cost": round(targeted_coupon_cost, 0),
        "targeted_uplift_revenue": round(targeted_uplift_revenue, 0),
        "targeted_net": round(targeted_uplift_revenue - targeted_coupon_cost, 0),
        "coupon_saving": round(baseline_coupon_cost - targeted_coupon_cost, 0),
        "coupon_saving_pct": round((baseline_coupon_cost - targeted_coupon_cost) / baseline_coupon_cost * 100, 1) if baseline_coupon_cost > 0 else 0
    }