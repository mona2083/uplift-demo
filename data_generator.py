import numpy as np
import pandas as pd

FEATURE_COLS = ["age", "past_spend", "visit_freq", "days_since_last", "avg_basket"]


def generate_demo_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)

    age             = rng.randint(18, 75, n_samples)
    past_spend      = rng.exponential(200, n_samples).clip(10, 2000)
    visit_freq      = rng.poisson(5, n_samples).clip(1, 30)
    days_since_last = rng.exponential(30, n_samples).clip(1, 365)
    avg_basket      = rng.normal(60, 20, n_samples).clip(10, 300)

    treatment = rng.binomial(1, 0.5, n_samples)

    # 購入確率のベース（顧客特性に依存）
    base_prob = (
        0.2
        + 0.003 * (visit_freq - 5)
        + 0.0002 * (past_spend - 200)
        - 0.002 * (days_since_last - 30)
        + 0.001 * (avg_basket - 60)
    ).clip(0.05, 0.95)

    # アップリフト効果（セグメント別に設定してデモ映えを重視）
    # Persuadables: visit_freq中程度 & past_spend中程度
    is_persuadable = (visit_freq >= 3) & (visit_freq <= 8) & (past_spend >= 50) & (past_spend <= 400)
    # Sleeping Dogs: 高頻度来店 & 高購買額（優良顧客はクーポンで鬱陶しがる）
    is_sleeping_dog = (visit_freq > 12) & (past_spend > 600)

    uplift_effect = np.where(
        is_persuadable,  0.35,   # 説得可能層：クーポンで購入率+35%
        np.where(
            is_sleeping_dog, -0.20,  # あまのじゃく層：クーポンで購入率-20%
            np.where(
                base_prob > 0.65, 0.02,  # 鉄板層：ほぼ効果なし
                0.05,                    # 無関心層：効果小
            )
        )
    )

    buy_prob = np.where(
        treatment == 1,
        (base_prob + uplift_effect).clip(0.01, 0.99),
        base_prob,
    )
    conversion = rng.binomial(1, buy_prob, n_samples)

    df = pd.DataFrame({
        "customer_id":    np.arange(1, n_samples + 1),
        "age":            age.astype(int),
        "past_spend":     past_spend.round(0).astype(int),
        "visit_freq":     visit_freq.astype(int),
        "days_since_last":days_since_last.round(0).astype(int),
        "avg_basket":     avg_basket.round(0).astype(int),
        "treatment":      treatment.astype(int),
        "conversion":     conversion.astype(int),
    })
    return df


def load_and_validate_csv(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"CSVの読み込みに失敗しました: {e}"

    required = ["customer_id", "treatment", "conversion"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        return None, f"必須カラムが不足しています: {missing}"

    feat_cols = [c for c in df.columns if c not in ["customer_id", "treatment", "conversion"]]
    if len(feat_cols) == 0:
        return None, "特徴量カラムが1つも見つかりません。"

    for col in ["treatment", "conversion"]:
        if not df[col].isin([0, 1]).all():
            return None, f"{col} カラムは0または1のみ使用可能です。"

    return df, ""
