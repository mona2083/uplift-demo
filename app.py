import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_generator import generate_demo_data, load_and_validate_csv, FEATURE_COLS
from model import TLearnerUplift, classify_segments, calculate_roi

st.set_page_config(page_title="Promo ROI Predictor", layout="wide")

SEGMENT_COLORS = {
    "Sure Things":   "#2d6a4f",
    "Persuadables":  "#1a4a7a",
    "Lost Causes":   "#888888",
    "Sleeping Dogs": "#c0392b",
}
SEGMENT_LABELS_JA = {
    "Sure Things":   "鉄板層",
    "Persuadables":  "説得可能層",
    "Lost Causes":   "無関心層",
    "Sleeping Dogs": "あまのじゃく層",
}
SEGMENT_DESC = {
    "en": {
        "Sure Things":   "Buy anyway — wasting coupons here",
        "Persuadables":  "Buy only with coupon — target here",
        "Lost Causes":   "Won't buy regardless",
        "Sleeping Dogs": "Coupon makes them leave",
    },
    "ja": {
        "Sure Things":   "クーポンなしでも買う → 配ると利益の無駄",
        "Persuadables":  "クーポンがあれば買う → ここだけに予算投下",
        "Lost Causes":   "何をしても買わない",
        "Sleeping Dogs": "クーポンを送ると鬱陶しがって離脱",
    },
}

LANG = {
    "ja": {
        "title":         "🎯 Promo ROI Predictor",
        "caption":       "Uplift Modeling で販促予算を最適化する",
        "data_source":   "データソース",
        "use_demo":      "デモデータを使う",
        "upload_csv":    "実データCSVをアップロード",
        "upload_label":  "CSVファイルをアップロード",
        "n_samples":     "生成件数",
        "seed":          "ランダムシード",
        "feat_select":   "使用する特徴量を選択",
        "threshold":     "閾値設定",
        "uplift_thresh": "アップリフト閾値（正：効果あり / 負：逆効果）",
        "buy_thresh":    "購入確率閾値（この値以上 = 購入傾向あり）",
        "run_btn":       "🚀 モデルを実行",
        "running":       "モデル学習・予測中...",
        "roi_params":    "ROIパラメータ",
        "coupon_cost":   "クーポン配布コスト（1枚あたり $）",
        "aov":           "平均注文額（$）",
        "margin":        "粗利率（%）",
        "seg_result":    "📊 4象限セグメント分類",
        "bubble_title":  "アップリフトスコア × 購入確率（コントロール群）",
        "roi_title":     "💰 ROI比較",
        "all_customers": "全員にクーポン配布",
        "targeted":      "説得可能層のみに配布",
        "coupon_spend":  "クーポン費用",
        "uplift_rev":    "増分売上（粗利）",
        "net":           "純利益",
        "saving":        "クーポン費用の節約",
        "seg_table":     "📋 セグメント別集計",
        "seg_col":       "セグメント",
        "n_col":         "人数",
        "pct_col":       "割合",
        "avg_uplift":    "平均アップリフト",
        "avg_p_treat":   "平均購入率(クーポンあり)",
        "avg_p_ctrl":    "平均購入率(クーポンなし)",
        "download":      "📥 予測スコアをダウンロード",
        "csv_label":     "customer_scores.csv",
        "no_data":       "データを選択して「モデルを実行」を押してください",
    },
    "en": {
        "title":         "🎯 Promo ROI Predictor",
        "caption":       "Optimize your promotional budget with Uplift Modeling",
        "data_source":   "Data Source",
        "use_demo":      "Use demo data",
        "upload_csv":    "Upload real data CSV",
        "upload_label":  "Upload CSV file",
        "n_samples":     "Number of samples",
        "seed":          "Random seed",
        "feat_select":   "Select features to use",
        "threshold":     "Threshold Settings",
        "uplift_thresh": "Uplift threshold (positive = effect / negative = backfire)",
        "buy_thresh":    "Purchase probability threshold (above = likely buyer)",
        "run_btn":       "🚀 Run Model",
        "running":       "Training model and predicting...",
        "roi_params":    "ROI Parameters",
        "coupon_cost":   "Coupon cost per customer ($)",
        "aov":           "Average order value ($)",
        "margin":        "Gross margin (%)",
        "seg_result":    "📊 4-Quadrant Segment Classification",
        "bubble_title":  "Uplift Score × Purchase Probability (Control)",
        "roi_title":     "💰 ROI Comparison",
        "all_customers": "Send to all customers",
        "targeted":      "Send to Persuadables only",
        "coupon_spend":  "Coupon cost",
        "uplift_rev":    "Incremental revenue (gross profit)",
        "net":           "Net profit",
        "saving":        "Coupon budget saved",
        "seg_table":     "📋 Segment Summary",
        "seg_col":       "Segment",
        "n_col":         "Count",
        "pct_col":       "Share",
        "avg_uplift":    "Avg Uplift",
        "avg_p_treat":   "Avg P(buy|coupon)",
        "avg_p_ctrl":    "Avg P(buy|no coupon)",
        "download":      "📥 Download Prediction Scores",
        "csv_label":     "customer_scores.csv",
        "no_data":       "Select data and click 'Run Model'",
    },
}

with st.sidebar:
    lang_choice = st.radio("🌐 Language / 言語", ["日本語", "English"], horizontal=True)
    lang = "ja" if lang_choice == "日本語" else "en"

T = LANG[lang]

if "result" not in st.session_state:
    st.session_state.result = None

st.title(T["title"])
st.caption(T["caption"])

# ── サイドバー設定 ────────────────────────────────────────────────
with st.sidebar:
    st.header(T["data_source"])
    data_mode = st.radio("", [T["use_demo"], T["upload_csv"]], label_visibility="collapsed")

    df_input     = None
    feat_cols    = FEATURE_COLS

    if data_mode == T["use_demo"]:
        n_samples = st.slider(T["n_samples"], 1000, 20000, 10000, step=1000)
        seed      = st.number_input(T["seed"], value=42, step=1)
        df_input  = generate_demo_data(n_samples=n_samples, random_state=int(seed))
        feat_cols = FEATURE_COLS
    else:
        uploaded = st.file_uploader(T["upload_label"], type=["csv"])
        if uploaded:
            df_input, err = load_and_validate_csv(uploaded)
            if err:
                st.error(err)
                df_input = None
            else:
                feat_cols = [c for c in df_input.columns if c not in ["customer_id", "treatment", "conversion"]]
                feat_cols = st.multiselect(T["feat_select"], feat_cols, default=feat_cols)

    st.divider()
    st.header(T["threshold"])
    uplift_thresh = st.slider(T["uplift_thresh"], -0.3, 0.3, 0.0, step=0.01)
    buy_thresh    = st.slider(T["buy_thresh"],     0.0,  1.0, 0.5, step=0.05)

    st.divider()
    st.header(T["roi_params"])
    coupon_cost = st.number_input(T["coupon_cost"], min_value=0.1, value=5.0, step=0.5)
    aov         = st.number_input(T["aov"],         min_value=1.0, value=60.0, step=5.0)
    margin      = st.slider(T["margin"], 5, 80, 30) / 100

    st.divider()
    run = st.button(T["run_btn"], type="primary", use_container_width=True, disabled=(df_input is None))

# ── モデル実行 ────────────────────────────────────────────────────
if run and df_input is not None:
    with st.spinner(T["running"]):
        model = TLearnerUplift()
        model.fit(df_input, feat_cols)
        scores  = model.predict(df_input)
        scored  = classify_segments(scores, uplift_thresh, buy_thresh)
        roi     = calculate_roi(scored, coupon_cost, aov, margin)
        st.session_state.result = {
            "scored": scored, "roi": roi, "df_input": df_input,
            "uplift_thresh": uplift_thresh, "buy_thresh": buy_thresh,
        }

result = st.session_state.result

if result is None:
    st.info(T["no_data"])
    st.stop()

scored        = result["scored"]
roi           = result["roi"]
uplift_thresh = result["uplift_thresh"]
buy_thresh    = result["buy_thresh"]

# ── 4象限バブルチャート ──────────────────────────────────────────
st.header(T["seg_result"])

seg_name = lambda s: f"{SEGMENT_LABELS_JA[s]}（{s}）" if lang == "ja" else s

fig = go.Figure()
for seg, color in SEGMENT_COLORS.items():
    sub = scored[scored["segment"] == seg]
    if len(sub) == 0:
        continue
    sample = sub.sample(min(500, len(sub)), random_state=42)
    fig.add_trace(go.Scatter(
        x=sample["uplift_score"],
        y=sample["p_control"],
        mode="markers",
        name=seg_name(seg),
        marker=dict(color=color, size=6, opacity=0.6),
        hovertemplate=(
            f"<b>{seg_name(seg)}</b><br>"
            "Uplift: %{x:.3f}<br>"
            "P(buy|no coupon): %{y:.3f}<extra></extra>"
        ),
    ))

fig.add_vline(x=uplift_thresh, line_dash="dash", line_color="gray",
              annotation_text=f"uplift={uplift_thresh}", annotation_position="top left")
fig.add_hline(y=buy_thresh,    line_dash="dash", line_color="gray",
              annotation_text=f"p={buy_thresh}",     annotation_position="bottom right")

desc = SEGMENT_DESC[lang]
for seg, (ax, ay) in {
    "Sure Things":   (uplift_thresh + 0.05, buy_thresh + 0.05),
    "Persuadables":  (uplift_thresh + 0.05, buy_thresh - 0.12),
    "Lost Causes":   (uplift_thresh - 0.15, buy_thresh - 0.12),
    "Sleeping Dogs": (uplift_thresh - 0.15, buy_thresh + 0.05),
}.items():
    fig.add_annotation(
        x=ax, y=ay,
        text=f"<b>{seg_name(seg)}</b><br><sub>{desc[seg]}</sub>",
        showarrow=False, font=dict(size=10, color=SEGMENT_COLORS[seg]),
        align="center",
    )

fig.update_layout(
    title=T["bubble_title"],
    xaxis_title="Uplift Score (P(buy|coupon) - P(buy|no coupon))",
    yaxis_title="P(buy | no coupon)",
    height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ── ROI比較 ──────────────────────────────────────────────────────
st.header(T["roi_title"])

col1, col2 = st.columns(2)

def _metric_color(val: float) -> str:
    return "normal" if val >= 0 else "inverse"

with col1:
    st.subheader(T["all_customers"])
    m1, m2, m3 = st.columns(3)
    m1.metric(T["coupon_spend"],  f"${roi['baseline_coupon_cost']:,.0f}")
    m2.metric(T["uplift_rev"],    f"${roi['baseline_uplift_revenue']:,.0f}")
    m3.metric(T["net"],           f"${roi['baseline_net']:,.0f}",
              delta_color=_metric_color(roi["baseline_net"]))

with col2:
    st.subheader(T["targeted"])
    m4, m5, m6 = st.columns(3)
    m4.metric(T["coupon_spend"],  f"${roi['targeted_coupon_cost']:,.0f}",
              delta=f"-${roi['coupon_saving']:,.0f}", delta_color="inverse")
    m5.metric(T["uplift_rev"],    f"${roi['targeted_uplift_revenue']:,.0f}")
    m6.metric(T["net"],           f"${roi['targeted_net']:,.0f}",
              delta_color=_metric_color(roi["targeted_net"]))

st.success(
    f"{'説得可能層のみをターゲットにすることで' if lang == 'ja' else 'By targeting Persuadables only,'} "
    f"**${roi['coupon_saving']:,.0f} ({roi['coupon_saving_pct']}%)** "
    f"{'のクーポン費用を節約できます。' if lang == 'ja' else 'in coupon budget can be saved.'}"
)

# ROI比較棒グラフ
fig2 = go.Figure()
cats     = [T["all_customers"], T["targeted"]]
costs    = [roi["baseline_coupon_cost"],    roi["targeted_coupon_cost"]]
revenues = [roi["baseline_uplift_revenue"], roi["targeted_uplift_revenue"]]
nets     = [roi["baseline_net"],            roi["targeted_net"]]

fig2.add_bar(name=T["coupon_spend"],  x=cats, y=costs,    marker_color="#c0392b", opacity=0.8)
fig2.add_bar(name=T["uplift_rev"],    x=cats, y=revenues, marker_color="#2d6a4f", opacity=0.8)
fig2.add_bar(name=T["net"],           x=cats, y=nets,     marker_color="#1a4a7a", opacity=0.8)
fig2.update_layout(barmode="group", height=380, title=T["roi_title"])
st.plotly_chart(fig2, use_container_width=True)

# ── セグメント別集計テーブル ─────────────────────────────────────
st.header(T["seg_table"])

rows = []
total = len(scored)
for seg in ["Sure Things", "Persuadables", "Lost Causes", "Sleeping Dogs"]:
    sub = scored[scored["segment"] == seg]
    rows.append({
        T["seg_col"]:     seg_name(seg),
        T["n_col"]:       len(sub),
        T["pct_col"]:     f"{len(sub)/total*100:.1f}%",
        T["avg_uplift"]:  f"{sub['uplift_score'].mean():.3f}" if len(sub) else "-",
        T["avg_p_treat"]: f"{sub['p_treatment'].mean():.3f}"  if len(sub) else "-",
        T["avg_p_ctrl"]:  f"{sub['p_control'].mean():.3f}"    if len(sub) else "-",
    })

seg_df = pd.DataFrame(rows)

def _highlight_seg(row):
    seg = row[T["seg_col"]]
    for k, v in {seg_name(s): SEGMENT_COLORS[s] for s in SEGMENT_COLORS}.items():
        if k == seg:
            return [f"color:{v};font-weight:500"] * len(row)
    return [""] * len(row)

st.dataframe(
    seg_df.style.apply(_highlight_seg, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ── セグメント分布パイチャート ────────────────────────────────────
fig3 = px.pie(
    seg_df,
    names=T["seg_col"],
    values=T["n_col"],
    color=T["seg_col"],
    color_discrete_map={seg_name(s): SEGMENT_COLORS[s] for s in SEGMENT_COLORS},
    title=T["seg_table"],
)
fig3.update_traces(textposition="inside", textinfo="percent+label")
fig3.update_layout(height=380)
st.plotly_chart(fig3, use_container_width=True)

# ── ダウンロード ─────────────────────────────────────────────────
st.header(T["download"])

download_df = scored.copy()
download_df["segment_ja"] = download_df["segment"].map(SEGMENT_LABELS_JA)
csv = download_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label=T["download"],
    data=csv,
    file_name=T["csv_label"],
    mime="text/csv",
    use_container_width=True,
)
