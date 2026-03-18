import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from data_generator import generate_demo_data, load_and_validate_csv, FEATURE_COLS
from model import TLearnerUplift, classify_segments, calculate_roi

st.set_page_config(page_title="Promo ROI Predictor", layout="wide")

PORTFOLIO_URL = "https://mona2083.github.io/portfolio-2026/index.html"

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
SEG_ORDER = ["Sure Things", "Persuadables", "Lost Causes", "Sleeping Dogs"]

FEAT_LABELS = {
    "age":             "年齢 / Age",
    "past_spend":      "過去購買額 / Past Spend",
    "visit_freq":      "来店頻度 / Visit Freq",
    "days_since_last": "最終来店からの日数 / Recency",
    "avg_basket":      "平均購買額 / Avg Basket",
}

LANG = {
    "ja": {
        "title":         "🎯 Promo ROI Predictor",
        "caption":       "Uplift Modeling で販促予算を最適化する",
        "portfolio_btn": "🔗 ポートフォリオを見る",
        "portfolio_label":"ポートフォリオ",
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
        "seg_profile":   "🔍 セグメント別キャラクター分析",
        "avg_label":     "平均値",
        "feat_profile":  "各セグメントの顧客特徴",
        "feat_profile_cap": "※ 数値は各特徴量を0〜1に正規化した相対値です。1.0に近いほどそのセグメントでその特徴が強く現れています。",
        "feat_dist_title": "📊 特徴量分布比較",
        "feat_imp_title":  "🔑 各セグメントを判別する特徴量の重要度",
        "feat_imp_cap":    "※ One-vs-Restのランダムフォレストによる重要度。値が大きいほどそのセグメントを他と区別するのにその特徴量が強く効いています。",
        "persona_title": "👤 セグメントペルソナ",
        "csv_desc":  "必要なカラム：",
        "csv_col1":  "• customer_id：顧客ID（整数）",
        "csv_col2":  "• treatment：クーポン配布（1=あり / 0=なし）",
        "csv_col3":  "• conversion：購入（1=あり / 0=なし）",
        "csv_col4":  "• 特徴量：年齢・購買額等（数値）",
        "csv_example": "例）customer_id,treatment,conversion,age,past_spend\n1,1,1,35,250\n2,0,0,52,80",
    },
    "en": {
        "title":         "🎯 Promo ROI Predictor",
        "caption":       "Optimize your promotional budget with Uplift Modeling",
        "portfolio_btn": "🔗 View Portfolio",
        "portfolio_label":"Portfolio",
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
        "seg_profile":   "🔍 Segment Character Analysis",
        "avg_label":     "Average",
        "feat_profile":  "Customer Profile by Segment",
        "feat_profile_cap": "Values are normalized (0–1) per feature. 1.0 means that segment scores highest on that feature across all segments.",
        "feat_dist_title": "📊 Feature Distribution by Segment",
        "feat_imp_title":  "🔑 Feature Importance for Segment Classification",
        "feat_imp_cap":    "One-vs-Rest Random Forest importance. Higher = that feature more strongly distinguishes the segment from others.",
        "persona_title": "👤 Segment Personas",
        "csv_desc":  "Required columns:",
        "csv_col1":  "• customer_id: customer ID (integer)",
        "csv_col2":  "• treatment: coupon sent (1=yes / 0=no)",
        "csv_col3":  "• conversion: purchased (1=yes / 0=no)",
        "csv_col4":  "• features: age, spend, etc. (numeric)",
        "csv_example": "e.g. customer_id,treatment,conversion,age,past_spend\n1,1,1,35,250\n2,0,0,52,80",
    },
}

with st.sidebar:
    lang_choice = st.radio("🌐 Language / 言語", ["日本語", "English"], horizontal=True)
    lang = "ja" if lang_choice == "日本語" else "en"

T = LANG[lang]

with st.sidebar:
    st.link_button(T["portfolio_btn"], PORTFOLIO_URL, use_container_width=True)
    st.divider()

if "result" not in st.session_state:
    st.session_state.result = None

head_l, head_r = st.columns([0.78, 0.22], vertical_alignment="center")
with head_l:
    st.title(T["title"])
    st.caption(T["caption"])
with head_r:
    st.link_button(T["portfolio_label"], PORTFOLIO_URL, use_container_width=True)

# ── サイドバー設定 ────────────────────────────────────────────────
with st.sidebar:
    st.header(T["data_source"])
    data_mode = st.radio("", [T["use_demo"], T["upload_csv"]], label_visibility="collapsed")

    df_input  = None
    feat_cols = FEATURE_COLS

    if data_mode == T["use_demo"]:
        n_samples = st.slider(T["n_samples"], 1000, 20000, 10000, step=1000)
        seed      = st.number_input(T["seed"], value=42, step=1)
        df_input  = generate_demo_data(n_samples=n_samples, random_state=int(seed))
        feat_cols = FEATURE_COLS
    else:
        uploaded = st.file_uploader(T["upload_label"], type=["csv"])
        st.caption(T["csv_desc"])
        st.caption(T["csv_col1"])
        st.caption(T["csv_col2"])
        st.caption(T["csv_col3"])
        st.caption(T["csv_col4"])
        with st.expander("📋 " + ("フォーマット例" if lang == "ja" else "Format example")):
            st.code(T["csv_example"], language="csv")
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
    coupon_cost = st.number_input(T["coupon_cost"], min_value=0.1, value=5.0,  step=0.5)
    aov         = st.number_input(T["aov"],         min_value=1.0, value=60.0, step=5.0)
    margin      = st.slider(T["margin"], 5, 80, 30) / 100

    st.divider()
    run = st.button(T["run_btn"], type="primary", use_container_width=True, disabled=(df_input is None))

# ── モデル実行 ────────────────────────────────────────────────────
if run and df_input is not None:
    with st.spinner(T["running"]):
        model  = TLearnerUplift()
        model.fit(df_input, feat_cols)
        scores = model.predict(df_input)
        scored = classify_segments(scores, uplift_thresh, buy_thresh)
        roi    = calculate_roi(scored, coupon_cost, aov, margin)
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

seg_name = lambda s: f"{SEGMENT_LABELS_JA[s]}（{s}）" if lang == "ja" else s

# ── 4象限バブルチャート ──────────────────────────────────────────
st.header(T["seg_result"])

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
fig.add_hline(y=buy_thresh, line_dash="dash", line_color="gray",
              annotation_text=f"p={buy_thresh}", annotation_position="bottom right")

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
    m1.metric(T["coupon_spend"], f"${roi['baseline_coupon_cost']:,.0f}")
    m2.metric(T["uplift_rev"],   f"${roi['baseline_uplift_revenue']:,.0f}")
    m3.metric(T["net"],          f"${roi['baseline_net']:,.0f}",
              delta_color=_metric_color(roi["baseline_net"]))

with col2:
    st.subheader(T["targeted"])
    m4, m5, m6 = st.columns(3)
    m4.metric(T["coupon_spend"], f"${roi['targeted_coupon_cost']:,.0f}",
              delta=f"-${roi['coupon_saving']:,.0f}", delta_color="inverse")
    m5.metric(T["uplift_rev"],   f"${roi['targeted_uplift_revenue']:,.0f}")
    m6.metric(T["net"],          f"${roi['targeted_net']:,.0f}",
              delta_color=_metric_color(roi["targeted_net"]))

st.success(
    f"{'説得可能層のみをターゲットにすることで' if lang == 'ja' else 'By targeting Persuadables only,'} "
    f"**${roi['coupon_saving']:,.0f} ({roi['coupon_saving_pct']}%)** "
    f"{'のクーポン費用を節約できます。' if lang == 'ja' else 'in coupon budget can be saved.'}"
)

fig2 = go.Figure()
cats     = [T["all_customers"], T["targeted"]]
costs    = [roi["baseline_coupon_cost"],    roi["targeted_coupon_cost"]]
revenues = [roi["baseline_uplift_revenue"], roi["targeted_uplift_revenue"]]
nets     = [roi["baseline_net"],            roi["targeted_net"]]
fig2.add_bar(name=T["coupon_spend"], x=cats, y=costs,    marker_color="#c0392b", opacity=0.8)
fig2.add_bar(name=T["uplift_rev"],   x=cats, y=revenues, marker_color="#2d6a4f", opacity=0.8)
fig2.add_bar(name=T["net"],          x=cats, y=nets,     marker_color="#1a4a7a", opacity=0.8)
fig2.update_layout(barmode="group", height=380, title=T["roi_title"])
st.plotly_chart(fig2, use_container_width=True)

# ── セグメント別キャラクター分析 ─────────────────────────────────
st.header(T["seg_profile"])

df_with_seg = result["df_input"].copy().merge(
    scored[["customer_id", "segment", "uplift_score", "p_treatment", "p_control"]],
    on="customer_id", how="left"
)
df_raw = result["df_input"].copy().merge(
    scored[["customer_id", "segment"]], on="customer_id", how="left"
)

feat_cols_used = [c for c in FEATURE_COLS if c in df_with_seg.columns]
feat_display   = [FEAT_LABELS.get(f, f) for f in feat_cols_used]

if feat_cols_used:

    # ── 特徴量分布比較（実数値・特徴量ごと） ────────────────────
    st.subheader(T["feat_profile"])

    raw_means = df_raw.groupby("segment")[feat_cols_used].mean().reset_index()
    with st.container():
        bar_cols = st.columns(len(feat_cols_used))
        for ci, feat in enumerate(feat_cols_used):
            with bar_cols[ci]:
                fig_bar = go.Figure()
                for seg in SEG_ORDER:
                    row_data = raw_means[raw_means["segment"] == seg]
                    if row_data.empty:
                        continue
                    fig_bar.add_bar(
                        x=[seg_name(seg)],
                        y=[row_data[feat].values[0]],
                        name=seg_name(seg),
                        marker_color=SEGMENT_COLORS[seg],
                        showlegend=False,
                    )
                fig_bar.update_layout(
                    title=FEAT_LABELS.get(feat, feat),
                    height=280,
                    margin=dict(t=40, b=20, l=10, r=10),
                    yaxis_title=T["avg_label"],
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # ── 特徴量重要度ヒートマップ ─────────────────────────────────
    st.subheader(T["feat_imp_title"])

    imp_rows = []
    for seg in SEG_ORDER:
        if seg not in df_raw["segment"].values:
            continue
        clf_ovr = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_ovr.fit(
            df_raw[feat_cols_used],
            (df_raw["segment"] == seg).astype(int),
        )
        row = {"segment": seg_name(seg)}
        for f, v in zip(feat_cols_used, clf_ovr.feature_importances_):
            row[FEAT_LABELS.get(f, f)] = round(v, 3)
        imp_rows.append(row)

    imp_df    = pd.DataFrame(imp_rows).set_index("segment")
    z_imp     = imp_df.values
    x_imp     = list(imp_df.columns)
    y_imp     = list(imp_df.index)

    fig_imp_heat = go.Figure(go.Heatmap(
        z=z_imp,
        x=x_imp,
        y=y_imp,
        colorscale="RdYlGn_r",
        zmin=0, zmax=z_imp.max(),
        text=[[f"{v:.3f}" for v in row] for row in z_imp],
        texttemplate="%{text}",
        textfont=dict(size=12, color="black"),
        colorbar=dict(title="Importance", thickness=14),
    ))
    fig_imp_heat.update_layout(
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_imp_heat, use_container_width=True)
    st.caption(T["feat_imp_cap"])

    # ── セグメントペルソナカード ──────────────────────────────────
    st.subheader(T["persona_title"])

    personas = {
        "Sure Things": {
            "en": {"icon": "💎", "tagline": "Your loyal regulars",             "traits": ["High visit frequency", "High past spend", "Short days since last visit", "Large average basket"]},
            "ja": {"icon": "💎", "tagline": "あなたの常連客",                  "traits": ["来店頻度：高", "過去購買額：高", "最終来店日：直近", "平均購買額：大"]},
        },
        "Persuadables": {
            "en": {"icon": "🎯", "tagline": "Price-sensitive occasional shoppers", "traits": ["Moderate visit frequency", "Medium past spend", "Longer days since last visit", "Responds to price incentives"]},
            "ja": {"icon": "🎯", "tagline": "価格に敏感な不定期客",            "traits": ["来店頻度：中程度", "過去購買額：中程度", "最終来店日：やや遠い", "価格インセンティブに反応"]},
        },
        "Lost Causes": {
            "en": {"icon": "🚪", "tagline": "Window shoppers, not buyers",     "traits": ["Low visit frequency", "Low past spend", "Very long since last visit", "Not price-motivated"]},
            "ja": {"icon": "🚪", "tagline": "見るだけ客",                      "traits": ["来店頻度：低", "過去購買額：低", "最終来店日：遠い", "価格で動かない"]},
        },
        "Sleeping Dogs": {
            "en": {"icon": "⚠️", "tagline": "VIPs who hate being sold to",    "traits": ["Very high visit frequency", "Very high past spend", "Recent visits", "Resent promotional noise"]},
            "ja": {"icon": "⚠️", "tagline": "売り込みを嫌うVIP客",            "traits": ["来店頻度：非常に高", "過去購買額：非常に高", "最終来店日：直近", "販促メッセージを鬱陶しがる"]},
        },
    }

    persona_cols = st.columns(4)
    for col, seg in zip(persona_cols, SEG_ORDER):
        with col:
            p         = personas[seg][lang]
            seg_count = len(scored[scored["segment"] == seg])
            pct       = seg_count / len(scored) * 100
            st.markdown(
                f"""
                <div style="background:{SEGMENT_COLORS[seg]};border-radius:10px;padding:16px;height:100%;">
                <div style="font-size:1.6rem;margin-bottom:6px;">{p['icon']}</div>
                <div style="font-family:serif;font-size:.95rem;color:#ffffff;
                font-weight:600;margin-bottom:4px;">{seg_name(seg)}</div>
                <div style="font-size:.75rem;color:#ffffffcc;margin-bottom:10px;font-style:italic;">
                {p['tagline']}</div>
                <div style="font-size:.82rem;font-weight:700;color:#ffffff;
                margin-bottom:8px;">{seg_count:,} {'人' if lang == 'ja' else 'customers'} ({pct:.1f}%)</div>
                {''.join(f'<div style="font-size:.75rem;color:#ffffffdd;margin-bottom:3px;">• {t}</div>' for t in p['traits'])}
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── セグメント別集計テーブル ─────────────────────────────────────
st.header(T["seg_table"])

rows  = []
total = len(scored)
for seg in SEG_ORDER:
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
            return [f"color:{v};font-weight:600;font-size:1rem"] * len(row)
    return [""] * len(row)

st.dataframe(
    seg_df.style.apply(_highlight_seg, axis=1),
    use_container_width=True,
    hide_index=True,
)

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