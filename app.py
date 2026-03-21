"""Streamlit app: Promo ROI Predictor (uplift modeling demo)."""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import streamlit as st

from constants import (
    PORTFOLIO_URL,
    SEGMENT_LABELS_JA,
)
from data_generator import FEATURE_COLS, generate_demo_data, load_and_validate_csv
from lang import LANG
from model import TLearnerUplift, classify_segments, calculate_roi
from visuals import (
    build_quadrant_bubble_figure,
    build_roi_comparison_chart,
    render_segment_character_section,
    render_segment_summary
)

st.set_page_config(page_title="Promo ROI Predictor", layout="wide")

@dataclass(frozen=True)
class SidebarState:
    lang: str
    T: dict
    df_input: pd.DataFrame | None
    feat_cols: list[str]
    uplift_thresh: float
    buy_thresh: float
    coupon_cost: float
    aov: float
    margin: float
    run: bool

def init_session_state() -> None:
    if "result" not in st.session_state:
        st.session_state.result = None

def render_header(T: dict) -> None:
    head_l, head_r = st.columns([0.78, 0.22], vertical_alignment="center")
    with head_l:
        st.title(T["title"])
        st.caption(T["caption"])
    with head_r:
        st.link_button(T["portfolio_label"], PORTFOLIO_URL, use_container_width=True)

def render_sidebar() -> SidebarState:
    with st.sidebar:
        lang_choice = st.radio("🌐 Language / 言語", ["日本語", "English"], horizontal=True)
        lang = "ja" if lang_choice == "日本語" else "en"
        T = LANG[lang]
        st.link_button(T["portfolio_btn"], PORTFOLIO_URL, use_container_width=True)
        st.divider()

        st.header(T["data_source"])
        data_mode = st.radio(T["data_mode_radio"], [T["use_demo"], T["upload_csv"]], label_visibility="collapsed")

        df_input: pd.DataFrame | None = None
        feat_cols = list(FEATURE_COLS)

        if data_mode == T["use_demo"]:
            n_samples = st.slider(T["n_samples"], 1000, 20000, 10000, step=1000)
            seed = st.number_input(T["seed"], value=42, step=1)
            df_input = generate_demo_data(n_samples=n_samples, random_state=int(seed))
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
        buy_thresh = st.slider(T["buy_thresh"], 0.0, 1.0, 0.5, step=0.05)

        st.divider()
        st.header(T["roi_params"])
        coupon_cost = st.number_input(T["coupon_cost"], min_value=0.1, value=5.0, step=0.5)
        aov = st.number_input(T["aov"], min_value=1.0, value=60.0, step=5.0)
        margin = st.slider(T["margin"], 5, 80, 30) / 100

        st.divider()
        run = st.button(T["run_btn"], type="primary", use_container_width=True, disabled=(df_input is None))

    return SidebarState(lang, T, df_input, feat_cols, uplift_thresh, buy_thresh, coupon_cost, aov, margin, run)

def run_model_pipeline(sb: SidebarState) -> None:
    if not sb.run or sb.df_input is None:
        return
    with st.spinner(sb.T["running"]):
        model = TLearnerUplift()
        model.fit(sb.df_input, sb.feat_cols)
        scores = model.predict(sb.df_input)
        scored_segments = classify_segments(scores, sb.uplift_thresh, sb.buy_thresh)
        roi = calculate_roi(scored_segments, sb.coupon_cost, sb.aov, sb.margin)
        st.session_state.result = {
            "scored": scored_segments, "roi": roi, "df_input": sb.df_input,
            "uplift_thresh": sb.uplift_thresh, "buy_thresh": sb.buy_thresh, "feat_cols": sb.feat_cols
        }

def render_roi_summary(roi: dict, lang: str, T: dict) -> None:
    st.header(T["roi_title"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(T["all_customers"])
        st.metric(T["coupon_spend"], f"${roi['baseline_coupon_cost']:,.0f}")
        st.metric(T["uplift_rev"], f"${roi['baseline_uplift_revenue']:,.0f}")
        st.metric(T["net"], f"${roi['baseline_net']:,.0f}")
    with c2:
        st.subheader(T["targeted"])
        st.metric(T["coupon_spend"], f"${roi['targeted_coupon_cost']:,.0f}", delta=f"-${roi['coupon_saving']:,.0f}", delta_color="inverse")
        st.metric(T["uplift_rev"], f"${roi['targeted_uplift_revenue']:,.0f}")
        st.metric(T["net"], f"${roi['targeted_net']:,.0f}")
    
    st.success(f"{T['saving'] if lang=='ja' else 'Coupon saving'}: **${roi['coupon_saving']:,.0f} ({roi['coupon_saving_pct']}%)**")
    st.plotly_chart(build_roi_comparison_chart(roi, T), use_container_width=True)

def main() -> None:
    init_session_state()
    sb = render_sidebar()
    render_header(sb.T)
    run_model_pipeline(sb)

    res = st.session_state.result
    if res is None:
        st.info(sb.T["no_data"])
        st.stop()

    st.header(res.get("T", sb.T)["seg_result"])
    st.plotly_chart(build_quadrant_bubble_figure(res["scored"], res["uplift_thresh"], res["buy_thresh"], sb.lang), use_container_width=True)
    
    render_roi_summary(res["roi"], sb.lang, sb.T)
    render_segment_character_section(res, res["scored"], sb.lang, sb.T)
    render_segment_summary(res["scored"], sb.lang, sb.T)

    st.header(sb.T["download"])
    csv = res["scored"].to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=sb.T["download"], data=csv, file_name="customer_uplift_scores.csv", mime="text/csv", use_container_width=True)

if __name__ == "__main__":
    main()