"""Streamlit app: Promo ROI Predictor (uplift modeling demo)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from constants import (
    FEAT_LABELS,
    PERSONAS,
    PORTFOLIO_URL,
    SEGMENT_COLORS,
    SEGMENT_DESC,
    SEGMENT_LABELS_JA,
    SEG_ORDER,
)
from data_generator import FEATURE_COLS, generate_demo_data, load_and_validate_csv
from lang import LANG
from model import TLearnerUplift, calculate_roi, classify_segments

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
        data_mode = st.radio(
            T["data_mode_radio"],
            [T["use_demo"], T["upload_csv"]],
            label_visibility="collapsed",
        )

        df_input: pd.DataFrame | None = None
        feat_cols = list(FEATURE_COLS)

        if data_mode == T["use_demo"]:
            n_samples = st.slider(T["n_samples"], 1000, 20000, 10000, step=1000)
            seed = st.number_input(T["seed"], value=42, step=1)
            df_input = generate_demo_data(n_samples=n_samples, random_state=int(seed))
            feat_cols = list(FEATURE_COLS)
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
                    feat_cols = [
                        c for c in df_input.columns if c not in ["customer_id", "treatment", "conversion"]
                    ]
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

    return SidebarState(
        lang=lang,
        T=T,
        df_input=df_input,
        feat_cols=feat_cols,
        uplift_thresh=uplift_thresh,
        buy_thresh=buy_thresh,
        coupon_cost=coupon_cost,
        aov=aov,
        margin=margin,
        run=run,
    )


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
            "scored": scored_segments,
            "roi": roi,
            "df_input": sb.df_input,
            "uplift_thresh": sb.uplift_thresh,
            "buy_thresh": sb.buy_thresh,
        }


def segment_display_name(seg: str, lang: str) -> str:
    if lang == "ja":
        return f"{SEGMENT_LABELS_JA[seg]}（{seg}）"
    return seg


def build_quadrant_bubble_figure(scored, uplift_thresh: float, buy_thresh: float, lang: str, T: dict) -> go.Figure:
    fig = go.Figure()
    for seg, color in SEGMENT_COLORS.items():
        sub = scored[scored["segment"] == seg]
        if len(sub) == 0:
            continue
        sample = sub.sample(min(500, len(sub)), random_state=42)
        name = segment_display_name(seg, lang)
        fig.add_trace(
            go.Scatter(
                x=sample["uplift_score"],
                y=sample["p_control"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=6, opacity=0.6),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Uplift: %{x:.3f}<br>"
                    "P(buy|no coupon): %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=uplift_thresh,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"uplift={uplift_thresh}",
        annotation_position="top left",
    )
    fig.add_hline(
        y=buy_thresh,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"p={buy_thresh}",
        annotation_position="bottom right",
    )

    desc = SEGMENT_DESC[lang]
    label_pos = {
        "Sure Things": (uplift_thresh + 0.05, buy_thresh + 0.05),
        "Persuadables": (uplift_thresh + 0.05, buy_thresh - 0.12),
        "Lost Causes": (uplift_thresh - 0.15, buy_thresh - 0.12),
        "Sleeping Dogs": (uplift_thresh - 0.15, buy_thresh + 0.05),
    }
    for seg, (ax, ay) in label_pos.items():
        fig.add_annotation(
            x=ax,
            y=ay,
            text=f"<b>{segment_display_name(seg, lang)}</b><br><sub>{desc[seg]}</sub>",
            showarrow=False,
            font=dict(size=10, color=SEGMENT_COLORS[seg]),
            align="center",
        )

    fig.update_layout(
        title=T["bubble_title"],
        xaxis_title="Uplift Score (P(buy|coupon) - P(buy|no coupon))",
        yaxis_title="P(buy | no coupon)",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _metric_color(val: float) -> str:
    return "normal" if val >= 0 else "inverse"


def render_roi_section(roi: dict, lang: str, T: dict) -> None:
    st.header(T["roi_title"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(T["all_customers"])
        m1, m2, m3 = st.columns(3)
        m1.metric(T["coupon_spend"], f"${roi['baseline_coupon_cost']:,.0f}")
        m2.metric(T["uplift_rev"], f"${roi['baseline_uplift_revenue']:,.0f}")
        m3.metric(
            T["net"],
            f"${roi['baseline_net']:,.0f}",
            delta_color=_metric_color(roi["baseline_net"]),
        )

    with col2:
        st.subheader(T["targeted"])
        m4, m5, m6 = st.columns(3)
        m4.metric(
            T["coupon_spend"],
            f"${roi['targeted_coupon_cost']:,.0f}",
            delta=f"-${roi['coupon_saving']:,.0f}",
            delta_color="inverse",
        )
        m5.metric(T["uplift_rev"], f"${roi['targeted_uplift_revenue']:,.0f}")
        m6.metric(
            T["net"],
            f"${roi['targeted_net']:,.0f}",
            delta_color=_metric_color(roi["targeted_net"]),
        )

    st.success(
        f"{'説得可能層のみをターゲットにすることで' if lang == 'ja' else 'By targeting Persuadables only,'} "
        f"**${roi['coupon_saving']:,.0f} ({roi['coupon_saving_pct']}%)** "
        f"{'のクーポン費用を節約できます。' if lang == 'ja' else 'in coupon budget can be saved.'}"
    )

    fig2 = go.Figure()
    cats = [T["all_customers"], T["targeted"]]
    costs = [roi["baseline_coupon_cost"], roi["targeted_coupon_cost"]]
    revenues = [roi["baseline_uplift_revenue"], roi["targeted_uplift_revenue"]]
    nets = [roi["baseline_net"], roi["targeted_net"]]
    fig2.add_bar(name=T["coupon_spend"], x=cats, y=costs, marker_color="#c0392b", opacity=0.8)
    fig2.add_bar(name=T["uplift_rev"], x=cats, y=revenues, marker_color="#2d6a4f", opacity=0.8)
    fig2.add_bar(name=T["net"], x=cats, y=nets, marker_color="#1a4a7a", opacity=0.8)
    fig2.update_layout(barmode="group", height=380, title=T["roi_title"])
    st.plotly_chart(fig2, use_container_width=True)


def render_segment_character_section(result: dict, scored, lang: str, T: dict) -> None:
    st.header(T["seg_profile"])
    df_raw = result["df_input"].copy().merge(scored[["customer_id", "segment"]], on="customer_id", how="left")

    feat_cols_used = [c for c in FEATURE_COLS if c in result["df_input"].columns]

    if not feat_cols_used:
        return

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
                        x=[segment_display_name(seg, lang)],
                        y=[row_data[feat].values[0]],
                        name=segment_display_name(seg, lang),
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

    st.subheader(T["feat_imp_title"])

    imp_rows = []
    for seg in SEG_ORDER:
        if seg not in df_raw["segment"].values:
            continue
        clf_ovr = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_ovr.fit(df_raw[feat_cols_used], (df_raw["segment"] == seg).astype(int))
        row = {"segment": segment_display_name(seg, lang)}
        for f, v in zip(feat_cols_used, clf_ovr.feature_importances_):
            row[FEAT_LABELS.get(f, f)] = round(v, 3)
        imp_rows.append(row)

    imp_df = pd.DataFrame(imp_rows).set_index("segment")
    z_imp = imp_df.values
    x_imp = list(imp_df.columns)
    y_imp = list(imp_df.index)

    fig_imp_heat = go.Figure(
        go.Heatmap(
            z=z_imp,
            x=x_imp,
            y=y_imp,
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=z_imp.max(),
            text=[[f"{v:.3f}" for v in row] for row in z_imp],
            texttemplate="%{text}",
            textfont=dict(size=12, color="black"),
            colorbar=dict(title="Importance", thickness=14),
        )
    )
    fig_imp_heat.update_layout(
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_imp_heat, use_container_width=True)
    st.caption(T["feat_imp_cap"])

    st.subheader(T["persona_title"])
    persona_cols = st.columns(4)
    for col, seg in zip(persona_cols, SEG_ORDER):
        with col:
            p = PERSONAS[seg][lang]
            seg_count = len(scored[scored["segment"] == seg])
            pct = seg_count / len(scored) * 100
            traits_html = "".join(
                f'<div style="font-size:.75rem;color:#ffffffdd;margin-bottom:3px;">• {t}</div>'
                for t in p["traits"]
            )
            cust_word = "人" if lang == "ja" else "customers"
            st.markdown(
                f"""
                <div style="background:{SEGMENT_COLORS[seg]};border-radius:10px;padding:16px;height:100%;">
                <div style="font-size:1.6rem;margin-bottom:6px;">{p['icon']}</div>
                <div style="font-family:serif;font-size:.95rem;color:#ffffff;
                font-weight:600;margin-bottom:4px;">{segment_display_name(seg, lang)}</div>
                <div style="font-size:.75rem;color:#ffffffcc;margin-bottom:10px;font-style:italic;">
                {p['tagline']}</div>
                <div style="font-size:.82rem;font-weight:700;color:#ffffff;
                margin-bottom:8px;">{seg_count:,} {cust_word} ({pct:.1f}%)</div>
                {traits_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def highlight_segment_row(row: pd.Series, seg_col: str, lang: str) -> list[str]:
    seg = row[seg_col]
    name_to_color = {segment_display_name(s, lang): SEGMENT_COLORS[s] for s in SEGMENT_COLORS}
    for name, v in name_to_color.items():
        if name == seg:
            return [f"color:{v};font-weight:600;font-size:1rem"] * len(row)
    return [""] * len(row)


def render_segment_summary(scored, lang: str, T: dict) -> None:
    st.header(T["seg_table"])

    rows = []
    total = len(scored)
    for seg in SEG_ORDER:
        sub = scored[scored["segment"] == seg]
        rows.append(
            {
                T["seg_col"]: segment_display_name(seg, lang),
                T["n_col"]: len(sub),
                T["pct_col"]: f"{len(sub) / total * 100:.1f}%" if total else "0%",
                T["avg_uplift"]: f"{sub['uplift_score'].mean():.3f}" if len(sub) else "-",
                T["avg_p_treat"]: f"{sub['p_treatment'].mean():.3f}" if len(sub) else "-",
                T["avg_p_ctrl"]: f"{sub['p_control'].mean():.3f}" if len(sub) else "-",
            }
        )

    seg_df = pd.DataFrame(rows)
    seg_col = T["seg_col"]

    def _apply_highlight(r):
        return highlight_segment_row(r, seg_col, lang)

    st.dataframe(seg_df.style.apply(_apply_highlight, axis=1), use_container_width=True, hide_index=True)

    fig3 = px.pie(
        seg_df,
        names=seg_col,
        values=T["n_col"],
        color=seg_col,
        color_discrete_map={segment_display_name(s, lang): SEGMENT_COLORS[s] for s in SEGMENT_COLORS},
        title=T["seg_table"],
    )
    fig3.update_traces(textposition="inside", textinfo="percent+label")
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)


def render_download_section(scored, T: dict) -> None:
    st.header(T["download"])
    download_df = scored.copy()
    download_df["segment_ja"] = download_df["segment"].map(SEGMENT_LABELS_JA)
    csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=T["download"],
        data=csv_bytes,
        file_name=T["csv_label"],
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    init_session_state()
    sb = render_sidebar()
    render_header(sb.T)

    run_model_pipeline(sb)

    result = st.session_state.result
    if result is None:
        st.info(sb.T["no_data"])
        st.stop()

    scored = result["scored"]
    roi = result["roi"]
    uplift_thresh = result["uplift_thresh"]
    buy_thresh = result["buy_thresh"]
    lang = sb.lang
    T = sb.T

    st.header(T["seg_result"])
    fig_bubble = build_quadrant_bubble_figure(scored, uplift_thresh, buy_thresh, lang, T)
    st.plotly_chart(fig_bubble, use_container_width=True)

    render_roi_section(roi, lang, T)
    render_segment_character_section(result, scored, lang, T)
    render_segment_summary(scored, lang, T)
    render_download_section(scored, T)


if __name__ == "__main__":
    main()
