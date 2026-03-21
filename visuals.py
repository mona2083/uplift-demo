import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from constants import (
    SEGMENT_COLORS, 
    SEGMENT_DESC, 
    FEAT_LABELS, 
    SEG_ORDER, 
    PERSONAS, 
    SEGMENT_LABELS_JA
)

def build_quadrant_bubble_figure(scored, uplift_thresh, buy_thresh, lang):
    fig = go.Figure()
    for seg, color in SEGMENT_COLORS.items():
        sub = scored[scored["segment"] == seg]
        if len(sub) == 0: continue
        # 表示を軽くするために最大500件をサンプリング
        sample = sub.sample(min(500, len(sub)), random_state=42)
        
        display_name = f"{SEGMENT_LABELS_JA[seg]}（{seg}）" if lang == "ja" else seg
        
        fig.add_trace(go.Scatter(
            x=sample["uplift_score"], y=sample["p_control"],
            mode="markers", name=display_name,
            marker=dict(color=color, size=6, opacity=0.6),
            hovertemplate=f"<b>{display_name}</b><br>Uplift: %{{x:.3f}}<br>Base P: %{{y:.3f}}<extra></extra>"
        ))
    
    fig.add_vline(x=uplift_thresh, line_dash="dash", line_color="gray")
    fig.add_hline(y=buy_thresh, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        xaxis_title="Uplift Score (P(buy|coupon) - P(buy|no-coupon))",
        yaxis_title="Base Purchase Probability (No-coupon)",
        height=500, legend=dict(orientation="h", y=1.1)
    )
    return fig

def build_roi_comparison_chart(roi, T):
    fig = go.Figure()
    cats = [T["all_customers"], T["targeted"]]
    fig.add_bar(name=T["coupon_spend"], x=cats, y=[roi["baseline_coupon_cost"], roi["targeted_coupon_cost"]], marker_color="#c0392b")
    fig.add_bar(name=T["uplift_rev"], x=cats, y=[roi["baseline_uplift_revenue"], roi["targeted_uplift_revenue"]], marker_color="#2d6a4f")
    fig.add_bar(name=T["net"], x=cats, y=[roi["baseline_net"], roi["targeted_net"]], marker_color="#1a4a7a")
    fig.update_layout(barmode="group", height=400)
    return fig

def render_segment_character_section(result, scored, lang, T):
    """セグメント別の特徴分析（ヒートマップ・ペルソナ）を描画"""
    st.header(T["seg_profile"])
    df_raw = result["df_input"].copy().merge(scored[["customer_id", "segment"]], on="customer_id", how="left")
    feat_cols_used = result.get("feat_cols", [])

    if not feat_cols_used:
        return

    st.subheader(T["feat_profile"])
    raw_means = df_raw.groupby("segment")[feat_cols_used].mean().reset_index()
    
    bar_cols = st.columns(len(feat_cols_used))
    for ci, feat in enumerate(feat_cols_used):
        with bar_cols[ci]:
            fig_bar = go.Figure()
            for seg in SEG_ORDER:
                row_data = raw_means[raw_means["segment"] == seg]
                if row_data.empty: continue
                
                display_name = f"{SEGMENT_LABELS_JA[seg]}" if lang == "ja" else seg
                fig_bar.add_bar(
                    x=[display_name], y=[row_data[feat].values[0]],
                    name=display_name, marker_color=SEGMENT_COLORS[seg],
                    showlegend=False
                )
            fig_bar.update_layout(
                title=FEAT_LABELS.get(feat, feat), height=250,
                margin=dict(t=40, b=20, l=10, r=10), yaxis_title=T["avg_label"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # 特徴量重要度（ランダムフォレスト）
    st.subheader(T["feat_imp_title"])
    imp_rows = []
    for seg in SEG_ORDER:
        if seg not in df_raw["segment"].values: continue
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(df_raw[feat_cols_used], (df_raw["segment"] == seg).astype(int))
        row = {"segment": f"{SEGMENT_LABELS_JA[seg]}" if lang == "ja" else seg}
        for f, v in zip(feat_cols_used, clf.feature_importances_):
            row[FEAT_LABELS.get(f, f)] = round(v, 3)
        imp_rows.append(row)

    imp_df = pd.DataFrame(imp_rows).set_index("segment")
    fig_imp = px.imshow(
        imp_df, text_auto=".3f", aspect="auto", 
        color_continuous_scale="RdYlGn_r", labels=dict(color="Importance")
    )
    fig_imp.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_imp, use_container_width=True)

    # ペルソナカードの表示
    st.subheader(T["persona_title"])
    persona_cols = st.columns(4)
    for col, seg in zip(persona_cols, SEG_ORDER):
        with col:
            p = PERSONAS[seg][lang]
            seg_count = len(scored[scored["segment"] == seg])
            pct = seg_count / len(scored) * 100
            display_name = f"{SEGMENT_LABELS_JA[seg]}" if lang == "ja" else seg
            traits_html = "".join(f'<div style="font-size:.75rem;color:#ffffffdd;margin-bottom:3px;">• {t}</div>' for t in p["traits"])
            st.markdown(f"""
                <div style="background:{SEGMENT_COLORS[seg]};border-radius:10px;padding:16px;height:100%;">
                <div style="font-size:1.6rem;margin-bottom:6px;">{p['icon']}</div>
                <div style="font-family:serif;font-size:.95rem;color:#ffffff;font-weight:600;margin-bottom:4px;">{display_name}</div>
                <div style="font-size:.75rem;color:#ffffffcc;margin-bottom:10px;font-style:italic;">{p['tagline']}</div>
                <div style="font-size:.82rem;font-weight:700;color:#ffffff;margin-bottom:8px;">{seg_count:,} ({pct:.1f}%)</div>
                {traits_html}
                </div>
                """, unsafe_allow_html=True)

def render_segment_summary(scored, lang, T):
    """セグメント別の集計テーブルと円グラフを描画"""
    st.header(T["seg_table"])
    rows = []
    total = len(scored)
    for seg in SEG_ORDER:
        sub = scored[scored["segment"] == seg]
        display_name = f"{SEGMENT_LABELS_JA[seg]}（{seg}）" if lang == "ja" else seg
        rows.append({
            T["seg_col"]: display_name,
            T["n_col"]: len(sub),
            T["pct_col"]: f"{len(sub) / total * 100:.1f}%" if total else "0%",
            T["avg_uplift"]: f"{sub['uplift_score'].mean():.3f}" if len(sub) else "-",
            T["avg_p_ctrl"]: f"{sub['p_control'].mean():.3f}" if len(sub) else "-",
        })
    
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    fig_pie = px.pie(
        summary_df, names=T["seg_col"], values=T["n_col"],
        color=T["seg_col"], color_discrete_map={r[T["seg_col"]]: SEGMENT_COLORS[s] for r, s in zip(rows, SEG_ORDER)},
        title=T["seg_table"]
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)