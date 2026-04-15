"""
CropCast — Forecast-First Dashboard
Rolling 5-year yield forecast platform for global fresh produce.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "cropcast" / "data" / "processed"
PLOTS_DIR     = BASE_DIR / "cropcast" / "models" / "plots"

st.set_page_config(
    page_title="CropCast — Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    features  = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    forecasts = pd.read_parquet(PROCESSED_DIR / "forecasts.parquet")
    return features, forecasts


features_df, forecasts_df = load_data()

LATEST_YEAR  = int(features_df["year"].max())
NOWCAST_YEAR = int(forecasts_df["year"].min())
HORIZON_YEAR = int(forecasts_df["year"].max())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909761.png", width=60)
st.sidebar.title("🌾 CropCast")
st.sidebar.caption(f"Latest FAO data: **{LATEST_YEAR}** · Forecast: **{NOWCAST_YEAR}–{HORIZON_YEAR}**")
st.sidebar.divider()

all_crops     = sorted(features_df["crop"].unique().tolist())
all_countries = sorted(features_df["country"].unique().tolist())

selected_crop      = st.sidebar.selectbox("Crop", all_crops, index=all_crops.index("Grapes") if "Grapes" in all_crops else 0)
selected_countries = st.sidebar.multiselect("Countries", all_countries, default=all_countries)

st.sidebar.divider()
view = st.sidebar.radio("View", ["Forecast", "Historical", "Risk"], index=0)

st.sidebar.divider()
st.sidebar.caption("**Data sources**")
st.sidebar.caption("FAO STAT · Open-Meteo · USDA NASS")
st.sidebar.caption("**Models**")
st.sidebar.caption("XGBoost + Prophet ensemble")
st.sidebar.caption("Walk-forward backtesting")
st.sidebar.caption("90% prediction intervals")

# ── Filter data ───────────────────────────────────────────────────────────────
hist_df = features_df[
    (features_df["crop"] == selected_crop) &
    (features_df["country"].isin(selected_countries))
].dropna(subset=["yield_mt_ha"]).copy()

fc_df = forecasts_df[
    (forecasts_df["crop"] == selected_crop) &
    (forecasts_df["country"].isin(selected_countries))
].copy()

nowcast_df = fc_df[fc_df["forecast_type"] == "nowcast"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🌾 CropCast — {selected_crop} Yield Intelligence")
st.caption(f"Rolling forecast · Latest FAO data: {LATEST_YEAR} · Nowcast: {NOWCAST_YEAR} · Horizon: {HORIZON_YEAR} · {len(selected_countries)} countries")
st.divider()

# ── Metric cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

latest_actual = hist_df[hist_df["year"] == LATEST_YEAR]
prev_actual   = hist_df[hist_df["year"] == LATEST_YEAR - 1]

with col1:
    val   = latest_actual["yield_mt_ha"].mean()
    prev  = prev_actual["yield_mt_ha"].mean()
    delta = val - prev
    st.metric(f"Avg yield {LATEST_YEAR} (actual)", f"{val:.1f} MT/HA", f"{delta:+.1f}")

with col2:
    val = nowcast_df["ensemble_forecast"].mean()
    st.metric(f"Avg yield {NOWCAST_YEAR} (nowcast)", f"{val:.1f} MT/HA", "")

with col3:
    final_fc = fc_df[fc_df["year"] == HORIZON_YEAR]["ensemble_forecast"].mean()
    change   = (final_fc - latest_actual["yield_mt_ha"].mean()) / latest_actual["yield_mt_ha"].mean() * 100
    st.metric(f"Avg yield {HORIZON_YEAR} (forecast)", f"{final_fc:.1f} MT/HA", f"{change:+.1f}% vs {LATEST_YEAR}")

with col4:
    total_prod = latest_actual["production_mt"].sum() / 1_000_000
    st.metric(f"Production {LATEST_YEAR}", f"{total_prod:.1f}M MT", "")

with col5:
    avg_conf = int(nowcast_df["confidence_pct"].mean()) if len(nowcast_df) > 0 else 90
    st.metric("Nowcast confidence", f"{avg_conf}%", "")

st.divider()

# ── FORECAST VIEW ─────────────────────────────────────────────────────────────
if view == "Forecast":

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Yield forecast {NOWCAST_YEAR}–{HORIZON_YEAR} with prediction intervals")

        colors = px.colors.qualitative.Set2
        fig = go.Figure()

        for i, country in enumerate(selected_countries[:8]):
            color = colors[i % len(colors)]
            c_hist = hist_df[hist_df["country"] == country].sort_values("year")
            c_fc   = fc_df[fc_df["country"] == country].sort_values("year")

            if len(c_hist) == 0:
                continue

            # Historical line
            fig.add_trace(go.Scatter(
                x=c_hist["year"], y=c_hist["yield_mt_ha"],
                mode="lines+markers", name=f"{country} (actual)",
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))

            if len(c_fc) == 0:
                continue

            # CI band
            fig.add_trace(go.Scatter(
                x=pd.concat([c_fc["year"], c_fc["year"][::-1]]),
                y=pd.concat([c_fc["pi_upper"], c_fc["pi_lower"][::-1]]),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)") if "rgb" in color else color,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))

            # Nowcast point
            nowcast_row = c_fc[c_fc["forecast_type"] == "nowcast"]
            if len(nowcast_row) > 0:
                connect_x = [c_hist["year"].iloc[-1], nowcast_row["year"].iloc[0]]
                connect_y = [c_hist["yield_mt_ha"].iloc[-1], nowcast_row["ensemble_forecast"].iloc[0]]
                fig.add_trace(go.Scatter(
                    x=connect_x, y=connect_y,
                    mode="lines", line=dict(color=color, width=2, dash="dot"),
                    showlegend=False,
                ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=c_fc["year"], y=c_fc["ensemble_forecast"],
                mode="lines+markers", name=f"{country} (forecast)",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            ))

        fig.add_vline(x=LATEST_YEAR + 0.5, line_dash="dash",
                      line_color="gray", opacity=0.5)
        fig.add_annotation(x=LATEST_YEAR + 0.6, y=1, yref="paper",
                           text="← Actual | Forecast →",
                           showarrow=False, font=dict(size=11, color="gray"))

        fig.update_layout(
            height=400, margin=dict(t=20, b=20),
            xaxis_title="Year", yaxis_title="Yield (MT/HA)",
            legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Nowcast {NOWCAST_YEAR}")
        nc = nowcast_df[nowcast_df["country"].isin(selected_countries)].sort_values(
            "ensemble_forecast", ascending=False
        )
        for _, row in nc.iterrows():
            actual_row = hist_df[(hist_df["country"] == row["country"]) &
                                  (hist_df["year"] == LATEST_YEAR)]
            actual_yield = actual_row["yield_mt_ha"].iloc[0] if len(actual_row) > 0 else None
            delta = row["ensemble_forecast"] - actual_yield if actual_yield else 0
            delta_str = f"{delta:+.1f}" if actual_yield else ""
            st.metric(
                label=row["country"],
                value=f"{row['ensemble_forecast']:.1f} MT/HA",
                delta=delta_str,
            )

    st.divider()

    # 5-year outlook table
    st.subheader("5-year forecast summary")
    pivot = fc_df[fc_df["country"].isin(selected_countries)].pivot_table(
        index="country",
        columns="year",
        values="ensemble_forecast",
        aggfunc="mean"
    ).round(1)
    pivot.columns = [f"{int(c)} {'(nowcast)' if c == NOWCAST_YEAR else ''}" for c in pivot.columns]
    st.dataframe(pivot, use_container_width=True)

# ── HISTORICAL VIEW ───────────────────────────────────────────────────────────
elif view == "Historical":
    st.subheader("Historical yield trends (2000–2024)")

    fig = px.line(
        hist_df.sort_values("year"),
        x="year", y="yield_mt_ha", color="country",
        markers=True,
        labels={"yield_mt_ha": "Yield (MT/HA)", "year": "Year"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=380, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Production by country — latest year")
        latest = hist_df[hist_df["year"] == LATEST_YEAR].sort_values("production_mt", ascending=True)
        fig2 = px.bar(latest, x="production_mt", y="country", orientation="h",
                      color="production_mt", color_continuous_scale="Greens",
                      labels={"production_mt": "MT", "country": ""})
        fig2.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Year-over-year yield change (%)")
        yoy = hist_df.dropna(subset=["yield_mt_ha_yoy_pct"])
        fig3 = px.box(yoy, x="country", y="yield_mt_ha_yoy_pct",
                      color="country", color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={"yield_mt_ha_yoy_pct": "YoY %", "country": ""})
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("View raw data"):
        cols = ["country", "year", "yield_mt_ha", "production_mt",
                "area_ha", "avg_temp_max_c", "total_precip_mm", "yield_mt_ha_yoy_pct"]
        available = [c for c in cols if c in hist_df.columns]
        st.dataframe(hist_df[available].sort_values(["country", "year"], ascending=[True, False]),
                     use_container_width=True, height=300)

# ── RISK VIEW ─────────────────────────────────────────────────────────────────
elif view == "Risk":
    st.subheader(f"Supply chain risk assessment — {NOWCAST_YEAR} nowcast")

    nc = nowcast_df[nowcast_df["country"].isin(selected_countries)].copy()

    if len(nc) > 0:
        nc["pi_width_pct"] = (nc["pi_upper"] - nc["pi_lower"]) / nc["ensemble_forecast"] * 100

        hist_weather = features_df[
            (features_df["crop"] == selected_crop) &
            (features_df["country"].isin(selected_countries))
        ].copy()

        if "avg_temp_max_c_anomaly" in hist_weather.columns:
            anomaly = hist_weather.groupby("country")["avg_temp_max_c_anomaly"].apply(
                lambda x: x.tail(3).mean()
            ).reset_index()
            anomaly.columns = ["country", "temp_anomaly"]
            nc = nc.merge(anomaly, on="country", how="left")
        else:
            nc["temp_anomaly"] = 0

        nc["temp_anomaly"] = nc["temp_anomaly"].fillna(0).abs()
        nc["risk_score"] = (nc["pi_width_pct"] * 0.4 + nc["temp_anomaly"] * 15).clip(0, 100).round(1)
        nc["risk_level"] = nc["risk_score"].apply(
            lambda x: "🔴 High" if x >= 60 else ("🟡 Medium" if x >= 35 else "🟢 Low")
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_risk = px.bar(
                nc.sort_values("risk_score", ascending=True),
                x="risk_score", y="country", orientation="h",
                color="risk_score",
                color_continuous_scale=["#639922", "#EF9F27", "#E24B4A"],
                range_color=[0, 100],
                labels={"risk_score": "Risk score", "country": ""},
            )
            fig_risk.update_layout(height=350, margin=dict(t=10, b=10),
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            display_cols = ["country", "ensemble_forecast", "pi_width_pct",
                            "temp_anomaly", "risk_score", "risk_level"]
            available = [c for c in display_cols if c in nc.columns]
            st.dataframe(
                nc[available].sort_values("risk_score", ascending=False).reset_index(drop=True),
                use_container_width=True, height=350,
            )

    st.divider()
    st.subheader("SHAP feature importance")
    slug = selected_crop.lower().replace(" ", "_").replace(",", "")
    shap_path = PLOTS_DIR / f"shap_{slug}.png"
    if shap_path.exists():
        st.image(str(shap_path), width=700)
    else:
        st.info(f"No SHAP plot for {selected_crop}")

st.divider()
st.caption("CropCast v2.0 · FAO STAT · Open-Meteo · XGBoost + Prophet ensemble · Walk-forward backtesting")
