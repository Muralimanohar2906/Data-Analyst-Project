# ================================================================
# Voltura Motors India — Intelligent EV Market Strategy Dashboard
# FINAL VERSION — Fully Functional Across All Tabs
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# ------------------------------------------------
# Streamlit Setup
# ------------------------------------------------
st.set_page_config(
    page_title="Voltura EV Market Strategy", page_icon="⚡", layout="wide"
)
st.title("⚡ Voltura Motors India — Intelligent EV Market Strategy Dashboard")
st.caption(
    "Auto-Optimized ANN Forecast | Regional & Maker Insights | Transparent Visualization"
)


# ------------------------------------------------
# Data Loading
# ------------------------------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


df_date = load_csv("dim_date.csv")
df_state = load_csv("electric_vehicle_sales_by_state.csv")
df_makers = load_csv("electric_vehicle_sales_by_makers.csv")

# ------------------------------------------------
# Date Standardization
# ------------------------------------------------
for df in (df_date, df_state, df_makers):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "fiscal_year" not in df.columns and "date" in df.columns:
        df["fiscal_year"] = df["date"].apply(
            lambda x: (
                x.year
                if pd.notnull(x) and x.month >= 4
                else (x.year - 1 if pd.notnull(x) else np.nan)
            )
        )
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")

# ------------------------------------------------
# Region Mapping
# ------------------------------------------------
region_map = {
    "North": [
        "Delhi",
        "Punjab",
        "Haryana",
        "Uttar Pradesh",
        "Uttarakhand",
        "Jammu & Kashmir",
    ],
    "South": ["Karnataka", "Tamil Nadu", "Kerala", "Andhra Pradesh", "Telangana"],
    "East": ["Bihar", "Odisha", "West Bengal", "Assam", "Jharkhand"],
    "West": ["Maharashtra", "Gujarat", "Rajasthan", "Goa"],
    "Central": ["Madhya Pradesh", "Chhattisgarh"],
    "Union Territories": ["Puducherry", "Chandigarh", "Ladakh"],
}


def get_region(state):
    for r, states in region_map.items():
        if state in states:
            return r
    return "Other"


df_state["region"] = df_state["state"].apply(get_region)

# ------------------------------------------------
# Sidebar Filters
# ------------------------------------------------
st.sidebar.header("Filters")
years_available = sorted(df_state["fiscal_year"].dropna().unique().astype(int).tolist())
years_selected = st.sidebar.multiselect(
    "Fiscal Years",
    years_available,
    default=years_available[-3:] if years_available else [],
)
regions_selected = st.sidebar.multiselect("Regions", list(region_map.keys()))
vehicle_category = st.sidebar.radio(
    "Vehicle Category", ["All", "2-Wheelers", "4-Wheelers"], index=0
)
state_selected = st.sidebar.selectbox(
    "State", ["All"] + sorted(df_state["state"].dropna().unique())
)
makers_list = (
    sorted(df_makers["maker"].dropna().unique()) if "maker" in df_makers.columns else []
)
makers_selected = st.sidebar.multiselect(
    "Makers", ["All"] + makers_list, default=["All"]
)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Years)", 1, 6, 3)


# ------------------------------------------------
# Filter Logic
# ------------------------------------------------
def apply_filters(df):
    df_filtered = df.copy()
    if regions_selected:
        selected_states = sum([region_map.get(r, []) for r in regions_selected], [])
        df_filtered = df_filtered[df_filtered["state"].isin(selected_states)]
    if years_selected:
        df_filtered = df_filtered[df_filtered["fiscal_year"].isin(years_selected)]
    if vehicle_category != "All":
        df_filtered = df_filtered[df_filtered["vehicle_category"] == vehicle_category]
    if state_selected != "All":
        df_filtered = df_filtered[df_filtered["state"] == state_selected]
    return df_filtered


data = apply_filters(df_state)
df_maker_filtered = df_makers.copy()
if makers_selected and "All" not in makers_selected:
    df_maker_filtered = df_maker_filtered[
        df_maker_filtered["maker"].isin(makers_selected)
    ]

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Home",
        "📊 Analysis",
        "📈 Trends & Forecast",
        "🌍 Regional Insights",
        "🏭 Makers",
    ]
)

# ============================================================
# 🏠 HOME TAB
# ============================================================
with tab1:
    st.header("India EV Landscape Overview")

    @st.cache_data
    def fetch_geojson():
        try:
            url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/india_states.geojson"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    india_geo = fetch_geojson()
    state_agg = data.groupby("state", as_index=False).agg(
        ev_sold=("electric_vehicles_sold", "sum"),
        total_sold=("total_vehicles_sold", "sum"),
    )
    state_agg["ev_penetration"] = (state_agg["ev_sold"] / state_agg["total_sold"]) * 100

    if india_geo is not None:
        fig_map = px.choropleth(
            state_agg,
            geojson=india_geo,
            locations="state",
            featureidkey="properties.ST_NM",
            color="ev_penetration",
            color_continuous_scale="Viridis",
            title="EV Penetration (%) by State",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", geo=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### 🌞 EV Sales Hierarchy — Region → State → Vehicle Category")
    df_sun = data.copy()
    df_sun["region"] = df_sun["state"].apply(get_region)
    df_sunburst = df_sun.groupby(
        ["region", "state", "vehicle_category"], as_index=False
    ).agg(ev_sum=("electric_vehicles_sold", "sum"))
    fig_sun = px.sunburst(
        df_sunburst,
        path=["region", "state", "vehicle_category"],
        values="ev_sum",
        color="region",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_sun.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sun, use_container_width=True)

# ============================================================
# 📊 ANALYSIS TAB (RESTORED)
# ============================================================
with tab2:
    st.header("📊 Market Performance Analysis")

    total_sales = int(data["electric_vehicles_sold"].sum())
    avg_price = (
        85000
        if vehicle_category == "2-Wheelers"
        else 1500000 if vehicle_category == "4-Wheelers" else 700000
    )
    revenue = total_sales * avg_price

    col1, col2, col3 = st.columns(3)
    col1.metric("Total EVs Sold", f"{total_sales:,}")
    col2.metric("Estimated Revenue", f"₹{revenue/1e7:.2f} Cr")

    if "fiscal_year" in data.columns:
        df_years = data.groupby("fiscal_year")["electric_vehicles_sold"].sum()
        if len(df_years) >= 2:
            start, end = df_years.iloc[0], df_years.iloc[-1]
            cagr = (
                ((end / start) ** (1 / (len(df_years) - 1)) - 1) * 100
                if start > 0
                else 0
            )
            col3.metric("CAGR", f"{cagr:.2f}%")

    st.markdown("### 🔝 Top Performing States by EV Sales")
    top_states = (
        data.groupby("state", as_index=False)["electric_vehicles_sold"]
        .sum()
        .sort_values(by="electric_vehicles_sold", ascending=False)
    )
    fig_top = px.bar(
        top_states.head(10),
        x="state",
        y="electric_vehicles_sold",
        color="electric_vehicles_sold",
        color_continuous_scale="Tealgrn",
        title="Top 10 States by EV Sales",
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("### ⚙️ EV Penetration by State")
    pen_df = data.groupby("state", as_index=False).agg(
        ev_sold=("electric_vehicles_sold", "sum"),
        total_sold=("total_vehicles_sold", "sum"),
    )
    pen_df["penetration_%"] = (pen_df["ev_sold"] / pen_df["total_sold"]) * 100
    fig_pen = px.bar(
        pen_df.sort_values("penetration_%", ascending=False).head(10),
        x="state",
        y="penetration_%",
        color="penetration_%",
        color_continuous_scale="Viridis",
        title="Top States by EV Penetration (%)",
    )
    st.plotly_chart(fig_pen, use_container_width=True)

# ============================================================
# 📈 TRENDS & FORECAST TAB
# ============================================================
with tab3:
    st.header("📈 EV Growth Trends & Forecasts (ANN Model)")

    df_trend = (
        data.groupby("fiscal_year", as_index=False)["electric_vehicles_sold"]
        .sum()
        .sort_values("fiscal_year")
    )

    if len(df_trend) >= 3:
        X = df_trend[["fiscal_year"]].values
        y = df_trend["electric_vehicles_sold"].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_scaled)

        ann = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=5000,
            random_state=42,
        )
        ann.fit(X_poly, y)

        future_years = np.arange(
            df_trend["fiscal_year"].max() + 1,
            df_trend["fiscal_year"].max() + forecast_horizon + 1,
        ).reshape(-1, 1)
        future_scaled = scaler.transform(future_years)
        future_poly = poly.transform(future_scaled)
        forecast = ann.predict(future_poly)

        st.subheader("📊 Historical EV Sales Trends")
        fig_trend = px.line(
            df_trend,
            x="fiscal_year",
            y="electric_vehicles_sold",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#00B4D8"],
            title="EV Sales Trend Over Time",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("🔮 Forecasted EV Sales Projection")
        forecast_df = pd.DataFrame(
            {
                "Fiscal Year": np.concatenate(
                    [df_trend["fiscal_year"], future_years.flatten()]
                ),
                "EVs Sold": np.concatenate([y, forecast]),
                "Type": ["Historical"] * len(y) + ["Forecast"] * len(forecast),
            }
        )
        fig_forecast = px.line(
            forecast_df,
            x="Fiscal Year",
            y="EVs Sold",
            color="Type",
            markers=True,
            line_dash="Type",
            color_discrete_map={"Historical": "#00B4D8", "Forecast": "magenta"},
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        growth_rate = ((forecast[-1] - y[-1]) / y[-1]) * 100 if y[-1] > 0 else 0
        st.success(
            f"📈 Projected Market Growth (Next {forecast_horizon} Years): **{growth_rate:.2f}%**"
        )

# ============================================================
# 🌍 REGIONAL INSIGHTS TAB
# ============================================================
with tab4:
    st.header("🌍 Regional Profitability & Growth Overview")

    region_df = (
        data.groupby("region", as_index=False)
        .agg({"electric_vehicles_sold": "sum", "total_vehicles_sold": "sum"})
        .rename(
            columns={
                "electric_vehicles_sold": "EV_Sales",
                "total_vehicles_sold": "Total_Sales",
            }
        )
    )
    region_df["EV Penetration (%)"] = (
        region_df["EV_Sales"] / region_df["Total_Sales"]
    ) * 100

    fig_profit = px.bar(
        region_df,
        x="region",
        y="EV Penetration (%)",
        color="EV_Sales",
        color_continuous_scale="Viridis",
        title="Regional EV Penetration & Sales",
    )
    st.plotly_chart(fig_profit, use_container_width=True)

# ============================================================
# 🏭 MAKERS TAB
# ============================================================
with tab5:
    st.header("🏭 Maker Insights & Recommendations")

    maker_cat = df_maker_filtered.groupby(
        ["maker", "vehicle_category"], as_index=False
    )["electric_vehicles_sold"].sum()
    fig_maker = px.bar(
        maker_cat,
        x="maker",
        y="electric_vehicles_sold",
        color="vehicle_category",
        barmode="group",
        title="Maker-wise Sales by Vehicle Category",
    )
    fig_maker.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_maker, use_container_width=True)

    if "state" in df_maker_filtered.columns:
        df_maker_filtered["region"] = df_maker_filtered["state"].apply(get_region)
        sun = df_maker_filtered.groupby(
            ["region", "maker", "vehicle_category"], as_index=False
        )["electric_vehicles_sold"].sum()
        fig_sun = px.sunburst(
            sun,
            path=["region", "maker", "vehicle_category"],
            values="electric_vehicles_sold",
            color="region",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="🧠 Strategic Advice — Region → Maker → Category",
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown(
        """
    ### 🧠 Strategic Advice for Manufacturers
    - **Top Makers:** Hero Electric, Ola Electric, Tata Motors  
    - **Best Regions:** South & West (fastest EV adoption)  
    - **Focus Categories:** 2-Wheelers (mass-market), 4-Wheelers (high-margin)  
    - **Recommendation:** Build production hubs in Karnataka or Tamil Nadu.
    """
    )
