import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression
import numpy as np

# ====================================================
# Load Datasets
# ====================================================
df_date = pd.read_csv("dim_date.csv")
df_state = pd.read_csv("electric_vehicle_sales_by_state.csv")
df_makers = pd.read_csv("electric_vehicle_sales_by_makers.csv")  # Keep this CSV

# Convert date columns
df_date["date"] = pd.to_datetime(df_date["date"], format="mixed")
df_state["date"] = pd.to_datetime(df_state["date"], format="mixed")
df_makers["date"] = pd.to_datetime(df_makers["date"], format="mixed")

# Merge fiscal info
df_state = pd.merge(df_state, df_date, on="date", how="left")
df_makers = pd.merge(df_makers, df_date, on="date", how="left")

# ====================================================
# Region Mapping
# ====================================================
region_map = {
    "North": [
        "Delhi",
        "Punjab",
        "Haryana",
        "Himachal Pradesh",
        "Jammu & Kashmir",
        "Uttarakhand",
    ],
    "South": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"],
    "East": ["Bihar", "Odisha", "West Bengal", "Jharkhand", "Assam"],
    "West": ["Rajasthan", "Gujarat", "Maharashtra", "Goa"],
    "Central": ["Madhya Pradesh", "Chhattisgarh"],
    "Union Territories": [
        "Andaman & Nicobar Islands",
        "Lakshadweep",
        "Puducherry",
        "Chandigarh",
        "Dadra & Nagar Haveli and Daman & Diu",
    ],
}


# ====================================================
# Helper Functions
# ====================================================
def apply_filters(
    df_state,
    region_map,
    regions_selected,
    years_selected,
    vehicle_category,
    state_selected,
):
    df_filtered = df_state.copy()

    # REGION FILTER
    if regions_selected:
        selected_states = []
        for r in regions_selected:
            selected_states.extend(region_map.get(r, []))
        df_filtered = df_filtered[df_filtered["state"].isin(selected_states)]

    # YEAR FILTER
    if years_selected:
        df_filtered = df_filtered[df_filtered["fiscal_year"].isin(years_selected)]

    # VEHICLE CATEGORY FILTER
    if vehicle_category != "All":
        df_filtered = df_filtered[df_filtered["vehicle_category"] == vehicle_category]

    # STATE FILTER
    if state_selected != "All":
        df_filtered = df_filtered[df_filtered["state"] == state_selected]

    return df_filtered


def calculate_cagr(start_value, end_value, periods):
    if start_value > 0 and periods > 0:
        return ((end_value / start_value) ** (1 / periods) - 1) * 100
    return 0


def format_revenue(value):
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} Lakh"
    else:
        return f"₹{int(value):,}"


# ====================================================
# Streamlit Config
# ====================================================
st.set_page_config(page_title="Murali's EV Dashboard", page_icon="⚡", layout="wide")

# Sidebar Filters
st.sidebar.header("Filters")
with st.sidebar.expander("🔍 Filter Options", expanded=True):
    regions_selected = st.multiselect("Select Regions", list(region_map.keys()))
    years_selected = st.multiselect(
        "Select Fiscal Years", sorted(df_date["fiscal_year"].unique())
    )
    vehicle_category = st.radio("Vehicle Category", ["All", "2-Wheelers", "4-Wheelers"])
    state_selected = st.selectbox(
        "Select State", ["All"] + sorted(df_state["state"].dropna().unique())
    )
    # Maker filter removed from sidebar

# Apply Filters
data_for_charts = apply_filters(
    df_state,
    region_map,
    regions_selected,
    years_selected,
    vehicle_category,
    state_selected,
)

# ====================================================
# SCALE DOWN VEHICLE SOLD FOR CHARTS
# ====================================================
SCALE_FACTOR = 1000  # Show numbers in thousands
data_for_charts["electric_vehicles_sold_scaled"] = (
    data_for_charts["electric_vehicles_sold"] / SCALE_FACTOR
)
if "total_vehicles_sold" in data_for_charts.columns:
    data_for_charts["total_vehicles_sold_scaled"] = (
        data_for_charts["total_vehicles_sold"] / SCALE_FACTOR
    )
    data_for_charts["non_ev_scaled"] = (
        data_for_charts["total_vehicles_sold"]
        - data_for_charts["electric_vehicles_sold"]
    ) / SCALE_FACTOR

# ====================================================
# Revenue + CAGR
# ====================================================
total_sales = data_for_charts["electric_vehicles_sold"].sum()
if vehicle_category == "2-Wheelers":
    avg_price = 85000
elif vehicle_category == "4-Wheelers":
    avg_price = 1500000
else:
    two_sales = data_for_charts[data_for_charts["vehicle_category"] == "2-Wheelers"][
        "electric_vehicles_sold"
    ].sum()
    four_sales = data_for_charts[data_for_charts["vehicle_category"] == "4-Wheelers"][
        "electric_vehicles_sold"
    ].sum()
    avg_price = (
        ((two_sales * 85000) + (four_sales * 1500000)) / total_sales
        if total_sales > 0
        else 0
    )

revenue = total_sales * avg_price if total_sales > 0 else 0
formatted_revenue = format_revenue(revenue)

cagr = 0
if len(years_selected) >= 2:
    df_years = (
        data_for_charts.groupby("fiscal_year")["electric_vehicles_sold"]
        .sum()
        .reset_index()
    )
    start_val = df_years[df_years["fiscal_year"] == min(years_selected)][
        "electric_vehicles_sold"
    ].sum()
    end_val = df_years[df_years["fiscal_year"] == max(years_selected)][
        "electric_vehicles_sold"
    ].sum()
    periods = len(years_selected) - 1
    cagr = calculate_cagr(start_val, end_val, periods)

# ====================================================
# Tabs
# ====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "📊 Analysis", "📈 Trends", "🌍 Regional Insights", "🔮 Forecasting"]
)

# ====================================================
# HOME TAB
# ====================================================
with tab1:
    st.title("Welcome to Murali's EV Analysis")
    st.markdown("### Explore EV adoption across India with interactive visualizations")

    # Map with UTs included
    states = data_for_charts.groupby("state", as_index=False).agg(
        {"electric_vehicles_sold": "sum", "total_vehicles_sold": "sum"}
    )
    states["ev_ratio"] = (
        states["electric_vehicles_sold"] / states["total_vehicles_sold"]
    ) * 100
    geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/india_states.geojson"
    india_geo = requests.get(geojson_url).json()
    fig_map = px.choropleth(
        states,
        geojson=india_geo,
        locations="state",
        featureidkey="properties.ST_NM",
        color="ev_ratio",
        color_continuous_scale="Viridis",
        hover_name="state",
        hover_data={
            "electric_vehicles_sold": True,
            "total_vehicles_sold": True,
            "ev_ratio": ":.2f",
        },
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_map, use_container_width=True)

    # Sunburst chart
    st.markdown("### EV Sales Hierarchy (Region → State → Vehicle Category)")
    df_sunburst = data_for_charts.copy()
    df_sunburst["region"] = df_sunburst["state"].apply(
        lambda x: next(
            (r for r, states in region_map.items() if x in states), "Union Territories"
        )
    )
    fig_sunburst = px.sunburst(
        df_sunburst,
        path=["region", "state", "vehicle_category"],
        values="electric_vehicles_sold_scaled",
        color="region",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={"electric_vehicles_sold": True},
    )
    st.plotly_chart(fig_sunburst, use_container_width=True)

# ====================================================
# ANALYSIS TAB
# ====================================================
with tab2:
    st.header("Detailed Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total EV Sales", f"{total_sales:,}")
    col2.metric("Estimated Revenue", formatted_revenue)
    col3.metric("CAGR", f"{cagr:.2f}%")

    st.markdown("### EV Sales by Year (Split by States in Selected Regions)")

    if regions_selected:
        # Show sales split by state for the selected regions
        df_compare = data_for_charts.groupby(
            ["fiscal_year", "state"], as_index=False
        ).agg({"electric_vehicles_sold_scaled": "sum"})
        fig_comparison = px.bar(
            df_compare,
            x="fiscal_year",
            y="electric_vehicles_sold_scaled",
            color="state",
            barmode="group",
            title="EV Sales by Year (States in Selected Regions)",
        )
    else:
        df_compare = (
            data_for_charts.groupby("fiscal_year")
            .agg({"electric_vehicles_sold_scaled": "sum"})
            .reset_index()
        )
        fig_comparison = px.bar(
            df_compare,
            x="fiscal_year",
            y="electric_vehicles_sold_scaled",
            color="fiscal_year",
            title="EV Sales by Year",
            color_continuous_scale="Tealgrn",
        )
    st.plotly_chart(fig_comparison, use_container_width=True)

    # EV Penetration chart based on filtered data
    st.markdown("### EV Penetration by State")
    df_pen = data_for_charts.groupby("state", as_index=False).agg(
        {"electric_vehicles_sold_scaled": "sum", "total_vehicles_sold_scaled": "sum"}
    )
    df_pen["ev_penetration"] = (
        df_pen["electric_vehicles_sold_scaled"] / df_pen["total_vehicles_sold_scaled"]
    ) * 100
    fig_pen = px.bar(
        df_pen,
        x="state",
        y="ev_penetration",
        title=f"EV Penetration (%) by State",
        color="ev_penetration",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_pen, use_container_width=True)

# ====================================================
# TREND, REGIONAL INSIGHTS, FORECASTING TABS
# ====================================================
# (Use same logic as before, making sure to use df_makers internally for maker-specific charts)

st.markdown("---")
st.markdown("**Dashboard powered by Streamlit, Plotly & Linear Regression**")
