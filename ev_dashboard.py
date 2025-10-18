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
df_makers = pd.read_csv("electric_vehicle_sales_by_makers.csv")  # Keep CSV

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
    # Maker filter removed

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
# SCALE DOWN VEHICLE SOLD
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

    # Map
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

    # EV Penetration chart
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
        title="EV Penetration (%) by State",
        color="ev_penetration",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_pen, use_container_width=True)

# ====================================================
# TRENDS TAB
# ====================================================
with tab3:
    st.header("EV Trends Over Time")

    # Historical EV sales (state-based)
    df_trend = (
        data_for_charts.groupby("fiscal_year")
        .agg({"electric_vehicles_sold_scaled": "sum"})
        .reset_index()
    )
    if len(df_trend) >= 2:
        X = df_trend["fiscal_year"].values.reshape(-1, 1)
        y = df_trend["electric_vehicles_sold_scaled"].values
        model = LinearRegression()
        model.fit(X, y)
        df_trend["predicted_sales"] = model.predict(X)
        future_years = np.array(
            [df_trend["fiscal_year"].max() + i for i in range(1, 4)]
        ).reshape(-1, 1)
        y_pred_future = model.predict(future_years)
    else:
        df_trend["predicted_sales"] = df_trend["electric_vehicles_sold_scaled"]
        future_years = np.array([])
        y_pred_future = np.array([])

    # EV Trend + Forecast Plot
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=df_trend["fiscal_year"],
            y=df_trend["electric_vehicles_sold_scaled"],
            mode="lines+markers",
            name="Historical EV Sales",
            line=dict(color="#2E86AB", width=3),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=df_trend["fiscal_year"],
            y=df_trend["predicted_sales"],
            mode="lines",
            name="Linear Regression Trend",
            line=dict(color="red", dash="dash", width=3),
        )
    )
    if len(future_years) > 0:
        fig_trend.add_trace(
            go.Scatter(
                x=future_years.flatten(),
                y=y_pred_future,
                mode="lines+markers",
                name="3-Year Forecast",
                line=dict(color="green", dash="dot", width=3),
                marker=dict(symbol="diamond", size=8),
            )
        )
    fig_trend.update_layout(
        title="EV Sales Trend with Regression & 3-Year Forecast",
        xaxis_title="Fiscal Year",
        yaxis_title="EVs Sold (K)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Trend Type",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # EV vs Non-EV comparison
    if "total_vehicles_sold_scaled" in data_for_charts.columns:
        df_compare = (
            data_for_charts.groupby("fiscal_year")
            .agg({"electric_vehicles_sold_scaled": "sum", "non_ev_scaled": "sum"})
            .reset_index()
        )
        df_compare_long = df_compare.melt(
            id_vars="fiscal_year",
            value_vars=["electric_vehicles_sold_scaled", "non_ev_scaled"],
            var_name="Type",
            value_name="Vehicles Sold",
        )
        df_compare_long["Type"] = df_compare_long["Type"].map(
            {"electric_vehicles_sold_scaled": "EV", "non_ev_scaled": "Non-EV"}
        )
        fig_compare = px.line(
            df_compare_long,
            x="fiscal_year",
            y="Vehicles Sold",
            color="Type",
            markers=True,
            title="EV vs Non-EV Sales Trend Over Time",
            color_discrete_map={"EV": "#1F77B4", "Non-EV": "#FF7F0E"},
        )
        fig_compare.update_layout(
            xaxis_title="Fiscal Year",
            yaxis_title="Vehicles Sold (K)",
            legend_title="Vehicle Type",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # Maker-wise Trend Chart
    st.markdown("### Maker-wise EV Sales Trend")
    df_maker_filtered = df_makers.copy()
    if years_selected:
        df_maker_filtered = df_maker_filtered[
            df_maker_filtered["fiscal_year"].isin(years_selected)
        ]
    if vehicle_category != "All":
        df_maker_filtered = df_maker_filtered[
            df_maker_filtered["vehicle_category"] == vehicle_category
        ]
    df_maker_trend = df_maker_filtered.groupby(
        ["fiscal_year", "maker"], as_index=False
    ).agg({"electric_vehicles_sold": "sum"})
    df_maker_trend["electric_vehicles_sold_scaled"] = (
        df_maker_trend["electric_vehicles_sold"] / SCALE_FACTOR
    )
    fig_maker = px.line(
        df_maker_trend,
        x="fiscal_year",
        y="electric_vehicles_sold_scaled",
        color="maker",
        markers=True,
        title="Maker-wise EV Sales Trend",
        labels={
            "electric_vehicles_sold_scaled": "EVs Sold (K)",
            "fiscal_year": "Fiscal Year",
        },
    )
    fig_maker.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_maker, use_container_width=True)

# ====================================================
# REGIONAL INSIGHTS TAB
# ====================================================
with tab4:
    st.header("EV Adoption by Region")
    if regions_selected:
        df_region = (
            data_for_charts.groupby("state")
            .agg({"electric_vehicles_sold_scaled": "sum", "non_ev_scaled": "sum"})
            .reset_index()
        )
        df_region["region"] = df_region["state"].apply(
            lambda x: next(
                (r for r, states in region_map.items() if x in states),
                "Union Territories",
            )
        )
        df_region_summary = df_region.groupby("region").sum().reset_index()
        fig_region = px.bar(
            df_region_summary,
            x="region",
            y=["electric_vehicles_sold_scaled", "non_ev_scaled"],
            barmode="stack",
            title="EV vs Non-EV Sales by Region",
        )
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("Select at least one region to see regional insights.")

# ====================================================
# FORECASTING TAB
# ====================================================
with tab5:
    st.header("EV Forecasting")
    st.markdown("Forecasting future EV sales using Linear Regression")
    df_forecast = (
        data_for_charts.groupby("fiscal_year")
        .agg({"electric_vehicles_sold_scaled": "sum"})
        .reset_index()
    )
    if len(df_forecast) >= 2:
        X = df_forecast["fiscal_year"].values.reshape(-1, 1)
        y = df_forecast["electric_vehicles_sold_scaled"].values
        model = LinearRegression()
        model.fit(X, y)
        future_years = np.array(
            [df_forecast["fiscal_year"].max() + i for i in range(1, 6)]
        ).reshape(-1, 1)
        y_pred_future = model.predict(future_years)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(
                x=df_forecast["fiscal_year"],
                y=df_forecast["electric_vehicles_sold_scaled"],
                mode="lines+markers",
                name="Historical Sales",
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=future_years.flatten(),
                y=y_pred_future,
                mode="lines+markers",
                name="Forecasted Sales",
                line=dict(color="green", dash="dot"),
            )
        )
        fig_forecast.update_layout(
            title="EV Sales Forecast (Next 5 Years)",
            xaxis_title="Fiscal Year",
            yaxis_title="EVs Sold (K)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Not enough data for forecasting.")

st.markdown("---")
st.markdown("**Dashboard powered by Streamlit, Plotly & Linear Regression**")
