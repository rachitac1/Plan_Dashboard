# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from io import StringIO

# Prophet import (may require `pip install prophet`)
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet forecasting is not available. Install it with: pip install prophet")

# ------------------------
# Utilities
# ------------------------
@st.cache_data
def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load {path}: {e}")
        return pd.DataFrame()

def week_to_date(df_weeks, calendar_df=None, week_col="week"):
    """
    Convert numeric week to datetime 'ds' column using calendar week_start if available,
    otherwise map week-> base_date + (week-1) * 7 days.
    """
    if calendar_df is not None and not calendar_df.empty:
        # Expect calendar has columns 'week' and 'week_start'
        cal = calendar_df.copy()
        cal['week'] = pd.to_numeric(cal['week'], errors="coerce")
        cal['week_start'] = pd.to_datetime(cal['week_start'], errors="coerce")
        mapping = cal.set_index('week')['week_start'].to_dict()
        df_weeks = df_weeks.copy()
        df_weeks['ds'] = df_weeks[week_col].map(mapping)
        # fallback for missing weeks
        missing = df_weeks['ds'].isna()
        if missing.any():
            base = pd.to_datetime("2025-01-06")  # default base Monday
            df_weeks.loc[missing, 'ds'] = df_weeks.loc[missing, week_col].apply(lambda w: base + pd.to_timedelta(int(w)-1, unit='W'))
        return pd.to_datetime(df_weeks['ds'])
    else:
        base = pd.to_datetime("2025-01-06")
        return pd.to_datetime(base + pd.to_timedelta(df_weeks[week_col].astype(int)-1, unit='W'))

# ------------------------
# Load data with upload option
# ------------------------
st.set_page_config(layout="wide", page_title="FreshBites — Supply Chain Dashboard")
st.title("FreshBites — Supply Chain Optimization Dashboard")

# File uploader section
st.sidebar.header("Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files (optional)", 
    type="csv", 
    accept_multiple_files=True,
    help="Upload your own CSV files to replace the default data"
)

# Function to load data with upload option
def load_data_with_upload(default_filename, upload_files):
    # Check if user uploaded a file with this name
    if upload_files:
        for file in upload_files:
            if file.name == default_filename or file.name == default_filename.replace("p1_", ""):
                return pd.read_csv(file)
    
    # Fall back to default file
    return load_csv_safe(default_filename)

# Load all data files
inv = load_data_with_upload("p1_inventory_history.csv", uploaded_files)
short = load_data_with_upload("p1_shortages.csv", uploaded_files)
calendar = load_data_with_upload("p1_calendar.csv", uploaded_files)
master_skus = load_data_with_upload("p1_master_skus.csv", uploaded_files)
demand_actual = load_data_with_upload("p1_demand_actual.csv", uploaded_files)
forecasts = load_data_with_upload("p1_forecasts.csv", uploaded_files)
safety_stock = load_data_with_upload("p1_safety_stock.csv", uploaded_files)
distribution_centers = load_data_with_upload("p1_distribution_centers.csv", uploaded_files)

# Create mapping for SKU IDs to names
if not master_skus.empty and 'sku_id' in master_skus.columns and 'sku_name' in master_skus.columns:
    sku_name_mapping = master_skus.set_index('sku_id')['sku_name'].to_dict()
else:
    sku_name_mapping = {}
    st.warning("Could not load SKU master data. Showing IDs instead of names.")

# Create mapping for DC IDs to names
if not distribution_centers.empty and 'dc_id' in distribution_centers.columns and 'dc_name' in distribution_centers.columns:
    dc_name_mapping = distribution_centers.set_index('dc_id')['dc_name'].to_dict()
else:
    dc_name_mapping = {}
    st.warning("Could not load DC master data. Showing IDs instead of names.")

# Validate required columns
for df_name, df, cols in [
    ("inventory", inv, ["sku_id","dc_id","week","inventory_end_units"]),
    ("shortages", short, ["sku_id","dc_id","week","shortage_units"])
]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in {df_name}: {missing}. Please check your CSVs.")
        st.stop()

# Ensure numeric week
inv['week'] = pd.to_numeric(inv['week'], errors='coerce')
short['week'] = pd.to_numeric(short['week'], errors='coerce')

# Quick merge for combined view (missing shortages => 0)
df = pd.merge(inv, short, on=["sku_id","dc_id","week"], how="left")
df['shortage_units'] = df['shortage_units'].fillna(0)
df['demand_units'] = df['inventory_end_units'] + df['shortage_units']
df['service_level_row'] = (df['inventory_end_units'] / (df['demand_units'] + 1e-9)).clip(0,1)

# Add SKU names and DC names for display
df['sku_name'] = df['sku_id'].map(sku_name_mapping).fillna(df['sku_id'])
df['dc_name'] = df['dc_id'].map(dc_name_mapping).fillna(df['dc_id'])

# Convert weeks to dates (for Prophet & nicer x-axis)
df['ds'] = week_to_date(df, calendar_df=calendar, week_col='week')

# ------------------------
# Sidebar - Filters
# ------------------------
st.sidebar.header("Filters & Scenario")

# Get unique values for filters with names if available
sku_options = sorted(df['sku_id'].unique().tolist())
sku_display_options = [f"{sku} - {sku_name_mapping.get(sku, 'Unknown')}" for sku in sku_options]
dc_options = sorted(df['dc_id'].unique().tolist())
dc_display_options = [f"{dc} - {dc_name_mapping.get(dc, 'Unknown')}" for dc in dc_options]

week_min = int(df['week'].min())
week_max = int(df['week'].max())

selected_sku_display = st.sidebar.selectbox("SKU", ["All"] + sku_display_options, index=0)
selected_dc_display = st.sidebar.selectbox("Distribution Center", ["All"] + dc_display_options, index=0)

# Extract IDs from display values
if selected_sku_display == "All":
    selected_sku = "All"
else:
    selected_sku = selected_sku_display.split(" - ")[0]

if selected_dc_display == "All":
    selected_dc = "All"
else:
    selected_dc = selected_dc_display.split(" - ")[0]

week_range = st.sidebar.slider("Week range", min_value=week_min, max_value=week_max, value=(week_min, week_max), step=1)

# What-if: demand multiplier
st.sidebar.markdown("---")
demand_mult = st.sidebar.slider("Demand multiplier (what-if)", 0.5, 2.0, 1.0, 0.05)
safety_mult = st.sidebar.slider("Safety stock multiplier (sim)", 0.5, 2.0, 1.0, 0.05)

# Filter the dataframe
df_f = df[(df['week'] >= week_range[0]) & (df['week'] <= week_range[1])].copy()
if selected_sku != "All":
    df_f = df_f[df_f['sku_id'] == selected_sku]
if selected_dc != "All":
    df_f = df_f[df_f['dc_id'] == selected_dc]

# Apply demand multiplier simulation (affects demand and shortages proportionally)
df_sim = df_f.copy()
df_sim['sim_demand'] = (df_sim['demand_units'] * demand_mult).round().astype(int)
df_sim['sim_shortage'] = (df_sim['shortage_units'] * demand_mult).round().astype(int)
# Note: inventory_end_units is left as planned ending inventory; in a full sim we would re-run optimizer.

# ------------------------
# Layout: Tabs
# ------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Inventory & Shortages", "KPIs & Insights", "Forecasting", "What-if Simulator"])

# ------------------------
# Tab 1: Inventory & Shortages
# ------------------------
with tab1:
    st.header("Inventory & Shortages Overview")

    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Inventory Over Time (filtered)")
        # Use sku_name and dc_name for display if available
        color_col = 'sku_name' if 'sku_name' in df_f.columns else 'sku_id'
        line_group_col = 'dc_name' if 'dc_name' in df_f.columns else 'dc_id'
        
        fig_inv = px.line(df_f, x='ds', y='inventory_end_units', color=color_col, line_group=line_group_col,
                          hover_data=['sku_id','sku_name','dc_id','dc_name','week'], markers=True, 
                          title="Ending Inventory Over Time")
        fig_inv.update_layout(xaxis_title="Date (week start)", yaxis_title="Units")
        st.plotly_chart(fig_inv, use_container_width=True)

        st.subheader("Shortages Over Time (filtered)")
        fig_sh = px.bar(df_f, x='ds', y='shortage_units', color=color_col, barmode='group',
                        hover_data=['sku_id','sku_name','dc_id','dc_name','week'], title="Shortages by Week")
        fig_sh.update_layout(xaxis_title="Date (week start)", yaxis_title="Shortage Units")
        st.plotly_chart(fig_sh, use_container_width=True)

    with c2:
        st.subheader("Download data")
        st.download_button("Download filtered inventory CSV", df_f.drop(columns=['ds']).to_csv(index=False), file_name="filtered_inventory.csv", mime="text/csv")
        st.download_button("Download filtered shortages CSV", df_f[['sku_id','sku_name','dc_id','dc_name','week','shortage_units']].to_csv(index=False), file_name="filtered_shortages.csv", mime="text/csv")

# ------------------------
# Tab 2: KPIs & Insights
# ------------------------
with tab2:
    st.header("KPIs & Insights")

    # KPI cards (wide)
    total_demand = int(df_f['demand_units'].sum())
    total_short = int(df_f['shortage_units'].sum())
    avg_inventory = float(df_f['inventory_end_units'].mean())
    service_level_overall = 100 * (1 - (total_short / (total_demand + 1e-9)))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Demand (units)", f"{total_demand:,}")
    k2.metric("Total Shortage (units)", f"{total_short:,}")
    k3.metric("Avg Inventory (units/week)", f"{avg_inventory:,.1f}")
    k4.metric("Service Level (%)", f"{service_level_overall:.2f}")

    st.markdown("### Service Level Heatmap (SKU × DC)")
    merged = df.copy()
    merged['service_level_pct'] = 100 * (1 - merged['shortage_units'] / (merged['demand_units'] + 1e-9))
    
    # Use names for display if available
    index_col = 'sku_name' if 'sku_name' in merged.columns else 'sku_id'
    columns_col = 'dc_name' if 'dc_name' in merged.columns else 'dc_id'
    
    heat = merged.groupby([index_col, columns_col])['service_level_pct'].mean().reset_index()
    heat_pivot = heat.pivot(index=index_col, columns=columns_col, values='service_level_pct').fillna(100)
    fig_heat = px.imshow(heat_pivot, text_auto=".1f", color_continuous_scale='RdYlGn', aspect="auto",
                        labels=dict(x="DC", y="SKU", color="Service level %"),
                        title="Avg Service Level by SKU & DC")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Inventory vs Shortage (scatter)")
    color_col_scatter = 'dc_name' if 'dc_name' in df_f.columns else 'dc_id'
    facet_col_scatter = 'sku_name' if 'sku_name' in df_f.columns else 'sku_id'
    
    scatter = px.scatter(df_f, x='inventory_end_units', y='shortage_units', color=color_col_scatter, facet_col=facet_col_scatter,
                         title="Inventory vs Shortage by SKU and DC", trendline="ols")
    st.plotly_chart(scatter, use_container_width=True)

# ------------------------
# Tab 3: Forecasting (Prophet)
# ------------------------
with tab3:
    st.header("Forecasting Inventory (Prophet)")
    
    if not PROPHET_AVAILABLE:
        st.warning("Prophet is not installed. Forecasting features are disabled.")
    else:
        st.markdown("Prophet requires a time series of dates (`ds`) and numeric target (`y`). We map week → week_start date using calendar.csv if available.")

        # Forecast UI - use display names but store IDs
        sku_options_fc = [f"{sku} - {sku_name_mapping.get(sku, 'Unknown')}" for sku in sku_options]
        dc_options_fc = [f"{dc} - {dc_name_mapping.get(dc, 'Unknown')}" for dc in dc_options]
        
        sku_fc_display = st.selectbox("Choose SKU to forecast", sku_options_fc, index=0)
        dc_fc_display = st.selectbox("Choose DC to forecast", dc_options_fc, index=0)
        
        sku_fc = sku_fc_display.split(" - ")[0]
        dc_fc = dc_fc_display.split(" - ")[0]

        df_fc = df[(df['sku_id'] == sku_fc) & (df['dc_id'] == dc_fc)].copy()
        df_fc = df_fc.sort_values('ds')

        if len(df_fc) < 6:
            st.warning("Not enough weekly observations to build a reliable Prophet model (need >= 6).")
        else:
            # Prepare data for Prophet
            prophet_df = df_fc[['ds','inventory_end_units']].rename(columns={'ds':'ds','inventory_end_units':'y'}).dropna()
            # Prophet expects ds to be datetime - our ds is already datetime from week_to_date
            try:
                model = Prophet(weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=8, freq='W')  # forecast 8 weeks ahead
                forecast = model.predict(future)

                fig_prophet = plot_plotly(model, forecast)
                sku_name = sku_name_mapping.get(sku_fc, sku_fc)
                dc_name = dc_name_mapping.get(dc_fc, dc_fc)
                fig_prophet.update_layout(title=f"Prophet Forecast — {sku_name} @ {dc_name}")
                st.plotly_chart(fig_prophet, use_container_width=True)

                # Show forecast table (last 8 rows)
                st.subheader("Forecast (next weeks)")
                forecast_out = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(8)
                st.dataframe(forecast_out.rename(columns={'ds':'week_start'}))
                st.download_button("Download forecast CSV", forecast_out.to_csv(index=False), file_name=f"forecast_{sku_fc}_{dc_fc}.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prophet model failed: {e}")

# ------------------------
# Tab 4: What-if Simulator
# ------------------------
with tab4:
    st.header("What-If Simulator (fast scenario)")

    st.markdown("This simple simulator scales historical demand/shortages by the demand multiplier and recomputes high-level KPIs. (For an exact plan re-run the MILP optimizer.)")

    st.write("Demand multiplier:", demand_mult, "Safety stock multiplier:", safety_mult)

    sim = df_f.copy()
    sim['sim_demand'] = (sim['demand_units'] * demand_mult).round().astype(int)
    sim['sim_shortage'] = (sim['shortage_units'] * demand_mult).round().astype(int)
    # naive simulated service level (fulfillment = sim_demand - sim_shortage)
    sim_total_d = sim['sim_demand'].sum()
    sim_total_short = sim['sim_shortage'].sum()
    sim_service = 100 * (1 - sim_total_short / (sim_total_d + 1e-9))

    s1, s2, s3 = st.columns(3)
    s1.metric("Sim Total Demand", f"{int(sim_total_d):,}")
    s2.metric("Sim Total Shortage", f"{int(sim_total_short):,}")
    s3.metric("Sim Service Level (%)", f"{sim_service:.2f}")

    st.markdown("### Simulated shortage heatmap (DC vs Week)")
    
    # Use DC names for display if available
    index_col_sim = 'dc_name' if 'dc_name' in sim.columns else 'dc_id'
    
    sim_pivot = sim.pivot_table(index=index_col_sim, columns='week', values='sim_shortage', aggfunc='sum').fillna(0)
    fig_sim = px.imshow(sim_pivot, labels=dict(x="Week", y="DC", color="Sim Shortage"), aspect="auto", title="Simulated Shortages (DC vs Week)")
    st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("---")
st.caption("Dashboard built for FreshBites supply planning hackathon. For accurate scenario runs, re-run the MILP optimizer with scenario inputs.")