import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="NexGen Logistics | CX & Risk Intelligence",
    page_icon="📦",
    layout="wide",
)

# ----------------------------
# Paths & Settings
# ----------------------------
DATA_DIR = Path("data")  # change if your CSVs are elsewhere

FILES = {
    "orders": DATA_DIR / "orders.csv",
    "delivery": DATA_DIR / "delivery_performance.csv",
    "routes": DATA_DIR / "routes_distance.csv",
    "feedback": DATA_DIR / "customer_feedback.csv",
    "costs": DATA_DIR / "cost_breakdown.csv",
    "fleet": DATA_DIR / "vehicle_fleet.csv",
    "inventory": DATA_DIR / "warehouse_inventory.csv",
}

# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path, **read_kwargs) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **read_kwargs)

@st.cache_data(show_spinner=True)
def load_data() -> dict:
    data = {k: load_csv(v) for k, v in FILES.items()}

    # Standardize column names (lower snake case)
    def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace("__+", "_", regex=True)
            .str.lower()
        )
        return df

    data = {k: clean_cols(v) for k, v in data.items()}

    # Parse dates if present
    for key, col in [("orders", "order_date"), ("feedback", "feedback_date"), ("inventory", "last_restocked_date")]:
        if not data[key].empty and col in data[key].columns:
            data[key][col] = pd.to_datetime(data[key][col], errors="coerce")

    # Derived columns for joins & KPIs
    orders = data["orders"].copy()
    delivery = data["delivery"].copy()
    routes = data["routes"].copy()
    costs = data["costs"].copy()
    feedback = data["feedback"].copy()

    # Delay metrics
    if not delivery.empty:
        if {"promised_delivery_days", "actual_delivery_days"}.issubset(delivery.columns):
            delivery["delay_days"] = delivery["actual_delivery_days"] - delivery["promised_delivery_days"]
            delivery["is_late"] = delivery["delay_days"] > 0

    # Master fact table
    master = orders
    for df, name in [(delivery, "delivery"), (routes, "routes"), (costs, "costs")]:
        if not df.empty and "order_id" in df.columns and "order_id" in master.columns:
            suffix = f"_{name}"
            master = master.merge(df, on="order_id", how="left", suffixes=("", suffix))

    # Bring feedback (1:N possible). Aggregate to order-level first.
    if not feedback.empty and "order_id" in feedback.columns:
        fb_agg = (
            feedback.groupby("order_id").agg(
                fb_rating_mean=("rating", "mean"),
                fb_count=("rating", "count"),
                would_recommend_share=("would_recommend", lambda s: np.mean(pd.to_numeric(s, errors="coerce")))
            )
            .reset_index()
        )
        master = master.merge(fb_agg, on="order_id", how="left")

    data["master"] = master
    return data

# ----------------------------
# KPI Calculations
# ----------------------------
@st.cache_data(show_spinner=False)
def compute_kpis(master: pd.DataFrame) -> dict:
    if master.empty:
        return {}
    kpis = {}

    # Delay rate
    if "is_late" in master.columns:
        kpis["delay_rate"] = float(master["is_late"].mean())
        kpis["avg_delay_days"] = float(master.get("delay_days", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().mean() or 0)

    # Ratings
    rating_cols = [c for c in ["customer_rating", "fb_rating_mean", "rating"] if c in master.columns]
    if rating_cols:
        kpis["avg_rating"] = float(master[rating_cols[0]].astype(float).mean())

    # Cost per order
    cost_cols = [c for c in master.columns if c.endswith("_cost") or c in [
        "fuel_cost", "labor_cost", "vehicle_maintenance", "insurance", "packaging_cost", "technology_platform_fee", "other_overhead", "delivery_cost_inr"
    ]]
    if cost_cols:
        kpis["avg_cost_per_order"] = float(master[cost_cols].replace([np.inf, -np.inf], np.nan).fillna(0).sum(axis=1).mean())

    # Priority failure rate (Express late)
    if {"priority", "is_late"}.issubset(master.columns):
        express = master[master["priority"].str.lower() == "express"]
        if not express.empty:
            kpis["express_failure_rate"] = float(express["is_late"].mean())

    # Risk score at segment level (0-1). Higher is worse.
    if {"customer_segment"}.issubset(master.columns):
        rating_src = master.get("customer_rating", master.get("fb_rating_mean"))
        tmp = master.assign(rating=rating_src)
        seg = tmp.groupby("customer_segment").agg(
            delay_rate=("is_late", "mean"),
            avg_rating=("rating", "mean"),
            orders=("order_id", "count")
        ).reset_index()
        if not seg.empty:
            seg["risk_score"] = seg["delay_rate"].fillna(0) * 0.6 + (5 - seg["avg_rating"].fillna(3)) / 5 * 0.4
            kpis["segment_risk_table"] = seg.sort_values("risk_score", ascending=False)

    return kpis

# ----------------------------
# UI Helpers
# ----------------------------
import matplotlib.pyplot as plt

def kpi_card(label: str, value, fmt: str = None):
    if value is None:
        st.metric(label, "-")
        return
    if fmt == "pct":
        st.metric(label, f"{value*100:,.1f}%")
    elif fmt == "inr":
        st.metric(label, f"₹{value:,.0f}")
    else:
        st.metric(label, f"{value:,.2f}")

# ----------------------------
# Sidebar Filters
# ----------------------------
def sidebar_filters(master: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    df = master.copy()
    if df.empty:
        return df

    # Date range
    if "order_date" in df.columns:
        min_d, max_d = df["order_date"].min(), df["order_date"].max()
        d_range = st.sidebar.date_input("Order date range", value=(min_d, max_d))
        if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
            df = df[(df["order_date"] >= pd.to_datetime(d_range[0])) & (df["order_date"] <= pd.to_datetime(d_range[1]))]

    # Priority
    if "priority" in df.columns:
        pri = st.sidebar.multiselect("Priority", sorted(df["priority"].dropna().unique().tolist()))
        if pri:
            df = df[df["priority"].isin(pri)]

    # Warehouse / Origin
    if "origin" in df.columns:
        origins = st.sidebar.multiselect("Origin Warehouse", sorted(df["origin"].dropna().unique().tolist()))
        if origins:
            df = df[df["origin"].isin(origins)]

    # Category
    if "product_category" in df.columns:
        cats = st.sidebar.multiselect("Product Category", sorted(df["product_category"].dropna().unique().tolist()))
        if cats:
            df = df[df["product_category"].isin(cats)]

    return df

# ----------------------------
# Pages
# ----------------------------

def page_overview(master: pd.DataFrame):
    st.subheader("Overview")
    kpis = compute_kpis(master)
    c1, c2, c3, c4 = st.columns(4)
    kpi_card("Delay rate", kpis.get("delay_rate"), fmt="pct")
    with c2:
        kpi_card("Avg delay (days)", kpis.get("avg_delay_days"))
    with c3:
        kpi_card("Avg rating", kpis.get("avg_rating"))
    with c4:
        kpi_card("Avg cost/order", kpis.get("avg_cost_per_order"), fmt="inr")

    st.markdown("---")

    # Charts section
    st.write("### Performance snapshots")
    cc1, cc2 = st.columns(2)

    # Delay by priority (matplotlib)
    if {"priority", "is_late"}.issubset(master.columns):
        by_p = (
            master.groupby("priority").agg(delay_rate=("is_late", "mean"), n=("order_id", "count")).reset_index()
        )
        with cc1:
            st.caption("Delay rate by priority")
            fig, ax = plt.subplots()
            ax.bar(by_p["priority"].astype(str), by_p["delay_rate"].astype(float))
            ax.set_ylabel("Delay rate")
            ax.set_xlabel("Priority")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

    # Rating distribution (matplotlib)
    rating_series = master.get("customer_rating", master.get("fb_rating_mean"))
    if rating_series is not None:
        with cc2:
            st.caption("Rating distribution")
            counts = rating_series.fillna(0).round().value_counts().sort_index()
            fig2, ax2 = plt.subplots()
            ax2.bar(counts.index.astype(str), counts.values)
            ax2.set_xlabel("Rating")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

    st.markdown("---")

    # Worst lanes table
    if {"origin", "destination", "is_late"}.issubset(master.columns):
        late_lanes = (
            master.groupby(["origin", "destination"]).agg(late_rate=("is_late", "mean"), n=("order_id", "count")).reset_index()
        ).sort_values(["late_rate", "n"], ascending=[False, False]).head(10)
        st.write("Worst-performing lanes (top 10)")
        st.dataframe(late_lanes)

    # Customer segments at risk
    seg_tbl = kpis.get("segment_risk_table")
    if seg_tbl is not None and not seg_tbl.empty:
        st.write("### Customer segments at risk")
        st.dataframe(seg_tbl)
        st.download_button(
            label="Download risk table (CSV)",
            data=seg_tbl.to_csv(index=False).encode("utf-8"),
            file_name="segment_risk.csv",
            mime="text/csv",
        )


    st.markdown("---")
    # Top late lanes
    if {"origin", "destination", "is_late"}.issubset(master.columns):
        late_lanes = (
            master.groupby(["origin", "destination"]).agg(late_rate=("is_late", "mean"), n=("order_id", "count")).reset_index()
        ).sort_values(["late_rate", "n"], ascending=[False, False]).head(10)
        st.write("Worst-performing lanes (top 10)")
        st.dataframe(late_lanes)

    # Customers at risk (simple rule-based)
    if {"customer_segment", "is_late"}.issubset(master.columns):
        risk = (
            master.assign(rating=master.get("customer_rating", master.get("fb_rating_mean")))
            .groupby("customer_segment")
            .agg(delay_rate=("is_late", "mean"), avg_rating=("rating", "mean"), orders=("order_id", "count"))
            .reset_index()
        )
        risk["risk_score"] = risk["delay_rate"] * 0.6 + (5 - risk["avg_rating"].fillna(3)) / 5 * 0.4
        st.write("Customer segments at risk")
        st.dataframe(risk.sort_values("risk_score", ascending=False))


def page_warehouse(master: pd.DataFrame, inventory: pd.DataFrame):
    st.subheader("Warehouse Watchtower")
    if {"origin", "is_late"}.issubset(master.columns):
        wh = (
            master.groupby("origin").agg(delay_rate=("is_late", "mean"), orders=("order_id", "count")).reset_index()
        ).sort_values("delay_rate", ascending=False)
        st.write("Warehouse performance (origin)")
        st.dataframe(wh)
        st.caption("Delay rate by origin")
        fig, ax = plt.subplots()
        ax.bar(wh["origin"].astype(str), wh["delay_rate"].astype(float))
        ax.set_ylabel("Delay rate")
        ax.set_xlabel("Origin")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    if not inventory.empty:
        st.write("Inventory snapshot")
        st.dataframe(inventory.head(50))


def page_priority(master: pd.DataFrame):
    st.subheader("Priority Performance")
    if {"priority", "is_late"}.issubset(master.columns):
        by_p = (
            master.groupby("priority").agg(delay_rate=("is_late", "mean"), avg_delay=("delay_days", "mean"), n=("order_id", "count")).reset_index()
        )
        st.dataframe(by_p)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Delay rate by priority")
            fig, ax = plt.subplots()
            ax.bar(by_p["priority"].astype(str), by_p["delay_rate"].astype(float))
            ax.set_ylabel("Delay rate")
            ax.set_xlabel("Priority")
            ax.set_ylim(0, 1)
            st.pyplot(fig)
        with c2:
            st.caption("Avg delay (days) by priority")
            fig2, ax2 = plt.subplots()
            ax2.bar(by_p["priority"].astype(str), by_p["avg_delay"].fillna(0).astype(float))
            ax2.set_ylabel("Avg delay (days)")
            ax2.set_xlabel("Priority")
            st.pyplot(fig2)


def page_cost(master: pd.DataFrame):
    st.subheader("Cost Intelligence")
    cost_cols = [c for c in master.columns if c.endswith("_cost") or c in [
        "fuel_cost", "labor_cost", "vehicle_maintenance", "insurance", "packaging_cost", "technology_platform_fee", "other_overhead", "delivery_cost_inr"
    ]]
    if cost_cols:
        st.write("Cost columns detected:", ", ".join(cost_cols))
        master["total_cost"] = master[cost_cols].fillna(0).sum(axis=1)
        by_cat = master.groupby("product_category").agg(avg_cost=("total_cost", "mean"), n=("order_id", "count")).reset_index()
        by_cat = by_cat.sort_values("avg_cost", ascending=False)
        st.write("Average cost by product category")
        st.dataframe(by_cat)
        st.caption("Avg cost by category")
        fig, ax = plt.subplots()
        ax.bar(by_cat["product_category"].astype(str), by_cat["avg_cost"].astype(float))
        ax.set_ylabel("Avg cost (INR)")
        ax.set_xlabel("Product category")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

# ----------------------------
# Voice of Customer (optional page)
# ----------------------------
def page_voice(master: pd.DataFrame, feedback: pd.DataFrame):
    st.subheader("Voice of Customer")
    if feedback is None or feedback.empty:
        st.info("No feedback data found.")
        return
    show_cols = [c for c in ["order_id","feedback_date","rating","feedback_text",
                             "issue_category","would_recommend"] if c in feedback.columns]
    st.dataframe(feedback.sort_values("feedback_date", ascending=False)[show_cols].head(100))

# ----------------------------
# Main
# ----------------------------
def main():
    st.title("📈 NexGen Logistics: Customer Experience & Risk Dashboard")
    st.caption("Streamlit skeleton with data loading, joins, filters, KPIs, charts, and risk table.")

    data = load_data()
    master_raw = data.get("master", pd.DataFrame())

    if master_raw is None or master_raw.empty:
        st.error("No data loaded. Place CSVs in a 'data/' folder or update FILES paths.")
        st.stop()

    master = sidebar_filters(master_raw)

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Warehouse", "Priority", "Cost", "Voice of Customer"],
        index=0,
    )

    if page == "Overview":
        page_overview(master)
    elif page == "Warehouse":
        page_warehouse(master, data.get("inventory", pd.DataFrame()))
    elif page == "Priority":
        page_priority(master)
    elif page == "Cost":
        page_cost(master)
    elif page == "Voice of Customer":
        page_voice(master, data.get("feedback", pd.DataFrame()))

    st.sidebar.markdown("---")
    st.sidebar.caption("Tips: Update FILES paths, refine KPI formulas, add charts.")

if __name__ == "__main__":
    main()
