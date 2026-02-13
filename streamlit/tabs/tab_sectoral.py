import streamlit as st
import plotly.express as px
import numpy as np


def _prepare_sectoral_df(df):
    """Allocate each posting's demand across its categories to avoid multi-tag overcounting."""
    sector_df = df[df["category"].notna() & (df["category"] != "Others")].copy()
    sector_df = sector_df.drop_duplicates(subset=["job_id", "category", "month_year"])
    sector_df["category_count"] = sector_df.groupby("job_id")["category"].transform("nunique").clip(lower=1)
    sector_df["num_applications_adj"] = sector_df["num_applications"] / sector_df["category_count"]
    sector_df["num_vacancies_adj"] = sector_df["num_vacancies"] / sector_df["category_count"]
    return sector_df


def get_demand_velocity(df):
    velocity_df = _prepare_sectoral_df(df)
    top_10 = velocity_df.groupby("category")["num_vacancies_adj"].sum().nlargest(10).index
    velocity_df = velocity_df[velocity_df["category"].isin(top_10)]
    agg_df = velocity_df.groupby(["month_year", "category"]).agg(
        num_applications=("num_applications_adj", "sum"),
        num_vacancies=("num_vacancies_adj", "sum"),
        unique_job_count=("job_id", "nunique"),
    ).reset_index()
    agg_df["bulk_factor"] = agg_df.apply(
        lambda x: x["num_applications"] / x["num_vacancies"] if x["num_vacancies"] > 0 else 0,
        axis=1,
    )
    return agg_df


def get_bulk_hiring_data(df):
    bulk_df = _prepare_sectoral_df(df)
    top_sectors = bulk_df.groupby("category")["num_vacancies_adj"].sum().nlargest(12).index
    bulk_filtered = bulk_df[bulk_df["category"].isin(top_sectors)]
    apps = bulk_filtered.pivot_table(index="category", columns="month_year",
                                     values="num_applications_adj", aggfunc="sum").fillna(0)
    vacs = bulk_filtered.pivot_table(index="category", columns="month_year",
                                     values="num_vacancies_adj", aggfunc="sum").fillna(0)
    jobs = bulk_filtered.pivot_table(index="category", columns="month_year",
                                     values="job_id", aggfunc="nunique").fillna(0).astype(int)
    bulk_factor = apps.divide(vacs.where(vacs > 0)).fillna(0)
    return bulk_factor, apps, vacs, jobs


def render(df):
    st.subheader("🏭 Sectoral Demand & Momentum")
    st.markdown("Objective: Identify \"What\" to teach by tracking the velocity of industry needs.")

    st.markdown("#### 📈 Demand Velocity (Bulk Factor)")
    st.caption("Bulk Factor = Applications ÷ Vacancies. Higher values indicate stronger competition.")
    st.caption("Multi-category postings are fractionally allocated across categories to avoid overcounting.")

    velocity_df = get_demand_velocity(df)
    if len(velocity_df) > 1:
        fig_vel = px.line(velocity_df, x="month_year", y="bulk_factor", color="category",
                          markers=True, line_shape="spline",
                          title="Top 10 Sectors: Bulk Factor Trend Over Time",
                          labels={"month_year": "Posting Date",
                                  "bulk_factor": "Bulk Factor (Apps/Vacancies)",
                                  "category": "Sector"},
                          custom_data=["num_applications", "num_vacancies", "unique_job_count"])
        fig_vel.update_traces(
            hovertemplate=(
                "Sector: %{fullData.name}<br>"
                "Month: %{x|%b %Y}<br>"
                "Bulk Factor: %{y:.2f}<br>"
                "Applications (allocated): %{customdata[0]:,.2f}<br>"
                "Vacancies (allocated): %{customdata[1]:,.2f}<br>"
                "Unique job_id: %{customdata[2]:,.0f}<extra></extra>"
            )
        )
        st.plotly_chart(fig_vel, use_container_width=True, key="demand_velocity_chart")
    else:
        st.warning("Not enough data points for time-series velocity.")

    st.markdown("#### 🗺️ Bulk Hiring Map")
    st.caption("Competition intensity by sector and time. Darker = higher bulk factor.")

    bulk_pivot, apps_pivot, vacs_pivot, jobs_pivot = get_bulk_hiring_data(df)
    fig_bulk = px.imshow(bulk_pivot, aspect="auto", color_continuous_scale="YlOrRd",
                         labels=dict(x="Month", y="Sector", color="Bulk Factor"))
    custom_data = np.dstack([
        apps_pivot.reindex(index=bulk_pivot.index, columns=bulk_pivot.columns).to_numpy(),
        vacs_pivot.reindex(index=bulk_pivot.index, columns=bulk_pivot.columns).to_numpy(),
        jobs_pivot.reindex(index=bulk_pivot.index, columns=bulk_pivot.columns).to_numpy(),
    ])
    fig_bulk.update_traces(
        customdata=custom_data,
        hovertemplate=(
            "Sector: %{y}<br>"
            "Month: %{x|%b %Y}<br>"
            "Bulk Factor: %{z:.2f}<br>"
            "Applications (allocated): %{customdata[0]:,.2f}<br>"
            "Vacancies (allocated): %{customdata[1]:,.2f}<br>"
            "Unique job_id: %{customdata[2]:,.0f}<extra></extra>"
        ),
    )
    st.plotly_chart(fig_bulk, use_container_width=True, key="bulk_hiring_map")
