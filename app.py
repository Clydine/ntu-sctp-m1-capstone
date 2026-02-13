import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="SG Job Market Dashboard for Curriculum Design",
    layout="wide",
    page_icon="📊",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50 !important;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .insight-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 20px;
        height: 100%;
    }
    .insight-text {
        color: #000000 !important;
        font-size: 16px;
        line-height: 1.5;
    }
    .finding-title {
        font-weight: bold;
        color: #0d47a1 !important;
        margin-bottom: 5px;
        font-size: 18px;
    }
    li {
        color: #000000 !important;
        margin-bottom: 5px;
    }
    b {
        font-weight: 700;
        color: #000;
    }
    [data-testid="stSelectbox"] label, [data-testid="stSelectbox"] p {
        color: #1e293b !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li:hover, [data-baseweb="popover"] [role="option"]:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    option {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    header {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================

@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_exploded_data():
    """Load the pre-exploded dataset (categories already expanded)."""
    df = pd.read_parquet("data/cleaned-sgjobdata-exploded.parquet")
    df["posting_date"] = pd.to_datetime(df["posting_date"])
    df["month_year"] = df["posting_date"].dt.to_period("M").dt.to_timestamp()
    df["average_salary"] = df["average_salary_cleaned"]
    df["exp_segment"] = pd.cut(
        df["min_exp"],
        bins=[0, 2, 5, 10, float("inf")],
        labels=["0-2 yrs (Entry/Junior)", ">2-5 yrs (Mid-Level)",
                ">5-10 yrs (Senior)", "10+ yrs (Expert)"],
        right=True,
    )
    return df


@st.cache_data(ttl=3600, show_spinner="Loading skills data...")
def load_skills_optimized():
    """Load pre-aggregated skills timeline data."""
    try:
        return pd.read_parquet("data/skills_optimized.parquet")
    except FileNotFoundError:
        st.info("Skills data file not found. Run scripts/build_skills_optimized.py first.")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading skills detail data...")
def load_withskills_data():
    """Load the full skills-enriched dataset (on-demand for Tab 4)."""
    return pd.read_parquet(
        "data/cleaned-sgjobdata-withskills.parquet",
        columns=["job_id", "jobtitle_cleaned", "skill"],
    )


# ==========================================
# 3. AGGREGATION FUNCTIONS
# ==========================================

def calculate_executive_metrics(df):
    return {
        "total_vacancies": df["num_vacancies"].sum(),
        "total_posts": len(df),
        "total_views": df["num_views"].sum(),
        "top_sector_vacancy": df.groupby("category")["num_vacancies"].sum().idxmax(),
        "top_sector_posts": df["category"].value_counts().idxmax(),
        "top_sector_views": df.groupby("category")["num_views"].sum().idxmax(),
    }


def get_top_sectors_data(df, metric="num_vacancies", limit=10):
    df_filtered = df[df["category"] != "Others"]
    if metric == "count":
        data = df_filtered["category"].value_counts().head(limit)
    else:
        data = df_filtered.groupby("category")[metric].sum().sort_values(ascending=False).head(limit)
    return data.reset_index(name="Value")


def filter_by_sector(df, sector):
    if sector == "All":
        return df
    return df[df["category"] == sector].copy()


def get_demand_velocity(df):
    velocity_df = df[df["category"] != "Others"]
    top_10 = velocity_df.groupby("category")["num_vacancies"].sum().nlargest(10).index
    velocity_df = velocity_df[velocity_df["category"].isin(top_10)]
    agg_df = velocity_df.groupby(["month_year", "category"]).agg(
        num_applications=("num_applications", "sum"),
        num_vacancies=("num_vacancies", "sum"),
    ).reset_index()
    agg_df["bulk_factor"] = agg_df.apply(
        lambda x: x["num_applications"] / x["num_vacancies"] if x["num_vacancies"] > 0 else 0,
        axis=1,
    )
    return agg_df


def get_bulk_hiring_data(df):
    bulk_df = df[df["category"] != "Others"]
    top_sectors = bulk_df.groupby("category")["num_vacancies"].sum().nlargest(12).index
    bulk_filtered = bulk_df[bulk_df["category"].isin(top_sectors)]
    apps = bulk_filtered.pivot_table(index="category", columns="month_year",
                                     values="num_applications", aggfunc="sum").fillna(0)
    vacs = bulk_filtered.pivot_table(index="category", columns="month_year",
                                     values="num_vacancies", aggfunc="sum").fillna(0)
    return (apps / vacs.replace(0, 1)).fillna(0)


def get_experience_metrics(df, sector="All"):
    if sector != "All":
        df = df[df["category"] == sector]
    pay_scale = df.groupby("exp_segment").apply(
        lambda g: (g["average_salary"] * g["num_vacancies"]).sum() / g["num_vacancies"].sum()
    ).reset_index(name="avg_salary")
    gate_df = df.groupby("exp_segment")["num_vacancies"].sum().reset_index()
    return pay_scale, gate_df


def get_education_metrics(df):
    metrics = df.groupby("category").agg(
        num_vacancies=("num_vacancies", "sum"),
        num_applications=("num_applications", "sum"),
        min_exp=("min_exp", "mean"),
        job_id=("job_id", "count"),
    ).reset_index()
    metrics["opp_score"] = metrics["num_vacancies"] / (metrics["min_exp"] + 1)
    metrics["comp_index"] = metrics.apply(
        lambda x: x["num_applications"] / x["num_vacancies"] if x["num_vacancies"] > 0 else 0,
        axis=1,
    )
    return metrics


# ==========================================
# 4. DASHBOARD
# ==========================================

def run_dashboard(df):
    if df.empty:
        st.error("No valid data found after loading. Please check your data files.")
        st.stop()

    min_d = df["posting_date"].min().strftime("%d %b %Y")
    max_d = df["posting_date"].max().strftime("%d %b %Y")

    st.title("🎓 SG Job Market Dashboard for Curriculum Design")
    st.markdown("Aligning Curriculum with Real-Time Market Structure")
    st.write(f"**Data Period:** {min_d} - {max_d}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Summary",
        "🏭 Sectoral Demand & Momentum",
        "🛠️ Experience Level",
        "🎓 Opportunity",
    ])

    # --- TAB 1: EXECUTIVE SUMMARY ---
    with tab1:
        st.subheader("High-Level Market Snapshot")
        metrics = calculate_executive_metrics(df)

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="👥 Total Vacancies", value=f"{metrics['total_vacancies']:,.0f}",
                      help=f"Top sector: {metrics['top_sector_vacancy']}")
            st.caption(f"🏆 **Top:** {metrics['top_sector_vacancy']}")
        with kpi2:
            st.metric(label="📝 Total Job Posts", value=f"{metrics['total_posts']:,.0f}",
                      help=f"Top sector: {metrics['top_sector_posts']}")
            st.caption(f"🏆 **Top:** {metrics['top_sector_posts']}")
        with kpi3:
            st.metric(label="👁️ Total Job Views", value=f"{metrics['total_views']:,.0f}",
                      help=f"Top sector: {metrics['top_sector_views']}")
            st.caption(f"🏆 **Top:** {metrics['top_sector_views']}")

        st.divider()

        c_head, c_opt = st.columns([3, 1])
        with c_head:
            st.markdown("#### 📊 Top 10 Sectors Breakdown")
        with c_opt:
            chart_metric = st.selectbox("View By:", ["Vacancies", "Job Posts", "Job Views"], index=0)

        if chart_metric == "Vacancies":
            chart_data = get_top_sectors_data(df, "num_vacancies", 10)
            x_label, bar_color = "Total Vacancies", "#2E86C1"
        elif chart_metric == "Job Posts":
            chart_data = get_top_sectors_data(df, "count", 10)
            x_label, bar_color = "Number of Posts", "#28B463"
        else:
            chart_data = get_top_sectors_data(df, "num_views", 10)
            x_label, bar_color = "Total Views", "#E67E22"

        altair_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Value:Q", title=x_label),
            y=alt.Y("category:N", sort="-x", title="Sector"),
            color=alt.value(bar_color),
            tooltip=["category", alt.Tooltip("Value:Q", format=",.0f")],
        ).properties(height=500, title=f"Top 10 Sectors by {chart_metric}")
        st.altair_chart(altair_chart, use_container_width=True)

    # --- TAB 2: SECTORAL DEMAND & MOMENTUM ---
    with tab2:
        st.subheader("🏭 Sectoral Demand & Momentum")
        st.markdown("Objective: Identify \"What\" to teach by tracking the velocity of industry needs.")

        st.markdown("#### 📈 Demand Velocity (Bulk Factor)")
        st.caption("Bulk Factor = Applications ÷ Vacancies. Higher values indicate stronger competition.")

        velocity_df = get_demand_velocity(df)
        if len(velocity_df) > 1:
            fig_vel = px.line(velocity_df, x="month_year", y="bulk_factor", color="category",
                              markers=True, line_shape="spline",
                              title="Top 10 Sectors: Bulk Factor Trend Over Time",
                              labels={"month_year": "Posting Date",
                                      "bulk_factor": "Bulk Factor (Apps/Vacancies)",
                                      "category": "Sector"})
            st.plotly_chart(fig_vel, use_container_width=True, key="demand_velocity_chart")
        else:
            st.warning("Not enough data points for time-series velocity.")

        st.markdown("#### 🗺️ Bulk Hiring Map")
        st.caption("Competition intensity by sector and time. Darker = higher bulk factor.")

        bulk_pivot = get_bulk_hiring_data(df)
        fig_bulk = px.imshow(bulk_pivot, aspect="auto", color_continuous_scale="YlOrRd",
                             labels=dict(x="Month", y="Sector", color="Bulk Factor"))
        st.plotly_chart(fig_bulk, use_container_width=True, key="bulk_hiring_map")

        # Skills in High Demand
        st.markdown("#### High Demand Skills")
        st.caption("Top 10 skills by unique job postings over time.")

        skills_df = load_skills_optimized()

        if not skills_df.empty:
            available_months = sorted(skills_df["month_year"].unique())
            month_labels = {}
            for month in available_months:
                date_obj = pd.to_datetime(month)
                month_labels[month] = date_obj.strftime("%b %Y")

            if len(available_months) > 0:
                st.markdown("### 📈 Skill Demand Timeline - Top 10 Most Popular Skills")

                skills_sectors = ["All"] + sorted(skills_df["category"].dropna().unique().tolist())
                col_skills_filter, col_skills_space = st.columns([1, 3])
                with col_skills_filter:
                    st.markdown("**Filter by Sector**")
                with col_skills_space:
                    selected_skills_sector = st.selectbox("", skills_sectors,
                                                          key="skills_sector_filter",
                                                          label_visibility="collapsed")

                skills_filtered = skills_df.copy()
                if selected_skills_sector != "All":
                    skills_filtered = skills_filtered[skills_filtered["category"] == selected_skills_sector]

                top_skills = skills_filtered.groupby("skill")["job_count"].sum().nlargest(10).index.tolist()

                if top_skills:
                    timeline_df = skills_filtered[skills_filtered["skill"].isin(top_skills)].copy()
                    timeline_df = timeline_df.groupby(["skill", "month_year"])["job_count"].sum().reset_index()
                    timeline_df["month_label"] = timeline_df["month_year"].map(month_labels)

                    fig = px.line(
                        timeline_df, x="month_label", y="job_count", color="skill", markers=True,
                        title=("Skill Demand Timeline - Top 10 Most Popular Skills"
                               if selected_skills_sector == "All"
                               else f"Top 10 Skills in {selected_skills_sector}"),
                        labels={"month_label": "Month-Year Period",
                                "job_count": "Number of Unique Job Postings",
                                "skill": "Skill"},
                    )
                    fig.update_layout(
                        height=600, hovermode="x unified",
                        legend=dict(title="Skills", orientation="v",
                                    yanchor="top", y=1, xanchor="left", x=1.02),
                    )
                    fig.update_traces(line=dict(width=2.5))
                    st.plotly_chart(fig, use_container_width=True, key="skills_demand_chart")
                else:
                    st.info(f"No skills data available for {selected_skills_sector}")
            else:
                st.info("No date information available in skills data.")
        else:
            st.info("Skills data file not found or empty.")

    # --- TAB 3: EXPERIENCE LEVEL ---
    with tab3:
        st.subheader("🛠️ Experience Analysis")
        st.markdown('Objective: Align the "Level" of training with market reality to ensure graduate ROI.')

        exp_comp_sectors = ["All"] + sorted(df["category"].unique().tolist())
        selected_exp_sector = st.selectbox("Filter by Sector:", exp_comp_sectors, key="tab3_sector_filter")

        df_exp = df.copy() if selected_exp_sector == "All" else df[df["category"] == selected_exp_sector]

        c3_new1, c3_new2 = st.columns(2)

        with c3_new1:
            st.markdown("#### Experience Level Distribution")
            st.caption("Distribution of total vacancies by experience level")

            exp_dist = df_exp.groupby("exp_segment")["num_vacancies"].sum().reset_index()
            exp_dist = exp_dist.sort_values("num_vacancies", ascending=False)
            max_idx = exp_dist["num_vacancies"].idxmax()
            explode = [0.1 if idx == max_idx else 0 for idx in exp_dist.index]

            distinct_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE"]
            fig_pie = go.Figure(data=[go.Pie(
                labels=exp_dist["exp_segment"],
                values=exp_dist["num_vacancies"],
                pull=explode, hole=0.3,
                marker=dict(colors=distinct_colors),
                textinfo="label+percent",
                textposition="auto",
                insidetextorientation="horizontal",
            )])
            fig_pie.update_layout(title="Vacancy Distribution by Experience Level",
                                  height=400, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True, key="exp_distribution_pie")

        with c3_new2:
            st.markdown("#### Average Salary Distribution by Experience")
            st.caption("Weighted salary ranges across experience levels")

            bins = [0, 2, 5, 10, float("inf")]
            labels = ["0-2 yrs (Entry)", ">2-5 yrs (Mid-Level)",
                      ">5-10 yrs (Senior)", "10+ yrs (Expert)"]
            colors = ["#FFB6C1", "#87CEEB", "#90EE90", "#FFD700"]

            df_plot = df_exp.copy()
            df_plot["exp_group"] = pd.cut(df_plot["min_exp"], bins=bins, labels=labels, right=False)
            df_plot["weight_cap"] = df_plot["num_vacancies"].clip(upper=5)
            df_expanded = df_plot.loc[df_plot.index.repeat(df_plot["weight_cap"].astype(int))].reset_index(drop=True)
            plot_df = df_expanded.dropna(subset=["exp_group", "average_salary"]).copy()

            vacancy_totals = df_plot.groupby("exp_group")["num_vacancies"].sum()

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=plot_df, x="exp_group", y="average_salary", palette=colors, linewidth=1, ax=ax)
            xticklabels = [f"{grp}\n(Total vacancies: {int(vacancy_totals.get(grp, 0))})" for grp in labels]
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel("Experience (Years)", fontsize=12)
            ax.set_ylabel("Average Salary (SGD)", fontsize=12)
            ax.set_title("Average Salary Distribution by Experience Group (weighted by num_vacancies)",
                         fontsize=14, fontweight="bold")
            plt.grid(True, axis="y", which="major", linestyle="--", alpha=0.6)
            plt.minorticks_on()
            plt.grid(True, which="minor", axis="y", linestyle=":", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # --- TAB 4: OPPORTUNITY ---
    with tab4:
        st.subheader("🎓 Opportunity")
        st.markdown('Objective: Identify "Blue Ocean" opportunities where job matching rates are highest.')

        p2_metrics = get_education_metrics(df)

        st.markdown("#### Supply vs Demand")
        st.caption("Treemap: Rectangle size = Vacancies (demand), Color = Applications (supply).")
        supply_demand = p2_metrics[p2_metrics["category"] != "Others"].copy()
        supply_demand = supply_demand.sort_values("num_vacancies", ascending=False).head(20)

        fig_supply_demand = px.treemap(
            supply_demand, path=[px.Constant("All Sectors"), "category"],
            values="num_vacancies", color="num_applications",
            color_continuous_scale="RdYlGn_r",
            labels={"num_vacancies": "Vacancies (Size)", "num_applications": "Applications (Color)"},
            title="Supply vs Demand Treemap",
            hover_data=["num_vacancies", "num_applications"],
        )
        st.plotly_chart(fig_supply_demand, use_container_width=True, key="supply_demand_treemap")

        # Hidden Demand Quadrant Analysis
        st.markdown('#### The "Hidden Demand"')
        st.caption("Quadrant analysis: High vacancies + Low applications = Hidden opportunities.")

        analysis_type = st.selectbox(
            "Analyze By:", ["Industry", "Job Title", "Skills"],
            key="hidden_demand_analysis_type",
            help="Choose the dimension for quadrant analysis",
        )

        if analysis_type == "Industry":
            hd_metrics = df.groupby("category").agg(
                num_vacancies=("num_vacancies", "sum"),
                num_applications=("num_applications", "sum"),
                min_exp=("min_exp", "mean"),
                job_id=("job_id", "count"),
            ).reset_index().rename(columns={"category": "name"})
            chart_title = "Hidden Demand Quadrant Analysis by Industry"
            hover_label = "Industry"

        elif analysis_type == "Job Title":
            ws = load_withskills_data()
            title_map = ws[["job_id", "jobtitle_cleaned"]].drop_duplicates()
            df_titled = df.merge(title_map, on="job_id", how="inner")
            hd_metrics = df_titled.groupby("jobtitle_cleaned").agg(
                num_vacancies=("num_vacancies", "sum"),
                num_applications=("num_applications", "sum"),
                min_exp=("min_exp", "mean"),
                job_id=("job_id", "count"),
            ).reset_index().rename(columns={"jobtitle_cleaned": "name"})
            chart_title = "Hidden Demand Quadrant Analysis by Job Title"
            hover_label = "Job Title"

        else:  # Skills
            try:
                ws = load_withskills_data()
                skills_with_data = ws.merge(
                    df[["job_id", "num_vacancies", "num_applications", "min_exp"]].drop_duplicates("job_id"),
                    on="job_id", how="left",
                )
                hd_metrics = skills_with_data.groupby("skill").agg(
                    num_vacancies=("num_vacancies", "sum"),
                    num_applications=("num_applications", "sum"),
                    min_exp=("min_exp", "mean"),
                    job_id=("job_id", "count"),
                ).reset_index().rename(columns={"skill": "name"})
                chart_title = "Hidden Demand Quadrant Analysis by Skills"
                hover_label = "Skill"
            except Exception as e:
                st.error(f"Failed to load skills data: {e}")
                hd_metrics = pd.DataFrame()

        if not hd_metrics.empty:
            hd_metrics["opp_score"] = hd_metrics["num_vacancies"] / (hd_metrics["min_exp"] + 1)
            hd_metrics["comp_index"] = hd_metrics.apply(
                lambda x: x["num_applications"] / x["num_vacancies"] if x["num_vacancies"] > 0 else 0,
                axis=1,
            )

        hidden_demand = hd_metrics.copy() if not hd_metrics.empty else pd.DataFrame()

        if len(hidden_demand) > 0:
            sample_size = 50 if analysis_type in ["Job Title", "Skills"] else len(hidden_demand)
            if len(hidden_demand) > sample_size:
                hidden_demand = hidden_demand.nlargest(sample_size, "num_vacancies")

            median_vac = hidden_demand["num_vacancies"].median()
            median_app = hidden_demand["num_applications"].median()

            def assign_quadrant(row):
                if row["num_vacancies"] >= median_vac and row["num_applications"] < median_app:
                    return "Hidden Opportunity"
                elif row["num_vacancies"] >= median_vac and row["num_applications"] >= median_app:
                    return "Competitive Market"
                elif row["num_vacancies"] < median_vac and row["num_applications"] < median_app:
                    return "Niche Market"
                else:
                    return "Oversupplied"

            hidden_demand["quadrant"] = hidden_demand.apply(assign_quadrant, axis=1)
            hidden_demand["display_text"] = hidden_demand.apply(
                lambda row: "" if row["quadrant"] == "Niche Market" else row["name"], axis=1
            )

            fig_hidden = px.scatter(
                hidden_demand, x="num_vacancies", y="num_applications",
                size="num_vacancies", color="quadrant",
                hover_name="name", text="display_text",
                labels={"num_vacancies": "Vacancies", "num_applications": "Applications",
                         "name": hover_label},
                title=chart_title,
                color_discrete_map={
                    "Hidden Opportunity": "#28B463",
                    "Competitive Market": "#E67E22",
                    "Niche Market": "#95A5A6",
                    "Oversupplied": "#E74C3C",
                },
            )
            fig_hidden.update_traces(textposition="top center", textfont_size=8)
            fig_hidden.add_hline(y=median_app, line_dash="dash", line_color="gray",
                                 annotation_text=f"Median Apps: {median_app:.0f}")
            fig_hidden.add_vline(x=median_vac, line_dash="dash", line_color="gray",
                                 annotation_text=f"Median Vacancies: {median_vac:.0f}")
            fig_hidden.update_layout(height=600)
            st.plotly_chart(fig_hidden, use_container_width=True, key="hidden_demand_chart")


# ==========================================
# 5. ENTRY POINT
# ==========================================

def main():
    df = load_exploded_data()
    run_dashboard(df)


if __name__ == "__main__":
    main()
