from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "metrics"

RQ2_DIR = RESULTS / "rq2_controlled_sensor_fault_experiment"
RQ1_DIR = RESULTS / "temperature_scaling_cross_subset"

RQ2_FILE = RQ2_DIR / "rq2_final_results_table.csv"
RQ2_SUMMARY_FILE = RQ2_DIR / "rq2_all_subsets_summary.csv"
RQ1_FILE = RQ1_DIR / "rq1_final_results_table.csv"
BOOTSTRAP_FILE = (
    RQ2_DIR / "uncertainty_error_analysis" /
    "engine_cluster_bootstrap" / "rq2_engine_cluster_bootstrap_results.csv"
)

st.set_page_config(
    page_title="SensorGuard | Predictive Maintenance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #0b1120; color: #e5e7eb; }
    [data-testid="stSidebar"] {
        background: #111a2e;
        border-right: 1px solid #26334d;
    }
    .hero {
        padding: 24px 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #123047, #172033);
        border: 1px solid #2c4968;
        margin-bottom: 20px;
    }
    .hero h1 { color: #f8fafc; margin-bottom: 6px; }
    .hero p { color: #cbd5e1; margin-bottom: 0; }
    .small-note { color: #94a3b8; font-size: 0.88rem; }
    div[data-testid="stMetric"] {
        background: #111a2e;
        border: 1px solid #26334d;
        padding: 16px;
        border-radius: 14px;
    }
    h2, h3 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


rq2 = load_csv(str(RQ2_FILE))
rq2_summary = load_csv(str(RQ2_SUMMARY_FILE))
rq1 = load_csv(str(RQ1_FILE))
bootstrap = load_csv(str(BOOTSTRAP_FILE))


def show_missing_files():
    missing = []
    for path in [RQ2_FILE, RQ2_SUMMARY_FILE, RQ1_FILE]:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        st.warning("Some result files are not available:")
        for item in missing:
            st.code(item)


def metric_value(frame, column):
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def chart_theme(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=48, b=12),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


with st.sidebar:
    st.markdown("## 🛡️ SensorGuard")
    st.caption("Predictive maintenance intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Fault Simulator",
            "Results Explorer",
            "Uncertainty Analysis",
            "Calibration",
            "Project Information",
        ],
    )

    st.divider()
    st.markdown("**Dataset**")
    st.caption("NASA C-MAPSS · FD001–FD004")
    st.markdown("**Mode**")
    st.caption("Saved experiment results")
    st.markdown(
        '<p class="small-note">Live inference is not connected in this version.</p>',
        unsafe_allow_html=True,
    )


st.markdown("""
<div class="hero">
  <h1>🛡️ SensorGuard</h1>
  <p>Robust and uncertainty-aware early fault prediction under
  progressive multi-sensor degradation.</p>
</div>
""", unsafe_allow_html=True)

show_missing_files()

subsets = ["FD001", "FD002", "FD003", "FD004"]

if page == "Overview":
    st.subheader("System overview")
    st.write(
        "Explore saved model evaluations, controlled synthetic sensor faults, "
        "and uncertainty-analysis results across four C-MAPSS subsets."
    )

    a, b, c, d = st.columns(4)
    a.metric("C-MAPSS subsets", "4")
    b.metric("RQ2 conditions", str(len(rq2)) if not rq2.empty else "—")
    c.metric("RQ1 subset summaries", str(len(rq1)) if not rq1.empty else "—")
    d.metric("Bootstrap rows", str(len(bootstrap)) if not bootstrap.empty else "—")

    if not rq2.empty and {"Subset", "F1", "AUROC"}.issubset(rq2.columns):
        st.subheader("Clean-condition model performance")
        clean = rq2[rq2["Scenario"].astype(str).str.lower() == "clean"].copy()
        if clean.empty:
            st.info("No clean-condition rows were found in the RQ2 table.")
        else:
            cols = st.columns(4)
            for i, subset in enumerate(subsets):
                part = clean[clean["Subset"] == subset]
                if part.empty:
                    cols[i].metric(subset, "—", "No clean row")
                else:
                    row = part.iloc[0]
                    f1 = pd.to_numeric(row["F1"], errors="coerce")
                    auroc = pd.to_numeric(row["AUROC"], errors="coerce")
                    cols[i].metric(
                        subset,
                        f"F1 {f1:.3f}" if pd.notna(f1) else "F1 —",
                        f"AUROC {auroc:.3f}" if pd.notna(auroc) else "AUROC —",
                    )

            plot = clean.copy()
            plot["F1"] = pd.to_numeric(plot["F1"], errors="coerce")
            plot["AUROC"] = pd.to_numeric(plot["AUROC"], errors="coerce")
            melted = plot.melt(
                id_vars="Subset",
                value_vars=["F1", "AUROC"],
                var_name="Metric",
                value_name="Score",
            )
            fig = px.bar(
                melted, x="Subset", y="Score", color="Metric",
                barmode="group", title="Clean-condition scores by subset",
                range_y=[0, 1],
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)

    st.subheader("Project modules")
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 🧪 Robustness")
    c1.caption("Controlled synthetic sensor degradation")
    c2.markdown("#### 🎯 Calibration")
    c2.caption("Temperature scaling and probabilistic scores")
    c3.markdown("#### 🔎 Uncertainty")
    c3.caption("Predictive variance and mutual information")


elif page == "Fault Simulator":
    st.subheader("Controlled fault-condition explorer")
    st.info(
        "This page explores previously evaluated synthetic fault conditions. "
        "It does not inject a new fault into a live model or run inference."
    )

    if rq2.empty:
        st.error("RQ2 results CSV is missing. Check the project results directory.")
    else:
        required = {"Subset", "Scenario", "F1", "AUROC"}
        if not required.issubset(rq2.columns):
            st.error(f"RQ2 CSV is missing columns: {sorted(required - set(rq2.columns))}")
        else:
            available_subsets = [x for x in subsets if x in rq2["Subset"].astype(str).unique()]
            chosen_subset = st.selectbox("Select C-MAPSS subset", available_subsets or subsets)
            part = rq2[rq2["Subset"] == chosen_subset].copy()

            scenarios = sorted(part["Scenario"].dropna().astype(str).unique())
            selected_scenarios = st.multiselect(
                "Select conditions to compare",
                scenarios,
                default=scenarios,
            )
            part = part[part["Scenario"].astype(str).isin(selected_scenarios)]

            for col in ["F1", "AUROC", "Brier"]:
                if col in part.columns:
                    part[col] = pd.to_numeric(part[col], errors="coerce")

            if not part.empty:
                score_options = [x for x in ["F1", "AUROC", "Brier"] if x in part.columns]
                selected_metric = st.selectbox("Metric", score_options)
                fig = px.bar(
                    part, x="Scenario", y=selected_metric,
                    color="Scenario", title=f"{chosen_subset}: {selected_metric} by condition",
                    hover_data=[c for c in ["FaultType", "Severity"] if c in part.columns],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(chart_theme(fig), use_container_width=True)
                st.dataframe(part, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download displayed results",
                    part.to_csv(index=False).encode("utf-8"),
                    file_name=f"{chosen_subset.lower()}_fault_results.csv",
                    mime="text/csv",
                )
            else:
                st.info("No conditions match the current selection.")


elif page == "Results Explorer":
    st.subheader("Results explorer")
    if rq2.empty:
        st.error("RQ2 results CSV is not available.")
    else:
        available_subsets = [x for x in subsets if x in rq2["Subset"].astype(str).unique()]
        chosen_subsets = st.multiselect(
            "Subsets",
            available_subsets,
            default=available_subsets,
        )
        part = rq2[rq2["Subset"].isin(chosen_subsets)].copy()

        metric_options = [c for c in ["F1", "AUROC", "Brier", "MeanPredictiveVariance",
                                      "MeanMutualInformation", "ClassificationChangeRate"]
                          if c in part.columns]
        if metric_options:
            selected_metric = st.selectbox("Metric to visualize", metric_options)
            part[selected_metric] = pd.to_numeric(part[selected_metric], errors="coerce")
            fig = px.bar(
                part, x="Scenario", y=selected_metric, color="Subset",
                barmode="group", title=f"{selected_metric} across subsets",
                hover_data=[c for c in ["FaultType", "Severity"] if c in part.columns],
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)

        st.dataframe(part, use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected rows",
            part.to_csv(index=False).encode("utf-8"),
            file_name="sensorguard_selected_results.csv",
            mime="text/csv",
        )


elif page == "Uncertainty Analysis":
    st.subheader("Uncertainty and error-detection analysis")
    st.caption(
        "Predictive variance and mutual information are uncertainty measures. "
        "Their ability to rank prediction errors is evaluated separately."
    )

    if not rq2.empty:
        available_metrics = [
            c for c in [
                "MeanPredictiveVariance", "MeanMutualInformation",
                "MeanPredictiveEntropy", "ClassificationChangeRate",
            ] if c in rq2.columns
        ]
        if available_metrics:
            selected_metric = st.selectbox("Uncertainty measure", available_metrics)
            part = rq2.copy()
            part[selected_metric] = pd.to_numeric(part[selected_metric], errors="coerce")
            fig = px.bar(
                part, x="Scenario", y=selected_metric, color="Subset",
                barmode="group", title=f"{selected_metric} by scenario",
                hover_data=[c for c in ["FaultType", "Severity"] if c in part.columns],
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)

    st.subheader("Engine-cluster bootstrap")
    if bootstrap.empty:
        st.warning("Bootstrap CSV was not found. Other uncertainty results remain available.")
    else:
        st.caption(f"Loaded {len(bootstrap)} bootstrap result rows.")
        st.dataframe(bootstrap, use_container_width=True, hide_index=True)
        st.download_button(
            "Download bootstrap results",
            bootstrap.to_csv(index=False).encode("utf-8"),
            file_name="sensorguard_bootstrap_results.csv",
            mime="text/csv",
        )


elif page == "Calibration":
    st.subheader("Temperature-scaling evaluation")
    st.write(
        "Calibration outcomes are metric-dependent. This page displays the "
        "existing summary counts; it does not fit a new calibration model."
    )

    if rq1.empty:
        st.error("RQ1 summary CSV is not available.")
    else:
        st.dataframe(rq1, use_container_width=True, hide_index=True)

        count_columns = [
            c for c in [
                "ECE_Improved", "ECE_Worsened",
                "LogLoss_Improved", "LogLoss_Worsened",
                "Brier_Improved", "Brier_Worsened",
            ] if c in rq1.columns
        ]
        if count_columns and "Subset" in rq1.columns:
            long = rq1.melt(
                id_vars="Subset", value_vars=count_columns,
                var_name="Outcome", value_name="Scenario count",
            )
            fig = px.bar(
                long, x="Subset", y="Scenario count", color="Outcome",
                barmode="group", title="Calibration metric changes by subset",
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)

    st.subheader("Reliability diagrams")
    image_paths = []
    for subset in subsets:
        image = RQ1_DIR / subset / "temperature_scaling_reliability.png"
        if image.exists():
            image_paths.append((subset, image))

    if image_paths:
        cols = st.columns(2)
        for i, (subset, image) in enumerate(image_paths):
            with cols[i % 2]:
                st.markdown(f"**{subset}**")
                st.image(str(image), use_container_width=True)
    else:
        st.caption("No saved reliability diagrams found at the expected paths.")


else:
    st.subheader("About SensorGuard")
    st.markdown("""
    **Objective:** Evaluate early fault prediction under controlled synthetic
    sensor degradation while examining predictive uncertainty and calibration.

    **Dataset:** NASA C-MAPSS FD001, FD002, FD003, and FD004.

    **Current application scope**
    - Displays previously computed experimental results.
    - Compares fault scenarios and dataset subsets.
    - Presents calibration and uncertainty-analysis outputs.

    **Important limitations**
    - Sensor faults in the controlled experiments are synthetic.
    - C-MAPSS engine degradation labels are not ground-truth sensor-failure labels.
    - This dashboard does not yet provide live model inference.
    - Results shown here are experimental outputs, not real-time equipment telemetry.
    """)

st.divider()
st.caption("SensorGuard · Research prototype · Saved experiment results")
