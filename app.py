# ============================================
# Cybersecurity Analytics Streamlit Dashboard
# ============================================

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# --------------------------------------------
# Application configuration
# --------------------------------------------

st.set_page_config(
    page_title="Cybersecurity Risk Analytics",
    page_icon="🛡️",
    layout="wide"
)

# Make the sidebar wider so full risk-profile names are easier to read.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 360px;
        max-width: 360px;
    }

    [data-testid="stSidebarContent"] {
        width: 360px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIRECTORY = Path(__file__).resolve().parent
EXPORT_DIRECTORY = BASE_DIRECTORY / "exports"

DASHBOARD_DATA_PATH = (
    EXPORT_DIRECTORY / "dashboard_data.csv"
)

MODEL_EVALUATION_PATH = (
    EXPORT_DIRECTORY / "model_evaluation.csv"
)

RISK_SUMMARY_PATH = (
    EXPORT_DIRECTORY / "risk_profile_summary.csv"
)

INPUT_OPTIONS_PATH = (
    BASE_DIRECTORY / "input_options.json"
)

# Uses the local FastAPI server by default.
# Later, this can be replaced with the Render URL.
API_URL = os.getenv(
    "API_URL",
    "https://raiyan-alif-cybersecurity-risk-analytics.onrender.com"
).rstrip("/")


# --------------------------------------------
# Load local data
# --------------------------------------------

@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    return pd.read_csv(DASHBOARD_DATA_PATH)


@st.cache_data
def load_model_evaluation() -> pd.DataFrame:
    return pd.read_csv(MODEL_EVALUATION_PATH)


@st.cache_data
def load_risk_summary() -> pd.DataFrame:
    return pd.read_csv(RISK_SUMMARY_PATH)


@st.cache_data
def load_input_options() -> dict:
    with INPUT_OPTIONS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


try:
    dashboard_df = load_dashboard_data()
    model_evaluation_df = load_model_evaluation()
    risk_summary_df = load_risk_summary()
    input_options = load_input_options()

except Exception as error:
    st.error(
        f"Unable to load the project files: {error}"
    )
    st.stop()


# --------------------------------------------
# Page heading
# --------------------------------------------

st.title("🛡️ Global Cybersecurity Risk Analytics")

st.write(
    "Explore global cybersecurity incidents, examine "
    "risk profiles and estimate incident resolution time."
)


# --------------------------------------------
# API status
# --------------------------------------------

with st.sidebar:

    st.header("API Connection")

    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health_response.status_code == 200:
            st.success("FastAPI connected")
        else:
            st.warning(
                "FastAPI returned an unexpected response."
            )

    except requests.RequestException:
        st.error("FastAPI is not currently reachable.")

    st.caption(f"API: {API_URL}")


# --------------------------------------------
# Dashboard filters
# --------------------------------------------

st.sidebar.header("Dashboard Filters")

minimum_year = int(
    dashboard_df["Year"].min()
)

maximum_year = int(
    dashboard_df["Year"].max()
)

selected_years = st.sidebar.slider(
    "Year range",
    minimum_year,
    maximum_year,
    (
        minimum_year,
        maximum_year
    )
)

attack_types = sorted(
    dashboard_df["Attack Type"]
    .dropna()
    .unique()
    .tolist()
)

selected_attacks = st.sidebar.multiselect(
    "Attack types",
    attack_types,
    default=attack_types
)

risk_profiles = sorted(
    dashboard_df["risk_profile"]
    .dropna()
    .unique()
    .tolist()
)

selected_profiles = st.sidebar.multiselect(
    "Risk profiles",
    risk_profiles,
    default=risk_profiles
)


filtered_df = dashboard_df[
    dashboard_df["Year"].between(
        selected_years[0],
        selected_years[1]
    )
    & dashboard_df["Attack Type"].isin(
        selected_attacks
    )
    & dashboard_df["risk_profile"].isin(
        selected_profiles
    )
].copy()


# --------------------------------------------
# Overview metrics
# --------------------------------------------

st.header("Incident Overview")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

total_financial_loss = (
    filtered_df["Financial Loss (in Million $)"].sum()
    if not filtered_df.empty
    else 0
)

# The source column is measured in millions of US dollars.
# Dividing by 1,000 converts the total to billions for readability.
total_financial_loss_billion = (
    total_financial_loss / 1000
)

average_affected_users = (
    filtered_df["Number of Affected Users"].mean()
    if not filtered_df.empty
    else 0
)

average_resolution_time = (
    filtered_df[
        "Incident Resolution Time (in Hours)"
    ].mean()
    if not filtered_df.empty
    else 0
)

metric_1.metric(
    "Incidents",
    f"{len(filtered_df):,}"
)

metric_2.metric(
    "Total Financial Loss",
    f"${total_financial_loss_billion:,.2f}B"
)

metric_3.metric(
    "Average Affected Users",
    f"{average_affected_users:,.0f}"
)

metric_4.metric(
    "Average Resolution Time",
    f"{average_resolution_time:.2f} hours"
)

if filtered_df.empty:
    st.warning(
        "No records match the selected filters."
    )
else:

    # ----------------------------------------
    # Financial loss over time
    # ----------------------------------------

    yearly_loss_df = (
        filtered_df
        .groupby(
            "Year",
            as_index=False
        )["Financial Loss (in Million $)"]
        .sum()
    )

    yearly_loss_chart = px.line(
        yearly_loss_df,
        x="Year",
        y="Financial Loss (in Million $)",
        markers=True,
        title="Total Financial Loss Over Time"
    )

    st.plotly_chart(
        yearly_loss_chart,
        use_container_width=True
    )


    chart_column_1, chart_column_2 = st.columns(2)


    # ----------------------------------------
    # Attack-type distribution
    # ----------------------------------------

    attack_distribution_df = (
        filtered_df[
            "Attack Type"
        ]
        .value_counts()
        .rename_axis(
            "Attack Type"
        )
        .reset_index(
            name="Incident Count"
        )
    )

    attack_chart = px.bar(
        attack_distribution_df,
        x="Attack Type",
        y="Incident Count",
        title="Attack-Type Distribution"
    )

    chart_column_1.plotly_chart(
        attack_chart,
        use_container_width=True
    )


    # ----------------------------------------
    # Risk-profile breakdown
    # ----------------------------------------

    profile_distribution_df = (
        filtered_df[
            "risk_profile"
        ]
        .value_counts()
        .rename_axis(
            "Risk Profile"
        )
        .reset_index(
            name="Incident Count"
        )
    )

    profile_chart = px.pie(
        profile_distribution_df,
        names="Risk Profile",
        values="Incident Count",
        hole=0.4,
        title="Cyber-Risk Profile Breakdown"
    )

    chart_column_2.plotly_chart(
        profile_chart,
        use_container_width=True
    )


    # ----------------------------------------
    # Loss and user-impact relationship
    # ----------------------------------------

    impact_chart = px.scatter(
        filtered_df,
        x="Number of Affected Users",
        y="Financial Loss (in Million $)",
        color="risk_profile",
        hover_data=[
            "Country",
            "Year",
            "Attack Type",
            "Target Industry"
        ],
        title=(
            "Financial Loss Versus "
            "Number of Affected Users"
        )
    )

    st.plotly_chart(
        impact_chart,
        use_container_width=True
    )


# --------------------------------------------
# Risk-profile summary
# --------------------------------------------

st.header("Risk-Profile Summary")

st.dataframe(
    risk_summary_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------
# Model evaluation
# --------------------------------------------

st.header("Regression Model Evaluation")

st.dataframe(
    model_evaluation_df,
    use_container_width=True,
    hide_index=True
)

st.info(
    "The final model is used as an operational "
    "estimate. Its evaluation showed limited "
    "predictive power, so predictions should not "
    "be treated as guaranteed resolution times."
)


# --------------------------------------------
# Prediction form
# --------------------------------------------

st.header("Predict Incident Resolution Time")

with st.form("prediction_form"):

    form_column_1, form_column_2 = st.columns(2)

    with form_column_1:

        country = st.selectbox(
            "Country",
            input_options["Country"]
        )

        year = st.number_input(
            "Year",
            min_value=2015,
            max_value=2024,
            value=2024,
            step=1
        )

        attack_type = st.selectbox(
            "Attack Type",
            input_options["Attack Type"]
        )

        target_industry = st.selectbox(
            "Target Industry",
            input_options["Target Industry"]
        )

        financial_loss = st.number_input(
            "Financial Loss (Million USD)",
            min_value=0.0,
            value=50.0,
            step=1.0
        )


    with form_column_2:

        affected_users = st.number_input(
            "Number of Affected Users",
            min_value=0,
            value=100000,
            step=1000
        )

        attack_source = st.selectbox(
            "Attack Source",
            input_options["Attack Source"]
        )

        vulnerability = st.selectbox(
            "Security Vulnerability Type",
            input_options[
                "Security Vulnerability Type"
            ]
        )

        defense_mechanism = st.selectbox(
            "Defense Mechanism Used",
            input_options[
                "Defense Mechanism Used"
            ]
        )


    submit_prediction = st.form_submit_button(
        "Predict Resolution Time",
        use_container_width=True
    )


if submit_prediction:

    request_data = {
        "country": country,
        "year": int(year),
        "attack_type": attack_type,
        "target_industry": target_industry,
        "financial_loss_million": float(
            financial_loss
        ),
        "number_of_affected_users": int(
            affected_users
        ),
        "attack_source": attack_source,
        "security_vulnerability_type": (
            vulnerability
        ),
        "defense_mechanism_used": (
            defense_mechanism
        )
    }

    try:
        prediction_response = requests.post(
            f"{API_URL}/predict",
            json=request_data,
            timeout=15
        )

        if prediction_response.status_code == 200:

            prediction_result = (
                prediction_response.json()
            )

            predicted_hours = prediction_result[
                "predicted_resolution_time_hours"
            ]

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Predicted Resolution Time",
                f"{predicted_hours:.2f} hours"
            )

        else:
            st.error(
                "Prediction failed: "
                f"{prediction_response.text}"
            )

    except requests.RequestException as error:
        st.error(
            f"Could not connect to FastAPI: {error}"
        )
