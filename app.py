import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MPLAD-AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD DATA
# =========================================================

work = joblib.load("mplad_work_clean.pkl")

scaler = joblib.load("mplad_scaler.pkl")

model = joblib.load("mplad_isolation_forest.pkl")

category_medians = joblib.load(
    "mplad_category_medians.pkl"
)

state_medians = joblib.load(
    "mplad_state_medians.pkl"
)

state_delay_medians = joblib.load(
    "mplad_state_delay_medians.pkl"
)


# =========================================================
# MODERN CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Hide Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }


    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }


    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        transition: all 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
    }


    /* Metric label */
    [data-testid="stMetricLabel"] {
        color: #64748b;
    }


    /* Metric value */
    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 700;
    }


    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }


    /* Form */
    [data-testid="stForm"] {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 25px;
        background-color: #ffffff;
    }


    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* Section spacing */
    .section-space {
        margin-top: 25px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    "## MPLAD-AI"
)

section = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Priority Review",
        "Check New Work"
    ]
)


# =========================================================
# NEW WORK ANALYSIS FUNCTION
# =========================================================

def analyze_new_work(new_work):

    # -----------------------------------------------------
    # 1. Validate state
    # -----------------------------------------------------

    if new_work["STATE_NAME"] not in state_medians.index:

        return {
            "error": "Invalid state. Please select a valid state."
        }


    # -----------------------------------------------------
    # 2. Validate category
    # -----------------------------------------------------

    if new_work["WORK_CATEGORY"] not in category_medians.index:

        return {
            "error": "Invalid work category. Please select a valid category."
        }


    # -----------------------------------------------------
    # 3. Validate amount
    # -----------------------------------------------------

    if new_work["SANCTION_AMOUNT"] <= 0:

        return {
            "error": "Sanction amount must be greater than zero."
        }


    # -----------------------------------------------------
    # 4. Validate delay
    # -----------------------------------------------------

    if new_work["SANCTION_DELAY_DAYS"] < 0:

        return {
            "error": "Sanction delay cannot be negative."
        }


    # -----------------------------------------------------
    # 5. Historical reference values
    # -----------------------------------------------------

    category_median = category_medians[
        new_work["WORK_CATEGORY"]
    ]

    state_median = state_medians[
        new_work["STATE_NAME"]
    ]

    state_delay_median = state_delay_medians[
        new_work["STATE_NAME"]
    ]


    # -----------------------------------------------------
    # 6. Cost deviation
    # -----------------------------------------------------

    cost_deviation = (
        new_work["SANCTION_AMOUNT"]
        / category_median
    )

    state_cost_deviation = (
        new_work["SANCTION_AMOUNT"]
        / state_median
    )


    # -----------------------------------------------------
    # 7. Delay deviation
    # -----------------------------------------------------

    state_delay_deviation = (
        new_work["SANCTION_DELAY_DAYS"]
        / state_delay_median
    )


    # -----------------------------------------------------
    # 8. Log transformations
    # -----------------------------------------------------

    log_cost_deviation = np.log1p(
        cost_deviation
    )

    log_state_cost_deviation = np.log1p(
        state_cost_deviation
    )

    log_sanction_delay = np.log1p(
        new_work["SANCTION_DELAY_DAYS"]
    )

    log_state_delay_deviation = np.log1p(
        state_delay_deviation
    )


    # -----------------------------------------------------
    # 9. Description words
    # -----------------------------------------------------

    description_words = len(
        new_work["WORK_DESCRIPTION"].split()
    )


    # -----------------------------------------------------
    # 10. ML input
    # -----------------------------------------------------

    new_ml_data = pd.DataFrame(
        [[
            log_cost_deviation,
            log_state_cost_deviation,
            log_sanction_delay,
            log_state_delay_deviation,
            description_words
        ]],
        columns=[
            "LOG_COST_DEVIATION",
            "LOG_STATE_COST_DEVIATION",
            "LOG_SANCTION_DELAY",
            "LOG_STATE_DELAY_DEVIATION",
            "DESCRIPTION_WORDS"
        ]
    )


    # -----------------------------------------------------
    # 11. Scale
    # -----------------------------------------------------

    new_ml_scaled = scaler.transform(
        new_ml_data
    )


    # -----------------------------------------------------
    # 12. AI prediction
    # -----------------------------------------------------

    new_prediction = model.predict(
        new_ml_scaled
    )[0]

    new_anomaly_score = model.decision_function(
        new_ml_scaled
    )[0]


    # -----------------------------------------------------
    # 13. Risk percentile calculations
    # -----------------------------------------------------

    cost_risk = (
        work["COST_DEVIATION"]
        <= cost_deviation
    ).mean()

    state_cost_risk = (
        work["STATE_COST_DEVIATION"]
        <= state_cost_deviation
    ).mean()

    delay_risk = (
        work["SANCTION_DELAY_DAYS"]
        <= new_work["SANCTION_DELAY_DAYS"]
    ).mean()

    state_delay_risk = (
        work["STATE_DELAY_DEVIATION"]
        <= state_delay_deviation
    ).mean()


    # -----------------------------------------------------
    # 14. Risk components
    # -----------------------------------------------------

    cost_risk_component = (
        cost_risk + state_cost_risk
    ) / 2

    delay_risk_component = (
        delay_risk + state_delay_risk
    ) / 2


    # -----------------------------------------------------
    # 15. Final risk score
    # -----------------------------------------------------

    base_risk_score = (
        (
            cost_risk_component
            + delay_risk_component
        ) / 2
    ) * 100


    # -----------------------------------------------------
    # 16. Risk level
    # -----------------------------------------------------

    if base_risk_score < 36.14:

        risk_level = "Low"

    elif base_risk_score < 49.86:

        risk_level = "Moderate"

    elif base_risk_score < 63.57:

        risk_level = "Elevated"

    else:

        risk_level = "High"


    # -----------------------------------------------------
    # 17. Reasons
    # -----------------------------------------------------

    reasons = []


    if cost_deviation >= 5:

        reasons.append(
            f"Cost is {cost_deviation:.1f}× "
            "the category median."
        )


    if state_cost_deviation >= 3:

        reasons.append(
            f"Cost is {state_cost_deviation:.1f}× "
            "the state median."
        )


    if new_work["SANCTION_DELAY_DAYS"] >= 300:

        reasons.append(
            f"Sanction delay is "
            f"{new_work['SANCTION_DELAY_DAYS']:.0f} days."
        )


    if state_delay_deviation >= 1.5:

        reasons.append(
            f"Sanction delay is "
            f"{state_delay_deviation:.2f}× "
            "the state median delay."
        )


    if len(reasons) == 0 and new_prediction == -1:

        reasons.append(
            "AI detected an unusual combination "
            "of analyzed patterns."
        )


    # -----------------------------------------------------
    # 18. AI status
    # -----------------------------------------------------

    if new_prediction == -1:

        ai_status = "Anomaly Detected"

    else:

        ai_status = "No Anomaly Detected"


    # -----------------------------------------------------
    # 19. Return
    # -----------------------------------------------------

    return {

        "risk_score":
            round(
                float(base_risk_score),
                2
            ),

        "risk_level":
            risk_level,

        "ai_status":
            ai_status,

        "anomaly_score":
            round(
                float(new_anomaly_score),
                6
            ),

        "cost_deviation":
            round(
                float(cost_deviation),
                2
            ),

        "state_cost_deviation":
            round(
                float(state_cost_deviation),
                2
            ),

        "sanction_delay_days":
            new_work[
                "SANCTION_DELAY_DAYS"
            ],

        "state_delay_deviation":
            round(
                float(state_delay_deviation),
                2
            ),

        "reasons":
            reasons
    }


# =========================================================
# OFFICER REVIEW GUIDANCE
# =========================================================

def get_review_action(risk_level, anomaly_label):
    """
    Returns a simple review action for the officer.

    The action is guidance only. It does not approve, reject,
    or confirm fraud for any work.
    """

    if risk_level == "High" and anomaly_label == -1:
        return (
            "Detailed Review",
            "The work has high relative review priority and "
            "the AI detected an unusual pattern in the analyzed data."
        )

    if risk_level == "High" and anomaly_label == 1:
        return (
            "Review Cost & Delay Justification",
            "The work has high relative review priority based on "
            "cost and delay indicators, although the AI did not "
            "detect an unusual overall pattern."
        )

    if risk_level == "Elevated" and anomaly_label == -1:
        return (
            "Review Unusual Pattern",
            "The AI detected an unusual pattern. Review the "
            "displayed indicators and supporting records."
        )

    if risk_level == "Elevated" and anomaly_label == 1:
        return (
            "Review Cost & Delay Indicators",
            "The work has elevated relative review priority. "
            "Check the displayed cost and delay indicators."
        )

    if risk_level == "Moderate" and anomaly_label == -1:
        return (
            "Check Unusual Pattern",
            "The AI detected an unusual pattern. "
            "Review the available supporting information."
        )

    if risk_level == "Moderate" and anomaly_label == 1:
        return (
            "Routine Review",
            "The work has moderate relative review priority "
            "and no unusual AI pattern was detected."
        )

    if anomaly_label == -1:
        return (
            "Check Unusual Pattern",
            "The AI detected an unusual pattern. "
            "Review the available supporting information."
        )

    return (
        "Routine Review",
        "The work has low relative review priority and "
        "no unusual AI pattern was detected."
    )


# =========================================================
# DASHBOARD
# =========================================================


if section == "Dashboard":

    st.title("MPLAD-AI")

    st.subheader(
        "AI-Powered MPLAD Risk Intelligence Platform"
    )

    st.caption(
        "Monitor MPLAD works and identify patterns requiring attention."
    )


    # -----------------------------------------------------
    # Dashboard numbers
    # -----------------------------------------------------

    total_works = len(work)

    anomaly_count = (
        work["ANOMALY_LABEL"] == -1
    ).sum()

    high_count = (
        work["RISK_LEVEL"] == "High"
    ).sum()

    elevated_count = (
        work["RISK_LEVEL"] == "Elevated"
    ).sum()

    moderate_count = (
        work["RISK_LEVEL"] == "Moderate"
    ).sum()

    low_count = (
        work["RISK_LEVEL"] == "Low"
    ).sum()


    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------

    st.markdown("## Overview")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Works",
            f"{total_works:,}",
            "Works analyzed"
        )


    with col2:

        st.metric(
            "AI Anomalies",
            f"{anomaly_count:,}",
            "Unusual patterns detected"
        )


    with col3:

        st.metric(
            "High Priority Works",
            f"{high_count:,}",
            "Relative review priority"
        )


    with col4:

        st.metric(
            "Elevated Priority Works",
            f"{elevated_count:,}",
            "Relative review priority"
        )


    # -----------------------------------------------------
    # Risk distribution
    # -----------------------------------------------------

    st.markdown("## Priority Risk Distribution")

    st.caption(
        "Relative review priority based on cost and delay indicators. "
        "These levels are dataset-relative and do not represent confirmed fraud."
    )


    risk_data = {
        "High": high_count,
        "Elevated": elevated_count,
        "Moderate": moderate_count,
        "Low": low_count
    }


    for risk_name, risk_count in risk_data.items():

        percentage = (
            risk_count / total_works
        )

        col1, col2, col3 = st.columns(
            [1, 5, 1]
        )


        with col1:

            st.write(
                f"**{risk_name}**"
            )


        with col2:

            st.progress(
                percentage
            )


        with col3:

            st.write(
                f"**{risk_count:,}**"
            )


    # -----------------------------------------------------
    # AI anomalies by state
    # -----------------------------------------------------

    st.markdown("## AI Anomalies by State")

    state_anomalies = (
        work[work["ANOMALY_LABEL"] == -1]
        .groupby("STATE_NAME")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    st.caption(
        "Top 10 states by number of AI-detected anomalies. "
        "An anomaly indicates an unusual pattern and does not confirm fraud."
    )

    if len(state_anomalies) > 0:

        st.bar_chart(
            state_anomalies,
            horizontal=True
        )

    else:

        st.info(
            "No AI anomalies were detected."
        )


    # -----------------------------------------------------
    # Key Insights
    # -----------------------------------------------------

    st.markdown("## Key Insights")

    long_delay_count = (
        work["SANCTION_DELAY_DAYS"] >= 300
    ).sum()

    states_covered = (
        work["STATE_NAME"].nunique()
    )

    insight1, insight2, insight3 = st.columns(3)


    with insight1:

        st.metric(
            "AI Anomalies",
            f"{anomaly_count:,}",
            "Unusual patterns detected"
        )


    with insight2:

        st.metric(
            "Long-Delay Works",
            f"{long_delay_count:,}",
            "300+ days to sanction"
        )


    with insight3:

        st.metric(
            "States Covered",
            f"{states_covered:,}",
            "States in the dataset"
        )


    st.caption(
        "These indicators support review prioritization "
        "and do not confirm fraud."
    )


    # -----------------------------------------------------
    # How MPLAD-AI helps
    # -----------------------------------------------------

    st.markdown("## How MPLAD-AI Helps")


    help_col1, help_col2, help_col3 = st.columns(3)


    with help_col1:

        st.info(
            "**01 — Analyze**\n\n"
            "Automatically analyzes MPLAD works "
            "using historical patterns."
        )


    with help_col2:

        st.info(
            "**02 — Prioritize**\n\n"
            "Prioritizes works showing unusual "
            "cost or delay patterns."
        )


    with help_col3:

        st.info(
            "**03 — Explain**\n\n"
            "Provides explainable risk indicators "
            "for officer review."
        )


# =========================================================
# PRIORITY REVIEW
# =========================================================

elif section == "Priority Review":

    st.title("Priority Review")

    st.caption(
        "Review works identified by the AI as unusual "
        "and prioritize them using the risk score."
    )


    # -----------------------------------------------------
    # FILTER OPTIONS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        anomaly_filter = st.selectbox(
            "AI Status",
            [
                "Anomaly Detected",
                "All Works",
                "No Anomaly Detected"
            ],
            index=0
        )


    with col2:

        risk_filter = st.selectbox(
            "Risk Level",
            [
                "All",
                "High",
                "Elevated",
                "Moderate",
                "Low"
            ],
            index=0
        )


    with col3:

        min_risk_score = st.number_input(
            "Minimum Risk Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0
        )


    # -----------------------------------------------------
    # FILTER DATA
    # -----------------------------------------------------

    priority_works = work.copy()


    if anomaly_filter == "Anomaly Detected":

        priority_works = priority_works[
            priority_works["ANOMALY_LABEL"] == -1
        ]


    elif anomaly_filter == "No Anomaly Detected":

        priority_works = priority_works[
            priority_works["ANOMALY_LABEL"] == 1
        ]


    if risk_filter != "All":

        priority_works = priority_works[
            priority_works["RISK_LEVEL"] == risk_filter
        ]


    priority_works = priority_works[
        priority_works["BASE_RISK_SCORE"] >= min_risk_score
    ]


    priority_works = priority_works.sort_values(
        "BASE_RISK_SCORE",
        ascending=False
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    st.markdown("## Review Queue")

    summary1, summary2, summary3 = st.columns(3)


    with summary1:

        st.metric(
            "Works in Queue",
            f"{len(priority_works):,}"
        )


    with summary2:

        queue_anomaly_count = (
            priority_works["ANOMALY_LABEL"] == -1
        ).sum()

        st.metric(
            "AI Anomalies",
            f"{queue_anomaly_count:,}"
        )


    with summary3:

        if len(priority_works) > 0:

            highest_score = priority_works[
                "BASE_RISK_SCORE"
            ].max()

        else:

            highest_score = 0


        st.metric(
            "Highest Risk Score",
            f"{highest_score:.2f}"
        )


    # -----------------------------------------------------
    # WORK TABLE
    # -----------------------------------------------------

    st.markdown("## Works Requiring Review")


    if len(priority_works) > 0:

        display_data = priority_works[
            [
                "STATE_NAME",
                "CONSTITUENCY",
                "SANCTION_AMOUNT",
                "BASE_RISK_SCORE",
                "RISK_LEVEL",
                "AI_STATUS",
                "RISK_REASONS"
            ]
        ].head(25).copy()


        # Work ID comes from the original dataframe index.

        display_data.insert(
            0,
            "WORK_ID",
            display_data.index
        )


        # Show the first explanation as the main reason.

        display_data["MAIN_REASON"] = (
            display_data["RISK_REASONS"]
            .apply(
                lambda reasons:
                reasons[0]
                if isinstance(reasons, list)
                and len(reasons) > 0
                else "AI detected an unusual pattern."
            )
        )


        display_data = display_data.drop(
            columns=["RISK_REASONS"]
        )


        display_data = display_data.rename(
            columns={
                "WORK_ID": "Work ID",
                "STATE_NAME": "State",
                "CONSTITUENCY": "Constituency",
                "SANCTION_AMOUNT": "Sanction Amount",
                "BASE_RISK_SCORE": "Risk Score",
                "RISK_LEVEL": "Risk Level",
                "AI_STATUS": "AI Status",
                "MAIN_REASON": "Main Reason"
            }
        )


        display_data["Sanction Amount"] = (
            display_data["Sanction Amount"]
            .apply(
                lambda x:
                f"₹{x:,.2f}"
            )
        )


        display_data["Risk Score"] = (
            display_data["Risk Score"]
            .apply(
                lambda x:
                f"{x:.2f}"
            )
        )


        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            column_config={

                "Work ID":
                    st.column_config.NumberColumn(
                        "Work ID",
                        width="small"
                    ),

                "State":
                    st.column_config.TextColumn(
                        "State",
                        width="medium"
                    ),

                "Constituency":
                    st.column_config.TextColumn(
                        "Constituency",
                        width="medium"
                    ),

                "Sanction Amount":
                    st.column_config.TextColumn(
                        "Sanction Amount",
                        width="medium"
                    ),

                "Risk Score":
                    st.column_config.TextColumn(
                        "Risk Score",
                        width="small"
                    ),

                "Risk Level":
                    st.column_config.TextColumn(
                        "Risk Level",
                        width="small"
                    ),

                "AI Status":
                    st.column_config.TextColumn(
                        "AI Status",
                        width="medium"
                    ),

                "Main Reason":
                    st.column_config.TextColumn(
                        "Main Reason",
                        width="large"
                    )
            }
        )


        # -------------------------------------------------
        # SELECT WORK
        # -------------------------------------------------

        st.markdown("## Work Details")


        selected_index = st.selectbox(
            "Select a work to inspect",
            priority_works.index.tolist(),
            format_func=lambda x:
            f"Work {x}"
        )


        selected_work = priority_works.loc[
            selected_index
        ]


        # -------------------------------------------------
        # DETAILS
        # -------------------------------------------------

        detail1, detail2, detail3, detail4 = st.columns(
            [1.2, 1.5, 1, 1]
        )


        with detail1:

            st.markdown("**State**")

            st.write(
                selected_work["STATE_NAME"]
            )


        with detail2:

            st.markdown("**Sanction Amount**")

            st.write(
                f"₹{selected_work['SANCTION_AMOUNT']:,.2f}"
            )


        with detail3:

            st.metric(
                "Risk Score",
                f"{selected_work['BASE_RISK_SCORE']:.2f}"
            )


        with detail4:

            st.metric(
                "Risk Level",
                selected_work["RISK_LEVEL"]
            )


        # -------------------------------------------------
        # AI STATUS
        # -------------------------------------------------

        st.markdown("### AI Assessment")


        assessment1, assessment2 = st.columns(2)


        with assessment1:

            st.markdown(
                "#### AI Pattern Status"
            )


            if selected_work["ANOMALY_LABEL"] == -1:

                st.warning(
                    "⚠️ Anomaly Detected"
                )

            else:

                st.success(
                    "✅ No Anomaly Detected"
                )


        with assessment2:

            st.markdown(
                "#### Priority Risk"
            )


            st.info(
                f"Risk Score: "
                f"{selected_work['BASE_RISK_SCORE']:.2f}/100\n\n"
                f"Risk Level: "
                f"{selected_work['RISK_LEVEL']}"
            )


        st.caption(
            "AI Pattern Status identifies unusual patterns "
            "in the analyzed data. Priority Risk represents "
            "relative review priority based on cost and delay "
            "indicators. An anomaly does not mean fraud is confirmed."
        )


        # -------------------------------------------------
        # OFFICER REVIEW GUIDANCE
        # -------------------------------------------------

        review_action, review_message = get_review_action(
            selected_work["RISK_LEVEL"],
            selected_work["ANOMALY_LABEL"]
        )

        st.markdown("### Recommended Review Action")

        st.info(
            f"**{review_action}**\n\n"
            f"{review_message}"
        )

        st.caption(
            "This is review guidance generated from the AI status "
            "and relative priority indicators. The system does not "
            "confirm fraud or make an official decision."
        )


        # -------------------------------------------------
        # REASONS
        # -------------------------------------------------

        if (
            selected_work["RISK_LEVEL"] == "High"
            and selected_work["ANOMALY_LABEL"] == 1
        ):
            reason_heading = (
                "### Why This Work Has High Review Priority"
            )

        elif selected_work["ANOMALY_LABEL"] == -1:
            reason_heading = (
                "### Why This Work Requires Review"
            )

        else:
            reason_heading = (
                "### Assessment Reasons"
            )

        st.markdown(reason_heading)


        reasons = selected_work[
            "RISK_REASONS"
        ]


        if isinstance(reasons, list) and len(reasons) > 0:

            for reason in reasons:

                st.info(reason)

        else:

            st.info(
                "No specific rule-based reason was identified."
            )


    else:

        st.info(
            "No works match the selected filters."
        )


# =========================================================
# CHECK NEW WORK
# =========================================================

elif section == "Check New Work":

    st.title("Check New Work")

    st.caption(
        "Enter work details to generate "
        "an AI-assisted risk assessment."
    )


    # -----------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------

    with st.form("new_work_form"):

        col1, col2 = st.columns(2)


        with col1:

            state = st.selectbox(
                "State",
                sorted(
                    state_medians.index.tolist()
                )
            )


            category = st.selectbox(
                "Work Category",
                sorted(
                    category_medians.index.tolist()
                )
            )


            sanction_amount = st.number_input(
                "Sanction Amount (₹)",
                min_value=1.0,
                value=300000.0,
                step=1000.0
            )


        with col2:

            sanction_delay = st.number_input(
                "Sanction Delay (Days)",
                min_value=0,
                value=0,
                step=1
            )


            description = st.text_area(
                "Work Description",
                height=140,
                placeholder=(
                    "Enter the description of the proposed work..."
                )
            )


        analyze_button = st.form_submit_button(
            "Analyze Work",
            use_container_width=True
        )


    # -----------------------------------------------------
    # RUN ANALYSIS
    # -----------------------------------------------------

    if analyze_button:

        if not description.strip():

            st.error(
                "Please enter a work description."
            )


        else:

            new_work = {

                "STATE_NAME":
                    state,

                "WORK_CATEGORY":
                    category,

                "SANCTION_AMOUNT":
                    sanction_amount,

                "SANCTION_DELAY_DAYS":
                    sanction_delay,

                "WORK_DESCRIPTION":
                    description
            }


            result = analyze_new_work(
                new_work
            )


            if "error" in result:

                st.error(
                    result["error"]
                )


            else:

                st.success(
                    "Work analysis completed successfully."
                )


                # -----------------------------------------
                # RESULT
                # -----------------------------------------

                st.markdown(
                    "## Analysis Result"
                )


                result_col1, result_col2, result_col3 = (
                    st.columns(3)
                )


                with result_col1:

                    st.metric(
                        "Risk Score",
                        result["risk_score"]
                    )


                with result_col2:

                    st.metric(
                        "Risk Level",
                        result["risk_level"]
                    )


                with result_col3:

                    st.metric(
                        "AI Status",
                        result["ai_status"]
                    )


                # -----------------------------------------
                # REASONS
                # -----------------------------------------

                st.markdown(
                    "### Assessment Reasons"
                )


                if result["reasons"]:

                    for reason in result["reasons"]:

                        st.warning(
                            reason
                        )

                else:

                    st.info(
                        "No specific risk reason was identified."
                    )


                # -----------------------------------------
                # ANALYSIS DETAILS
                # -----------------------------------------

                st.markdown(
                    "### Analysis Details"
                )


                detail_col1, detail_col2 = (
                    st.columns(2)
                )


                with detail_col1:

                    st.metric(
                        "Category Cost Deviation",
                        f"{result['cost_deviation']}×"
                    )


                    st.metric(
                        "State Cost Deviation",
                        f"{result['state_cost_deviation']}×"
                    )


                with detail_col2:

                    st.metric(
                        "Sanction Delay",
                        f"{result['sanction_delay_days']} days"
                    )


                    st.metric(
                        "State Delay Deviation",
                        f"{result['state_delay_deviation']}×"
                    )