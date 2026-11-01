"""
SmartML: Interactive Machine Learning Model Selection, Evaluation and Comparison Platform
Home Dashboard & Application Entrypoint.
"""

import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="SmartML - Machine Learning Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.ui_styles import apply_custom_styles, render_header, render_sidebar_status, render_metric_card
from utils.state import init_session_state, set_new_dataset
from utils.data_loader import load_sample_dataset


def main():
    apply_custom_styles()
    init_session_state()
    render_sidebar_status()

    # Main Hero Header
    render_header(
        title="SmartML Platform",
        subtitle="Interactive Machine Learning Model Selection, Empirical Evaluation, and Comparative Analysis System. Perform real machine learning workflows on your datasets.",
        icon="🤖"
    )

    # Status Overview Grid
    st.markdown("### 📌 Workspace Overview")
    df = st.session_state.get("df")
    task = st.session_state.get("task", "None")
    target = st.session_state.get("target_col", "None")
    trained = st.session_state.get("is_model_trained", False)
    trained_model = st.session_state.get("trained_model_name", "None")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if df is not None:
            render_metric_card("Dataset Status", f"{len(df)} Rows", f"{len(df.columns)} Columns", "Loaded", "green")
        else:
            render_metric_card("Dataset Status", "None", "No data loaded yet", "Pending", "amber")

    with col2:
        render_metric_card("Active Task", task.upper(), f"Target: {target if target else 'None'}", "Supervised", "blue")

    with col3:
        if trained:
            render_metric_card("Trained Model", trained_model, "Ready for evaluation", "Active", "green")
        else:
            render_metric_card("Trained Model", "Untrained", "No model fitted", "Pending", "amber")

    with col4:
        comp_df = st.session_state.get("comparison_df")
        comp_count = len(comp_df) if comp_df is not None else 0
        render_metric_card("Comparison Pool", f"{comp_count} Models", "Evaluated on same split", "Ready" if comp_count > 0 else "Pending", "purple" if comp_count > 0 else "amber")

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Start Sample Data Loader
    st.markdown("### ⚡ Quick Start: Load Benchmark Datasets")
    st.markdown("Explore the platform immediately using verified, real-world benchmark datasets:")

    qc1, qc2 = st.columns(2)
    with qc1:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
                <h4 style="margin: 0 0 0.4rem 0; color: #1E3A8A;">📈 Regression Benchmark: Student Performance</h4>
                <p style="margin: 0 0 0.8rem 0; font-size: 0.86rem; color: #475569;">
                    Predict continuous exam scores (0–100) based on study hours, attendance rate, sleep duration, and previous grades.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Load Student Performance Dataset", key="btn_load_sample_reg", use_container_width=True):
            sample_df, sample_target, sample_task, desc = load_sample_dataset("regression")
            if sample_df is not None:
                set_new_dataset(sample_df, "Sample Regression: Student Performance", sample_target, sample_task)
                st.success("Loaded Student Performance Regression dataset! Head to 'Dataset Explorer' or 'Model Selection'.")
                st.rerun()

    with qc2:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
                <h4 style="margin: 0 0 0.4rem 0; color: #1E3A8A;">🎯 Classification Benchmark: Loan Risk Approval</h4>
                <p style="margin: 0 0 0.8rem 0; font-size: 0.86rem; color: #475569;">
                    Predict binary loan approval ('Approved' vs. 'Rejected') from credit score, annual income, debt ratio, and education level.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Load Loan Approval Dataset", key="btn_load_sample_clf", use_container_width=True):
            sample_df, sample_target, sample_task, desc = load_sample_dataset("classification")
            if sample_df is not None:
                set_new_dataset(sample_df, "Sample Classification: Loan Approval", sample_target, sample_task)
                st.success("Loaded Loan Approval Classification dataset! Head to 'Dataset Explorer' or 'Model Selection'.")
                st.rerun()

    st.markdown("---")

    # Interactive Navigation Cards
    st.markdown("### 🧭 Interactive Workflow Guide")
    st.markdown("Follow the structured machine learning lifecycle:")

    nav_c1, nav_c2, nav_c3, nav_c4 = st.columns(4)

    with nav_c1:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>1. Dataset Explorer</h4>
                <p>Upload CSV/Excel data, inspect column types, missing values, duplicates, and select target/features.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.page_link("pages/1_Dataset_Explorer.py", label="Open Dataset Explorer", icon="📊")

    with nav_c2:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>2. Model Selection</h4>
                <p>Validate model compatibility, train algorithms, view real metrics, residuals, and confusion matrices.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.page_link("pages/2_Model_Selection.py", label="Open Model Selection", icon="⚙️")

    with nav_c3:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>3. Model Comparison</h4>
                <p>Train all compatible algorithms on identical test splits, compare bar charts, and view rankings.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.page_link("pages/3_Model_Comparison.py", label="Open Model Comparison", icon="📈")

    with nav_c4:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>4. Make Predictions</h4>
                <p>Supply custom values for new unseen samples and perform live inference with confidence scores.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.page_link("pages/4_Make_Predictions.py", label="Open Make Predictions", icon="🔮")

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Features & Educational Principles
    st.markdown("### 🎓 Core Principles of SmartML")
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown(
            """
            **🔬 Zero Data Leakage**
            - Strict featurization ordering.
            - Train/test splitting occurs *before* fitting scalers or encoders.
            - Target column is strictly isolated from predictors.
            """
        )
    with exp2:
        st.markdown(
            """
            **📊 Real Computations**
            - No synthetic or hardcoded metrics.
            - All scores, residuals, and curves are calculated dynamically at runtime from Scikit-Learn models.
            """
        )
    with exp3:
        st.markdown(
            """
            **⚖️ Fair Comparison**
            - All competing algorithms are trained and tested on the exact same held-out validation split.
            - Generalization gaps (train vs. test) are clearly visualized.
            """
        )

    # Footer
    st.markdown(
        """
        <div class="footer-note">
            SmartML &bull; Interactive Machine Learning Model Selection, Evaluation and Comparison Platform
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
