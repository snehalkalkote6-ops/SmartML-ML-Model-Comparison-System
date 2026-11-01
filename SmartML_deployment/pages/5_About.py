"""
SmartML: Educational Knowledge Base & System Documentation
Page 5: About SmartML
"""

import streamlit as st

st.set_page_config(
    page_title="About SmartML - Knowledge Base",
    page_icon="📚",
    layout="wide"
)

from utils.ui_styles import apply_custom_styles, render_header, render_sidebar_status


def main():
    apply_custom_styles()
    render_sidebar_status()

    render_header(
        title="About SmartML & Machine Learning Guide",
        subtitle="Educational reference manual explaining machine learning workflows, algorithm mechanics, evaluation metric mathematics, and diagnostic principles.",
        icon="📚"
    )

    t1, t2, t3, t4 = st.tabs([
        "📖 Platform Overview",
        "🧠 Algorithm Taxonomy",
        "📐 Evaluation Metrics Manual",
        "🛡️ Data Leakage & Best Practices"
    ])

    with t1:
        st.markdown(
            """
            ### What is SmartML?
            **SmartML** is an interactive, analytical machine learning platform designed to help students, developers, and data practitioners:
            1. **Understand** the core difference between Regression and Classification tasks.
            2. **Verify** algorithm compatibility with real dataset properties before training.
            3. **Execute** real Scikit-learn pipelines with strictly leak-free preprocessing.
            4. **Evaluate** models using rigorous, standardized statistical metrics and interactive Plotly visualizations.
            5. **Compare** candidate models on identical test splits and identify empirical trade-offs.
            6. **Deploy** fitted pipelines for live single-sample inference.

            ---

            ### System Architecture Overview
            ```
            [ User Dataset (CSV/Excel) ]  OR  [ Benchmark Datasets ]
                         │
                         ▼
             [ Data Loader & Profiler ] ──► (Missing checks, constant features, task recommendation)
                         │
                         ▼
            [ Strict Leak-Free Preprocessor ]
                         │  (Split Train/Test BEFORE fitting)
                         ├────────────────────────┐
                         ▼                        ▼
               [ Fit on X_train ]         [ Transform X_test ]
               • Median Imputation        • Median Imputation
               • Standard Scaling         • Standard Scaling
               • One-Hot Encoding         • One-Hot Encoding
                         │                        │
                         ▼                        ▼
                  [ Model Training ] ──► [ Model Evaluation ]
                         │               • MAE, MSE, RMSE, R²
                         │               • Accuracy, F1, ROC-AUC
                         │               • Confusion Matrix & Residuals
                         ▼
             [ Live Inference Pipeline ] ◄── (Single-record transformation)
            ```
            """
        )

    with t2:
        st.markdown(
            """
            ### Supervised Learning Algorithm Taxonomy

            #### 📈 Regression Models
            - **Linear Regression (OLS):**
              - *Formulation:* $\\hat{y} = \\beta_0 + \\beta_1 X_1 + \\dots + \\beta_p X_p$
              - *Mechanics:* Fits coefficients by minimizing the Residual Sum of Squares (RSS).
              - *Strengths:* Highly interpretable, zero hyperparameter tuning, fast training.
              - *Limitations:* Vulnerable to outliers, assumes strictly linear relationships.
            - **Polynomial Regression:**
              - *Formulation:* $\\hat{y} = \\beta_0 + \\beta_1 X + \\beta_2 X^2 + \\dots + \\beta_d X^d$
              - *Mechanics:* Applies a non-linear feature transformation before fitting OLS.
              - *Strengths:* Models curves, acceleration, and non-linear interactions.
              - *Limitations:* Combinatorial feature explosion with high dimensions; prone to extreme overfitting.
            - **Multiple Regression:**
              - *Formulation:* OLS regression specifically involving $p \\ge 2$ independent features.
              - *Strengths:* Quantifies partial regression coefficients holding other variables constant.
              - *Limitations:* Requires $\\ge 2$ features; vulnerable to multicollinearity.
            - **Ridge Regression (L2 Regularization):**
              - *Formulation:* Minimizes $\\text{RSS} + \\alpha \\sum_{j=1}^p \\beta_j^2$
              - *Mechanics:* Penalizes large weights, shrinking coefficients toward zero.
              - *Strengths:* Prevents overfitting on collinear or high-variance features.
              - *Limitations:* Requires feature standardization; does not perform hard feature elimination.
            - **Random Forest Regressor:**
              - *Formulation:* Ensemble averaging over $B$ bootstrap decision trees: $\\hat{y} = \\frac{1}{B} \\sum_{b=1}^B T_b(x)$
              - *Strengths:* Models non-linear interactions effortlessly, resistant to outliers, handles mixed scales.
              - *Limitations:* Black-box predictions, memory intensive, cannot extrapolate beyond training extremes.

            ---

            #### 🎯 Classification Models
            - **Logistic Regression:**
              - *Formulation:* $P(y=1|X) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta^T X)}}$
              - *Mechanics:* Models the log-odds (logit) using the logistic sigmoid link function.
              - *Strengths:* Calibrated class probabilities, fast, interpretable odds-ratios.
              - *Limitations:* Cannot model non-linear decision boundaries without manual feature transformations.
            - **k-Nearest Neighbors (k-NN):**
              - *Formulation:* $\\hat{y} = \\text{mode}(\\{y_i : x_i \\in \\mathcal{N}_k(x)\\})$
              - *Mechanics:* Distance-based instance learning using Euclidean distance in scaled space.
              - *Strengths:* Non-parametric, intuitive, adapts to irregular multi-modal cluster boundaries.
              - *Limitations:* Sensitive to the curse of dimensionality; inference scales linearly with dataset size ($O(N)$).
            - **Support Vector Machine (SVM):**
              - *Formulation:* Maximizes margin $\\frac{2}{\\|w\\|}$ subject to $y_i (w^T \\phi(x_i) + b) \\ge 1 - \\xi_i$
              - *Mechanics:* Projects features into infinite-dimensional Hilbert space via the Radial Basis Function (RBF) kernel.
              - *Strengths:* Highly effective in high dimensions, robust against points far outside the margin.
              - *Limitations:* Computationally heavy ($O(N^2)$ to $O(N^3)$), requires careful feature scaling.
            - **HMM (Hidden Markov / State-Emission Classifier):**
              - *Formulation:* Computes joint posterior $P(S_t = c | X_t) \\propto P(X_t | S_t = c) \\cdot P(S_t = c | S_{t-1})$
              - *Mechanics:* Combines class-conditional Gaussian feature emission densities with Markovian transition probabilities between records.
              - *Strengths:* Models sequential structure, temporal continuity, and generative probabilistic distributions.
              - *Limitations:* Assumes feature independence conditioned on state and stationary Markov transitions.
            - **Random Forest Classifier:**
              - *Formulation:* Majority voting over $B$ randomized decision trees.
              - *Strengths:* Exceptional tabular classification accuracy, resists noise, captures complex interactions.
              - *Limitations:* Slower inference than linear models, larger memory footprint.
            """
        )

    with t3:
        st.markdown(
            """
            ### Evaluation Metrics: Formulas & Interpretation

            #### 📈 Regression Metrics
            | Metric | Formula | Interpretation |
            | :--- | :--- | :--- |
            | **Mean Absolute Error (MAE)** | $\\frac{1}{N} \\sum_{i=1}^N \\|y_i - \\hat{y}_i\\|$ | Average magnitude of error in original target units. Robust to extreme outliers. |
            | **Mean Squared Error (MSE)** | $\\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$ | Average squared residual. Heavily penalizes large prediction mistakes. |
            | **Root Mean Squared Error (RMSE)** | $\\sqrt{\\text{MSE}}$ | Square root of MSE. Interpretable in original units while penalizing outliers. |
            | **R² Score (Determination)** | $1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$ | Percentage of target variance captured by the model. 1.0 is perfect; negative is worse than baseline mean. |

            ---

            #### 🎯 Classification Metrics
            | Metric | Formula | Interpretation |
            | :--- | :--- | :--- |
            | **Accuracy** | $\\frac{TP + TN}{TP + TN + FP + FN}$ | Overall fraction of correct predictions. Misleading if classes are heavily imbalanced. |
            | **Precision** | $\\frac{TP}{TP + FP}$ | Fraction of positive predictions that were true positives (avoids False Alarms). |
            | **Recall (Sensitivity)** | $\\frac{TP}{TP + FN}$ | Fraction of actual positive samples identified (avoids Missed Detections). |
            | **F1-Score** | $2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ | Harmonic mean of precision and recall. Ideal for imbalanced datasets. |
            | **ROC-AUC** | $\\int_0^1 \\text{TPR}(FPR) dFPR$ | Ability of classifier to rank positive cases higher than negative cases across all thresholds. |

            #### Averaging Strategies for Multiclass:
            - **Macro Average:** Computes metric independently per class and averages equally: $\\frac{1}{K} \\sum_{c=1}^K M_c$. Treats all classes equally regardless of frequency.
            - **Weighted Average:** Weights metric for each class by its support (number of true instances): $\\sum_{c=1}^K \\frac{N_c}{N} M_c$. Reflects real-world class prevalence.
            """
        )

    with t4:
        st.markdown(
            """
            ### Preventing Data Leakage & Machine Learning Best Practices

            #### 1. Strict Featurization Ordering
            A critical error in naive machine learning workflows is fitting transformations (such as `StandardScaler`, `SimpleImputer`, or `OneHotEncoder`) on the entire dataset before splitting.
            - **Why is it dangerous?** The scaler learns the global mean $\\mu$ and variance $\\sigma^2$ of the test set, creating optimistic bias and unrealistic benchmark metrics.
            - **The SmartML Solution:** The dataset is split into training and testing sets **first**. The `ColumnTransformer` is fitted strictly on `X_train`, and then used to transform both `X_train` and `X_test`.

            #### 2. Target Leakage Isolation
            - When predicting a target column, any feature that directly encodes or is derived after the target event must be excluded.
            - SmartML automatically removes the target column from candidate feature inputs.

            #### 3. Generalization Gap (Overfitting Diagnostics)
            - Comparing test metrics alone can mask severe overfitting.
            - SmartML computes metrics on both the training split and the held-out test split.
            - If Training $R^2 = 0.99$ but Test $R^2 = 0.55$, the model has memorized sample noise rather than learning generalizable signals.

            #### 4. Handling Missing Values and Categorical Encoders
            - Missing numbers are imputed with the median to resist extreme outliers.
            - Categorical columns are imputed with the most frequent category and transformed using one-hot encoding with `handle_unknown='ignore'`.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748B; font-size: 0.85rem;">
            <b>SmartML v2.0 Platform</b> &bull; Third-Year B.Tech Computer Engineering Machine Learning Mini-Project
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
