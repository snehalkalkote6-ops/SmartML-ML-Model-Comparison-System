# SmartML: Interactive Machine Learning Model Selection, Evaluation and Comparison Platform

An interactive, analytical web application built for machine learning practitioners, students, and engineers. SmartML provides an end-to-end environment to upload tabular datasets, profile features, automatically verify algorithm compatibility, train Scikit-learn models with strict data leakage prevention, evaluate real statistical performance metrics, and compare multiple models on identical validation splits.

---

## 🌟 Key Features

### 1. Dataset Input, Profiling & Exploration
- **Flexible File Ingestion**: Upload custom `.csv`, `.xlsx`, or `.xls` files, or immediately explore built-in benchmark datasets (**Student Academic Performance** for Regression, **Loan Approval Risk** for Classification).
- **Comprehensive Data Profiling**: Inspect row/column counts, memory footprints, missing cell counts and percentages, duplicate rows, and statistical summaries (`describe()`).
- **Flexible Feature & Target Selection**: Choose any column as the dependent target ($y$) with the remaining columns automatically isolated as candidate predictors ($X$) to prevent data leakage.
- **Intelligent Task Recommendation**: Evaluates target column data types, cardinality, and distribution to recommend **Regression** vs. **Classification**, complete with transparent reasoning.

### 2. Intelligent Model Compatibility Checker
- Evaluates candidate machine learning algorithms against dataset characteristics:
  - Validates minimum feature counts (e.g., Multiple Linear Regression strictly requires $\ge 2$ features).
  - Flags potential feature combinatorial explosions (e.g., Polynomial degree expansion with $> 15$ features).
  - Automatically verifies class distribution, sample sizes, and scaling requirements (e.g., k-NN and SVM requiring standardized continuous features).
- Outlines algorithm mechanics, optimal use cases, and theoretical limitations.

### 3. Leak-Free Preprocessing Pipeline
- **Strict Featurization Ordering**: The dataset is split into training and testing sets **before** fitting any transformation pipelines.
- **Scikit-Learn `ColumnTransformer`**:
  - **Numerical Features**: Imputed using the median and scaled using `StandardScaler`.
  - **Categorical Features**: Imputed using the most frequent value and encoded via `OneHotEncoder(handle_unknown='ignore')`.
  - **Classification Targets**: Transformed via `LabelEncoder` with inverse mapping preserved for inference.
  - **Target Isolation**: Target column is strictly isolated from predictors, eliminating target leakage.

### 4. Real Empirical Evaluation & Interactive Visualizations
- **Regression**:
  - Metrics: **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score (Coefficient of Determination)**.
  - Interactive Plotly Visualizations: **Actual vs. Predicted** (with ideal $y = x$ reference line), **Residual Plot** (against fitted values), and **Residual Distribution Histogram**.
- **Classification**:
  - Metrics: **Accuracy**, **Precision** (Macro & Weighted), **Recall** (Macro & Weighted), **F1-Score** (Macro & Weighted), and **ROC-AUC**.
  - Interactive Plotly Visualizations: **Annotated Confusion Matrix Heatmap** (with counts and cell percentages), and **Multi-Class / Binary ROC Curves** with chance baselines.

### 5. Automated Multi-Model Comparison & Benchmarking
- Trains all compatible algorithms on the **exact same train/test split** to guarantee fair benchmarking.
- Displays a unified comparison table with test metrics, train metrics, and training runtimes.
- Dynamic ranking bar charts allowing sorting by RMSE, MAE, $R^2$, Accuracy, or F1-Score.
- **Overfitting & Generalization Gap Analysis**: Side-by-side grouped charts comparing training scores against held-out test scores.
- Automated architectural trade-off summaries detailing compute, interpretability, and boundary capacity.

### 6. Live Single-Sample Inference
- Automatically generates dynamic input fields matching the feature types and observed dataset ranges.
- Transforms new records through the fitted `ColumnTransformer`.
- Outputs continuous numerical predictions (Regression) or predicted classes with confidence probabilities (Classification).

---

## 🧠 Supported Machine Learning Algorithms

| Prediction Task | Algorithm | Type | Primary Strengths |
| :--- | :--- | :--- | :--- |
| **Regression** | **Linear Regression** | Ordinary Least Squares | High interpretability, analytical baseline |
| **Regression** | **Polynomial Regression** | Basis Expansion + OLS | Captures non-linear curves and synergies |
| **Regression** | **Multiple Regression** | Multi-variable OLS | Models partial effects of multiple predictors |
| **Regression** | **Ridge Regression** | L2 Regularized Linear | Shrinks weights, controls multicollinearity |
| **Regression** | **Random Forest Regressor** | Tree Ensemble Bagging | Captures non-linear relationships, resists outliers |
| **Classification** | **Logistic Regression** | Generalized Linear Model | Calibrated probabilities, fast baseline |
| **Classification** | **k-Nearest Neighbors (k-NN)**| Instance-based Non-parametric | Intuitive distance-based local voting |
| **Classification** | **Support Vector Machine (SVM)**| Maximum Margin (RBF Kernel)| Effective in high-dimensional spaces |
| **Classification** | **HMM** | Generative State-Emission | Models class transitions & Gaussian emissions |
| **Classification** | **Random Forest Classifier** | Tree Ensemble Bagging | High classification power, resists overfitting |

---

## 📁 Project Structure

```
SmartML/
├── app.py                          # Main Streamlit application & Home Dashboard
├── requirements.txt                # Python project dependencies
├── README.md                       # Comprehensive documentation & setup guide
├── pages/
│   ├── 1_Dataset_Explorer.py       # Data upload, profiling, task recommendation
│   ├── 2_Model_Selection.py        # Compatibility checks, hyperparameter tuning, model training
│   ├── 3_Model_Comparison.py       # Multi-model benchmarking on identical splits
│   ├── 4_Make_Predictions.py       # Dynamic input form for live single-record inference
│   └── 5_About.py                  # Educational manual, metric formulas, and ML guide
├── utils/
│   ├── data_loader.py              # File parsing, profiling, and validation
│   ├── preprocessing.py            # Leak-free ColumnTransformer pipeline
│   ├── model_registry.py           # Model taxonomy, compatibility checker, and HMM classifier
│   ├── evaluation.py               # Regression & classification metric calculations
│   ├── visualization.py            # Plotly interactive charting functions
│   ├── recommendations.py          # Evidence-based diagnostics & ranking logic
│   ├── state.py                    # Session state management and model invalidation
│   └── ui_styles.py                # Modern CSS styling, badges, and layout helpers
├── data/
│   ├── sample_regression.csv       # Student Exam Performance dataset (100 rows)
│   └── sample_classification.csv   # Loan Risk Approval dataset (80 rows)
└── tests/
    └── test_smartml.py             # Automated unit and integration test suite
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.13, or 3.14 supported)
- Virtual environment recommended

### 1. Clone or Open the Repository
```bash
cd "d:\JSPM_Btech\Sem 5\ML\SmartML"
```

### 2. Activate Your Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
Verify that all data loaders, preprocessing pipelines, models, and metric calculations operate correctly:
```bash
python -m unittest tests/test_smartml.py
```

### 5. Launch the Web Application

**Option A — Via Terminal:**
```bash
streamlit run app.py
```

**Option B — Windows 1-Click Launchers:**
Double-click `run_app.bat` (automatically detects `.venv`) or `run_simple.bat`.

Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 🧪 Verification & Testing Results

The automated integration test suite (`tests/test_smartml.py`) executes 6 comprehensive test suites covering:
1. **Dataset Loading**: Verifies both benchmark datasets load without corruption.
2. **Profiling & Task Detection**: Validates continuous vs. categorical target detection.
3. **Leak-Free Preprocessing**: Validates train/test split isolation and single-record transformations.
4. **All Regression Models**: Fits and verifies metrics for Linear, Polynomial, Multiple, Ridge, and Random Forest Regressors.
5. **All Classification Models**: Fits and verifies metrics for Logistic Regression, k-NN, SVM, HMM, and Random Forest Classifiers.
6. **Recommendations & Rankings**: Tests empirical overfitting detection and model sorting.

**Result**: `Ran 6 tests in 0.658s — OK (All Passed)`.

---

## 📌 Known Limitations & Best Practices

1. **Dataset Size for Complex Ensembles**: Random Forest and SVM training run in memory. For extremely large datasets (> 100,000 rows), tree depth and estimator counts should be adjusted to manage compute time.
2. **High-Cardinality Categorical Columns**: Categorical features with hundreds of unique text values will create large one-hot sparse matrices. Filtering or frequency encoding is recommended for high-cardinality columns.
3. **Single Train/Test Split**: While held-out validation is standard for initial model selection, small datasets (< 30 samples) exhibit higher variance. Formal deployment should incorporate k-fold cross-validation.
