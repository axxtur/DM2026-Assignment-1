# Project Setup and Implementation Log

This document summarizes the steps taken to resolve environment issues and complete missing implementations for the Data Mining assignment.

## 1. Environment Setup & Dependency Resolution

### Kernel Issues
- **Problem:** Jupyter kernel failed to start due to missing `pygments` module.
- **Solution:** Installed `pygments` and `ipykernel` in the Python 3.11 environment.

### Missing Libraries
- **Problem:** Several required Data Science libraries were missing.
- **Solution:** Installed the following packages via pip:
    - `scikit-learn`
    - `scipy`
    - `seaborn`
    - `graphviz` (Python interface)

## 2. Code Debugging & Implementation

### ModuleNotFoundError: 'graphviz'
- **Problem:** Importing `model.utils` failed because `graphviz` was not installed.
- **Solution:** Installed the `graphviz` Python package.

### ImportError: 'MAE_grad' missing
- **Problem:** `from model.gradients import MAE_grad` failed because the function was not defined in `model/gradients.py`.
- **Solution:** 
    - Analyzed `model/gradients.py`.
    - Implemented the `MAE_grad` function (Mean Absolute Error gradient).

### Placeholder Implementations in `model/metrics.py`
- **Problem:** Several core functions were marked as `TODO` or `pass`.
- **Solution:** Fully implemented the following functions:
    - `MAE(y, y_pred)`: Added logic for Mean Absolute Error.
    - `evaluate_linear_regression(y_true, y_pred)`: Integrated `sklearn.metrics` to compute MSE, MAE, RMSE, and R-squared.
    - `evaluate_binary_classifier(y_true, y_pred)`: Integrated `sklearn.metrics` for Accuracy, Precision, Recall, and F1-score, and added a Confusion Matrix visualization using `matplotlib`.

## 3. Linear Regression Task Execution

The task involved testing the linear regression model across multiple scenarios:
- **Datasets:** Evaluated datasets `linear_data_A.npz`, `linear_data_B.npz`, `linear_data_C.npz`, and `linear_data_D.npz`.
- **Hyperparameters:** For each dataset, tested three different learning rates: `0.1`, `0.01`, and `0.001`.
- **Validation:** Recorded results via screenshots for each combination, capturing:
    - The regression plot (Loss curve / Fit).
    - The 4 core metrics (MSE, MAE, RMSE, R-squared).
- **Organization:** Results were structured into a directory hierarchy: `Linear_Regression/Dataset_[A-D]/LR_[Value]/`.

## 4. Logistic Regression Task Execution

Similar to the linear regression task, the logistic model was evaluated:
- **Datasets:** Evaluated datasets `logistic_data_A.npz`, `logistic_data_B.npz`, `logistic_data_C.npz`, and `logistic_data_D.npz`.
- **Hyperparameters:** Tested learning rates `0.1`, `0.01`, and `0.001`.
- **Extended Testing:** Conducted additional runs with varying iteration counts (`500`, `1000`, `1500`) to observe convergence behavior.
- **Validation:** Captured screenshots for each run, including:
    - The decision boundary or loss curve.
    - The Confusion Matrix and classification metrics (Accuracy, Precision, Recall, F1-score).
- **Organization:** Results were stored in `logistic_regression/logistic_data_[A-D]/`.

## 5. Real World Classification (Iris Dataset)

### Task 3: Data Preprocessing
- **Label Encoding:** Successfully converted the target 'Species' into numerical format.
- **Missing Values (3a & 3b):** 
    - Implemented `KNNImputer(n_neighbors=5)` to handle missing values in columns like `SepalWidthCm`, `PetalLengthCm`, etc.
    - Reported `median` and `std` before and after imputation.
    - **Observation:** The median remained robust, while the standard deviation decreased slightly, confirming that KNN imputation preserves the central tendency while smoothing variance.

### Task 4: Data Exploration
- **Visualization (4a):** Plotted a histogram for `PetalWidthCm`.
    - **Observation:** The distribution is bimodal, indicating distinct clusters that likely correspond to different species within the dataset.
- **Pearson Correlation (4b & 4c):**
    - Used `sklearn.feature_selection.r_regression` to identify relationships specifically with the `PetalWidthCm` column.
    - Identified the feature with the largest positive correlation (excluding `Species`, `Id`, and `PetalWidthCm`).
    - Identified the top 5 features with the strongest negative correlations.
- **Boxplot Analysis (4d):** Created boxplots for all features identified in 4b and 4c to visualize distributions and identify outliers.

## 6. Task 5: Regularization Analysis

### L2 Regularization Comparison
- **Experiment Setup (5a):** Compared the standard logistic regression (no regularization) with L2 regularization using varying `reg_lambda` values: `0.01`, `1`, and `100`.
- **Loss Curve Analysis:** 
    - Plotted four comparative loss curves.
    - **Observation:** Low lambda values (0.01) showed minimal impact, behaving similarly to no regularization. Moderate values (1) stabilized the gap between training and validation loss. High values (100) led to underfitting, indicated by significantly higher overall loss.
- **Performance Metrics (5b):** Evaluated models using `evaluate_binary_classifier`.
    - Captured Accuracy, Precision, Recall, and F1-score.
    - Comparison showed that while small regularization preserves high metrics, excessive regularization degrades predictive power.

## 7. Final Summary
- **Linear Regression:** 12 scenarios (4 datasets x 3 LRs) documented with plots and metrics (MSE, MAE, RMSE, R2).
- **Logistic Regression:** 12 scenarios documented with decision boundaries, confusion matrices, and metrics.
- **Real-World Application:** Successfully applied preprocessing (KNN Imputation), exploration (Correlation/Boxplots), and regularization tuning to the Iris dataset.
- **Code Integrity:** All `TODO` sections in `model/` and notebooks are resolved. Final verification of requirements for Assignment 1 completed.
