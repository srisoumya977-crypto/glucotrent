# ============================================================
# GLUCOTREND - REGULARISATION
# Ridge (L2), Lasso (L1), Elastic Net
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
   accuracy_score,
   precision_score,
   recall_score,
   f1_score,
   classification_report
)

# ============================================================
# 1. FILE SETTINGS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_NAME = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
output_folder = os.path.join(BASE_DIR, "outputs", "Regularization_ridg_laso_elastic_outputs")
TARGET_COLUMN = "target_binary"

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# 2. READ PREPROCESSED DATASET
# ============================================================
df_original = pd.read_csv(FILE_NAME)

print("\n================================================")
print("ORIGINAL PREPROCESSED DATASET")
print("================================================")
print(df_original.head())
print("\nShape:", df_original.shape)
print("\nColumns:")
print(df_original.columns.tolist())

# Create copy
df = df_original.copy(deep=True)

# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================
print("\n================================================")
print("MISSING VALUE CHECK")
print("================================================")
print(df.isnull().sum())

df_model = df.copy(deep=True)
if df_model.isnull().sum().sum() > 0:
   print("\nMissing values detected. Cleaning model copy...")
   df_model = df_model.dropna().copy()

# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================
ALL_TARGET_CANDIDATES = [
    "glucose_lead_1", "glucose_lead_3", "glucose_lead_6",
    "target_binary", "target_ada3", "target_5class", "target_trend"
]

X = df_model.drop(columns=ALL_TARGET_CANDIDATES, errors="ignore").copy()
y = df_model[TARGET_COLUMN].copy()

# ============================================================
# 5. CONVERT TARGET TO 0 AND 1
# ============================================================
print("\n================================================")
print("TARGET VALUES")
print("================================================")
print(y.unique())

if y.dtype == "object":
   y = y.astype(str).str.strip().str.lower()
   target_mapping = {
       "yes": 1,
       "no": 0,
       "placed": 1,
       "not placed": 0,
       "true": 1,
       "false": 0,
       "in_range": 0,
       "out_of_range": 1
   }
   y = y.map(target_mapping)
elif y.dtype == bool:
   y = y.astype(int)
else:
   unique_values = sorted(y.dropna().unique())
   if set(unique_values) != {0, 1}:
       if len(unique_values) == 2:
           mapping = {unique_values[0]: 0, unique_values[1]: 1}
           y = y.map(mapping)

if y.isnull().any():
   raise ValueError("Target values could not be converted to 0 and 1.")

y = y.astype(int)
print("\nTarget distribution:")
print(y.value_counts())

# Check features are numeric
non_numeric_columns = X.select_dtypes(exclude=np.number).columns.tolist()
if len(non_numeric_columns) > 0:
   print("\nNon-numeric columns found:", non_numeric_columns)
   raise ValueError("Your preprocessed features contain non-numeric columns.")

print("\nAll predictor variables are numeric.")

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n================================================")
print("TRAIN-TEST SPLIT")
print("================================================")
print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# ============================================================
# 7. MODEL TRAINING
# ============================================================
# Ridge (L2)
print("\n================================================")
print("RIDGE REGULARISATION - L2")
print("================================================")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_values = ridge.predict(X_test)
ridge_predictions = (ridge_values >= 0.5).astype(int)
print("Ridge model trained successfully.")

# Lasso (L1)
print("\n================================================")
print("LASSO REGULARISATION - L1")
print("================================================")
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_values = lasso.predict(X_test)
lasso_predictions = (lasso_values >= 0.5).astype(int)
print("Lasso model trained successfully.")

# Elastic Net
print("\n================================================")
print("ELASTIC NET")
print("================================================")
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
elastic.fit(X_train, y_train)
elastic_values = elastic.predict(X_test)
elastic_predictions = (elastic_values >= 0.5).astype(int)
print("Elastic Net model trained successfully.")

# ============================================================
# 8. EVALUATION
# ============================================================
def evaluate_model(model_name, y_actual, y_predicted):
   accuracy = accuracy_score(y_actual, y_predicted)
   precision = precision_score(y_actual, y_predicted, zero_division=0)
   recall = recall_score(y_actual, y_predicted, zero_division=0)
   f1 = f1_score(y_actual, y_predicted, zero_division=0)
   
   print("\n--------------------------------------------")
   print(model_name)
   print("--------------------------------------------")
   print("Accuracy :", round(accuracy, 4))
   print("Precision:", round(precision, 4))
   print("Recall   :", round(recall, 4))
   print("F1 Score :", round(f1, 4))
   print("\nClassification Report:")
   print(classification_report(y_actual, y_predicted, zero_division=0))
   
   return [accuracy, precision, recall, f1]

ridge_results = evaluate_model("Ridge (L2)", y_test, ridge_predictions)
lasso_results = evaluate_model("Lasso (L1)", y_test, lasso_predictions)
elastic_results = evaluate_model("Elastic Net", y_test, elastic_predictions)

# Comparison Table
comparison = pd.DataFrame({
   "Model": ["Ridge (L2)", "Lasso (L1)", "Elastic Net"],
   "Accuracy": [ridge_results[0], lasso_results[0], elastic_results[0]],
   "Precision": [ridge_results[1], lasso_results[1], elastic_results[1]],
   "Recall": [ridge_results[2], lasso_results[2], elastic_results[2]],
   "F1_Score": [ridge_results[3], lasso_results[3], elastic_results[3]]
})

print("\n================================================")
print("REGULARISATION MODEL COMPARISON")
print("================================================")
print(comparison)

# Save Comparison CSV
comparison_file = os.path.join(BASE_DIR, "dataset", "regularisation_model_comparison_M2.csv")
comparison.to_csv(comparison_file, index=False)
print(f"\nSaved: {comparison_file}")

# Coefficients Table
coefficient_table = pd.DataFrame({
   "Feature": X.columns,
   "Ridge_L2": ridge.coef_,
   "Lasso_L1": lasso.coef_,
   "Elastic_Net": elastic.coef_
})

print("\n================================================")
print("REGULARISATION COEFFICIENTS")
print("================================================")
print(coefficient_table)

# Save Coefficients CSV
coefficients_file = os.path.join(BASE_DIR, "dataset", "regularisation_coefficients_M2.csv")
coefficient_table.to_csv(coefficients_file, index=False)
print(f"\nSaved: {coefficients_file}")

# ============================================================
# 9. LASSO SPARSITY AND SELECTION
# ============================================================
zero_lasso = coefficient_table[np.isclose(coefficient_table["Lasso_L1"], 0, atol=1e-6)].copy()

print("\n================================================")
print("LASSO SPARSITY / FEATURE SELECTION")
print("================================================")
print("Total features:", len(X.columns))
print("Zero Lasso coefficients:", len(zero_lasso))
print("Non-zero Lasso coefficients:", len(X.columns) - len(zero_lasso))

if len(zero_lasso) > 0:
   print("\nFeatures whose Lasso coefficient became zero:")
   print(zero_lasso[["Feature", "Lasso_L1"]].to_string(index=False))
else:
   print("\nNo coefficients became exactly zero.")

# Save Lasso Feature Selection
lasso_features = coefficient_table.copy()
lasso_features["Selected_by_Lasso"] = (np.abs(lasso_features["Lasso_L1"]) > 1e-6)
lasso_selection_file = os.path.join(BASE_DIR, "dataset", "lasso_feature_selection_M2.csv")
lasso_features.to_csv(lasso_selection_file, index=False)
print(f"\nSaved: {lasso_selection_file}")

# ============================================================
# 10. PLOTS
# ============================================================
# 1. Geometric Diamond vs Circle (L1 vs L2)
beta1 = np.linspace(-2, 2, 500)
beta2 = np.linspace(-2, 2, 500)
BETA1, BETA2 = np.meshgrid(beta1, beta2)

C = 1.5
L1 = np.abs(BETA1) + np.abs(BETA2)
L2 = BETA1 ** 2 + BETA2 ** 2

plt.figure(figsize=(8, 8))
plt.contour(BETA1, BETA2, L1, levels=[C], colors="red", linewidths=2.5, linestyles="-")
plt.contour(BETA1, BETA2, L2, levels=[C ** 2], colors="blue", linewidths=2.5, linestyles="-")
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Coefficient β1")
plt.ylabel("Coefficient β2")
plt.title("Geometric Picture of L1 and L2 Regularisation")
plt.text(1.15, 0.05, "L1\nDiamond (Lasso)", fontsize=12, color="red")
plt.text(0.45, 1.25, "L2\nCircle (Ridge)", fontsize=12, color="blue")
plt.grid(True)
plt.axis("equal")
plt.savefig(os.path.join(output_folder, "L1_vs_L2_Regularisation.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Lasso Sparsity Graph
plt.figure(figsize=(12, 6))
feature_numbers = np.arange(len(X.columns))
plt.stem(feature_numbers, lasso.coef_)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Feature Number")
plt.ylabel("Lasso Coefficient")
plt.title("Lasso L1 Regularisation - Feature Sparsity")
plt.xticks(feature_numbers, X.columns, rotation=90)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Lasso_Sparsity_Graph.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3. Coefficient Comparison Graph
plt.figure(figsize=(14, 7))
x = np.arange(len(X.columns))
width = 0.25
plt.bar(x - width, ridge.coef_, width, label="Ridge (L2)", color="skyblue")
plt.bar(x, lasso.coef_, width, label="Lasso (L1)", color="coral")
plt.bar(x + width, elastic.coef_, width, label="Elastic Net", color="lightgreen")
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Ridge vs Lasso vs Elastic Net Coefficients")
plt.xticks(x, X.columns, rotation=90)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Coefficient_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# 4. Model Performance Graph
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
plt.figure(figsize=(10, 6))
x_m = np.arange(len(metrics))
plt.bar(x_m - width, ridge_results, width, label="Ridge", color="skyblue")
plt.bar(x_m, lasso_results, width, label="Lasso", color="coral")
plt.bar(x_m + width, elastic_results, width, label="Elastic Net", color="lightgreen")
plt.xticks(x_m, metrics)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Regularisation Model Performance Comparison")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Model_Performance.png"), dpi=300, bbox_inches="tight")
plt.close()

# Verify original dataset untouched
print("\n================================================")
print("VERIFY ORIGINAL GLUCOTREND PREPROCESSED DATASET")
print("================================================")
print("Original dataset shape:", df_original.shape)
print("Original dataset still contains:", len(df_original), "records")
print("\nThe original preprocessed dataset was NOT modified.")
print("\n================================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("================================================")
