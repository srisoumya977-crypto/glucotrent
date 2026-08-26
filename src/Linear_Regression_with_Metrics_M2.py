# ============================================================
# LINEAR REGRESSION - GLUCOTREND DATASET
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# 1. FILE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "Linear_Regression_with_Metrics_M2")
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("LINEAR REGRESSION - GLUCOTREND")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================
feature_columns = [
    "glucose",
    "glucose_lag_1",
    "glucose_lag_3",
    "glucose_lag_6",
    "glucose_roll_mean_1h"
]
target_column = "glucose_lead_1"

# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================
required_columns = feature_columns + [target_column]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print("\nERROR! Missing columns:")
    print(missing_columns)
    print("\nAvailable columns:")
    print(list(df.columns))
    raise ValueError("Required columns not found in preprocessed dataset.")

# ============================================================
# 5. CREATE MODEL DATA AND HANDLE NULLS/INF
# ============================================================
model_df = df[required_columns].copy()

# Replace infinite values with NaN
model_df = model_df.replace([np.inf, -np.inf], np.nan)

# Fill missing features with median
for col in feature_columns:
    if model_df[col].isnull().any():
        median_val = model_df[col].median()
        model_df[col] = model_df[col].fillna(median_val)
        print(f"Filled missing values in {col} with median = {median_val}")

# Drop rows where target is missing
model_df = model_df.dropna(subset=[target_column])

# Safety check
if model_df.isnull().sum().sum() > 0:
    raise ValueError("NaN values are still present after cleaning.")

# ============================================================
# 6. DEFINE X AND Y AND SPLIT
# ============================================================
X = model_df[feature_columns].astype(float)
y = model_df[target_column].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 7. TRAIN MODEL
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed.")

# ============================================================
# 8. COEFFICIENTS AND EQUATION
# ============================================================
print("\nIntercept (b0):", model.intercept_)
coefficient_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
})
print(coefficient_df)

equation = f"{target_column} = {model.intercept_:.4f}"
for feature, coef in zip(feature_columns, model.coef_):
    equation += f" + ({coef:.4f} * {feature})"

print("\nLinear Regression Equation:")
print(equation)

# ============================================================
# 9. EVALUATION
# ============================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ============================================================
# 10. SAVE RESULTS
# ============================================================
import joblib

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred
results["Residual"] = results["Actual"] - results["Predicted"]
results["Absolute_Error"] = abs(results["Residual"])

results.to_csv(os.path.join(OUTPUT_FOLDER, "linear_regression_predictions.csv"), index=False)
coefficient_df.to_csv(os.path.join(OUTPUT_FOLDER, "linear_regression_coefficients.csv"), index=False)

# Save model to pkl
joblib.dump(model, os.path.join(OUTPUT_FOLDER, "linear_regression.pkl"))

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
})
metrics_df.to_csv(os.path.join(OUTPUT_FOLDER, "linear_regression_metrics.csv"), index=False)

with open(os.path.join(OUTPUT_FOLDER, "linear_regression_equation.txt"), "w", encoding="utf-8") as f:
    f.write("Linear Regression Equation\n")
    f.write("=" * 40 + "\n")
    f.write(equation + "\n")

# ============================================================
# 11. PLOTS
# ============================================================
# Actual vs Predicted Graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color="teal")
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="orange", linewidth=2)
plt.xlabel("Actual Glucose (mg/dL)")
plt.ylabel("Predicted Glucose (mg/dL)")
plt.title("Linear Regression: Actual vs Predicted Glucose")
plt.grid(True)
plt.savefig(os.path.join(IMAGE_FOLDER, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

# Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, results["Residual"], alpha=0.4, color="coral")
plt.axhline(y=0, linestyle="--", color="black", linewidth=1.5)
plt.xlabel("Predicted Glucose (mg/dL)")
plt.ylabel("Residual (mg/dL)")
plt.title("Residual Plot - Linear Regression")
plt.grid(True)
plt.savefig(os.path.join(IMAGE_FOLDER, "residual_plot.png"), dpi=300, bbox_inches="tight")
plt.close()

# Feature Coefficients
plt.figure(figsize=(8, 5))
plt.bar(coefficient_df["Feature"], coefficient_df["Coefficient"], color="skyblue", edgecolor="grey")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Linear Regression Feature Coefficients")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.savefig(os.path.join(IMAGE_FOLDER, "feature_coefficients.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Output Folder:", OUTPUT_FOLDER)
