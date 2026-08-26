# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. LOAD DATASET
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

data = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", data.shape)

# ============================================================
# 2. CHECK MISSING / INFINITE VALUES
# ============================================================
print("\nMissing values before cleaning:", data.isnull().sum().sum())
print("Infinite values before cleaning:", np.isinf(data.select_dtypes(include=np.number)).sum().sum())

# Replace infinity with NaN
data = data.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================
# List all engineered targets to exclude them from the features (prevent target leakage)
ALL_TARGET_CANDIDATES = [
    "glucose_lead_1", "glucose_lead_3", "glucose_lead_6",
    "target_binary", "target_ada3", "target_5class", "target_trend"
]

X_df = data.drop(columns=ALL_TARGET_CANDIDATES, errors="ignore").copy()
y_series = data["glucose_lead_1"].copy()

# Convert everything to numeric if possible
X_df = X_df.apply(pd.to_numeric, errors="coerce")
y_series = pd.to_numeric(y_series, errors="coerce")

# Fill missing feature values using median
X_df = X_df.fillna(X_df.median())

# Remove rows where target is missing
valid_rows = y_series.notna()
X_df = X_df.loc[valid_rows]
y_series = y_series.loc[valid_rows]

# Convert to NumPy
X = X_df.values
y = y_series.values

print("\nDataset after cleaning:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 4. FINAL SAFETY CHECK
# ============================================================
print("\nNaN in X:", np.isnan(X).sum())
print("NaN in y:", np.isnan(y).sum())
print("Inf in X:", np.isinf(X).sum())
print("Inf in y:", np.isinf(y).sum())

# ============================================================
# 5. CREATE IMAGE OUTPUT FOLDER
# ============================================================
IMAGE_FOLDER = os.path.join(BASE_DIR, "outputs", "Linear_Regression_CFNE_GD_Compare_M2")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("\nImage output folder:", IMAGE_FOLDER)

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ============================================================
# 7. FEATURE SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 8. CLOSED FORM SOLUTION (NORMAL EQUATION)
# ============================================================
# Add bias column
X_train_bias = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_bias = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# theta = (X^T X)^(-1) X^T y
# pinv() is used because it is numerically more stable.
theta = np.linalg.pinv(X_train_bias).dot(y_train)

# Prediction
pred_normal = X_test_bias.dot(theta)
pred_normal = np.asarray(pred_normal).ravel()

# Check Normal Equation Predictions
print("\nNormal Equation prediction check:")
print("NaN:", np.isnan(pred_normal).sum())
print("Inf:", np.isinf(pred_normal).sum())

if np.isnan(pred_normal).any() or np.isinf(pred_normal).any():
    raise ValueError("Normal Equation produced NaN or Inf predictions.")

# Metrics
mse_normal = mean_squared_error(y_test, pred_normal)
r2_normal = r2_score(y_test, pred_normal)

print("\n------ Closed Form Normal Equation ------")
print("Intercept + Coefficients:")
print(theta)
print("MSE:", mse_normal)
print("R2 Score:", r2_normal)

# ============================================================
# 9. GRADIENT DESCENT
# ============================================================
X_train_gd = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_gd = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

m = len(y_train)

# Initialize parameters
theta_gd = np.zeros(X_train_gd.shape[1])

# Hyperparameters
learning_rate = 0.01
epochs = 1000
loss_history = []

# GD Iterations
for epoch in range(epochs):
    predictions = X_train_gd.dot(theta_gd)
    errors = predictions - y_train
    gradients = (2 / m) * X_train_gd.T.dot(errors)
    theta_gd -= learning_rate * gradients
    loss = np.mean(errors ** 2)
    loss_history.append(loss)

# GD Prediction
pred_gd = X_test_gd.dot(theta_gd)
pred_gd = np.asarray(pred_gd).ravel()

# Check Gradient Descent Predictions
print("\nGradient Descent prediction check:")
print("NaN:", np.isnan(pred_gd).sum())
print("Inf:", np.isinf(pred_gd).sum())

if np.isnan(pred_gd).any() or np.isinf(pred_gd).any():
    raise ValueError("Gradient Descent produced NaN or Inf predictions.")

# Metrics
mse_gd = mean_squared_error(y_test, pred_gd)
r2_gd = r2_score(y_test, pred_gd)

print("\n------ Gradient Descent ------")
print("Intercept + Coefficients:")
print(theta_gd)
print("MSE:", mse_gd)
print("R2 Score:", r2_gd)

# ============================================================
# 10. COMPARISON SUMMARY
# ============================================================
print("\n=========== Comparison ===========")
print("Normal Equation - MSE:", mse_normal, "| R2:", r2_normal)
print("Gradient Descent - MSE:", mse_gd, "| R2:", r2_gd)

# ============================================================
# 11. PLOTS AND VISUALIZATIONS
# ============================================================
# Image 1: Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_normal, alpha=0.4, label="Normal Equation", color="teal")
plt.scatter(y_test, pred_gd, alpha=0.4, label="Gradient Descent", color="coral")

minimum = min(y_test.min(), pred_normal.min(), pred_gd.min())
maximum = max(y_test.max(), pred_normal.max(), pred_gd.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", label="Perfect Prediction", color="black")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
image1 = os.path.join(IMAGE_FOLDER, "actual_vs_predicted.png")
plt.savefig(image1, dpi=300, bbox_inches="tight")
plt.close()

# Image 2: Residual Comparison
normal_residuals = y_test - pred_normal
gd_residuals = y_test - pred_gd

plt.figure(figsize=(9, 6))
plt.scatter(pred_normal, normal_residuals, alpha=0.4, label="Normal Equation", color="teal")
plt.scatter(pred_gd, gd_residuals, alpha=0.4, label="Gradient Descent", color="coral")
plt.axhline(y=0, linestyle="--", color="black")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
image2 = os.path.join(IMAGE_FOLDER, "residual_comparison.png")
plt.savefig(image2, dpi=300, bbox_inches="tight")
plt.close()

# Image 3: Gradient Descent Loss Curve
plt.figure(figsize=(9, 6))
plt.plot(range(1, epochs + 1), loss_history, color="blue", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.tight_layout()
image3 = os.path.join(IMAGE_FOLDER, "gradient_descent_loss.png")
plt.savefig(image3, dpi=300, bbox_inches="tight")
plt.close()

# Save comparison metrics
comparison_metrics_df = pd.DataFrame({
    "Method": ["Normal Equation", "Gradient Descent"],
    "MSE": [mse_normal, mse_gd],
    "R2": [r2_normal, r2_gd]
})
comparison_metrics_df.to_csv(os.path.join(IMAGE_FOLDER, "comparison_metrics.csv"), index=False)

# Save Image Information
image_info = pd.DataFrame({
    "Image": [
        "actual_vs_predicted.png",
        "residual_comparison.png",
        "gradient_descent_loss.png"
    ],
    "Description": [
        "Actual values versus predictions from both methods",
        "Residual comparison between Normal Equation and Gradient Descent",
        "MSE loss across Gradient Descent epochs"
    ]
})
image_info.to_csv(os.path.join(IMAGE_FOLDER, "image_information.csv"), index=False)

print("\n==========================================")
print("PROCESS COMPLETED SUCCESSFULLY")
print("==========================================")
print("Generated images stored in:", IMAGE_FOLDER)
