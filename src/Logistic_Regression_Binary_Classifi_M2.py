# ============================================================
# LOGISTIC REGRESSION - BINARY CLASSIFICATION
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# 1. SETTINGS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "Logistic_Regression_Binary_Classify_M2")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD GLUCOTREND DATASET
# ============================================================
df = pd.read_csv(DATASET)

print("\n========== DATASET ==========")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())
print("\nDataset shape:", df.shape)

# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================
FEATURES = ["glucose", "glucose_roll_mean_1h", "glucose_lag_1"]
TARGET = "target_binary"

# Ensure target is loaded correctly
# In final_preprocess_M2.csv, target_binary might be strings or label-encoded ints.
# If strings, map 'in_range' -> 0, 'out_of_range' -> 1
if df[TARGET].dtype == 'object':
    y = np.where(df[TARGET].str.lower().str.strip() == 'out_of_range', 1, 0)
else:
    # If already label-encoded, in_range (alphabetically first) is 0, out_of_range is 1.
    y = df[TARGET].values.astype(int)

X = df[FEATURES].values

print("\nFeatures:", FEATURES)
print("Target:", TARGET)
print("\nTarget distribution:")
print(pd.Series(y).value_counts())

# ============================================================
# 4. SPLIT DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.25,
   random_state=42,
   stratify=y
)

# ============================================================
# 5. STANDARDIZE FEATURES
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import joblib
joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "binary_scaler.pkl"))

# ============================================================
# 6. SIGMOID FUNCTION
# ============================================================
def sigmoid(z):
   z = np.clip(z, -500, 500)
   return 1 / (1 + np.exp(-z))

# Draw Sigmoid Graph
z_values = np.linspace(-10, 10, 500)
sigmoid_values = sigmoid(z_values)

plt.figure(figsize=(8, 6))
plt.plot(z_values, sigmoid_values, color="blue", linewidth=3)
plt.axhline(0.5, color="red", linestyle="--", label="Threshold = 0.5")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_FOLDER, "01_sigmoid.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 7. CROSS-ENTROPY LOSS
# ============================================================
def cross_entropy_loss(y_true, y_probability):
   epsilon = 1e-15
   y_probability = np.clip(y_probability, epsilon, 1 - epsilon)
   return -np.mean(y_true * np.log(y_probability) + (1 - y_true) * np.log(1 - y_probability))

# ============================================================
# 8. TRAIN USING GRADIENT DESCENT
# ============================================================
number_of_features = X_train_scaled.shape[1]
weights = np.zeros(number_of_features)
bias = 0.0
learning_rate = 0.05
epochs = 3000
loss_history = []

m = len(y_train)

for epoch in range(epochs):
   z = np.dot(X_train_scaled, weights) + bias
   probability = sigmoid(z)
   loss = cross_entropy_loss(y_train, probability)
   loss_history.append(loss)
   
   error = probability - y_train
   dw = (1 / m) * np.dot(X_train_scaled.T, error)
   db = (1 / m) * np.sum(error)
   
   weights -= learning_rate * dw
   bias -= learning_rate * db

# Save trained weights & bias for the Flask web application prediction form
weights_bias_df = pd.DataFrame({
    "Parameter": [f"weight_{f}" for f in FEATURES] + ["bias"],
    "Value": list(weights) + [bias]
})
weights_bias_df.to_csv(os.path.join(OUTPUT_FOLDER, "weights_bias.csv"), index=False)

print("\n========== MODEL PARAMETERS ==========")
for feature, weight in zip(FEATURES, weights):
   print(f"{feature}: {weight:.6f}")
print(f"Bias: {bias:.6f}")

# Cross-entropy Loss Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), loss_history, color="purple", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Logistic Regression - Cross-Entropy Loss")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_FOLDER, "02_cross_entropy_loss.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 9. PREDICTIONS AND EVALUATION
# ============================================================
def predict_probability(X_val):
   z = np.dot(X_val, weights) + bias
   return sigmoid(z)

def predict(X_val, threshold=0.5):
   prob = predict_probability(X_val)
   return (prob >= threshold).astype(int)

test_probability = predict_probability(X_test_scaled)
y_pred = predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["In Range", "Out of Range"])
plt.yticks([0, 1], ["In Range", "Out of Range"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(2):
   for j in range(2):
       plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="black")

plt.savefig(os.path.join(OUTPUT_FOLDER, "03_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 10. DECISION BOUNDARY AS HYPERPLANE
# ============================================================
# We visualize a 2D cross-section of the 3D features by fixing
# Feature 3 (glucose_lag_1) to its average raw value in the dataset.
mean_lag_1 = df["glucose_lag_1"].mean()

glucose_values = np.linspace(df["glucose"].min() - 10, df["glucose"].max() + 10, 300)
glucose_roll_values = np.linspace(df["glucose_roll_mean_1h"].min() - 10, df["glucose_roll_mean_1h"].max() + 10, 300)

GLUCOSE, GLUCOSE_ROLL = np.meshgrid(glucose_values, glucose_roll_values)
GLUCOSE_LAG = np.full_like(GLUCOSE, mean_lag_1)

grid = np.column_stack([
   GLUCOSE.ravel(),
   GLUCOSE_ROLL.ravel(),
   GLUCOSE_LAG.ravel()
])

# Scale the grid
grid_scaled = scaler.transform(grid)

# Probabilities
grid_probability = predict_probability(grid_scaled)
grid_probability = grid_probability.reshape(GLUCOSE.shape)

plt.figure(figsize=(10, 7))

# Contour probability shading
contour = plt.contourf(GLUCOSE, GLUCOSE_ROLL, grid_probability, levels=50, cmap="RdYlGn_r", alpha=0.35)
plt.colorbar(contour, label="Out-of-Range Probability")

# Hyperplane boundary (Probability = 0.5)
plt.contour(GLUCOSE, GLUCOSE_ROLL, grid_probability, levels=[0.5], colors="black", linewidths=3)

# Scatter original data points (sub-sampled for visualization clarity)
sample_df = df.sample(n=min(len(df), 1000), random_state=42)
plt.scatter(
   sample_df[sample_df[TARGET] == 'in_range']["glucose"],
   sample_df[sample_df[TARGET] == 'in_range']["glucose_roll_mean_1h"],
   color="green", edgecolor="black", s=30, label="In Range", alpha=0.5
)
plt.scatter(
   sample_df[sample_df[TARGET] == 'out_of_range']["glucose"],
   sample_df[sample_df[TARGET] == 'out_of_range']["glucose_roll_mean_1h"],
   color="red", edgecolor="black", s=40, label="Out of Range", alpha=0.8
)

plt.xlabel("Current Glucose (mg/dL)")
plt.ylabel("1-Hour Rolling Mean Glucose (mg/dL)")
plt.title("Logistic Regression Decision Boundary\nFixed glucose_lag_1 = mean")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(os.path.join(OUTPUT_FOLDER, "04_decision_boundary_hyperplane.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 11. SAVE PREDICTION RESULTS
# ============================================================
results = pd.DataFrame({
   "Actual": y_test,
   "Predicted": y_pred,
   "Range_Out_Probability": test_probability
})
results.to_csv(os.path.join(OUTPUT_FOLDER, "prediction_results.csv"), index=False)

print("\n============================================")
print("PROGRAM COMPLETED")
print("============================================")
print("All outputs stored in:", OUTPUT_FOLDER)
