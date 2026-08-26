# ==============================================================
# MULTINOMIAL LOGISTIC REGRESSION
# SOFTMAX REGRESSION FOR MULTI-CLASS TREND PREDICTION
# ==============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
   accuracy_score,
   precision_score,
   recall_score,
   f1_score,
   log_loss,
   confusion_matrix,
   classification_report
)

warnings.filterwarnings("ignore")

# ==============================================================
# 1. FILE SETTINGS
# ==============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
TARGET_COLUMN = "target_trend"
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "Multinomial_Logistic_Regre_for_Multiclass_M2")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 75)
print("MULTINOMIAL LOGISTIC REGRESSION")
print("SOFTMAX REGRESSION - MULTI-CLASS GLUCOSE TREND PREDICTION")
print("=" * 75)

# ==============================================================
# 2. READ DATASET
# ==============================================================
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

df = pd.read_csv(DATASET_FILE)
print("\nDataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# Create copy
data = df.copy(deep=True)

# ==============================================================
# 3. SEPARATE FEATURES AND TARGET
# ==============================================================
ALL_TARGET_CANDIDATES = [
    "glucose_lead_1", "glucose_lead_3", "glucose_lead_6",
    "target_binary", "target_ada3", "target_5class", "target_trend"
]

X = data.drop(columns=ALL_TARGET_CANDIDATES, errors="ignore").copy()

# Fit LabelEncoder using target_trend values
# In final_preprocess_M2.csv, target_trend is label encoded to numeric already.
# We will recover original strings to make inverse transform readable.
raw_engineered_path = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")
engineered_df = pd.read_csv(raw_engineered_path)
target_strings = engineered_df["target_trend"].fillna("Stable").str.lower().str.strip()

label_encoder = LabelEncoder()
label_encoder.fit(target_strings)

class_names = label_encoder.classes_
number_of_classes = len(class_names)

print("\nClass Mapping:")
for i, name in enumerate(class_names):
    print(f"{i} --> {name}")

# Align target array (y) as numeric classes
y_encoded = data[TARGET_COLUMN].values.astype(int)

# ==============================================================
# 4. SAVE DATASET INFORMATION
# ==============================================================
class_counts = pd.Series(label_encoder.inverse_transform(y_encoded)).value_counts()

information_file = os.path.join(OUTPUT_FOLDER, "dataset_information.txt")
with open(information_file, "w", encoding="utf-8") as file:
   file.write("PREPROCESSED GLUCOTREND DATASET INFORMATION\n")
   file.write("=" * 75 + "\n\n")
   file.write(f"Dataset file: {DATASET_FILE}\n")
   file.write(f"Rows: {data.shape[0]}\n")
   file.write(f"Columns: {data.shape[1]}\n")
   file.write(f"Target column: {TARGET_COLUMN}\n")
   file.write(f"Number of classes: {number_of_classes}\n")
   file.write("\nFeatures used:\n")
   for column in X.columns:
       file.write(f"- {column}\n")
   file.write("\nClass distribution:\n")
   file.write(str(class_counts))

# ==============================================================
# 5. TRAIN-TEST SPLIT
# ==============================================================
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y_encoded,
   test_size=0.20,
   random_state=42,
   stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ==============================================================
# 6. TRAINING MODEL
# ==============================================================
print("\nTraining multinomial model...")
model = LogisticRegression(
   solver="lbfgs",
   max_iter=2000,
   random_state=42
)
model.fit(X_train, y_train)
print("Model training completed successfully.")

# Predictions & Probabilities
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cross_entropy = log_loss(y_test, y_probability, labels=np.arange(number_of_classes))

print("\n" + "=" * 75)
print("MODEL RESULTS")
print("=" * 75)
print(f"Accuracy             : {accuracy:.4f}")
print(f"Accuracy (%)         : {accuracy * 100:.2f}%")
print(f"Precision (Weighted) : {precision:.4f}")
print(f"Recall (Weighted)    : {recall:.4f}")
print(f"F1 Score (Weighted)  : {f1:.4f}")
print(f"Cross-Entropy Loss   : {cross_entropy:.4f}")

report = classification_report(
   y_test,
   y_pred,
   target_names=[str(x) for x in class_names],
   zero_division=0
)
print("\nClassification Report:\n", report)

# ==============================================================
# 7. SAVE MODEL METRICS
# ==============================================================
metrics_file = os.path.join(OUTPUT_FOLDER, "model_metrics.txt")
with open(metrics_file, "w", encoding="utf-8") as file:
   file.write("MULTINOMIAL LOGISTIC REGRESSION RESULTS\n")
   file.write("=" * 75 + "\n\n")
   file.write(f"Dataset: {DATASET_FILE}\n")
   file.write(f"Target: {TARGET_COLUMN}\n")
   file.write(f"Number of classes: {number_of_classes}\n\n")
   file.write("CLASS MAPPING\n")
   for i, class_name in enumerate(class_names):
        file.write(f"{i} = {class_name}\n")
   file.write("\nMODEL PERFORMANCE\n")
   file.write(f"Accuracy: {accuracy:.4f}\n")
   file.write(f"Precision: {precision:.4f}\n")
   file.write(f"Recall: {recall:.4f}\n")
   file.write(f"F1 Score: {f1:.4f}\n")
   file.write(f"Cross-Entropy Loss: {cross_entropy:.4f}\n\n")
   file.write("CLASSIFICATION REPORT\n")
   file.write(report)

# ==============================================================
# 8. PLOT CONFUSION MATRIX
# ==============================================================
cm = confusion_matrix(y_test, y_pred, labels=np.arange(number_of_classes))

plt.figure(figsize=(8, 6))
sns.heatmap(
   cm,
   annot=True,
   fmt="d",
   cmap="Blues",
   xticklabels=class_names,
   yticklabels=class_names
)
plt.title("Confusion Matrix - Multinomial Logistic Regression")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"), dpi=300)
plt.close()

# ==============================================================
# 9. SAVE PREDICTIONS & SOFTMAX PROBABILITIES
# ==============================================================
prediction_output = X_test.copy()
prediction_output["Actual_Class"] = label_encoder.inverse_transform(y_test)
prediction_output["Predicted_Class"] = label_encoder.inverse_transform(y_pred)
prediction_output["Correct"] = (y_test == y_pred)

for i, class_name in enumerate(class_names):
   prob_col = f"Probability_{str(class_name).replace(' ', '_')}"
   prediction_output[prob_col] = y_probability[:, i]

prediction_output.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), index=False)

probability_output = pd.DataFrame(y_probability, columns=[f"Probability_{str(c).replace(' ', '_')}" for c in class_names])
probability_output.to_csv(os.path.join(OUTPUT_FOLDER, "softmax_probabilities.csv"), index=False)

# ==============================================================
# 10. SAVE CLASS DISTRIBUTION GRAPH
# ==============================================================
class_distribution = pd.DataFrame({
   "Class": class_names,
   "Count": [np.sum(y_encoded == i) for i in range(number_of_classes)]
})
class_distribution.to_csv(os.path.join(OUTPUT_FOLDER, "class_distribution.csv"), index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=class_distribution, x="Class", y="Count", hue="Class", legend=False)
plt.title("Glucose Trend Class Distribution")
plt.xlabel("Trend Class")
plt.ylabel("Number of Readings")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "class_distribution.png"), dpi=300)
plt.close()

# ==============================================================
# 11. SAVE MODEL AND ENCODER PKL
# ==============================================================
joblib.dump(model, os.path.join(OUTPUT_FOLDER, "multinomial_logistic_regression.pkl"))
joblib.dump(label_encoder, os.path.join(OUTPUT_FOLDER, "label_encoder.pkl"))

# Save coefficients and intercepts
coefficients = pd.DataFrame(model.coef_, columns=X.columns, index=[str(x) for x in class_names])
coefficients.to_csv(os.path.join(OUTPUT_FOLDER, "model_coefficients.csv"))

intercepts = pd.DataFrame({"Class": [str(x) for x in class_names], "Intercept": model.intercept_})
intercepts.to_csv(os.path.join(OUTPUT_FOLDER, "model_intercepts.csv"), index=False)

print("\n" + "=" * 75)
print("COMPLETED SUCCESSFULLY")
print("=" * 75)
print("Trained model files saved in:", OUTPUT_FOLDER)
