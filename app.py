import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ============================================================
# CUSTOM ROUTE TO SERVE GENERATED OUTPUT PLOTS
# ============================================================
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

# ============================================================
# 1. HOME PAGE
# ============================================================
@app.route("/")
def home():
    return render_template("home.html")

# ============================================================
# 2. ABOUT PAGE
# ============================================================
@app.route("/about")
def about():
    return render_template("about.html")

# ============================================================
# 3. DATASET PAGE
# ============================================================
@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

# ============================================================
# 4. PREPROCESSING PAGE
# ============================================================
@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")

# ============================================================
# 5. VISUALIZATION PAGE
# ============================================================
@app.route("/visualization")
def visualization():
    boxplot_dir = os.path.join(OUTPUTS_DIR, "Boxplots_correlation")
    boxplot_files = []
    
    if os.path.exists(boxplot_dir):
        # List all boxplots except the heatmap
        boxplot_files = [
            f for f in os.listdir(boxplot_dir)
            if f.endswith(".png") and f != "correlation_heatmap.png"
        ]
        boxplot_files.sort()
        
    return render_template(
        "visualization.html",
        boxplot_files=boxplot_files
    )

# ============================================================
# 6. MODELS COMPARISON PAGE
# ============================================================
@app.route("/models")
def models():
    # A. Multiple Linear Regression metrics
    lr_metrics = {}
    lr_metrics_path = os.path.join(OUTPUTS_DIR, "Linear_Regression_with_Metrics_M2", "linear_regression_metrics.csv")
    if os.path.exists(lr_metrics_path):
        df_lr = pd.read_csv(lr_metrics_path)
        lr_metrics = dict(zip(df_lr["Metric"], df_lr["Value"]))

    # B. Closed-Form Normal Equation vs Gradient Descent comparison metrics
    cfne_gd_metrics = []
    cfne_gd_path = os.path.join(OUTPUTS_DIR, "Linear_Regression_CFNE_GD_Compare_M2", "comparison_metrics.csv")
    if os.path.exists(cfne_gd_path):
        cfne_gd_metrics = pd.read_csv(cfne_gd_path).to_dict(orient="records")

    # C. Binary Logistic Regression metrics
    bin_metrics = {}
    bin_pred_path = os.path.join(OUTPUTS_DIR, "Logistic_Regression_Binary_Classify_M2", "prediction_results.csv")
    if os.path.exists(bin_pred_path):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        df_bin = pd.read_csv(bin_pred_path)
        bin_metrics = {
            "Accuracy": accuracy_score(df_bin["Actual"], df_bin["Predicted"]),
            "Precision": precision_score(df_bin["Actual"], df_bin["Predicted"], zero_division=0),
            "Recall": recall_score(df_bin["Actual"], df_bin["Predicted"], zero_division=0),
            "F1_Score": f1_score(df_bin["Actual"], df_bin["Predicted"], zero_division=0)
        }

    # D. Multinomial Logistic Regression metrics
    multi_metrics = {}
    multi_pred_path = os.path.join(OUTPUTS_DIR, "Multinomial_Logistic_Regre_for_Multiclass_M2", "predictions.csv")
    if os.path.exists(multi_pred_path):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        df_multi = pd.read_csv(multi_pred_path)
        multi_metrics = {
            "Accuracy": accuracy_score(df_multi["Actual_Class"], df_multi["Predicted_Class"]),
            "Precision": precision_score(df_multi["Actual_Class"], df_multi["Predicted_Class"], average="weighted", zero_division=0),
            "Recall": recall_score(df_multi["Actual_Class"], df_multi["Predicted_Class"], average="weighted", zero_division=0),
            "F1_Score": f1_score(df_multi["Actual_Class"], df_multi["Predicted_Class"], average="weighted", zero_division=0)
        }

    # E. Regularization model comparison
    reg_models = []
    reg_path = os.path.join(DATASET_DIR, "regularisation_model_comparison_M2.csv")
    if os.path.exists(reg_path):
        reg_models = pd.read_csv(reg_path).to_dict(orient="records")

    return render_template(
        "models.html",
        lr_metrics=lr_metrics,
        cfne_gd_metrics=cfne_gd_metrics,
        bin_metrics=bin_metrics,
        multi_metrics=multi_metrics,
        reg_models=reg_models
    )

# ============================================================
# 7. INTERACTIVE PREDICTION PAGE
# ============================================================
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    prediction_result = None
    error_message = None
    form_data = {}

    if request.method == "POST":
        try:
            # 1. Parse user inputs
            glucose = float(request.form.get("glucose"))
            glucose_lag_1 = float(request.form.get("glucose_lag_1"))
            glucose_lag_3 = float(request.form.get("glucose_lag_3"))
            glucose_lag_6 = float(request.form.get("glucose_lag_6"))
            glucose_roll_mean_1h = float(request.form.get("glucose_roll_mean_1h"))
            
            form_data = {
                "glucose": glucose,
                "glucose_lag_1": glucose_lag_1,
                "glucose_lag_3": glucose_lag_3,
                "glucose_lag_6": glucose_lag_6,
                "glucose_roll_mean_1h": glucose_roll_mean_1h
            }

            # 2. Linear Regression (Regression Forecast)
            lr_model_path = os.path.join(OUTPUTS_DIR, "Linear_Regression_with_Metrics_M2", "linear_regression.pkl")
            if os.path.exists(lr_model_path):
                lr_model = joblib.load(lr_model_path)
                X_lr = np.array([[glucose, glucose_lag_1, glucose_lag_3, glucose_lag_6, glucose_roll_mean_1h]])
                forecast_val = lr_model.predict(X_lr)[0]
            else:
                forecast_val = "Linear Regression model not trained."

            # 3. Binary Logistic Regression (Range Warning status)
            weights_bias_path = os.path.join(OUTPUTS_DIR, "Logistic_Regression_Binary_Classify_M2", "weights_bias.csv")
            binary_scaler_path = os.path.join(OUTPUTS_DIR, "Logistic_Regression_Binary_Classify_M2", "binary_scaler.pkl")
            
            if os.path.exists(weights_bias_path) and os.path.exists(binary_scaler_path):
                wb_df = pd.read_csv(weights_bias_path)
                binary_weights = wb_df[wb_df["Parameter"].str.startswith("weight_")]["Value"].values
                binary_bias = wb_df[wb_df["Parameter"] == "bias"]["Value"].values[0]
                binary_scaler = joblib.load(binary_scaler_path)
                
                X_bin = np.array([[glucose, glucose_roll_mean_1h, glucose_lag_1]])
                X_bin_scaled = binary_scaler.transform(X_bin)[0]
                
                z = np.dot(X_bin_scaled, binary_weights) + binary_bias
                prob = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                range_status = "Out of Range" if prob >= 0.5 else "In Range"
                range_prob = prob
            else:
                range_status = "Binary classification model not trained."
                range_prob = None

            # 4. Multinomial Logistic Regression (ROC Trend Class)
            multinomial_model_path = os.path.join(OUTPUTS_DIR, "Multinomial_Logistic_Regre_for_Multiclass_M2", "multinomial_logistic_regression.pkl")
            label_encoder_path = os.path.join(OUTPUTS_DIR, "Multinomial_Logistic_Regre_for_Multiclass_M2", "label_encoder.pkl")
            final_scaler_path = os.path.join(DATASET_DIR, "final_scaler.pkl")
            final_preprocess_path = os.path.join(DATASET_DIR, "final_preprocess_M2.csv")
            raw_engineered_path = os.path.join(DATASET_DIR, "target_engineered_M2.csv")
            
            if (os.path.exists(multinomial_model_path) and 
                os.path.exists(label_encoder_path) and 
                os.path.exists(final_scaler_path) and 
                os.path.exists(final_preprocess_path) and 
                os.path.exists(raw_engineered_path)):
                
                multinomial_model = joblib.load(multinomial_model_path)
                label_encoder = joblib.load(label_encoder_path)
                final_scaler = joblib.load(final_scaler_path)
                
                # Load preprocessed schema columns of X
                df_prep = pd.read_csv(final_preprocess_path)
                ALL_TARGET_CANDIDATES = [
                    "glucose_lead_1", "glucose_lead_3", "glucose_lead_6",
                    "target_binary", "target_ada3", "target_5class", "target_trend"
                ]
                feature_cols = [c for c in df_prep.columns if c not in ALL_TARGET_CANDIDATES]
                
                # Fetch raw medians of scale features to serve as defaults
                df_raw = pd.read_csv(raw_engineered_path)
                default_features = {}
                for col in feature_cols:
                    if col in df_raw.columns:
                        default_features[col] = df_raw[col].median()
                    else:
                        default_features[col] = 0.0 # Fallback for label-encoded categories
                
                # Overwrite defaults with user input raw values
                user_features = default_features.copy()
                user_features["glucose"] = glucose
                user_features["glucose_lag_1"] = glucose_lag_1
                user_features["glucose_lag_3"] = glucose_lag_3
                user_features["glucose_lag_6"] = glucose_lag_6
                user_features["glucose_roll_mean_1h"] = glucose_roll_mean_1h
                
                X_pred_df = pd.DataFrame([user_features])[feature_cols]
                
                # Standard scaler is only fitted on numeric scaling variables
                scale_cols = list(final_scaler.feature_names_in_)
                X_pred_df[scale_cols] = final_scaler.transform(X_pred_df[scale_cols])
                
                # Predict trend class
                trend_idx = multinomial_model.predict(X_pred_df)[0]
                trend_class = label_encoder.inverse_transform([trend_idx])[0]
            else:
                trend_class = "Multinomial model not trained."

            prediction_result = {
                "forecast": f"{forecast_val:.2f} mg/dL" if isinstance(forecast_val, float) else forecast_val,
                "range_status": range_status,
                "range_probability": f"{range_prob * 100:.2f}%" if range_prob is not None else "N/A",
                "trend": trend_class.title()
            }

        except Exception as e:
            error_message = f"Error processing prediction: {str(e)}"

    return render_template(
        "prediction.html",
        prediction_result=prediction_result,
        error_message=error_message,
        form_data=form_data
    )

# ============================================================
# 8. ANALYTICS DASHBOARD PAGE
# ============================================================
@app.route("/dashboard")
def dashboard():
    # Precalculate summary stats from preprocessed dataset
    stats = {}
    final_preprocess_path = os.path.join(DATASET_DIR, "final_preprocess_M2.csv")
    
    if os.path.exists(final_preprocess_path):
        df_p = pd.read_csv(final_preprocess_path)
        raw_path = os.path.join(DATASET_DIR, "target_engineered_M2.csv")
        df_raw = pd.read_csv(raw_path)
        
        total_readings = len(df_raw)
        
        # Calculate time in range from raw target distribution
        bin_counts = df_raw["target_binary"].value_counts()
        in_range_cnt = bin_counts.get("In_Range", 0)
        out_range_cnt = bin_counts.get("Out_of_Range", 0)
        
        time_in_range_pct = (in_range_cnt / total_readings) * 100 if total_readings > 0 else 0
        avg_glucose = df_raw["glucose"].mean()
        
        stats = {
            "total_readings": f"{total_readings:,}",
            "in_range_pct": f"{time_in_range_pct:.2f}%",
            "out_range_pct": f"{(out_range_cnt / total_readings) * 100:.2f}%" if total_readings > 0 else "0%",
            "avg_glucose": f"{avg_glucose:.2f} mg/dL"
        }
    else:
        stats = {
            "total_readings": "--",
            "in_range_pct": "--%",
            "out_range_pct": "--%",
            "avg_glucose": "-- mg/dL"
        }
        
    return render_template("dashboard.html", stats=stats)

# ============================================================
# 9. EVALUATION REPORTS PAGE
# ============================================================
@app.route("/reports")
def reports():
    class_balance_report = ""
    report_path = os.path.join(OUTPUTS_DIR, "Target_Variable_Analysis_M2", "target_class_balance_report.txt")
    
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            class_balance_report = f.read()
            
    return render_template("reports.html", class_balance_report=class_balance_report)

# ============================================================
# 10. CONTACT PAGE
# ============================================================
@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
