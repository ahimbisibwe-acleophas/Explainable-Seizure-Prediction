import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json

app = Flask(__name__)

# Load model + scaler
model = tf.keras.models.load_model("cnn_seizure_model.h5")
scaler = joblib.load("scaler.pkl")

# Try to load a saved feature schema; otherwise fall back to scaler.feature_names_in_
FEATURE_COLUMNS = []
try:
    with open("feature_columns.json", "r") as f:
        FEATURE_COLUMNS = json.load(f)
except Exception:
    FEATURE_COLUMNS = list(getattr(scaler, "feature_names_in_", []))

html_template = """... (unchanged HTML) ..."""

@app.route('/')
def index():
    return render_template_string(html_template, prediction=None)

# Optional helper to inspect server schema
@app.route('/schema', methods=['GET'])
def schema():
    return jsonify({
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS
    })

def align_to_schema(df_numeric: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """
    Return a DF that has exactly required_cols in the same order,
    filling missing with 0 and dropping extras.
    """
    if not required_cols:  # fallback if we do not have schema
        return df_numeric

    # Ensure all required columns exist; fill missing with 0
    aligned = pd.DataFrame(index=df_numeric.index, columns=required_cols, dtype=float)
    for c in required_cols:
        aligned[c] = df_numeric[c] if c in df_numeric.columns else 0.0

    # Fill any NaNs with column means (or zeros)
    aligned = aligned.fillna(aligned.mean(numeric_only=True)).fillna(0.0)
    return aligned

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        raw = pd.read_csv(file)

        # Keep only numeric columns (your app expects pre-encoded numeric features)
        df_num = raw.select_dtypes(include=[np.number]).copy()

        # Drop any label columns if they slipped through
        for label_col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
            if label_col in df_num.columns:
                df_num = df_num.drop(columns=[label_col])

        if df_num.shape[0] == 0:
            return render_template_string(html_template, prediction=None)

        # Align to training schema / scaler expectation
        required = FEATURE_COLUMNS or list(getattr(scaler, "feature_names_in_", df_num.columns))
        X = align_to_schema(df_num, required)

        # Take first row (your UI uses the first row only)
        features = np.array(X.iloc[0]).reshape(1, -1)

        # Scale and reshape for Conv1D
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(html_template, prediction=prediction, probability=pred_proba)

    except Exception as e:
        # Return details to help debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
