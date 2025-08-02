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

# Load model and scaler
model = tf.keras.models.load_model("cnn_model_47features.h5")
scaler = joblib.load("scaler.pkl")

# Load feature schema if available
FEATURE_COLUMNS = []
try:
    with open("feature_columns.json", "r") as f:
        FEATURE_COLUMNS = json.load(f)
except Exception:
    FEATURE_COLUMNS = list(getattr(scaler, "feature_names_in_", []))

# HTML template (unchanged from your current code)
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Epileptic Seizure Predictor</title>
</head>
<body style="font-family: sans-serif; max-width: 700px; margin: auto; padding: 2em; background-color: #f9f9f9;">
    <h1 style="text-align: center; color: #333;">Epileptic Seizure Predictor</h1>

    <form method="POST" action="/predict_csv" enctype="multipart/form-data" style="margin-top: 2em;">
        <label for="file"><strong>Select EEG Feature CSV File:</strong></label><br><br>
        <input type="file" id="file" name="file" accept=".csv" required>
        <br><br>
        <button type="submit" style="padding: 10px 20px; font-size: 1em;">Run Prediction</button>
    </form>

    {% if prediction is not none %}
        <hr style="margin-top: 3em;">
        <h2 style="color: #444;">Prediction Result</h2>
        <p style="font-size: 1.2em;">
            <strong>Outcome:</strong>
            <span style="color: {{ 'red' if prediction == 1 else 'green' }};">
                {{ 'Seizure Likely' if prediction == 1 else 'No Seizure Detected' }}
            </span>
        </p>
        <p style="font-size: 1.2em;">
            <strong>Prediction Probability:</strong> {{ '%.2f'|format(probability * 100) }}%
        </p>
    {% endif %}
</body>
</html>
"""

# Optional schema endpoint to inspect server features
@app.route('/schema', methods=['GET'])
def schema():
    return jsonify({
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS
    })

def align_to_schema(df_numeric, required_cols):
    """
    Ensure uploaded CSV matches training schema: correct columns and order.
    """
    if not required_cols:
        return df_numeric

    aligned = pd.DataFrame(index=df_numeric.index, columns=required_cols, dtype=float)
    for col in required_cols:
        aligned[col] = df_numeric[col] if col in df_numeric.columns else 0.0

    # Fill any remaining NaNs
    aligned = aligned.fillna(aligned.mean(numeric_only=True)).fillna(0.0)
    return aligned

@app.route('/')
def index():
    return render_template_string(html_template, prediction=None)

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        raw = pd.read_csv(file)

        # Keep only numeric columns
        df_num = raw.select_dtypes(include=[np.number])

        # Drop known label columns
        for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
            if col in df_num.columns:
                df_num = df_num.drop(columns=[col])

        if df_num.shape[0] == 0:
            return render_template_string(html_template, prediction=None)

        # Align to schema
        required = FEATURE_COLUMNS or list(getattr(scaler, "feature_names_in_", df_num.columns))
        X = align_to_schema(df_num, required)

        # First row only
        features = np.array(X.iloc[0]).reshape(1, -1)

        # Scale and reshape for Conv1D
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        # Predict
        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(html_template, prediction=prediction, probability=pred_proba)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
