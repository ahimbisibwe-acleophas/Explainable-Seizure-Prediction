# Disable GPU usage and suppress TF logs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Load the CNN model
model = tf.keras.models.load_model("cnn_model_47features.h5")

# Get number of input features expected by the model
EXPECTED_FEATURES = model.input_shape[-1]  # e.g., 47

# HTML UI Template
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

# Feature alignment utility
def align_to_expected_features(df, expected_n):
    """
    Align uploaded DataFrame to model's expected number of input features.
    Fills missing with 0s or trims extras.
    """
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] > expected_n:
        df = df.iloc[:, :expected_n]
    elif df.shape[1] < expected_n:
        for i in range(df.shape[1], expected_n):
            df[f'dummy_{i}'] = 0.0
    return df

@app.route('/')
def index():
    return render_template_string(html_template, prediction=None)

@app.route('/schema', methods=['GET'])
def schema():
    return jsonify({
        "expected_features": EXPECTED_FEATURES
    })

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        raw = pd.read_csv(file)

        # Drop label columns if they exist
        for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
            if col in raw.columns:
                raw = raw.drop(columns=[col])

        # Align to expected number of features
        X = align_to_expected_features(raw, EXPECTED_FEATURES)

        # Take first row
        features = np.array(X.iloc[0]).reshape(1, 1, EXPECTED_FEATURES)

        # Predict
        pred_proba = float(model.predict(features)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(html_template, prediction=prediction, probability=pred_proba)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
