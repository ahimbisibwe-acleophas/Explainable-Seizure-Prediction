import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Load model and scaler
model = tf.keras.models.load_model("cnn_seizure_model.h5")
scaler = joblib.load("scaler.pkl")

# ✅ HTML UI template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Seizure Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { margin-top: 80px; background-color: #f8f9fa; }
        #result { margin-top: 20px; display: none; }
    </style>
</head>
<body>
<div class="container text-center">
    <h2>🧠 CNN Seizure Prediction</h2>
    <p class="text-muted">Upload a CSV file with patient features</p>

    <form method="POST" action="/predict_csv" enctype="multipart/form-data">
        <div class="mb-3">
            <input type="file" name="file" accept=".csv" class="form-control" required>
        </div>
        <button type="submit" class="btn btn-primary">🔍 Predict</button>
    </form>

    {% if prediction is not none %}
    <div id="result" class="alert {{ 'alert-warning' if prediction == 1 else 'alert-success' }}">
        <strong>{{ '⚠️ Seizure Likely!' if prediction == 1 else '✅ No Seizure Detected' }}</strong><br>
        Probability: {{ (probability * 100) | round(2) }}%
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(html_template, prediction=None)

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        # ✅ Read CSV and extract first row as feature vector
        df = pd.read_csv(file)
        if df.shape[0] == 0:
            return render_template_string(html_template, prediction=None)

        features = df.iloc[0].values.astype(float).reshape(1, -1)

        # ✅ Scale and reshape
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        # ✅ Predict
        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(html_template, prediction=prediction, probability=pred_proba)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ API endpoint for raw JSON input (optional)
@app.route('/predict', methods=['POST'])
def predict_json():
    try:
        data = request.get_json(force=True)
        features = data.get("features")
        if features is None:
            return jsonify({"error": "Missing 'features' in request."}), 400

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])
        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": pred_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
