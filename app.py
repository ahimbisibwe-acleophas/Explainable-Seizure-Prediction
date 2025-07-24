import os

# ✅ Suppress TensorFlow GPU usage and verbose logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load model and scaler
model = tf.keras.models.load_model("cnn_seizure_model.h5")
scaler = joblib.load("scaler.pkl")

# ✅ HTML template with upload form and results
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Seizure Prediction</title>
</head>
<body style="font-family: Arial; background-color: #f4f4f4; padding: 40px;">
    <h2>🧠 Upload EEG Features CSV</h2>
    <form action="/predict_csv" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required><br><br>
        <input type="submit" value="Predict" style="padding: 10px; background-color: #4CAF50; color: white; border: none; cursor: pointer;">
    </form>

    {% if prediction is not none %}
        <h3 style="margin-top: 30px;">🔍 Prediction Result:</h3>
        <p><strong>Status:</strong> {{ '⚠️ Seizure Imminent' if prediction == 1 else '✅ No Seizure Detected' }}</p>
        <p><strong>Probability:</strong> {{ probability | round(4) }}</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template, prediction=None)

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        df = pd.read_csv(file)

        # ✅ Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # ✅ Drop label column if exists
        for label_col in ['Class', 'Label', 'Target']:
            if label_col in df.columns:
                df = df.drop(columns=[label_col])

        if df.shape[0] == 0:
            return render_template_string(html_template, prediction=None)

        # ✅ Extract the first row
        features = df.iloc[0].values.reshape(1, -1)

        # ✅ Scale features
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        # ✅ Predict
        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(html_template, prediction=prediction, probability=pred_proba)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ API endpoint for JSON-based prediction
@app.route('/predict', methods=['POST'])
def predict_json():
    try:
        data = request.get_json(force=True)
        features = data.get("features")

        if features is None:
            return jsonify({"error": "Missing 'features' in request. Please provide a list."}), 400

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

# ✅ Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
