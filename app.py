import os

# Suppress TensorFlow GPU usage and logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # Force TensorFlow to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         # Suppress TF info/warnings

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

#  Initialize Flask app
app = Flask(__name__)

#  Load pre-trained CNN model and scaler
model = tf.keras.models.load_model("cnn_seizure_model.h5")
scaler = joblib.load("scaler.pkl")

#  HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title> Seizure Prediction</title>
</head>
<body style="font-family:sans-serif; max-width:600px; margin:auto; padding:2em;">
    <h2> CNN Seizure Prediction Interface</h2>
    <form method="POST" action="/predict_csv" enctype="multipart/form-data">
        <label><strong>Upload your features CSV file:</strong></label><br><br>
        <input type="file" name="file" accept=".csv" required>
        <br><br>
        <button type="submit"> Predict</button>
    </form>

    {% if prediction is not none %}
        <hr>
        <h3> Prediction Result</h3>
        <p><strong>Prediction:</strong> {{ 'Seizure Likely' if prediction == 1 else 'No Seizure Detected' }}</p>
        <p><strong>Probability:</strong> {{ '%.2f'|format(probability * 100) }}%</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template, prediction=None)

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        df = pd.read_csv(file)

        #  Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        #  Drop known label columns if present
        for label_col in ['Class', 'Label', 'Target']:
            if label_col in df.columns:
                df = df.drop(columns=[label_col])

        if df.shape[0] == 0:
            return render_template_string(html_template, prediction=None)

        #  Reshape input for scaler and Conv1D
        features = np.array(df.iloc[0]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        #  Predict
        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(html_template, prediction=prediction, probability=pred_proba)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#  Main run
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
