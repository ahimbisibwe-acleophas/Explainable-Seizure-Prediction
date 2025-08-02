# Disable GPU usage and suppress TensorFlow logs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template_string, send_file
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular

# Initialize Flask app
app = Flask(__name__)

# Load the trained CNN model
model = tf.keras.models.load_model("cnn_model_47features.h5")
EXPECTED_FEATURES = model.input_shape[-1]  # e.g., 47

# HTML Template
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
        <br>
        <form method="POST" action="/explain_lime" enctype="multipart/form-data">
            <input type="hidden" name="file" value="{{ filename }}">
            <label><strong>Re-upload CSV for Explanation:</strong></label><br><br>
            <input type="file" name="file" accept=".csv" required>
            <button type="submit" style="padding: 8px 16px;">Get LIME Explanation</button>
        </form>
        <br>
        <form method="POST" action="/explain_shap" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <button type="submit" style="padding: 8px 16px;">Get SHAP Explanation</button>
        </form>
    {% endif %}
</body>
</html>
"""

# Helper: Align features to expected number
def align_to_expected_features(df, expected_n):
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
    return jsonify({"expected_features": EXPECTED_FEATURES})

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        raw = pd.read_csv(file)

        for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
            if col in raw.columns:
                raw = raw.drop(columns=[col])

        X = align_to_expected_features(raw, EXPECTED_FEATURES)
        features = np.array(X.iloc[0]).reshape(1, 1, EXPECTED_FEATURES)

        pred_proba = float(model.predict(features)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(html_template, prediction=prediction, probability=pred_proba)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explain_lime', methods=['POST'])
def explain_lime():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)

        for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
            if col in df.columns:
                df = df.drop(columns=[col])

        X = align_to_expected_features(df, EXPECTED_FEATURES)
        sample = X.iloc[0].values.reshape(1, -1)

        def predict_fn(x):
            return model.predict(x.reshape(x.shape[0], 1, x.shape[1]))

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            mode="classification",
            class_names=["No Seizure", "Seizure"]
        )

        exp = explainer.explain_instance(
            data_row=sample[0],
            predict_fn=predict_fn,
            num_features=10
        )

        explanation_path = "/tmp/lime_explanation.html"
        exp.save_to_file(explanation_path)
        return send_file(explanation_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explain_shap', methods=['POST'])
def explain_shap():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)

        for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
            if col in df.columns:
                df = df.drop(columns=[col])

        X = align_to_expected_features(df, EXPECTED_FEATURES)

        background = X.sample(n=min(100, len(X)), random_state=42).values
        explainer = shap.KernelExplainer(
            model=lambda x: model.predict(x.reshape((x.shape[0], 1, x.shape[1]))),
            data=background
        )

        shap_values = explainer.shap_values(X.iloc[:1].values)

        shap.force_plot(
            explainer.expected_value[0],
            shap_values[0][0],
            features=X.iloc[0],
            matplotlib=True,
            show=False
        )

        shap_path = "/tmp/shap_explanation.png"
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()

        return send_file(shap_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
