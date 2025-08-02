import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, send_file, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import lime
import lime.lime_tabular
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# === Flask App Setup ===
app = Flask(__name__)

# === Load Model and Preprocessor ===
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")
EXPECTED_FEATURES = 47

# === HTML Template ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Seizure Prediction</title>
</head>
<body style="font-family:Arial; max-width:600px; margin:auto;">
    <h2>CNN Seizure Prediction</h2>
    <form action="/predict_csv" method="post" enctype="multipart/form-data">
        <label>Upload CSV File:</label><br><br>
        <input type="file" name="file"><br><br>
        <input type="submit" value="Predict">
    </form>
    <br><hr><br>
    <form action="/explain_lime" method="post" enctype="multipart/form-data">
        <label>Explain with LIME:</label><br><br>
        <input type="file" name="file"><br><br>
        <input type="submit" value="Explain LIME">
    </form>
    <br><hr><br>
    <form action="/explain_shap" method="post" enctype="multipart/form-data">
        <label>Explain with SHAP:</label><br><br>
        <input type="file" name="file"><br><br>
        <input type="submit" value="Explain SHAP">
    </form>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# === Helper Function ===
def align_to_expected_features(df, expected_n):
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] > expected_n:
        df = df.iloc[:, :expected_n]
    elif df.shape[1] < expected_n:
        for i in range(df.shape[1], expected_n):
            df[f'dummy_{i}'] = 0.0
    return df

# === Prediction Route ===
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = align_to_expected_features(df, EXPECTED_FEATURES)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]))
    preds = (preds > 0.5).astype(int).flatten()

    return jsonify({"predictions": preds.tolist()})

# === LIME Explanation ===
@app.route('/explain_lime', methods=['POST'])
def explain_lime():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = align_to_expected_features(df, EXPECTED_FEATURES)
    X_scaled = scaler.transform(X)

    def predict_fn(x):
        return model.predict(x.reshape(x.shape[0], 1, x.shape[1]))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        mode='classification',
        feature_names=X.columns.tolist(),
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        data_row=X_scaled[0],
        predict_fn=predict_fn,
        num_features=EXPECTED_FEATURES
    )

    fig = exp.as_pyplot_figure()
    lime_path = "/tmp/lime_explanation.png"
    fig.savefig(lime_path, bbox_inches='tight')
    plt.close(fig)

    return send_file(lime_path, mimetype='image/png')

# === SHAP Explanation ===
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
        X_scaled = scaler.transform(X)

        def predict_wrapper(x):
            return model.predict(x.reshape(x.shape[0], 1, x.shape[1]))

        background = X_scaled[:50]
        explainer = shap.KernelExplainer(predict_wrapper, background)
        sample = X_scaled[[0]]

        shap_values = explainer.shap_values(sample)

        if len(shap_values[0][0]) != EXPECTED_FEATURES:
            return jsonify({
                "error": f"Feature length mismatch: SHAP returned {len(shap_values[0][0])}, expected {EXPECTED_FEATURES}"
            }), 500

        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value[0],
            shap_values[0][0],
            X.columns.tolist(),
            show=False
        )

        shap_path = "/tmp/shap_explanation.html"
        with open(shap_path, "w") as f:
            f.write(shap.save_html(force_plot))

        return send_file(shap_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run Locally ===
if __name__ == '__main__':
    app.run(debug=True)
