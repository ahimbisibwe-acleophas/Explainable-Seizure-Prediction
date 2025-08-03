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

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("cnn_model_47features.h5", compile=False)
EXPECTED_FEATURES = model.input_shape[-1]

# Basic HTML page (same as yours)
html_template = """<html>...unchanged HTML...</html>"""  # Use your original HTML template here

# Feature alignment helper
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
        X = X.reindex(sorted(X.columns), axis=1)
        sample = X.iloc[0].values

        def predict_fn(x):
            x_reshaped = x.reshape(x.shape[0], 1, x.shape[1])
            preds = model.predict(x_reshaped)
            return np.hstack([1 - preds, preds])

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            mode="classification",
            class_names=["No Seizure", "Seizure"],
            discretize_continuous=True
        )

        exp = explainer.explain_instance(
            data_row=sample,
            predict_fn=predict_fn,
            num_features=25  # UPDATED: now showing 25 features
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
        X = X.reindex(sorted(X.columns), axis=1)
        feature_names = X.columns.tolist()

        background = X.sample(n=min(100, len(X)), random_state=42)
        background_tensor = tf.convert_to_tensor(background.values.reshape(-1, 1, EXPECTED_FEATURES), dtype=tf.float32)

        sample = X.iloc[[0]].values.reshape(1, 1, EXPECTED_FEATURES)
        sample_2d = X.iloc[[0]].values  # Needed for shap_values display

        explainer = shap.GradientExplainer(model, background_tensor)
        shap_values = explainer.shap_values(sample)

        shap_path = "/tmp/shap_explanation.png"
        shap.plots._waterfall.waterfall_legacy(
            shap_values[0][0],
            feature_names=feature_names,
            max_display=25,  # Show up to 25 features
            show=False
        )

        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        return send_file(shap_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
