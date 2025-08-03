import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import lime
import lime.lime_tabular
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("model.h5")
EXPECTED_FEATURES = joblib.load("expected_features.pkl")

def align_to_expected_features(df, expected_features):
    df = df.copy()
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features]

@app.route('/')
def home():
    return render_template_string("""
    <h2>Upload CSV for Prediction and Explanation</h2>
    <form method="POST" action="/predict_csv" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Predict">
    </form>
    <form method="POST" action="/explain_lime" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Explain with LIME">
    </form>
    <form method="POST" action="/explain_shap" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Explain with SHAP">
    </form>
    """)

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
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
        X_reshaped = X.values.reshape(X.shape[0], 1, X.shape[1])

        preds = model.predict(X_reshaped)
        results = pd.DataFrame({"Prediction_Probability": preds.flatten()})
        results["Prediction_Label"] = (results["Prediction_Probability"] > 0.5).astype(int)

        output_path = "/tmp/predictions.csv"
        results.to_csv(output_path, index=False)
        return send_file(output_path, mimetype='text/csv')

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

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            mode="classification",
            class_names=["No Seizure", "Seizure"],
            discretize_continuous=True
        )

        exp = explainer.explain_instance(
            data_row=sample,
            predict_fn=predict_fn,
            num_features=25
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

        sample = X.iloc[[0]]
        background = shap.sample(X, nsamples=50, random_state=0)

        def wrapped_model(x):
            reshaped = x.reshape(x.shape[0], 1, x.shape[1])
            return model.predict(reshaped)

        explainer = shap.KernelExplainer(wrapped_model, background.values)
        shap_values = explainer.shap_values(sample.values)

        shap.summary_plot(
            shap_values[0], sample.values,
            feature_names=X.columns.tolist(),
            show=False,
            plot_type="bar"
        )

        shap_path = "/tmp/shap_explanation.png"
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()

        return send_file(shap_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
