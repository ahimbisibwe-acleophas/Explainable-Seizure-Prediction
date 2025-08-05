# Disable GPU usage and suppress TensorFlow logs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
from lime import lime_tabular

app = Flask(__name__)

model = tf.keras.models.load_model("cnn_model_47features.h5", compile=False)
EXPECTED_FEATURES = model.input_shape[-1]

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Epileptic Seizure Predictor</title>
    <style>
        body {
            font-family: sans-serif;
            max-width: 900px;
            margin: auto;
            padding: 2em;
            background-color: #f9f9f9;
        }
        .two-columns {
            display: flex;
            gap: 20px;
            margin-top: 1em;
        }
        .column {
            flex: 1;
            background: #fff;
            padding: 1em;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <h1 style="text-align: center; color: #333;">Epileptic Seizure Predictor</h1>

    <form method="POST" action="/predict_and_explain" enctype="multipart/form-data" style="margin-top: 2em;">
        <label for="file"><strong>Select EEG Feature CSV File:</strong></label><br><br>
        <input type="file" id="file" name="file" accept=".csv" required><br><br>
        <button type="submit" style="padding: 10px 20px; font-size: 1em;">Run Prediction and Explain</button>
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

        <div class="two-columns">
            <div class="column">
                <h3>LIME Explanation</h3>
                <iframe src="/static/lime_explanation.html" width="100%" height="400px" style="border: none;"></iframe>
            </div>
            <div class="column">
                <h3>Feature Contributions</h3>
                <ul>
                    {% for feature, weight in contributions %}
                    <li><strong>{{ feature }}</strong>: {{ '%.4f'|format(weight) }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    {% endif %}
</body>
</html>
"""

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

@app.route('/predict_and_explain', methods=['POST'])
def predict_and_explain():
    try:
        file = request.files.get('file')
        if not file:
            return render_template_string(html_template, prediction=None)

        df = pd.read_csv(file)

        # Drop label columns if present
        for col in ['Class', 'Label', 'Target', 'Pre-Ictal Alert']:
            if col in df.columns:
                df = df.drop(columns=[col])

        X = align_to_expected_features(df, EXPECTED_FEATURES)
        X = X.reindex(sorted(X.columns), axis=1)

        features = np.array(X.iloc[0]).reshape(1, 1, EXPECTED_FEATURES)
        pred_proba = float(model.predict(features)[0][0])
        prediction = int(pred_proba > 0.5)

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
            num_features=min(10, EXPECTED_FEATURES)
        )

        explanation_path = "static/lime_explanation.html"
        os.makedirs("static", exist_ok=True)
        exp.save_to_file(explanation_path)

        # Get feature contributions for display
        contributions = [(label, weight) for label, weight in exp.as_list()]

        return render_template_string(html_template,
                                      prediction=prediction,
                                      probability=pred_proba,
                                      contributions=contributions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
