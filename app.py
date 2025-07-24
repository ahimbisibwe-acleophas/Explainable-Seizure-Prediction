import os
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ✅ Suppress TensorFlow GPU usage and verbose logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load model and scaler
model = tf.keras.models.load_model("cnn_seizure_model.h5")
scaler = joblib.load("scaler.pkl")

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Seizure Prediction</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background-color: #f8f9fa;
    }
    .container {
      margin-top: 80px;
    }
    #result {
      margin-top: 20px;
      display: none;
    }
  </style>
</head>
<body>
  <div class="container text-center">
    <h2>🧠 CNN Seizure Prediction System</h2>
    <p class="text-muted">Upload feature file (CSV format) to predict seizures</p>

    <div class="mb-3">
      <input class="form-control" type="file" id="fileInput" accept=".csv">
    </div>
    
    <button class="btn btn-primary" onclick="predict()">🧪 Predict</button>

    <div id="result" class="alert mt-4" role="alert"></div>
  </div>

  <script>
    async function predict() {
      const input = document.getElementById('fileInput');
      if (!input.files.length) {
        alert('Please upload a CSV file.');
        return;
      }

      const file = input.files[0];
      const reader = new FileReader();

      reader.onload = async function(event) {
        const csv = event.target.result;
        const rows = csv.trim().split('\n');
        const firstRow = rows[0].split(',').map(parseFloat);

        if (firstRow.some(isNaN)) {
          alert('Invalid data format in CSV. Ensure only numeric features are included.');
          return;
        }

        try {
          const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: firstRow })
          });

          const result = await response.json();
          const resultDiv = document.getElementById('result');
          
          if (result.error) {
            resultDiv.className = 'alert alert-danger';
            resultDiv.innerText = "❌ " + result.error;
          } else {
            resultDiv.className = result.prediction === 1
              ? 'alert alert-warning'
              : 'alert alert-success';

            resultDiv.innerHTML = `
              <strong>${result.prediction === 1 ? '⚠️ Seizure Likely!' : '✅ No Seizure Detected'}</strong><br>
              Probability: ${(result.probability * 100).toFixed(2)}%
            `;
          }
          resultDiv.style.display = 'block';
        } catch (error) {
          alert('Prediction failed: ' + error);
        }
      };

      reader.readAsText(file);
    }
  </script>
</body>
</html>

'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None, probability=None)

@app.route('/web_predict', methods=['POST'])
def web_predict():
    try:
        file = request.files['file']
        if not file:
            return render_template_string(HTML_TEMPLATE, prediction=None, probability=None)

        # ✅ Read CSV, extract features (first row)
        df = pd.read_csv(file)
        features = df.values[0].reshape(1, -1)  # only first row used

        # ✅ Scale and reshape
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        # ✅ Predict
        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=round(pred_proba, 4))

    except Exception as e:
        return f"Error: {e}", 500

# ✅ API endpoint remains available
@app.route('/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        features = data.get("features")

        if features is None:
            return jsonify({"error": "Missing 'features' in request"}), 400

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
