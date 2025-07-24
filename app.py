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

# ✅ HTML template with upload form
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧠 CNN Seizure Prediction</title>
</head>
<body>
  <div class="card">
    <h1>Seizure Predictor</h1>
    <form id="upload-form">
      <input type="file" id="csv-file" name="file" accept=".csv" required>
      <button type="submit" class="predict-button">Predict</button>
    </form>
    <div id="results" class="section"></div>
  </div>

  <script>
    document.getElementById("upload-form").addEventListener("submit", async function(event) {
      event.preventDefault();
      const file = document.getElementById("csv-file").files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const html = await response.text();
      document.getElementById("results").innerHTML = response.ok ? html : `<pre>${html}</pre>`;
    });
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
