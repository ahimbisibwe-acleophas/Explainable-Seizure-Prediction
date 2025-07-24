import os

# ✅ Suppress TensorFlow GPU usage and verbose logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # Force TensorFlow to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         # Suppress TensorFlow warnings/info logs

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load pre-trained CNN model and feature scaler
model = tf.keras.models.load_model("cnn_seizure_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "🧠 CNN Seizure Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Parse incoming JSON
        data = request.get_json(force=True)
        features = data.get("features")

        if features is None:
            return jsonify({
                "error": "Missing 'features' in request. Please provide a list of features."
            }), 400

        # ✅ Convert features to NumPy array
        features = np.array(features).reshape(1, -1)  # Shape: (1, num_features)

        # ✅ Apply scaler
        features_scaled = scaler.transform(features)

        # ✅ Reshape for Conv1D input (batch_size, time_step, features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        # ✅ Get prediction
        pred_proba = float(model.predict(features_scaled)[0][0])
        prediction = int(pred_proba > 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": pred_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Main entrypoint (for local or cloud run)
if __name__ == '__main__':
    # Use Render-provided port if available
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
