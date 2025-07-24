import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU only

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("cnn_seizure_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "🧠 CNN Seizure Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json(force=True)
        features = data.get("features")

        if features is None:
            return jsonify({"error": "Missing 'features' in request"}), 400

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)  # Shape: (1, num_features)

        # Standardize features using saved scaler
        features_scaled = scaler.transform(features)

        # Reshape for Conv1D input (batch, time_step=1, features)
        features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

        # Get prediction probability
        pred_proba = model.predict(features_scaled)[0][0]

        # Binary classification output
        prediction = int(pred_proba > 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": float(pred_proba)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True)
