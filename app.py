from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
import scipy.signal as signal

# Initializing Flask app
app = Flask(__name__)

# Load trained CNN model
model = tf.keras.models.load_model("cnn_seizure_prediction_model.keras")

# Band power definitions (Hz)
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# Feature extraction using Welch's method
def extract_band_power(eeg_data, sfreq=256):
    freqs, psd = signal.welch(eeg_data, fs=sfreq, nperseg=sfreq)
    features = []

    for band in bands.values():
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        band_power = np.mean(psd[..., idx_band], axis=-1)
        features.append(band_power)

    return np.stack(features, axis=-1)  # Shape: (channels, 5)

# Preprocess input
def preprocess_input(raw_data):
    """
    raw_data: dict with keys 'eeg' and 'sfreq' (optional)
    'eeg' should be a 2D list/array (channels x samples)
    """
    eeg = np.array(raw_data["eeg"])
    sfreq = raw_data.get("sfreq", 256)

    # Extract band power features per channel
    band_power = np.array([extract_band_power(chan, sfreq) for chan in eeg])

    # Flatten features (e.g., shape (channels, 5) → (channels*5,))
    feature_vector = band_power.flatten()

    # Reshape for CNN input (e.g., (54, 1) if 10 channels)
    reshaped = feature_vector.reshape(-1, 1)

    return reshaped

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = preprocess_input(data)

        # Add batch dimension: (1, features, 1)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)[0][0]
        label = int(prediction >= 0.5)

        return jsonify({
            "prediction": float(prediction),
            "label": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(debug=True)
