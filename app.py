from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import gdown

app = Flask(__name__)

# Google Drive file IDs
model_id = "17BMDk5lx0IbqhWjifUNSKPuy_Ecs-b3k"
encoder_id = "1jseiOszY07Kn3FFOOBAhbt9OeMcuE5zF"

# File paths
model_path = "sentiment_model.pkl"
encoder_path = "text_encoder.pkl"

# Download if missing
if not os.path.exists(model_path):
    print("Downloading sentiment model...")
    gdown.download(f"https://drive.google.com/uc?id={model_id}", model_path, quiet=False, use_cookies=False)

if not os.path.exists(encoder_path):
    print("Downloading text encoder...")
    gdown.download(f"https://drive.google.com/uc?id={encoder_id}", encoder_path, quiet=False, use_cookies=False)

# Load models
classifier = joblib.load(model_path)
encoder = joblib.load(encoder_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing \"text\" in request'}), 400

    input_text = data['text']

    encoded = encoder.encode([input_text])
    pred = classifier.predict(encoded)[0]
    proba = max(classifier.predict_proba(encoded)[0])

    return jsonify({
        'class': 'Positive' if pred == 1 else 'Negative',
        'confidence': round(float(proba), 2)
    })

if __name__ == '__main__':
    print("✅ Starting Flask app...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

