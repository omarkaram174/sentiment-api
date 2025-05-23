from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load your models
classifier = joblib.load('sentiment_model.pkl')
encoder = joblib.load('text_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request'}), 400

    input_text = data['text']

    encoded = encoder.encode([input_text])
    pred = classifier.predict(encoded)[0]
    proba = max(classifier.predict_proba(encoded)[0])

    return jsonify({
        'class': 'Positive' if pred == 1 else 'Negative',
        'confidence': round(float(proba), 2)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
