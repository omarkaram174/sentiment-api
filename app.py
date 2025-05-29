from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

# Load classifier
print("🔁 Loading classifier...")
classifier = joblib.load("sentiment_model.pkl")

# Load Hugging Face model + tokenizer
print("🔁 Loading sentence transformer model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to get sentence embedding
def get_embedding(texts):
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request'}), 400

    input_text = data['text']
    embedding = get_embedding([input_text])
    pred = classifier.predict(embedding)[0]
    proba = max(classifier.predict_proba(embedding)[0])

    return jsonify({
        'class': 'Positive' if pred == 1 else 'Negative',
        'confidence': round(float(proba), 2)
    })

# Run the app
if __name__ == '__main__':
    print("✅ Starting Flask app from entrypoint...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
