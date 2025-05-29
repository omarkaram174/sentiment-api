from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the saved model and vectorizer
print("🔁 Loading classifier and vectorizer...")
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing \"text\" in request'}), 400

    input_text = [data['text']]
    transformed = vectorizer.transform(input_text)
    pred = model.predict(transformed)[0]
    proba = max(model.predict_proba(transformed)[0])

    return jsonify({
        'class': 'Positive' if pred == 1 else 'Negative',
        'confidence': round(float(proba), 2)
    })

if __name__ == '__main__':
    print("✅ Starting Flask app...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
