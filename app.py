from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['job_text']
    cleaned = ' '.join(text.lower().split())

    # Keyword rule check
    suspicious_keywords = ['₹', 'earn', 'form filling', 'registration fee', 'click here', 'no experience', 'work from home', 'pay to start']
    flagged = any(keyword in cleaned for keyword in suspicious_keywords)

    # ML prediction
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Result
    if prediction == 1 or flagged:
        result = "🔴 FAKE Job Posting (Suspicious)"
    else:
        result = "🟢 Real Job Posting"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

