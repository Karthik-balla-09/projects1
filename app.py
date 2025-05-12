from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Home page route (GET request)
@app.route('/')
def home():
    return render_template('index.html')

# Predict route (POST request from form)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get text input from form
        news_article = request.form['news_text']
        
        # Vectorize the input
        vect_input = vectorizer.transform([news_article])
        
        # Predict
        prediction = model.predict(vect_input)[0]
        
        # Convert to label
        result = "Real News" if prediction == 1 else "Fake News"
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)