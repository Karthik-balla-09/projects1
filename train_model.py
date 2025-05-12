import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the dataset
data = pd.read_csv("news.csv")

# Check if required columns exist
if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("CSV must have 'text' and 'label' columns")

# Drop missing values
data.dropna(subset=['text', 'label'], inplace=True)

# Convert labels to int if not already
data['label'] = data['label'].astype(int)

# Features and target
X = data['text']
y = data['label']

# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save model and vectorizer
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')