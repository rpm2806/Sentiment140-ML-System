import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import re
import string
import nltk
from nltk.corpus import stopwords
import joblib
import os
import sys

# Setup
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Download resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def train():
    print("--- Starting Training (SVM) ---")
    
    # 1. Load Data
    if os.path.exists('train_data.csv'):
        print("Loading Train Data...")
        train_df = pd.read_csv('train_data.csv')
        print(f"Loaded {len(train_df)} rows.")
    else:
        print("Error: train_data.csv not found!")
        return

    # 2. Preprocess
    print("Cleaning text...")
    # Fast cleaning
    train_df['clean_text'] = train_df['sentence'].apply(clean_text)

    # 3. Vectorize
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['sentiment']

    # 4. Train
    print("Training Linear SVM...")
    svm_model = LinearSVC(dual=False, random_state=42)
    svm_model.fit(X_train, y_train)
    print("Training complete.")

    # 5. Save
    print("Saving artifacts...")
    joblib.dump(vectorizer, 'saved_models/tfidf_vectorizer.pkl')
    joblib.dump(svm_model, 'saved_models/svm_model.pkl')
    print("✅ Models saved to 'saved_models/'")

if __name__ == "__main__":
    train()
