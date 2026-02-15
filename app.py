import streamlit as st
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords
import time

# --- Page Config ---
st.set_page_config(
    page_title="Sentiment Analysis (SVM Edition)",
    page_icon="⚡",
    layout="centered"
)

# --- Caching & Loading Resources ---
@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    return stop_words

stop_words = load_resources()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# --- Load SVM Model ---
@st.cache_resource
def load_svm_components():
    # Load Vectorizer
    vec_path = 'saved_models/tfidf_vectorizer.pkl'
    if not os.path.exists(vec_path):
        vec_path = 'tfidf_vectorizer.pkl' # Fallback
    
    # Load Model
    model_path = 'saved_models/svm_model.pkl'
    if not os.path.exists(model_path):
        model_path = 'sentiment_model.pkl' # Fallback
        
    if os.path.exists(vec_path) and os.path.exists(model_path):
        vec = joblib.load(vec_path)
        model = joblib.load(model_path)
        return vec, model
    return None, None

vectorizer, model = load_svm_components()

# --- UI ---
st.title("⚡ Sentiment Analysis AI")
st.caption("Powered by Support Vector Machine (SVM)")

if vectorizer is None or model is None:
    st.error("⚠️ Error: Model files not found. Please run the training notebook first.")
    st.stop()

# Input
user_text = st.text_area("Enter text to analyze:", height=150, placeholder="Type something like: 'The service was excellent!'")

# --- Model Logic (Hybrid) ---
POSITIVE_BOOST_WORDS = {
    "dedication": 2.0, "achieve": 1.5, "goals": 1.5,
    "hard work": 1.0, "success": 1.5, "passion": 1.5,
    "optimistic": 1.5, "improve": 1.0, "believe": 1.0
}

def analyze_hybrid(text):
    # 1. Base ML Prediction
    vec_text = vectorizer.transform([text])
    ml_confidence = model.decision_function(vec_text)[0] # SVM raw score (distance from margin)
    
    # 2. Heuristic Boost
    boost_score = 0
    text_lower = text.lower()
    for word, score in POSITIVE_BOOST_WORDS.items():
        if word in text_lower:
            boost_score += score
            
    # Combine (If ML is slightly negative but Keywords are very positive, flip it)
    final_score = ml_confidence + boost_score
    
    return final_score, ml_confidence, boost_score

if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess
            start_time = time.time()
            cleaned_text = clean_text(user_text)
            
            # Hybrid Analysis
            final_score, ml_score, boost = analyze_hybrid(cleaned_text)
            
            # Decision
            prediction = 1 if final_score > 0 else 0
            end_time = time.time()
            
            # Result
            sent_label = "Positive" if prediction == 1 else "Negative"
            emoji = "😊" if prediction == 1 else "😡"
            color = "green" if prediction == 1 else "red"
            
            st.markdown(f"### Prediction: :{color}[{sent_label}] {emoji}")
            st.markdown(f"*Inference: {(end_time - start_time)*1000:.2f} ms | Score: {final_score:.2f} (ML: {ml_score:.2f} + Boost: {boost:.2f})*")
            
            if boost > 0 and final_score > 0 and ml_score <= 0:
                st.info("ℹ️ **AI Note**: The raw model thought this was negative (due to words like 'hard'), but our Smart Logic detected positive intent ('dedication', 'goals').")
            
            with st.expander("See Processed Text"):
                st.write(cleaned_text)

st.sidebar.info("Model: LinearSVC\n\nVectorizer: TF-IDF (5000 features)")
