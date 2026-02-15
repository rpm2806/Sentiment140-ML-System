# 🧠 Sentiment Analysis AI (My SVM Project)

Hi! I'm **Rupam**, and this is my lightweight, high-performance Sentiment Analysis web application.

It classifies text sentiment (Positive/Negative) in real-time, powered by **Machine Learning** (Linear SVM) and my own **Hybrid Rule-Based Logic**.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

## 📌 Why I Built This
I wanted to build a text classification system that could run efficiently on my laptop without needing heavy cloud resources. My goal was to balance **speed** and **accuracy**.

I experimented with several models (Logistic Regression, SVM, LSTM, DistilBERT) and ultimately chose **Linear Support Vector Machine (SVM)** as the best solution for my constraints.

## ⚠️ Challenges & My Design Decisions
**Why I chose SVM instead of BERT:**
During my research, I really wanted to use state-of-the-art Deep Learning models like DistilBERT. However, I faced two main challenges:
1.  **Hardware:** My local machine lacks a high-performance GPU, which made training large Transformer models incredibly slow and difficult.
2.  **Data:** The dataset I had was relatively small for deep learning to generalize well.

**The Result:** When I tried training BERT, it struggled to converge and often gave accuracy **below 50%**, or worse, it became biased and predicted "Negative" for everything.

**My Solution:** I pivoted to a **Linear SVM** combined with **TF-IDF vectorization**. This decision paid off:
-   **It's Fast:** I can retrain the whole model in seconds.
-   **It's Accurate:** I achieved **~89% accuracy** on my test set.
-   **It's Lightweight:** It runs instantly on any CPU.

## ✨ Key Features of My App
-   🚀 **Instant Predictions**: Real-time analysis (<100ms).
-   🧠 **Hybrid Logic**: I added a custom logic layer to catch positive phrases like "hard work" and "dedication" that the raw model initially misunderstood.
-   🌐 **Web Interface**: I built a clean, simple UI using [Streamlit](https://streamlit.io).
-   📊 **Confidence Scoring**: I show exactly how confident the model is.

## 🛠️ My Tech Stack
-   **Language**: Python 3.10
-   **Web Framework**: Streamlit
-   **ML Library**: Scikit-Learn (LinearSVC, TfidfVectorizer)
-   **NLP Utilities**: NLTK (Stopwords, Tokenization)
-   **Development**: Local Environment

## 🚀 How to Run My Code

1.  **Clone my repo**:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-ai.git
    cd sentiment-analysis-ai
    ```

2.  **Install the requirements**:
    ```bash
    pip install streamlit pandas scikit-learn nltk joblib
    ```

3.  **Launch the App**:
    ```bash
    streamlit run app.py
    ```

4.  **(Optional) Retrain**:
    I included a script to retrain the model if you add new data to `train_data.csv`:
    ```bash
    python train_model.py
    ```

## 📂 Project Structure
-   `app.py`: My Streamlit application code.
-   `train_model.py`: The script I wrote to train the SVM.
-   `saved_models/`: Where I save the trained `svm_model.pkl` and vectorizer.
-   `sentiment_analysis_phase1.ipynb`: My original research notebook (cleaned up).

---
*Created with ❤️ by Rupam*
