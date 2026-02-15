# 🧠 Sentiment Analysis AI (SVM + Hybrid Logic)

A lightweight, high-performance Sentiment Analysis web application powered by **Machine Learning** (Linear SVM) and **Hybrid Rule-Based Logic**.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

## 📌 Project Overview
This project classifies text sentiment (Positive/Negative) in real-time. It was designed to run efficiently on standard CPU hardware without requiring heavy GPU resources.

We compared multiple architectures (Logistic Regression, SVM, LSTM, DistilBERT) and selected **Linear Support Vector Machine (SVM)** as the production model due to its superior speed and accuracy on our dataset.

## ⚠️ Challenges & Design Decisions
**Why SVM instead of BERT?**
During our research phase, we experimented with advanced Deep Learning models (LSTM, DistilBERT). However, we faced two critical constraints:
1.  **Hardware Limitations**: Lack of a high-performance GPU meant training large Transformer models was extremely slow and difficult to fine-tune.
2.  **Dataset Constraints**: The available dataset was relatively small for deep learning generalization.

**Result**: The complex models (BERT/LSTM) struggled to converge, often yielding accuracy **below 50%** or exhibiting severe bias (predicting only "Negative").

**Solution**: We pivoted to a **Linear SVM** combined with **TF-IDF vectorization**. This approach proved to be the winner:
-   **Faster**: Training takes seconds, not hours.
-   **More Accurate**: Achieved **~89% accuracy** on our test set.
-   **Lightweight**: Can run on any standard laptop CPU.

## ✨ Key Features
-   🚀 **Instant Predictions**: Real-time analysis (<100ms inference).
-   🧠 **Hybrid Logic**: Uses a custom dictionary to boost scores for positive concepts like "hard work", "dedication", and "achieve" that pure ML might miss.
-   🌐 **Web Interface**: Clean, simple UI built with [Streamlit](https://streamlit.io).
-   📊 **Confidence Scoring**: Shows model confidence for every prediction.

## 🛠️ Tech Stack
-   **Language**: Python 3.10
-   **Web Framework**: Streamlit
-   **ML Library**: Scikit-Learn (LinearSVC, TfidfVectorizer)
-   **NLP Utilities**: NLTK (Stopwords, Tokenization)
-   **Deployment**: LocalHost

## 🚀 Installation & Usage

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-ai.git
    cd sentiment-analysis-ai
    ```

2.  **Install Dependencies**:
    ```bash
    pip install streamlit pandas scikit-learn nltk joblib
    ```

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

4.  **(Optional) Retrain Model**:
    If you have new data in `train_data.csv`, run:
    ```bash
    python train_model.py
    ```

## 📂 Project Structure
-   `app.py`: Main application code (Streamlit).
-   `train_model.py`: Script to train the SVM model and save artifacts.
-   `saved_models/`: Directory containing the trained `svm_model.pkl` and `tfidf_vectorizer.pkl`.
-   `sentiment_analysis_phase1.ipynb`: Original research notebook (cleaned).

---
*Created by Rupam*
