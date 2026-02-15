# 🧠 Sentiment Analysis AI

A fast, lightweight Sentiment Analysis web application powered by a Linear Support Vector Machine (SVM).

## Features
-   🚀 **Instant Predictions**: Real-time sentiment analysis.
-   ⚡ **Lightweight**: Uses SVM + TF-IDF (runs on any CPU).
-   🧠 **Hybrid Logic**: Keyword boosting for "hard work", "dedication" etc.
-   🌐 **Web Interface**: Built with [Streamlit](https://streamlit.io).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-ai.git
    cd sentiment-analysis-ai
    ```

2.  Install dependencies:
    ```bash
    pip install streamlit pandas scikit-learn nltk joblib
    ```

3.  (Optional) Retrain the model:
    ```bash
    python train_model.py
    ```

## Usage

Run the app:
```bash
streamlit run app.py
```

## Model Details
-   **Algorithm**: LinearSVC (Scikit-Learn)
-   **Features**: TF-IDF Vectorizer (5000 max features)
-   **Accuracy**: ~89% on IMDB Dataset
