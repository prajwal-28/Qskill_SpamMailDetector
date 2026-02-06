# Spam Mail Detector


## 📖 Overview
**Spam Mail Detector** is a classifier designed to classify SMS messages as either "Spam" or "Ham" (non-spam). Leveraging the **UCI SMS Spam Collection** dataset, this project utilizes Natural Language Processing (NLP) techniques and supervised classification algorithms to achieve high-accuracy detection.

This project demonstrates a complete ML pipeline: from data ingestion and text preprocessing to feature extraction (TF-IDF) and model evaluation.

## ✨ Key Features
*   **Robust Preprocessing**: Automated text cleaning pipeline including lowercasing, punctuation removal, and stopword filtering.
*   **Feature Extraction**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text data into meaningful numeric vectors.
*   **Multiple Classifiers**: Implements and compares **Naive Bayes (MultinomialNB)** and **Logistic Regression** models.
*   **High Performance**: Achieves **>98% accuracy** on the test dataset.
*   **Detailed Metrics**: detailed classification reports including Precision, Recall, and F1-Score.

## 🛠 Tech Stack
*   **Language**: Python 3.x
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn
*   **NLP**: NLTK (Natural Language Toolkit)
*   **Visualization**: Matplotlib, Seaborn (included in dependencies)

## 📂 Project Structure
```text
spam_mail/
├── data/                  # Dataset storage (SMS Spam Collection)
│   ├── SMSSpamCollection
│   └── ...
├── src/                   # Source code
│   ├── preprocess.py      # Text cleaning and data loading logic
│   └── train.py           # Model training and evaluation script
├── notebooks/             # Jupyter Notebooks for EDA (if applicable)
├── venv/                  # Virtual environment
└── requirements.txt       # Project dependencies
```

## 🚀 Installation & Use

### Prerequisites
Ensure you have Python installed on your system.

### 1. Clone the Repository (or navigate to directory)
```bash
cd d:/spam_mail
```

### 2. Set Up Virtual Environment
It is recommended to use a virtual environment.
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline
Execute the training script to load data, train models, and view results.
```bash
python src/train.py
```

## 📊 Results

The models were evaluated on an 80/20 train-test split. The **Naive Bayes** classifier proved to be the most effective for this task.

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|-------|----------|------------------|---------------|-----------------|
| **Naive Bayes** | **98.03%** | **1.00** | **0.85** | **0.92** |
| Logistic Regression | 95.96% | 0.97 | 0.72 | 0.83 |

> **Note**: Naive Bayes achieved **zero false positives**, making it the safer choice for a spam filter to avoid blocking legitimate emails.
