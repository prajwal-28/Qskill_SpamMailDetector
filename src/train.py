import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import preprocess

def train_and_evaluate():
    # 1. Load Data
    data_path = "d:/spam_mail/data/SMSSpamCollection"
    print("Loading data...")
    df = preprocess.load_data(data_path)
    
    # 2. Preprocess Data
    print("Preprocessing data (this might take a moment)...")
    # Apply preprocessing to the 'message' column
    df['processed_message'] = df['message'].apply(preprocess.preprocess_text)
    
    # 3. Feature Extraction
    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features for efficiency
    X = vectorizer.fit_transform(df['processed_message'])
    y = df['label']
    
    # 4. Split Data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Model Training & Evaluation
    
    # Naive Bayes
    print("\n--- Naive Bayes ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_nb))
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_lr))

    return nb_model, lr_model, vectorizer

if __name__ == "__main__":
    train_and_evaluate()
