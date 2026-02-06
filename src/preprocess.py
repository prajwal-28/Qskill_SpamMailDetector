import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    # The dataset is tab-separated and has no header
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    return df

def preprocess_text(text):
    """
    Preprocesses the text:
    1. Lowercasing
    2. Removing punctuation
    3. Tokenization
    4. Removing stopwords
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Join back to string (optional: depending on if we want list of tokens or string)
    # TF-IDF vectorizer usually expects a string
    return " ".join(filtered_tokens)

if __name__ == "__main__":
    data_path = "d:/spam_mail/data/SMSSpamCollection"
    try:
        print(f"Loading data from {data_path}...")
        df = load_data(data_path)
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print(df.head())
        
        print("\nPreprocessing first 5 messages...")
        df['processed_message'] = df['message'].apply(preprocess_text)
        print(df[['message', 'processed_message']].head())
        
    except Exception as e:
        print(f"Error: {e}")
