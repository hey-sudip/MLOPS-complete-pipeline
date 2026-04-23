import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Fix for newer nltk versions
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk_resources()

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()


# import os
# import logging
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
# import nltk

# # ==============================
# # NLTK Setup (Safe for production)
# # ==============================
# def download_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')

#     try:
#         nltk.data.find('corpora/stopwords')
#     except LookupError:
#         nltk.download('stopwords')

#     # Fix for newer nltk versions
#     try:
#         nltk.data.find('tokenizers/punkt_tab')
#     except LookupError:
#         nltk.download('punkt_tab')

# download_nltk_resources()

# # ==============================
# # Logger Setup
# # ==============================
# log_dir = 'logs'
# os.makedirs(log_dir, exist_ok=True)

# logger = logging.getLogger('data_preprocessing')
# logger.setLevel(logging.DEBUG)
# logger.propagate = False

# if not logger.handlers:
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.DEBUG)

#     log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
#     file_handler = logging.FileHandler(log_file_path)
#     file_handler.setLevel(logging.DEBUG)

#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     console_handler.setFormatter(formatter)
#     file_handler.setFormatter(formatter)

#     logger.addHandler(console_handler)
#     logger.addHandler(file_handler)

# # ==============================
# # Global objects (optimized)
# # ==============================
# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# # ==============================
# # Text Transformation
# # ==============================
# def transform_text(text):
#     if pd.isna(text):
#         return ""

#     text = text.lower()
#     tokens = nltk.word_tokenize(text)

#     tokens = [word for word in tokens if word.isalnum()]
#     tokens = [word for word in tokens if word not in stop_words]
#     tokens = [ps.stem(word) for word in tokens]

#     return " ".join(tokens)

# # ==============================
# # Data Preprocessing
# # ==============================
# def preprocess_df(df, text_column='text', target_column=None):
#     try:
#         logger.debug('Starting preprocessing')

#         df = df.copy()

#         # Encode target only if exists
#         if target_column and target_column in df.columns:
#             encoder = LabelEncoder()
#             df[target_column] = encoder.fit_transform(df[target_column])
#             logger.debug('Target column encoded')

#         # Remove duplicates
#         df = df.drop_duplicates(keep='first')
#         logger.debug('Duplicates removed')

#         # Transform text
#         if text_column in df.columns:
#             df[text_column] = df[text_column].apply(transform_text)
#             logger.debug('Text transformed')
#         else:
#             raise KeyError(f"{text_column} column not found")

#         return df

#     except Exception as e:
#         logger.error(f"Preprocessing failed: {e}")
#         raise

# # ==============================
# # Main Pipeline
# # ==============================
# def main(text_column='text', target_column='target'):
#     try:
#         logger.debug('Loading data...')

#         train_path = './data/raw/train.csv'
#         test_path = './data/raw/test.csv'

#         train_data = pd.read_csv(train_path)
#         test_data = pd.read_csv(test_path)

#         logger.debug('Data loaded successfully')

#         # Process train
#         train_processed = preprocess_df(train_data, text_column, target_column)

#         # Process test (no target encoding)
#         test_processed = preprocess_df(test_data, text_column, target_column=None)

#         # Save output
#         output_dir = './data/interim'
#         os.makedirs(output_dir, exist_ok=True)

#         train_processed.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
#         test_processed.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)

#         logger.debug(f'Processed data saved to {output_dir}')

#     except FileNotFoundError as e:
#         logger.error(f'File not found: {e}')
#     except Exception as e:
#         logger.error(f'Pipeline failed: {e}')
#         print(f"Error: {e}")

# # ==============================
# # Entry Point
# # ==============================
# if __name__ == '__main__':
#     main()