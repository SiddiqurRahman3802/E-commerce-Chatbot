import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Union
from collections import Counter

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the Bitext retail e-commerce dataset.
    
    Args:
        file_path: Path to the dataset CSV file
    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(file_path)

def clean_text(text: str) -> str:
    """
    Clean and standardize text data.
    
    Args:
        text: Input text to clean
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep essential punctuation
    text = re.sub(r'[^\w\s\.,?!]', '', text)
    
    return text.strip()

def extract_language_features(tags: str) -> Dict[str, int]:
    """
    Extract language variation features from tags.
    
    Args:
        tags: String containing language variation tags
    Returns:
        Dictionary of language features
    """
    features = {
        'is_polite': 'P' in tags,
        'is_colloquial': 'Q' in tags,
        'has_offensive_language': 'W' in tags,
        'has_typos': 'Z' in tags,
        'is_basic_syntax': 'B' in tags,
        'is_question': 'I' in tags,
        'is_complex': 'C' in tags,
        'has_negation': 'N' in tags,
        'has_abbreviations': 'E' in tags,
        'is_keyword_mode': 'K' in tags
    }
    return features

def preprocess_ecommerce_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the e-commerce dataset with feature engineering.
    
    Args:
        df: Raw DataFrame from the Bitext dataset
    Returns:
        Preprocessed DataFrame with additional features
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Clean instruction and response text
    df_processed['clean_instruction'] = df_processed['instruction'].apply(clean_text)
    df_processed['clean_response'] = df_processed['response'].apply(clean_text)
    
    # Extract language features
    language_features = df_processed['tags'].apply(extract_language_features)
    df_processed = pd.concat([
        df_processed,
        pd.DataFrame(language_features.tolist())
    ], axis=1)
    
    # Add e-commerce specific features
    df_processed['instruction_length'] = df_processed['clean_instruction'].apply(len)
    df_processed['response_length'] = df_processed['clean_response'].apply(len)
    df_processed['word_count'] = df_processed['clean_instruction'].apply(lambda x: len(x.split()))
    
    # Create category and intent encodings
    df_processed['category_code'] = pd.Categorical(df_processed['category']).codes
    df_processed['intent_code'] = pd.Categorical(df_processed['intent']).codes
    
    # Extract question-related features
    df_processed['has_question_mark'] = df_processed['clean_instruction'].apply(lambda x: '?' in x)
    df_processed['starts_with_question_word'] = df_processed['clean_instruction'].apply(
        lambda x: any(x.startswith(w) for w in ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 'would'])
    )
    
    # Add e-commerce keyword features
    ecommerce_keywords = {
        'price': ['price', 'cost', 'expensive', 'cheap'],
        'shipping': ['shipping', 'delivery', 'ship', 'track'],
        'payment': ['pay', 'payment', 'card', 'refund'],
        'product': ['product', 'item', 'order', 'cart'],
        'return': ['return', 'exchange', 'refund', 'cancel']
    }
    
    for category, keywords in ecommerce_keywords.items():
        df_processed[f'has_{category}_keywords'] = df_processed['clean_instruction'].apply(
            lambda x: any(keyword in x for keyword in keywords)
        )
    
    # Add urgency indicators
    urgency_words = ['urgent', 'asap', 'emergency', 'immediately', 'quick']
    df_processed['is_urgent'] = df_processed['clean_instruction'].apply(
        lambda x: any(word in x for word in urgency_words)
    )
    
    # Add sentiment indicators (basic)
    positive_words = ['please', 'thank', 'good', 'great', 'help']
    negative_words = ['bad', 'wrong', 'issue', 'problem', 'error', 'complaint']
    
    df_processed['has_positive_tone'] = df_processed['clean_instruction'].apply(
        lambda x: any(word in x for word in positive_words)
    )
    df_processed['has_negative_tone'] = df_processed['clean_instruction'].apply(
        lambda x: any(word in x for word in negative_words)
    )
    
    return df_processed

