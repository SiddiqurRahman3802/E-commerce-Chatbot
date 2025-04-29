import pandas as pd
import numpy as np
import re
from typing import Optional, Dict

class EcommerceDataProcessor:
    """
    Wraps data loading and preprocessing for the Bitext retail e-commerce dataset.
    """

    def __init__(self, file_path: str):
        """
        Args:
            file_path: Path to the CSV dataset.
        """
        self.file_path: str = file_path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Load raw data from CSV into a DataFrame.
        """
        self.df = pd.read_csv(self.file_path)
        return self.df

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a piece of text: lowercase, remove HTML tags, non-alphanumeric chars,
        and collapse whitespace.

        Args:
            text: raw text string
        Returns:
            cleaned text string
        """
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def extract_language_features(tags: str) -> Dict[str, bool]:
        """
        Extract boolean features from the `tags` string.
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

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the e-commerce dataset with feature engineering.

        Returns:
            Preprocessed DataFrame with additional features
        """
        df_processed = self.df.copy()

        # Clean instruction and response text
        df_processed['clean_instruction'] = df_processed['instruction'].apply(self.clean_text)
        df_processed['clean_response'] = df_processed['response'].apply(self.clean_text)

        # Extract language features from tags
        language_features = df_processed['tags'].apply(self.extract_language_features)
        df_processed = pd.concat([df_processed, pd.DataFrame(language_features.tolist())], axis=1)

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
                lambda x, kw=keywords: any(word in x for word in kw)
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

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline: load -> preprocess.

        Returns:
            Final preprocessed DataFrame.
        """
        self.load()
        return self.preprocess()
