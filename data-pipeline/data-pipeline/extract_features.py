"""
Extract ML features from parsed emails for spam detection.

This script:
1. Reads parsed email Parquet from Azure Blob Storage
2. Extracts text, structural, and metadata features
3. Fits TF-IDF vectorizer on corpus
4. Writes feature Parquet files to Feast offline store location

Input:  abfss://datasets@<storage_account>.dfs.core.windows.net/processed/emails/
Output: abfss://feast@<storage_account>.dfs.core.windows.net/features/
"""

import os
import re
import logging
import pickle
from datetime import datetime
from typing import Optional

import ray
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from adlfs import AzureBlobFileSystem
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Azure configuration
STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

INPUT_CONTAINER = "datasets"
INPUT_PATH = "processed/emails/all_emails.parquet"
OUTPUT_CONTAINER = "feast"
OUTPUT_PATH = "features"

# Feature engineering parameters
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)


def get_azure_fs() -> AzureBlobFileSystem:
    """Create Azure Blob FileSystem client."""
    return AzureBlobFileSystem(
        account_name=STORAGE_ACCOUNT_NAME,
        account_key=STORAGE_ACCOUNT_KEY
    )


# ============================================================================
# Text Feature Extraction
# ============================================================================

def count_urls(text: str) -> int:
    """Count URLs in text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return len(re.findall(url_pattern, text or ""))


def count_emails_in_text(text: str) -> int:
    """Count email addresses in text body."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return len(re.findall(email_pattern, text or ""))


def uppercase_ratio(text: str) -> float:
    """Calculate ratio of uppercase characters."""
    if not text:
        return 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)


def count_exclamations(text: str) -> int:
    """Count exclamation marks."""
    return (text or "").count("!")


def count_question_marks(text: str) -> int:
    """Count question marks."""
    return (text or "").count("?")


def avg_word_length(text: str) -> float:
    """Calculate average word length."""
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def word_count(text: str) -> int:
    """Count words in text."""
    return len((text or "").split())


def char_count(text: str) -> int:
    """Count characters in text."""
    return len(text or "")


# ============================================================================
# Structural Feature Extraction
# ============================================================================

def has_html(html_body: Optional[str]) -> int:
    """Check if email has HTML body."""
    return 1 if html_body else 0


def html_to_text_ratio(text_body: str, html_body: Optional[str]) -> float:
    """Calculate HTML to text length ratio."""
    text_len = len(text_body or "")
    html_len = len(html_body or "")
    if text_len == 0:
        return float(html_len) if html_len > 0 else 0.0
    return html_len / text_len


def subject_length(subject: str) -> int:
    """Calculate subject length."""
    return len(subject or "")


def subject_has_re(subject: str) -> int:
    """Check if subject starts with RE: or Re:."""
    return 1 if (subject or "").lower().startswith("re:") else 0


def subject_has_fwd(subject: str) -> int:
    """Check if subject starts with FWD: or Fwd:."""
    s = (subject or "").lower()
    return 1 if s.startswith("fwd:") or s.startswith("fw:") else 0


def subject_all_caps(subject: str) -> int:
    """Check if subject is all uppercase."""
    s = subject or ""
    alpha_chars = [c for c in s if c.isalpha()]
    if len(alpha_chars) < 3:
        return 0
    return 1 if all(c.isupper() for c in alpha_chars) else 0


# ============================================================================
# Header Feature Extraction
# ============================================================================

def has_x_mailer(x_mailer: Optional[str]) -> int:
    """Check if X-Mailer header is present."""
    return 1 if x_mailer else 0


def sender_domain_length(sender_domain: str) -> int:
    """Calculate sender domain length."""
    return len(sender_domain or "")


def sender_has_numbers(sender_email: str) -> int:
    """Check if sender email contains numbers."""
    local_part = (sender_email or "").split("@")[0]
    return 1 if any(c.isdigit() for c in local_part) else 0


def received_hop_count_feature(hop_count: int) -> int:
    """Return received hop count (already extracted)."""
    return hop_count or 0


# ============================================================================
# Temporal Feature Extraction
# ============================================================================

def extract_hour(date: Optional[datetime]) -> int:
    """Extract hour from datetime."""
    if pd.isna(date):
        return -1
    return date.hour


def extract_day_of_week(date: Optional[datetime]) -> int:
    """Extract day of week (0=Monday, 6=Sunday)."""
    if pd.isna(date):
        return -1
    return date.weekday()


def is_weekend(date: Optional[datetime]) -> int:
    """Check if email was sent on weekend."""
    if pd.isna(date):
        return -1
    return 1 if date.weekday() >= 5 else 0


def is_night_hour(date: Optional[datetime]) -> int:
    """Check if sent between 10pm and 6am."""
    if pd.isna(date):
        return -1
    hour = date.hour
    return 1 if hour >= 22 or hour < 6 else 0


# ============================================================================
# Spam Indicator Features
# ============================================================================

SPAM_KEYWORDS = [
    "free", "winner", "click here", "act now", "limited time",
    "congratulations", "urgent", "money", "cash", "prize",
    "viagra", "pharmacy", "buy now", "order now", "subscribe",
    "unsubscribe", "remove", "opt-out", "credit card", "wire transfer"
]


def spam_keyword_count(text: str) -> int:
    """Count spam-related keywords in text."""
    text_lower = (text or "").lower()
    return sum(1 for kw in SPAM_KEYWORDS if kw in text_lower)


def has_unsubscribe(text: str) -> int:
    """Check for unsubscribe link/text."""
    return 1 if "unsubscribe" in (text or "").lower() else 0


# ============================================================================
# Main Feature Extraction Pipeline
# ============================================================================

@ray.remote
def extract_features_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    """Extract non-TF-IDF features for a batch of emails."""
    
    features = pd.DataFrame()
    features["email_id"] = df_batch["email_id"]
    features["label"] = df_batch["label"]
    
    # Text features
    features["url_count"] = df_batch["body_text"].apply(count_urls)
    features["email_count_in_body"] = df_batch["body_text"].apply(count_emails_in_text)
    features["uppercase_ratio"] = df_batch["body_text"].apply(uppercase_ratio)
    features["exclamation_count"] = df_batch["body_text"].apply(count_exclamations)
    features["question_mark_count"] = df_batch["body_text"].apply(count_question_marks)
    features["avg_word_length"] = df_batch["body_text"].apply(avg_word_length)
    features["word_count"] = df_batch["body_text"].apply(word_count)
    features["char_count"] = df_batch["body_text"].apply(char_count)
    
    # Structural features
    features["has_html"] = df_batch["body_html"].apply(has_html)
    features["html_to_text_ratio"] = df_batch.apply(
        lambda r: html_to_text_ratio(r["body_text"], r["body_html"]), axis=1
    )
    features["subject_length"] = df_batch["subject"].apply(subject_length)
    features["subject_has_re"] = df_batch["subject"].apply(subject_has_re)
    features["subject_has_fwd"] = df_batch["subject"].apply(subject_has_fwd)
    features["subject_all_caps"] = df_batch["subject"].apply(subject_all_caps)
    
    # Header features
    features["has_x_mailer"] = df_batch["x_mailer"].apply(has_x_mailer)
    features["sender_domain_length"] = df_batch["sender_domain"].apply(sender_domain_length)
    features["sender_has_numbers"] = df_batch["sender_email"].apply(sender_has_numbers)
    features["received_hop_count"] = df_batch["received_hop_count"].apply(received_hop_count_feature)
    
    # Temporal features
    features["hour_of_day"] = df_batch["date"].apply(extract_hour)
    features["day_of_week"] = df_batch["date"].apply(extract_day_of_week)
    features["is_weekend"] = df_batch["date"].apply(is_weekend)
    features["is_night_hour"] = df_batch["date"].apply(is_night_hour)
    
    # Spam indicator features
    features["spam_keyword_count"] = df_batch["body_text"].apply(spam_keyword_count)
    features["has_unsubscribe"] = df_batch["body_text"].apply(has_unsubscribe)
    
    # Add event timestamp for Feast
    features["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    
    return features


def fit_tfidf_vectorizer(texts: pd.Series) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on corpus."""
    logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")
    
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        min_df=5,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # Fill NaN with empty string
    texts_clean = texts.fillna("")
    vectorizer.fit(texts_clean)
    
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return vectorizer


def transform_tfidf(texts: pd.Series, vectorizer: TfidfVectorizer) -> np.ndarray:
    """Transform texts to TF-IDF vectors."""
    texts_clean = texts.fillna("")
    return vectorizer.transform(texts_clean).toarray()


def main():
    """Main entry point."""
    logger.info("Starting feature extraction job")
    
    if not STORAGE_ACCOUNT_NAME or not STORAGE_ACCOUNT_KEY:
        raise ValueError("Azure storage credentials not set")
    
    # Initialize Ray
    ray.init(address="auto")
    logger.info(f"Connected to Ray cluster: {ray.cluster_resources()}")
    
    fs = get_azure_fs()
    
    # Load parsed emails
    input_path = f"{INPUT_CONTAINER}/{INPUT_PATH}"
    logger.info(f"Loading emails from {input_path}")
    
    with fs.open(input_path, "rb") as f:
        df = pd.read_parquet(f)
    
    logger.info(f"Loaded {len(df)} emails")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # ========================================================================
    # Extract non-TF-IDF features using Ray
    # ========================================================================
    
    batch_size = 500
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    logger.info(f"Extracting features in {len(batches)} batches")
    
    # Put batches in object store
    batch_refs = [ray.put(batch) for batch in batches]
    
    # Process batches in parallel
    futures = [extract_features_batch.remote(ref) for ref in batch_refs]
    
    # Collect results
    feature_batches = ray.get(futures)
    features_df = pd.concat(feature_batches, ignore_index=True)
    
    logger.info(f"Extracted {len(features_df.columns) - 3} non-TF-IDF features")  # -3 for email_id, label, event_timestamp
    
    # ========================================================================
    # Fit and apply TF-IDF on full corpus (needs to be done centrally)
    # ========================================================================
    
    logger.info("Fitting TF-IDF vectorizer")
    vectorizer = fit_tfidf_vectorizer(df["body_text"])
    
    logger.info("Transforming text to TF-IDF vectors")
    tfidf_matrix = transform_tfidf(df["body_text"], vectorizer)
    
    # Create TF-IDF feature columns
    tfidf_columns = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_columns)
    tfidf_df["email_id"] = df["email_id"].values
    
    # Merge TF-IDF features with other features
    features_df = features_df.merge(tfidf_df, on="email_id", how="left")
    
    logger.info(f"Final feature matrix shape: {features_df.shape}")
    
    # ========================================================================
    # Create sender features (aggregated by domain)
    # ========================================================================
    
    logger.info("Computing sender domain features")
    
    sender_features = df.groupby("sender_domain").agg(
        email_count=("email_id", "count"),
        spam_count=("label", lambda x: (x == "spam").sum()),
        ham_count=("label", lambda x: (x == "ham").sum()),
    ).reset_index()
    
    sender_features["spam_ratio"] = sender_features["spam_count"] / sender_features["email_count"]
    sender_features["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    
    logger.info(f"Computed features for {len(sender_features)} sender domains")
    
    # ========================================================================
    # Write feature files to Feast offline store location
    # ========================================================================
    
    # Email features
    email_features_path = f"{OUTPUT_CONTAINER}/{OUTPUT_PATH}/email_features/data.parquet"
    logger.info(f"Writing email features to {email_features_path}")
    
    table = pa.Table.from_pandas(features_df, preserve_index=False)
    with fs.open(email_features_path, "wb") as f:
        pq.write_table(table, f)
    
    # Sender features
    sender_features_path = f"{OUTPUT_CONTAINER}/{OUTPUT_PATH}/sender_features/data.parquet"
    logger.info(f"Writing sender features to {sender_features_path}")
    
    table = pa.Table.from_pandas(sender_features, preserve_index=False)
    with fs.open(sender_features_path, "wb") as f:
        pq.write_table(table, f)
    
    # Save TF-IDF vectorizer for inference
    vectorizer_path = f"{OUTPUT_CONTAINER}/{OUTPUT_PATH}/artifacts/tfidf_vectorizer.pkl"
    logger.info(f"Saving TF-IDF vectorizer to {vectorizer_path}")
    
    vectorizer_bytes = pickle.dumps(vectorizer)
    with fs.open(vectorizer_path, "wb") as f:
        f.write(vectorizer_bytes)
    
    # ========================================================================
    # Feature statistics summary
    # ========================================================================
    
    logger.info("\n=== Feature Extraction Summary ===")
    logger.info(f"Total emails processed: {len(df)}")
    logger.info(f"Email features shape: {features_df.shape}")
    logger.info(f"Sender domains: {len(sender_features)}")
    logger.info(f"TF-IDF features: {len(tfidf_columns)}")
    
    # Class distribution
    spam_count = (features_df["label"] == "spam").sum()
    ham_count = (features_df["label"] == "ham").sum()
    logger.info(f"Class distribution - Spam: {spam_count}, Ham: {ham_count}")
    
    logger.info("Feature extraction complete!")
    
    return {
        "total_emails": len(df),
        "email_features_shape": features_df.shape,
        "sender_domains": len(sender_features),
        "tfidf_features": len(tfidf_columns),
    }


if __name__ == "__main__":
    main()
