"""
Feast Feature View Definitions for Spam Detection

IMPORTANT: Set these environment variables before running `feast apply`:
    export AZURE_STORAGE_ACCOUNT_NAME=<your-storage-account>
    export AZURE_STORAGE_ACCOUNT_KEY=<your-storage-key>
"""

import os
from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

from entities import email, sender_domain

# ============================================================================
# Configuration
# ============================================================================

STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

if not STORAGE_ACCOUNT_NAME or not STORAGE_ACCOUNT_KEY:
    raise ValueError("AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY environment variables are required")

FEAST_CONTAINER = "feast"
FEATURES_PATH = "features"

# Use abfss:// path format
BASE_PATH = f"abfss://{FEAST_CONTAINER}@{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/{FEATURES_PATH}"

# Storage options for fsspec/adlfs
STORAGE_OPTIONS = {
    "account_name": STORAGE_ACCOUNT_NAME,
    "account_key": STORAGE_ACCOUNT_KEY,
}

# ============================================================================
# Data Sources
# NOTE: We define the full schema here to prevent Feast from trying to infer
# it from Azure (which fails due to PyArrow authentication issues)
# ============================================================================

# Email features schema - includes all columns from the parquet file
email_features_schema = [
    Field(name="email_id", dtype=String),
    Field(name="label", dtype=String),
    Field(name="url_count", dtype=Int64),
    Field(name="email_count_in_body", dtype=Int64),
    Field(name="uppercase_ratio", dtype=Float32),
    Field(name="exclamation_count", dtype=Int64),
    Field(name="question_mark_count", dtype=Int64),
    Field(name="avg_word_length", dtype=Float32),
    Field(name="word_count", dtype=Int64),
    Field(name="char_count", dtype=Int64),
    Field(name="has_html", dtype=Int64),
    Field(name="html_to_text_ratio", dtype=Float32),
    Field(name="subject_length", dtype=Int64),
    Field(name="subject_has_re", dtype=Int64),
    Field(name="subject_has_fwd", dtype=Int64),
    Field(name="subject_all_caps", dtype=Int64),
    Field(name="has_x_mailer", dtype=Int64),
    Field(name="sender_domain_length", dtype=Int64),
    Field(name="sender_has_numbers", dtype=Int64),
    Field(name="received_hop_count", dtype=Int64),
    Field(name="hour_of_day", dtype=Int64),
    Field(name="day_of_week", dtype=Int64),
    Field(name="is_weekend", dtype=Int64),
    Field(name="is_night_hour", dtype=Int64),
    Field(name="spam_keyword_count", dtype=Int64),
    Field(name="has_unsubscribe", dtype=Int64),
] + [Field(name=f"tfidf_{i}", dtype=Float32) for i in range(500)]

email_features_source = FileSource(
    name="email_features_source",
    path=f"{BASE_PATH}/email_features/data.parquet",
    timestamp_field="event_timestamp",
    description="Email-level features for spam detection",
)

# Sender features schema
sender_features_schema = [
    Field(name="sender_domain", dtype=String),
    Field(name="email_count", dtype=Int64),
    Field(name="spam_count", dtype=Int64),
    Field(name="ham_count", dtype=Int64),
    Field(name="spam_ratio", dtype=Float32),
]

sender_features_source = FileSource(
    name="sender_features_source",
    path=f"{BASE_PATH}/sender_features/data.parquet",
    timestamp_field="event_timestamp",
    description="Aggregated features per sender domain",
)

# ============================================================================
# Feature Views
# ============================================================================

email_text_features = FeatureView(
    name="email_text_features",
    entities=[email],
    ttl=timedelta(days=365),
    schema=[
        Field(name="url_count", dtype=Int64),
        Field(name="uppercase_ratio", dtype=Float32),
        Field(name="exclamation_count", dtype=Int64),
        Field(name="question_mark_count", dtype=Int64),
        Field(name="word_count", dtype=Int64),
        Field(name="char_count", dtype=Int64),
        Field(name="spam_keyword_count", dtype=Int64),
        Field(name="has_unsubscribe", dtype=Int64),
    ],
    source=email_features_source,
    online=True,
    tags={"team": "ml", "project": "spam-detection"},
)

email_structural_features = FeatureView(
    name="email_structural_features",
    entities=[email],
    ttl=timedelta(days=365),
    schema=[
        Field(name="has_html", dtype=Int64),
        Field(name="html_to_text_ratio", dtype=Float32),
        Field(name="subject_length", dtype=Int64),
        Field(name="subject_has_re", dtype=Int64),
        Field(name="subject_has_fwd", dtype=Int64),
        Field(name="has_x_mailer", dtype=Int64),
        Field(name="sender_domain_length", dtype=Int64),
        Field(name="received_hop_count", dtype=Int64),
    ],
    source=email_features_source,
    online=True,
    tags={"team": "ml", "project": "spam-detection"},
)

email_temporal_features = FeatureView(
    name="email_temporal_features",
    entities=[email],
    ttl=timedelta(days=365),
    schema=[
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_night_hour", dtype=Int64),
    ],
    source=email_features_source,
    online=True,
    tags={"team": "ml", "project": "spam-detection"},
)

sender_domain_features = FeatureView(
    name="sender_domain_features",
    entities=[sender_domain],
    ttl=timedelta(days=30),
    schema=[
        Field(name="email_count", dtype=Int64),
        Field(name="spam_count", dtype=Int64),
        Field(name="ham_count", dtype=Int64),
        Field(name="spam_ratio", dtype=Float32),
    ],
    source=sender_features_source,
    online=True,
    tags={"team": "ml", "project": "spam-detection"},
)

# TF-IDF features - offline only
tfidf_schema = [
    Field(name=f"tfidf_{i}", dtype=Float32)
    for i in range(500)
]

email_tfidf_features = FeatureView(
    name="email_tfidf_features",
    entities=[email],
    ttl=timedelta(days=365),
    schema=tfidf_schema,
    source=email_features_source,
    online=False,
    tags={"team": "ml", "project": "spam-detection"},
)

# ============================================================================
# Feature Services
# ============================================================================

from feast import FeatureService

spam_detection_training_service = FeatureService(
    name="spam_detection_training",
    features=[
        email_text_features,
        email_structural_features,
        email_temporal_features,
        email_tfidf_features,
    ],
)

spam_detection_inference_service = FeatureService(
    name="spam_detection_inference",
    features=[
        email_text_features,
        email_structural_features,
        email_temporal_features,
        sender_domain_features,
    ],
)
