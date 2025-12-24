"""
Feast Feature Definitions for Serving

This module defines the feature views used for real-time and batch inference.
Features are optimized for serving with proper online store configuration.

Feature Views:
- sender_domain_features: Aggregated statistics per sender domain
- email_text_features: Text-based features (url_count, etc.)
- email_structural_features: Email structure features
- email_temporal_features: Time-based features

Usage:
    # Apply feature definitions
    cd feast/
    feast apply
    
    # Materialize features to online store
    feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
"""

import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float32, Int64, String

# =============================================================================
# Configuration
# =============================================================================

STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
FEAST_CONTAINER = "feast"
FEATURES_PATH = "features"

BASE_PATH = f"abfss://{FEAST_CONTAINER}@{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/{FEATURES_PATH}"


# =============================================================================
# Entities
# =============================================================================

email = Entity(
    name="email",
    description="Unique identifier for an email message",
    join_keys=["email_id"],
)

sender_domain = Entity(
    name="sender_domain",
    description="Email sender domain (e.g., gmail.com)",
    join_keys=["sender_domain"],
)


# =============================================================================
# Data Sources
# =============================================================================

email_features_source = FileSource(
    name="email_features_source",
    path=f"{BASE_PATH}/email_features/data.parquet",
    timestamp_field="event_timestamp",
    description="Email-level features for spam detection",
)

sender_features_source = FileSource(
    name="sender_features_source",
    path=f"{BASE_PATH}/sender_features/data.parquet",
    timestamp_field="event_timestamp",
    description="Aggregated features per sender domain",
)


# =============================================================================
# Feature Views for Online Serving
# =============================================================================

# Sender domain features - most important for serving
sender_domain_features = FeatureView(
    name="sender_domain_features",
    entities=[sender_domain],
    ttl=timedelta(days=30),
    schema=[
        Field(name="email_count", dtype=Int64, description="Total emails from this domain"),
        Field(name="spam_count", dtype=Int64, description="Spam emails from this domain"),
        Field(name="ham_count", dtype=Int64, description="Ham emails from this domain"),
        Field(name="spam_ratio", dtype=Float32, description="Ratio of spam emails"),
    ],
    source=sender_features_source,
    online=True,  # Enable online serving
    tags={
        "team": "ml",
        "project": "spam-detection",
        "serving": "online",
    },
)

# Email text features
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
    tags={
        "team": "ml",
        "project": "spam-detection",
        "serving": "online",
    },
)

# Email structural features
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
    tags={
        "team": "ml",
        "project": "spam-detection",
        "serving": "online",
    },
)

# Email temporal features
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
    tags={
        "team": "ml",
        "project": "spam-detection",
        "serving": "online",
    },
)

# TF-IDF features - offline only (too large for online store)
tfidf_schema = [Field(name=f"tfidf_{i}", dtype=Float32) for i in range(500)]

email_tfidf_features = FeatureView(
    name="email_tfidf_features",
    entities=[email],
    ttl=timedelta(days=365),
    schema=tfidf_schema,
    source=email_features_source,
    online=False,  # Too large for online serving
    tags={
        "team": "ml",
        "project": "spam-detection",
        "serving": "offline",
    },
)


# =============================================================================
# Feature Services
# =============================================================================

# Feature service for real-time inference
spam_detection_serving_service = FeatureService(
    name="spam_detection_serving",
    features=[
        sender_domain_features,
    ],
    description="Feature service for real-time spam detection inference",
    tags={
        "serving": "online",
        "latency": "low",
    },
)

# Feature service for batch inference (includes all features)
spam_detection_batch_service = FeatureService(
    name="spam_detection_batch",
    features=[
        email_text_features,
        email_structural_features,
        email_temporal_features,
        email_tfidf_features,
        sender_domain_features,
    ],
    description="Feature service for batch spam detection inference",
    tags={
        "serving": "offline",
    },
)

# Feature service for training (same as batch)
spam_detection_training_service = FeatureService(
    name="spam_detection_training",
    features=[
        email_text_features,
        email_structural_features,
        email_temporal_features,
        email_tfidf_features,
    ],
    description="Feature service for model training",
    tags={
        "serving": "offline",
        "training": "true",
    },
)
