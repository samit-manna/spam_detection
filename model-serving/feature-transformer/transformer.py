"""
Feature Transformer Service

Real-time feature extraction service for spam detection inference.
Accepts raw email JSON input and returns feature vectors for model prediction.

Features extracted:
- Text features: url_count, uppercase_ratio, exclamation_count, etc.
- Structural features: has_html, subject_length, etc.
- Temporal features: hour_of_day, is_weekend, etc.
- Optional: Feast online features (sender_domain_features)

Usage:
    uvicorn transformer:app --host 0.0.0.0 --port 8080
"""

import os
import re
import json
import logging
import pickle
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import hashlib

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "feast")
TFIDF_VECTORIZER_PATH = os.environ.get(
    "TFIDF_VECTORIZER_PATH",
    "features/artifacts/tfidf_vectorizer.pkl"
)

FEAST_ENABLED = os.environ.get("FEAST_ENABLED", "false").lower() == "true"
FEAST_REDIS_HOST = os.environ.get("FEAST_REDIS_HOST", "feast-redis.feast.svc.cluster.local")
FEAST_REDIS_PORT = int(os.environ.get("FEAST_REDIS_PORT", "6379"))

# Number of TF-IDF features (must match training)
TFIDF_MAX_FEATURES = 500

# Feature order (must match data pipeline extract_features.py EXACTLY - 528 features total)
# Order: Text(8) -> Structural(10) -> Temporal(4) -> SpamIndicators(2) -> TF-IDF(500) -> Sender(4)
FEATURE_ORDER = [
    # Text features (8) - from extract_features_batch
    "url_count", "email_count_in_body", "uppercase_ratio", "exclamation_count", 
    "question_mark_count", "avg_word_length", "word_count", "char_count",
    # Structural features (10) - from extract_features_batch
    "has_html", "html_to_text_ratio", "subject_length", "subject_has_re",
    "subject_has_fwd", "subject_all_caps", "has_x_mailer", "sender_domain_length", 
    "sender_has_numbers", "received_hop_count",
    # Temporal features (4)
    "hour_of_day", "day_of_week", "is_weekend", "is_night_hour",
    # Spam indicator features (2) - these come AFTER temporal in extract_features_batch
    "spam_keyword_count", "has_unsubscribe",
    # TF-IDF features (500)
] + [f"tfidf_{i}" for i in range(TFIDF_MAX_FEATURES)] + [
    # Sender domain features (4) - from Redis/Feast
    "email_count", "spam_count", "ham_count", "spam_ratio"
]


# =============================================================================
# Data Models
# =============================================================================

class EmailInput(BaseModel):
    """Input schema for a single email."""
    email_id: str = Field(..., description="Unique email identifier")
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body text (plain text)")
    body_html: Optional[str] = Field(default=None, description="HTML body (optional)")
    sender: str = Field(..., description="Sender email address")
    date: Optional[str] = Field(default=None, description="Email date (ISO format)")
    x_mailer: Optional[str] = Field(default=None, description="X-Mailer header")
    received_hop_count: int = Field(default=0, description="Number of received headers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "abc123",
                "subject": "You've won a prize!",
                "body": "Click here to claim your FREE prize now!",
                "sender": "spam@example.com",
                "date": "2024-01-15T10:30:00Z"
            }
        }


class BatchEmailInput(BaseModel):
    """Input schema for batch email processing."""
    emails: List[EmailInput] = Field(..., description="List of emails to process")


class FeatureOutput(BaseModel):
    """Output schema for feature vector."""
    email_id: str
    features: List[float]
    feature_names: List[str]
    sender_domain: str
    feast_features_included: bool


class BatchFeatureOutput(BaseModel):
    """Output schema for batch feature extraction."""
    outputs: List[FeatureOutput]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    tfidf_loaded: bool
    scaler_loaded: bool
    feast_enabled: bool


# =============================================================================
# Spam Keywords
# =============================================================================

SPAM_KEYWORDS = [
    "free", "winner", "click here", "act now", "limited time",
    "congratulations", "urgent", "money", "cash", "prize",
    "viagra", "pharmacy", "buy now", "order now", "subscribe",
    "unsubscribe", "remove", "opt-out", "credit card", "wire transfer"
]


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def count_urls(text: str) -> int:
    """Count URLs in text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return len(re.findall(url_pattern, text or ""))


def count_emails_in_text(text: str) -> int:
    """Count email addresses in text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return len(re.findall(email_pattern, text or ""))


def avg_word_length(text: str) -> float:
    """Calculate average word length."""
    words = (text or "").split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


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


def word_count(text: str) -> int:
    """Count words in text."""
    return len((text or "").split())


def char_count(text: str) -> int:
    """Count characters in text."""
    return len(text or "")


def spam_keyword_count(text: str) -> int:
    """Count spam-related keywords."""
    text_lower = (text or "").lower()
    return sum(1 for kw in SPAM_KEYWORDS if kw in text_lower)


def has_unsubscribe(text: str) -> int:
    """Check for unsubscribe link/text."""
    return 1 if "unsubscribe" in (text or "").lower() else 0


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
    """Check if subject starts with RE:."""
    return 1 if (subject or "").lower().startswith("re:") else 0


def subject_has_fwd(subject: str) -> int:
    """Check if subject starts with FWD:."""
    s = (subject or "").lower()
    return 1 if s.startswith("fwd:") or s.startswith("fw:") else 0


def subject_all_caps(subject: str) -> int:
    """Check if subject is all uppercase."""
    s = subject or ""
    # Only consider alphabetic characters
    alpha_chars = [c for c in s if c.isalpha()]
    if not alpha_chars:
        return 0
    return 1 if all(c.isupper() for c in alpha_chars) else 0


def has_x_mailer_header(x_mailer: Optional[str]) -> int:
    """Check if X-Mailer header is present."""
    return 1 if x_mailer else 0


def sender_domain_length(sender_email: str) -> int:
    """Calculate sender domain length."""
    domain = extract_sender_domain(sender_email)
    return len(domain)


def sender_has_numbers(sender_email: str) -> int:
    """Check if sender email address contains numbers."""
    local_part = sender_email.split("@")[0] if "@" in sender_email else sender_email
    return 1 if any(c.isdigit() for c in local_part) else 0


def extract_sender_domain(sender_email: str) -> str:
    """Extract domain from email address."""
    if "@" in sender_email:
        return sender_email.split("@")[-1].lower()
    return sender_email.lower()


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string to datetime."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def extract_hour(date: Optional[datetime]) -> int:
    """Extract hour from datetime."""
    if date is None:
        return 12  # Default to noon
    return date.hour


def extract_day_of_week(date: Optional[datetime]) -> int:
    """Extract day of week (0=Monday, 6=Sunday)."""
    if date is None:
        return 2  # Default to Wednesday
    return date.weekday()


def is_weekend(date: Optional[datetime]) -> int:
    """Check if email was sent on weekend."""
    if date is None:
        return 0
    return 1 if date.weekday() >= 5 else 0


def is_night_hour(date: Optional[datetime]) -> int:
    """Check if sent between 10pm and 6am."""
    if date is None:
        return 0
    hour = date.hour
    return 1 if hour >= 22 or hour < 6 else 0


# =============================================================================
# TF-IDF Vectorizer
# =============================================================================

class TFIDFTransformer:
    """TF-IDF transformer loaded from Azure Blob Storage."""
    
    def __init__(self):
        self.vectorizer = None
        self.loaded = False
    
    def load_from_azure(self):
        """Load TF-IDF vectorizer from Azure Blob Storage."""
        if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
            logger.warning("Azure storage credentials not set, TF-IDF will return zeros")
            return
        
        try:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={AZURE_STORAGE_ACCOUNT_NAME};"
                f"AccountKey={AZURE_STORAGE_ACCOUNT_KEY};"
                f"EndpointSuffix=core.windows.net"
            )
            
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service.get_blob_client(
                container=AZURE_STORAGE_CONTAINER,
                blob=TFIDF_VECTORIZER_PATH
            )
            
            logger.info(f"Loading TF-IDF vectorizer from {TFIDF_VECTORIZER_PATH}")
            blob_data = blob_client.download_blob().readall()
            self.vectorizer = pickle.loads(blob_data)
            self.loaded = True
            logger.info(f"TF-IDF vectorizer loaded. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            
        except Exception as e:
            logger.error(f"Failed to load TF-IDF vectorizer: {e}")
            self.loaded = False
    
    def transform(self, text: str) -> np.ndarray:
        """Transform text to TF-IDF vector."""
        if not self.loaded or self.vectorizer is None:
            return np.zeros(TFIDF_MAX_FEATURES)
        
        try:
            tfidf_vector = self.vectorizer.transform([text or ""]).toarray()[0]
            return tfidf_vector
        except Exception as e:
            logger.error(f"TF-IDF transform error: {e}")
            return np.zeros(TFIDF_MAX_FEATURES)


# Global TF-IDF transformer
tfidf_transformer = TFIDFTransformer()


# =============================================================================
# Standard Scaler for Feature Normalization
# =============================================================================

# Configuration for scaler (models container has the scaler)
SCALER_CONTAINER = os.environ.get("SCALER_CONTAINER", "models")
SCALER_PATH = os.environ.get("SCALER_PATH", "hpo/scaler.pkl")

class FeatureScaler:
    """Standard scaler loaded from Azure Blob Storage - must match training."""
    
    def __init__(self):
        self.scaler = None
        self.loaded = False
    
    def load_from_azure(self):
        """Load StandardScaler from Azure Blob Storage."""
        if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
            logger.warning("Azure storage credentials not set, scaler will be disabled")
            return
        
        try:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={AZURE_STORAGE_ACCOUNT_NAME};"
                f"AccountKey={AZURE_STORAGE_ACCOUNT_KEY};"
                f"EndpointSuffix=core.windows.net"
            )
            
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service.get_blob_client(
                container=SCALER_CONTAINER,
                blob=SCALER_PATH
            )
            
            logger.info(f"Loading scaler from {SCALER_CONTAINER}/{SCALER_PATH}")
            blob_data = blob_client.download_blob().readall()
            self.scaler = pickle.loads(blob_data)
            self.loaded = True
            logger.info(f"Scaler loaded. Expected features: {self.scaler.n_features_in_}")
            
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            self.loaded = False
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Scale features using the trained StandardScaler."""
        if not self.loaded or self.scaler is None:
            logger.warning("Scaler not loaded, returning unscaled features")
            return features
        
        try:
            # Handle NaN and Inf values before scaling
            features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Reshape for single sample if needed
            if len(features_clean.shape) == 1:
                features_clean = features_clean.reshape(1, -1)
            
            scaled = self.scaler.transform(features_clean)
            
            # Handle any NaN/Inf that might result from scaling
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            return scaled.flatten()
        except Exception as e:
            logger.error(f"Scaler transform error: {e}")
            return features


# Global feature scaler
feature_scaler = FeatureScaler()


# =============================================================================
# Feast Client (Optional) - Direct Redis Access
# =============================================================================

# Redis configuration for Feast online store
FEAST_REDIS_PASSWORD = os.environ.get("FEAST_REDIS_PASSWORD", "")
FEAST_REDIS_SSL = os.environ.get("FEAST_REDIS_SSL", "false").lower() == "true"

class FeastClient:
    """Feast online store client - queries Redis directly for sender domain features."""
    
    def __init__(self):
        self.enabled = FEAST_ENABLED
        self.redis_client = None
    
    async def initialize(self):
        """Initialize Redis client for Feast online store."""
        if not self.enabled:
            logger.info("Feast features disabled")
            return
        
        try:
            import redis
            
            # Connect directly to Redis (Feast online store)
            self.redis_client = redis.Redis(
                host=FEAST_REDIS_HOST,
                port=FEAST_REDIS_PORT,
                password=FEAST_REDIS_PASSWORD or None,
                ssl=FEAST_REDIS_SSL,
                ssl_cert_reqs=None if FEAST_REDIS_SSL else None,
                decode_responses=False  # Binary data from Feast
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Feast Redis client initialized: {FEAST_REDIS_HOST}:{FEAST_REDIS_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Feast Redis client: {e}")
            self.enabled = False
    
    def _get_feast_key(self, sender_domain: str) -> str:
        """
        Generate the Feast Redis key for sender_domain entity.
        Feast uses format: project:feature_view:entity_key
        """
        # Feast stores features with a serialized key
        # Format depends on Feast version, but commonly:
        # spam_detection:sender_domain_features:sender_domain=<domain>
        return f"spam_detection:sender_domain_features:sender_domain={sender_domain}"
    
    def _parse_feast_value(self, data: bytes) -> Dict[str, float]:
        """Parse Feast serialized feature values from Redis."""
        default_features = {
            "email_count": 0.0,
            "spam_count": 0.0,
            "ham_count": 0.0,
            "spam_ratio": 0.5
        }
        
        if not data:
            return default_features
        
        try:
            # Feast uses protobuf serialization - try to decode
            # For simpler setups, it might be JSON or msgpack
            import struct
            
            # Try JSON first (some Feast configs use JSON)
            try:
                import json
                decoded = json.loads(data.decode('utf-8'))
                if isinstance(decoded, dict):
                    return {
                        "email_count": float(decoded.get("email_count", 0)),
                        "spam_count": float(decoded.get("spam_count", 0)),
                        "ham_count": float(decoded.get("ham_count", 0)),
                        "spam_ratio": float(decoded.get("spam_ratio", 0.5))
                    }
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            
            # Try protobuf (Feast default)
            try:
                from feast.protos.feast.types import Value_pb2
                from google.protobuf.json_format import MessageToDict
                
                # Feast stores as ValueProto
                value = Value_pb2.Value()
                value.ParseFromString(data)
                decoded = MessageToDict(value)
                
                if "floatListVal" in decoded:
                    values = decoded["floatListVal"].get("val", [])
                    if len(values) >= 4:
                        return {
                            "email_count": float(values[0]),
                            "spam_count": float(values[1]),
                            "ham_count": float(values[2]),
                            "spam_ratio": float(values[3])
                        }
            except Exception:
                pass
            
            logger.debug(f"Could not parse Feast value, using defaults")
            return default_features
            
        except Exception as e:
            logger.warning(f"Error parsing Feast value: {e}")
            return default_features
    
    async def get_sender_features(self, sender_domain: str) -> Dict[str, float]:
        """Get sender domain features directly from Redis."""
        default_features = {
            "email_count": 0.0,
            "spam_count": 0.0,
            "ham_count": 0.0,
            "spam_ratio": 0.5  # Neutral default
        }
        
        if not self.enabled or self.redis_client is None:
            return default_features
        
        try:
            # Try different key patterns used by Feast
            key_patterns = [
                f"spam_detection:sender_domain_features:sender_domain={sender_domain}",
                f"feast:sender_domain_features:{sender_domain}",
                f"sender_domain_features:{sender_domain}",
            ]
            
            for key in key_patterns:
                data = self.redis_client.get(key)
                if data:
                    logger.debug(f"Found Feast data at key: {key}")
                    return self._parse_feast_value(data)
            
            # Also try HGETALL for hash-based storage
            for key in key_patterns:
                data = self.redis_client.hgetall(key)
                if data:
                    logger.debug(f"Found Feast hash data at key: {key}")
                    return {
                        "email_count": float(data.get(b"email_count", 0)),
                        "spam_count": float(data.get(b"spam_count", 0)),
                        "ham_count": float(data.get(b"ham_count", 0)),
                        "spam_ratio": float(data.get(b"spam_ratio", 0.5))
                    }
            
            logger.debug(f"No Feast features found for {sender_domain}, using defaults")
            return default_features
            
        except Exception as e:
            logger.warning(f"Redis lookup failed for {sender_domain}: {e}")
            return default_features


# Global Feast client
feast_client = FeastClient()


# =============================================================================
# Feature Extractor
# =============================================================================

class FeatureExtractor:
    """Main feature extraction class."""
    
    def __init__(self, tfidf: TFIDFTransformer, feast: FeastClient, scaler: FeatureScaler):
        self.tfidf = tfidf
        self.feast = feast
        self.scaler = scaler
    
    async def extract_features(self, email: EmailInput) -> FeatureOutput:
        """Extract all features for a single email."""
        
        # Parse date
        email_date = parse_date(email.date)
        
        # Extract sender domain
        sender_domain = extract_sender_domain(email.sender)
        
        # Text features (8 features) - matches extract_features_batch order
        text_features = {
            "url_count": count_urls(email.body),
            "email_count_in_body": count_emails_in_text(email.body),
            "uppercase_ratio": uppercase_ratio(email.body),
            "exclamation_count": count_exclamations(email.body),
            "question_mark_count": count_question_marks(email.body),
            "avg_word_length": avg_word_length(email.body),
            "word_count": word_count(email.body),
            "char_count": char_count(email.body),
        }
        
        # Structural features (10 features)
        structural_features = {
            "has_html": has_html(email.body_html),
            "html_to_text_ratio": html_to_text_ratio(email.body, email.body_html),
            "subject_length": subject_length(email.subject),
            "subject_has_re": subject_has_re(email.subject),
            "subject_has_fwd": subject_has_fwd(email.subject),
            "subject_all_caps": subject_all_caps(email.subject),
            "has_x_mailer": has_x_mailer_header(email.x_mailer),
            "sender_domain_length": sender_domain_length(email.sender),
            "sender_has_numbers": sender_has_numbers(email.sender),
            "received_hop_count": email.received_hop_count,
        }
        
        # Temporal features (4 features)
        temporal_features = {
            "hour_of_day": extract_hour(email_date),
            "day_of_week": extract_day_of_week(email_date),
            "is_weekend": is_weekend(email_date),
            "is_night_hour": is_night_hour(email_date),
        }
        
        # Spam indicator features (2 features) - come after temporal in extract_features_batch
        spam_indicator_features = {
            "spam_keyword_count": spam_keyword_count(email.body),
            "has_unsubscribe": has_unsubscribe(email.body),
        }
        
        # TF-IDF features (500 features)
        tfidf_features = self.tfidf.transform(email.body)
        
        # Feast features (sender domain) - 4 features
        feast_features = await self.feast.get_sender_features(sender_domain)
        
        # Combine all features in EXACT order matching extract_features_batch
        feature_vector = []
        
        # Text features (8)
        for name in ["url_count", "email_count_in_body", "uppercase_ratio", 
                     "exclamation_count", "question_mark_count", "avg_word_length",
                     "word_count", "char_count"]:
            feature_vector.append(float(text_features[name]))
        
        # Structural features (10)
        for name in ["has_html", "html_to_text_ratio", "subject_length",
                     "subject_has_re", "subject_has_fwd", "subject_all_caps",
                     "has_x_mailer", "sender_domain_length", "sender_has_numbers",
                     "received_hop_count"]:
            feature_vector.append(float(structural_features[name]))
        
        # Temporal features (4)
        for name in ["hour_of_day", "day_of_week", "is_weekend", "is_night_hour"]:
            feature_vector.append(float(temporal_features[name]))
        
        # Spam indicator features (2) - AFTER temporal
        for name in ["spam_keyword_count", "has_unsubscribe"]:
            feature_vector.append(float(spam_indicator_features[name]))
        
        # TF-IDF features (500)
        feature_vector.extend(tfidf_features.tolist())
        
        # Feast/Sender features (4)
        for name in ["email_count", "spam_count", "ham_count", "spam_ratio"]:
            feature_vector.append(feast_features[name])
        
        # CRITICAL: Apply StandardScaler to match training
        # The model was trained on scaled features!
        feature_array = np.array(feature_vector)
        scaled_features = self.scaler.transform(feature_array)
        
        return FeatureOutput(
            email_id=email.email_id,
            features=scaled_features.tolist(),
            feature_names=FEATURE_ORDER,
            sender_domain=sender_domain,
            feast_features_included=self.feast.enabled
        )
    
    async def extract_batch(self, emails: List[EmailInput]) -> BatchFeatureOutput:
        """Extract features for a batch of emails."""
        outputs = []
        for email in emails:
            output = await self.extract_features(email)
            outputs.append(output)
        
        return BatchFeatureOutput(outputs=outputs, count=len(outputs))


# Global feature extractor
feature_extractor: Optional[FeatureExtractor] = None


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize resources."""
    global feature_extractor
    
    logger.info("Initializing Feature Transformer Service...")
    
    # Load TF-IDF vectorizer
    tfidf_transformer.load_from_azure()
    
    # Load StandardScaler (CRITICAL - model was trained on scaled features!)
    feature_scaler.load_from_azure()
    
    # Initialize Feast client
    await feast_client.initialize()
    
    # Create feature extractor with scaler
    feature_extractor = FeatureExtractor(tfidf_transformer, feast_client, feature_scaler)
    
    logger.info("Feature Transformer Service initialized")
    
    yield
    
    logger.info("Shutting down Feature Transformer Service")


app = FastAPI(
    title="Feature Transformer Service",
    description="Real-time feature extraction for spam detection",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        tfidf_loaded=tfidf_transformer.loaded,
        scaler_loaded=feature_scaler.loaded,
        feast_enabled=feast_client.enabled
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    if feature_extractor is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.post("/transform", response_model=FeatureOutput)
async def transform_single(email: EmailInput):
    """Transform a single email to feature vector."""
    if feature_extractor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return await feature_extractor.extract_features(email)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transform/batch", response_model=BatchFeatureOutput)
async def transform_batch(batch: BatchEmailInput):
    """Transform a batch of emails to feature vectors."""
    if feature_extractor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if len(batch.emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum batch size is 100")
    
    try:
        return await feature_extractor.extract_batch(batch.emails)
    except Exception as e:
        logger.error(f"Batch feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-info")
async def get_feature_info():
    """Get information about features."""
    return {
        "total_features": len(FEATURE_ORDER),
        "text_features": 8,
        "structural_features": 8,
        "temporal_features": 4,
        "tfidf_features": TFIDF_MAX_FEATURES,
        "sender_features": 4,
        "feature_order": FEATURE_ORDER,
        "tfidf_loaded": tfidf_transformer.loaded,
        "feast_enabled": feast_client.enabled
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
