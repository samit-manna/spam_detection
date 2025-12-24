"""
Feast Setup and Materialization Script

This script:
1. Initializes the Feast feature store
2. Applies feature definitions
3. Materializes features to the online store (Redis)

Usage:
    python setup_feast.py --apply --materialize
    
Environment variables:
    AZURE_STORAGE_ACCOUNT_NAME: Azure storage account name
    AZURE_STORAGE_ACCOUNT_KEY: Azure storage account key
    FEAST_REDIS_HOST: Redis host for online store
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

from feast import FeatureStore
from feast.repo_config import RepoConfig
from feast.infra.offline_stores.file import FileOfflineStoreConfig
from feast.infra.online_stores.redis import RedisOnlineStoreConfig

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
FEAST_REDIS_HOST = os.environ.get("FEAST_REDIS_HOST", "feast-redis.feast.svc.cluster.local")
FEAST_REDIS_PORT = int(os.environ.get("FEAST_REDIS_PORT", "6379"))
FEAST_PROJECT = os.environ.get("FEAST_PROJECT", "spam_detection")


def create_feast_config() -> RepoConfig:
    """Create Feast repository configuration."""
    
    if not AZURE_STORAGE_ACCOUNT_NAME:
        raise ValueError("AZURE_STORAGE_ACCOUNT_NAME environment variable is required")
    
    # Create configuration
    config = RepoConfig(
        project=FEAST_PROJECT,
        provider="local",
        offline_store=FileOfflineStoreConfig(type="file"),
        online_store=RedisOnlineStoreConfig(
            type="redis",
            connection_string=f"redis://{FEAST_REDIS_HOST}:{FEAST_REDIS_PORT}/0"
        ),
        entity_key_serialization_version=3,
    )
    
    return config


def initialize_feature_store(repo_path: str = ".") -> FeatureStore:
    """Initialize the Feast feature store."""
    
    logger.info(f"Initializing Feast feature store from {repo_path}")
    
    try:
        fs = FeatureStore(repo_path=repo_path)
        logger.info(f"Feature store initialized for project: {fs.project}")
        return fs
    except Exception as e:
        logger.error(f"Failed to initialize feature store: {e}")
        raise


def apply_feature_definitions(fs: FeatureStore):
    """Apply feature definitions to the feature store."""
    
    logger.info("Applying feature definitions...")
    
    try:
        # This runs `feast apply` programmatically
        fs.apply([])  # Empty list means apply all from repo
        
        # List applied feature views
        feature_views = fs.list_feature_views()
        logger.info(f"Applied {len(feature_views)} feature views:")
        for fv in feature_views:
            logger.info(f"  - {fv.name}: {len(fv.features)} features, online={fv.online}")
        
        # List entities
        entities = fs.list_entities()
        logger.info(f"Applied {len(entities)} entities:")
        for entity in entities:
            logger.info(f"  - {entity.name}: join_keys={entity.join_keys}")
        
        # List feature services
        services = fs.list_feature_services()
        logger.info(f"Applied {len(services)} feature services:")
        for service in services:
            logger.info(f"  - {service.name}")
        
    except Exception as e:
        logger.error(f"Failed to apply feature definitions: {e}")
        raise


def materialize_features(
    fs: FeatureStore,
    start_date: datetime = None,
    end_date: datetime = None,
    feature_views: list = None
):
    """
    Materialize features from offline to online store.
    
    Args:
        fs: FeatureStore instance
        start_date: Start date for materialization
        end_date: End date for materialization
        feature_views: List of feature view names to materialize (None = all)
    """
    
    # Default to last 30 days
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    logger.info(f"Materializing features from {start_date} to {end_date}")
    
    try:
        if feature_views:
            # Materialize specific feature views
            for fv_name in feature_views:
                logger.info(f"Materializing feature view: {fv_name}")
                fs.materialize(
                    start_date=start_date,
                    end_date=end_date,
                    feature_views=[fv_name]
                )
        else:
            # Materialize all online-enabled feature views
            fs.materialize(
                start_date=start_date,
                end_date=end_date
            )
        
        logger.info("Materialization complete")
        
    except Exception as e:
        logger.error(f"Failed to materialize features: {e}")
        raise


def materialize_incremental(fs: FeatureStore, end_date: datetime = None):
    """
    Incrementally materialize features since last materialization.
    
    This is more efficient than full materialization for regular updates.
    """
    
    if end_date is None:
        end_date = datetime.utcnow()
    
    logger.info(f"Running incremental materialization up to {end_date}")
    
    try:
        fs.materialize_incremental(end_date=end_date)
        logger.info("Incremental materialization complete")
        
    except Exception as e:
        logger.error(f"Failed to materialize features incrementally: {e}")
        raise


def test_online_features(fs: FeatureStore):
    """Test online feature retrieval."""
    
    logger.info("Testing online feature retrieval...")
    
    try:
        # Test sender domain features
        test_entities = [
            {"sender_domain": "gmail.com"},
            {"sender_domain": "yahoo.com"},
            {"sender_domain": "unknown-domain.com"},
        ]
        
        features = fs.get_online_features(
            features=[
                "sender_domain_features:email_count",
                "sender_domain_features:spam_count",
                "sender_domain_features:ham_count",
                "sender_domain_features:spam_ratio",
            ],
            entity_rows=test_entities
        ).to_dict()
        
        logger.info("Online feature retrieval results:")
        for i, entity in enumerate(test_entities):
            logger.info(f"  {entity['sender_domain']}:")
            logger.info(f"    email_count: {features['email_count'][i]}")
            logger.info(f"    spam_count: {features['spam_count'][i]}")
            logger.info(f"    ham_count: {features['ham_count'][i]}")
            logger.info(f"    spam_ratio: {features['spam_ratio'][i]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Online feature test failed: {e}")
        return False


def get_feature_stats(fs: FeatureStore):
    """Get statistics about materialized features."""
    
    logger.info("Getting feature store statistics...")
    
    stats = {
        "project": fs.project,
        "feature_views": [],
        "entities": [],
        "feature_services": [],
    }
    
    # Feature views
    for fv in fs.list_feature_views():
        stats["feature_views"].append({
            "name": fv.name,
            "features": len(fv.features),
            "online": fv.online,
            "ttl": str(fv.ttl),
        })
    
    # Entities
    for entity in fs.list_entities():
        stats["entities"].append({
            "name": entity.name,
            "join_keys": entity.join_keys,
        })
    
    # Feature services
    for service in fs.list_feature_services():
        stats["feature_services"].append({
            "name": service.name,
            "feature_views": [fv.name for fv in service.feature_view_projections],
        })
    
    return stats


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Setup and manage Feast feature store"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply feature definitions"
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize features to online store"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run incremental materialization"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test online feature retrieval"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print feature store statistics"
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=".",
        help="Path to Feast feature repository"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to materialize (default: 30)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Feast Feature Store Setup")
    logger.info("=" * 60)
    
    # Initialize feature store
    fs = initialize_feature_store(args.repo_path)
    
    # Apply feature definitions
    if args.apply:
        apply_feature_definitions(fs)
    
    # Materialize features
    if args.materialize:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)
        materialize_features(fs, start_date, end_date)
    
    # Incremental materialization
    if args.incremental:
        materialize_incremental(fs)
    
    # Test online features
    if args.test:
        success = test_online_features(fs)
        if not success:
            sys.exit(1)
    
    # Print statistics
    if args.stats:
        import json
        stats = get_feature_stats(fs)
        print(json.dumps(stats, indent=2))
    
    logger.info("=" * 60)
    logger.info("Feast setup complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
