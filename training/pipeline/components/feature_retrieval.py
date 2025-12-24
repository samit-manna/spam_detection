"""
Feature Retrieval Component

Retrieves features from Feast server for training and test sets.
Joins all feature views (email_text, email_structural, email_temporal, 
email_tfidf, sender_domain) using get_historical_features.
"""
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Input, Output, Metrics, InputPath, OutputPath


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
)
def feature_retrieval(
    acr_name: str,
    image_tag: str,
    train_entity_path: str,
    test_entity_path: str,
    feast_server_url: str,  # feast-service.feast.svc.cluster.local:6566
    storage_account: str,
    container_name: str,
    include_sender_features: bool,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("feature_count", int), ("train_samples", int), ("test_samples", int), ("train_features_path", str), ("test_features_path", str)]):
    """
    Retrieve features from Feast for training and test entity dataframes.
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        train_entity_path: Blob path to training entity parquet
        test_entity_path: Blob path to test entity parquet
        feast_server_url: Feast server gRPC endpoint
        storage_account: Azure storage account name
        container_name: Azure blob container name
        include_sender_features: Whether to include sender_domain_features
        metrics: Output metrics object
        
    Returns:
        feature_count: Number of features retrieved
        train_samples: Number of training samples
        test_samples: Number of test samples
        train_features_path: Blob path to training features
        test_features_path: Blob path to test features
    """
    import os
    import pandas as pd
    from feast import FeatureStore
    
    # Get connection string
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
    
    storage_options = {
        "connection_string": connection_string
    }
    
    print(f"Train entity path: {train_entity_path}")
    print(f"Test entity path: {test_entity_path}")
    
    # Load entity dataframes with fallback logic
    read_path = f"abfs://{train_entity_path}"
    print(f"Reading train entities from: {read_path}")
    
    try:
        train_entity_df = pd.read_parquet(read_path, storage_options=storage_options)
        print(f"Train entities: {len(train_entity_df)} samples")
    except Exception as e:
        print(f"Error reading with abfs://: {e}")
        
        # Try abfss://
        read_path_secure = f"abfss://{train_entity_path.replace('/', '@' + storage_account + '.dfs.core.windows.net/', 1)}"
        print(f"Trying: {read_path_secure}")
        
        try:
            train_entity_df = pd.read_parquet(read_path_secure, storage_options=storage_options)
            print(f"Train entities: {len(train_entity_df)} samples using abfss://")
        except Exception as e2:
            print(f"Error reading with abfss://: {e2}")
            
            # Last attempt: BlobServiceClient
            print("Trying direct Azure Blob read...")
            from azure.storage.blob import BlobServiceClient
            import io
            
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            parts = train_entity_path.split('/', 1)
            container = parts[0]
            path = parts[1] if len(parts) > 1 else ""
            
            container_client = blob_service.get_container_client(container)
            blob_client = container_client.get_blob_client(path)
            blob_data = blob_client.download_blob().readall()
            train_entity_df = pd.read_parquet(io.BytesIO(blob_data))
            print(f"Train entities: {len(train_entity_df)} samples using BlobServiceClient")
    
    # Load test entities (same fallback logic)
    read_path = f"abfs://{test_entity_path}"
    print(f"Reading test entities from: {read_path}")
    
    try:
        test_entity_df = pd.read_parquet(read_path, storage_options=storage_options)
        print(f"Test entities: {len(test_entity_df)} samples")
    except Exception as e:
        print(f"Error reading with abfs://: {e}")
        
        # Try abfss://
        read_path_secure = f"abfss://{test_entity_path.replace('/', '@' + storage_account + '.dfs.core.windows.net/', 1)}"
        print(f"Trying: {read_path_secure}")
        
        try:
            test_entity_df = pd.read_parquet(read_path_secure, storage_options=storage_options)
            print(f"Test entities: {len(test_entity_df)} samples using abfss://")
        except Exception as e2:
            print(f"Error reading with abfss://: {e2}")
            
            # Last attempt: BlobServiceClient
            print("Trying direct Azure Blob read...")
            from azure.storage.blob import BlobServiceClient
            import io
            
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            parts = test_entity_path.split('/', 1)
            container = parts[0]
            path = parts[1] if len(parts) > 1 else ""
            
            container_client = blob_service.get_container_client(container)
            blob_client = container_client.get_blob_client(path)
            blob_data = blob_client.download_blob().readall()
            test_entity_df = pd.read_parquet(io.BytesIO(blob_data))
            print(f"Test entities: {len(test_entity_df)} samples using BlobServiceClient")
    
    print(f"Entity columns: {train_entity_df.columns.tolist()}")
    
    # For offline features, read directly from parquet files
    # Remote Feast server is for online serving only
    print(f"Using direct offline store access (parquet files)")
    print(f"Features stored in: {container_name}/features/")
    
    # Define feature references for all feature views
    # Email features (entity: email_id)
    email_features = [
        # email_text_features (8 features)
        "email_text_features:url_count",
        "email_text_features:uppercase_ratio",
        "email_text_features:exclamation_count",
        "email_text_features:question_mark_count",
        "email_text_features:word_count",
        "email_text_features:char_count",
        "email_text_features:spam_keyword_count",
        "email_text_features:has_unsubscribe",
        
        # email_structural_features (8 features)
        "email_structural_features:has_html",
        "email_structural_features:html_to_text_ratio",
        "email_structural_features:subject_length",
        "email_structural_features:subject_has_re",
        "email_structural_features:subject_has_fwd",
        "email_structural_features:has_x_mailer",
        "email_structural_features:sender_domain_length",
        "email_structural_features:received_hop_count",
        
        # email_temporal_features (4 features)
        "email_temporal_features:hour_of_day",
        "email_temporal_features:day_of_week",
        "email_temporal_features:is_weekend",
        "email_temporal_features:is_night_hour",
    ]
    
    # Add TF-IDF features (500 features)
    tfidf_features = [f"email_tfidf_features:tfidf_{i}" for i in range(500)]
    email_features.extend(tfidf_features)
    
    # Sender domain features (entity: sender_domain)
    sender_features = []
    if include_sender_features and "sender_domain" in train_entity_df.columns:
        sender_features = [
            "sender_domain_features:email_count",
            "sender_domain_features:spam_count",
            "sender_domain_features:ham_count",
            "sender_domain_features:spam_ratio",
        ]
        print("Including sender_domain_features")
    else:
        print("Skipping sender_domain_features")
    
    all_features = email_features + sender_features
    print(f"Total features to retrieve: {len(all_features)}")
    
    def retrieve_features(entity_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Retrieve features for a given entity dataframe from parquet files."""
        print(f"\nRetrieving features for {dataset_name}...")
        
        feast_entity_df = entity_df.copy()
        feast_entity_df["event_timestamp"] = pd.to_datetime(feast_entity_df["event_timestamp"])
        
        # Read features directly from offline store parquet files
        feature_df = feast_entity_df.copy()
        
        # Read email features parquet
        email_features_path = f"abfs://{container_name}/features/email_features/data.parquet"
        print(f"Reading email features from: {email_features_path}")
        
        try:
            email_feat_df = pd.read_parquet(email_features_path, storage_options=storage_options)
            print(f"Email features shape: {email_feat_df.shape}")
            print(f"Email features columns (first 10): {email_feat_df.columns.tolist()[:10]}")
            
            # Merge on email_id
            feature_df = feature_df.merge(
                email_feat_df,
                on="email_id",
                how="left",
                suffixes=('', '_feat')
            )
            print(f"After email features merge: {feature_df.shape}")
            
        except Exception as e2:
            print(f"Error reading email features: {e2}")
            raise
        
        # Read sender features if needed
        if include_sender_features and "sender_domain" in feast_entity_df.columns:
            sender_features_path = f"abfs://{container_name}/features/sender_features/data.parquet"
            print(f"Reading sender features from: {sender_features_path}")
            
            try:
                sender_feat_df = pd.read_parquet(sender_features_path, storage_options=storage_options)
                print(f"Sender features shape: {sender_feat_df.shape}")
                
                # Merge on sender_domain
                feature_df = feature_df.merge(
                    sender_feat_df,
                    on="sender_domain",
                    how="left",
                    suffixes=('', '_sender')
                )
                print(f"After sender features merge: {feature_df.shape}")
                
            except Exception as e3:
                print(f"Error reading sender features: {e3}")
                # Continue without sender features
                pass
        
        return feature_df
    
    # Retrieve features for train and test sets
    train_features_df = retrieve_features(train_entity_df, "training")
    test_features_df = retrieve_features(test_entity_df, "test")
    
    # Check for nulls and handle them
    train_null_counts = train_features_df.isnull().sum()
    if train_null_counts.any():
        null_cols = train_null_counts[train_null_counts > 0]
        print(f"\nWarning: Null values in training data:\n{null_cols.head(10)}")
        
        # Fill numeric nulls with 0
        numeric_cols = train_features_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        train_features_df[numeric_cols] = train_features_df[numeric_cols].fillna(0)
        test_features_df[numeric_cols] = test_features_df[numeric_cols].fillna(0)
    
    # Count actual features (exclude entity columns, label, and duplicate suffixes)
    exclude_cols = ["email_id", "sender_domain", "event_timestamp", "label", "event_timestamp_feat", "created_timestamp"]
    feature_cols = [c for c in train_features_df.columns if c not in exclude_cols and not c.endswith('_feat') and not c.endswith('_sender')]
    feature_count = len(feature_cols)
    
    print(f"\nFinal feature count: {feature_count}")
    print(f"Train shape: {train_features_df.shape}")
    print(f"Test shape: {test_features_df.shape}")
    
    # Write outputs to blob storage
    train_output_path = f"abfs://{container_name}/training/train_features.parquet"
    test_output_path = f"abfs://{container_name}/training/test_features.parquet"
    
    print(f"Writing training features to: {train_output_path}")
    train_features_df.to_parquet(train_output_path, storage_options=storage_options, index=False)
    
    print(f"Writing test features to: {test_output_path}")
    test_features_df.to_parquet(test_output_path, storage_options=storage_options, index=False)
    
    # Return paths for downstream components
    train_path_output = f"{container_name}/training/train_features.parquet"
    test_path_output = f"{container_name}/training/test_features.parquet"
    
    # Log metrics
    metrics.log_metric("feature_count", feature_count)
    metrics.log_metric("train_samples", len(train_features_df))
    metrics.log_metric("test_samples", len(test_features_df))
    metrics.log_metric("email_features", 520)  # 8+8+4+500
    metrics.log_metric("sender_features", len(sender_features))
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["feature_count", "train_samples", "test_samples", "train_features_path", "test_features_path"])
    return outputs(feature_count, len(train_features_df), len(test_features_df), train_path_output, test_path_output)
