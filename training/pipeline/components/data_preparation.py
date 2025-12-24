"""
Data Preparation Component

Reads processed email data, creates entity dataframe with train/test split.
This prepares the entity_df needed for Feast feature retrieval.
"""
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Output, Metrics, OutputPath


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
)
def data_preparation(
    acr_name: str,
    image_tag: str,
    entity_data_path: str,  # datasets/processed/emails/all_emails.parquet
    storage_account: str,   # mltrainingsdevsajvnm5w
    container_name: str,    # feast
    test_split_ratio: float,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("train_count", int), ("test_count", int), ("spam_ratio", float), ("train_entity_path", str), ("test_entity_path", str)]):
    """
    Prepare entity dataframes for Feast feature retrieval.
    
    Reads the processed email data, extracts entity columns (email_id, 
    sender_domain, event_timestamp, label), and performs stratified 
    train/test split.
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        entity_data_path: Path to all_emails.parquet in blob storage
        storage_account: Azure storage account name
        container_name: Azure blob container name
        test_split_ratio: Fraction of data for test set (e.g., 0.2)
        train_entity_path: Output path for training entity parquet
        test_entity_path: Output path for test entity parquet
        metrics: Output metrics object
        
    Returns:
        train_count: Number of training samples
        test_count: Number of test samples  
        spam_ratio: Ratio of spam in dataset
    """
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from datetime import datetime
    
    # Get connection string
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
    
    print(f"Storage account: {storage_account}")
    print(f"Container: {container_name}")
    print(f"Entity data path: {entity_data_path}")
    
    # Use pandas with fsspec for reading from Azure
    # The abfs:// protocol works with adlfs backend
    # Format: abfs://container/path
    read_path = f"abfs://{container_name}/{entity_data_path}"
    print(f"Reading from: {read_path}")
    
    storage_options = {
        "connection_string": connection_string
    }
    
    # Read the parquet file
    try:
        df = pd.read_parquet(read_path, storage_options=storage_options)
        print(f"Successfully loaded {len(df)} records")
    except Exception as e:
        print(f"Error reading with abfs://: {e}")
        
        # Try alternative: abfss:// (ADLS Gen2 secure)
        read_path_secure = f"abfss://{container_name}@{storage_account}.dfs.core.windows.net/{entity_data_path}"
        print(f"Trying secure path: {read_path_secure}")
        
        try:
            df = pd.read_parquet(read_path_secure, storage_options=storage_options)
            print(f"Successfully loaded {len(df)} records using abfss://")
        except Exception as e2:
            print(f"Error reading with abfss://: {e2}")
            
            # Last attempt: use Azure Blob directly
            print("Trying direct Azure Blob read...")
            from azure.storage.blob import BlobServiceClient
            import io
            
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service.get_container_client(container_name)
            blob_client = container_client.get_blob_client(entity_data_path)
            
            # Download to memory
            blob_data = blob_client.download_blob().readall()
            df = pd.read_parquet(io.BytesIO(blob_data))
            print(f"Successfully loaded {len(df)} records using BlobServiceClient")
    
    print(f"Columns: {df.columns.tolist()}")
    
    # Validate required columns
    required_columns = ["email_id", "label"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for event_timestamp, create if missing
    if "event_timestamp" not in df.columns:
        print("Warning: event_timestamp not found, creating default timestamps")
        df["event_timestamp"] = datetime.utcnow()
    
    # Check for sender_domain
    has_sender_domain = "sender_domain" in df.columns
    if not has_sender_domain:
        print("Warning: sender_domain not found, will skip sender_domain_features")
    
    # Select entity columns
    entity_columns = ["email_id", "event_timestamp", "label"]
    if has_sender_domain:
        entity_columns.append("sender_domain")
    
    entity_df = df[entity_columns].copy()
    
    # Ensure event_timestamp is datetime
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
    
    # Encode labels if they are strings
    if entity_df["label"].dtype == 'object':
        print(f"Converting string labels to binary: {entity_df['label'].unique()}")
        entity_df["label"] = entity_df["label"].map({"ham": 0, "spam": 1})
        print(f"After encoding: {entity_df['label'].unique()}")
    
    # Data quality checks
    null_counts = entity_df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Null values found:\n{null_counts[null_counts > 0]}")
        entity_df = entity_df.dropna(subset=["email_id", "label"])
        print(f"After dropping nulls: {len(entity_df)} records")
    
    # Validate labels are binary
    unique_labels = entity_df["label"].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be binary (0 or 1), found: {unique_labels}")
    
    # Calculate statistics
    spam_ratio = entity_df["label"].mean()
    print(f"Spam ratio: {spam_ratio:.2%}")
    print(f"Class distribution:\n{entity_df['label'].value_counts()}")
    
    # Stratified train/test split
    train_df, test_df = train_test_split(
        entity_df,
        test_size=test_split_ratio,
        stratify=entity_df["label"],
        random_state=42,
    )
    
    print(f"Train set: {len(train_df)} samples (spam ratio: {train_df['label'].mean():.2%})")
    print(f"Test set: {len(test_df)} samples (spam ratio: {test_df['label'].mean():.2%})")
    
    # Write outputs to blob storage
    train_output_path = f"abfs://{container_name}/training/train_entities.parquet"
    test_output_path = f"abfs://{container_name}/training/test_entities.parquet"
    
    print(f"Writing training entities to: {train_output_path}")
    train_df.to_parquet(train_output_path, storage_options=storage_options, index=False)
    
    print(f"Writing test entities to: {test_output_path}")
    test_df.to_parquet(test_output_path, storage_options=storage_options, index=False)
    
    # Return paths for downstream components
    train_path_output = f"{container_name}/training/train_entities.parquet"
    test_path_output = f"{container_name}/training/test_entities.parquet"
    
    # Log metrics
    metrics.log_metric("train_samples", len(train_df))
    metrics.log_metric("test_samples", len(test_df))
    metrics.log_metric("spam_ratio", spam_ratio)
    metrics.log_metric("total_samples", len(entity_df))
    metrics.log_metric("has_sender_domain", int(has_sender_domain))
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["train_count", "test_count", "spam_ratio", "train_entity_path", "test_entity_path"])
    return outputs(len(train_df), len(test_df), float(spam_ratio), train_path_output, test_path_output)
