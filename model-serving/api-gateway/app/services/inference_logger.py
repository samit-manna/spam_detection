"""
Async Inference Logger for Model Monitoring

Logs inference requests and predictions asynchronously for drift analysis.
Features:
- Non-blocking async logging (doesn't impact inference latency)
- Batching for efficient writes (configurable buffer size and flush interval)
- Parquet output with date partitioning
- Azure Blob Storage integration

Usage:
    from app.services.inference_logger import InferenceLogger
    
    logger = InferenceLogger(
        storage_account_name="...",
        container_name="data",
    )
    await logger.start()
    
    # After each prediction:
    await logger.log_prediction(
        inference_id="...",
        model_name="spam-detector",
        model_version="1.0",
        features={...},
        prediction="spam",
        spam_probability=0.87,
        latency_ms=23.5,
    )
    
    # On shutdown:
    await logger.stop()
"""

import os
import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.identity.aio import DefaultAzureCredential


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Schema Models
# =============================================================================

class InferenceRecord(BaseModel):
    """Schema for a single inference log record."""
    inference_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    
    # Input features (flattened for parquet storage)
    features: Dict[str, float]
    
    # Prediction output
    prediction: str  # "spam" or "ham"
    spam_probability: float
    
    # Metadata
    latency_ms: float
    api_version: str = "1.0.0"
    environment: str = "production"
    
    # Optional context
    email_id: Optional[str] = None
    sender_domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class InferenceBatch(BaseModel):
    """A batch of inference records for writing."""
    records: List[InferenceRecord]
    batch_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    record_count: int


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LoggerConfig:
    """Configuration for inference logger."""
    
    # Azure Storage
    azure_storage_account_name: str = ""
    azure_storage_account_key: Optional[str] = None
    azure_storage_container: str = "data"
    
    # Buffering
    buffer_size: int = 100
    flush_interval_seconds: int = 60
    
    # Paths
    log_path_prefix: str = "inference-logs"
    
    # Model info
    model_name: str = "spam-detector"
    
    # Behavior
    enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Local fallback
    local_fallback_enabled: bool = True
    local_fallback_path: str = "/tmp/inference-logs"


# =============================================================================
# Async Buffer
# =============================================================================

class AsyncBuffer:
    """Thread-safe async buffer for inference records."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._records: List[InferenceRecord] = []
        self._lock = asyncio.Lock()
        self._first_record_time: Optional[datetime] = None
    
    async def add(self, record: InferenceRecord) -> bool:
        """
        Add a record to the buffer.
        
        Returns:
            True if buffer is full and should be flushed
        """
        async with self._lock:
            if not self._first_record_time and not self._records:
                self._first_record_time = datetime.now(timezone.utc)
            
            self._records.append(record)
            return len(self._records) >= self.max_size
    
    async def drain(self) -> List[InferenceRecord]:
        """
        Drain all records from buffer.
        
        Returns:
            List of all records, clears the buffer
        """
        async with self._lock:
            records = self._records.copy()
            self._records = []
            self._first_record_time = None
            return records
    
    async def size(self) -> int:
        """Get current buffer size."""
        async with self._lock:
            return len(self._records)
    
    async def first_record_time(self) -> Optional[datetime]:
        """Get timestamp of first record in buffer."""
        async with self._lock:
            return self._first_record_time


# =============================================================================
# Parquet Writer
# =============================================================================

class ParquetWriter:
    """Writes inference records to parquet format."""
    
    # Define parquet schema
    SCHEMA = pa.schema([
        ("inference_id", pa.string()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("model_name", pa.string()),
        ("model_version", pa.string()),
        ("prediction", pa.string()),
        ("spam_probability", pa.float64()),
        ("latency_ms", pa.float64()),
        ("api_version", pa.string()),
        ("environment", pa.string()),
        ("email_id", pa.string()),
        ("sender_domain", pa.string()),
        # Features stored as JSON string for flexibility
        ("features_json", pa.string()),
        ("metadata_json", pa.string()),
    ])
    
    @classmethod
    def records_to_parquet(
        cls,
        records: List[InferenceRecord],
        output_path: str,
    ) -> str:
        """
        Convert inference records to parquet file.
        
        Args:
            records: List of inference records
            output_path: Path to write parquet file
            
        Returns:
            Path to written file
        """
        # Convert to dict format
        data = {
            "inference_id": [],
            "timestamp": [],
            "model_name": [],
            "model_version": [],
            "prediction": [],
            "spam_probability": [],
            "latency_ms": [],
            "api_version": [],
            "environment": [],
            "email_id": [],
            "sender_domain": [],
            "features_json": [],
            "metadata_json": [],
        }
        
        for record in records:
            data["inference_id"].append(record.inference_id)
            data["timestamp"].append(record.timestamp)
            data["model_name"].append(record.model_name)
            data["model_version"].append(record.model_version)
            data["prediction"].append(record.prediction)
            data["spam_probability"].append(record.spam_probability)
            data["latency_ms"].append(record.latency_ms)
            data["api_version"].append(record.api_version)
            data["environment"].append(record.environment)
            data["email_id"].append(record.email_id or "")
            data["sender_domain"].append(record.sender_domain or "")
            data["features_json"].append(json.dumps(record.features))
            data["metadata_json"].append(json.dumps(record.metadata or {}))
        
        # Create table and write
        table = pa.Table.from_pydict(data, schema=cls.SCHEMA)
        pq.write_table(table, output_path, compression="snappy")
        
        return output_path
    
    @classmethod
    def records_to_dataframe(cls, records: List[InferenceRecord]) -> pd.DataFrame:
        """Convert records to pandas DataFrame."""
        data = []
        for record in records:
            row = record.model_dump()
            row["features_json"] = json.dumps(row.pop("features"))
            row["metadata_json"] = json.dumps(row.pop("metadata") or {})
            data.append(row)
        
        return pd.DataFrame(data)


# =============================================================================
# Azure Blob Writer
# =============================================================================

class AsyncBlobWriter:
    """Async Azure Blob Storage writer."""
    
    def __init__(self, config: LoggerConfig):
        self.config = config
        self._client: Optional[BlobServiceClient] = None
        self._container_client: Optional[ContainerClient] = None
    
    async def connect(self):
        """Initialize blob service connection."""
        if not self.config.azure_storage_account_name:
            raise ValueError("Azure storage account name is required")
        
        if self.config.azure_storage_account_key:
            # Use account key
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.config.azure_storage_account_name};"
                f"AccountKey={self.config.azure_storage_account_key};"
                f"EndpointSuffix=core.windows.net"
            )
            self._client = BlobServiceClient.from_connection_string(connection_string)
        else:
            # Use managed identity
            credential = DefaultAzureCredential()
            account_url = f"https://{self.config.azure_storage_account_name}.blob.core.windows.net"
            self._client = BlobServiceClient(account_url, credential=credential)
        
        self._container_client = self._client.get_container_client(
            self.config.azure_storage_container
        )
        
        logger.info(f"Connected to Azure Blob: {self.config.azure_storage_account_name}")
    
    async def close(self):
        """Close blob service connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._container_client = None
    
    def _generate_blob_path(self, timestamp: datetime) -> str:
        """
        Generate blob path with date partitioning.
        
        Format: inference-logs/year=YYYY/month=MM/day=DD/hour=HH/batch_UUID.parquet
        """
        batch_id = str(uuid.uuid4())[:8]
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        return (
            f"{self.config.log_path_prefix}/"
            f"year={timestamp.year}/"
            f"month={timestamp.month:02d}/"
            f"day={timestamp.day:02d}/"
            f"hour={timestamp.hour:02d}/"
            f"batch_{ts_str}_{batch_id}.parquet"
        )
    
    async def write_batch(
        self,
        records: List[InferenceRecord],
        retry_count: int = 0,
    ) -> str:
        """
        Write a batch of records to blob storage.
        
        Args:
            records: List of inference records
            retry_count: Current retry attempt
            
        Returns:
            Blob path where records were written
        """
        if not records:
            return ""
        
        if not self._container_client:
            await self.connect()
        
        # Generate blob path based on first record timestamp
        timestamp = records[0].timestamp
        blob_path = self._generate_blob_path(timestamp)
        
        try:
            # Write to temp file first
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
            
            ParquetWriter.records_to_parquet(records, tmp_path)
            
            # Upload to blob
            blob_client = self._container_client.get_blob_client(blob_path)
            
            with open(tmp_path, "rb") as f:
                await blob_client.upload_blob(f, overwrite=True)
            
            # Cleanup temp file
            os.unlink(tmp_path)
            
            logger.info(f"Wrote {len(records)} records to {blob_path}")
            return blob_path
            
        except Exception as e:
            logger.error(f"Failed to write batch to blob: {e}")
            
            # Retry logic
            if retry_count < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay_seconds * (retry_count + 1))
                return await self.write_batch(records, retry_count + 1)
            
            # Fallback to local storage
            if self.config.local_fallback_enabled:
                return await self._write_local_fallback(records, timestamp)
            
            raise
    
    async def _write_local_fallback(
        self,
        records: List[InferenceRecord],
        timestamp: datetime,
    ) -> str:
        """Write to local file as fallback."""
        fallback_dir = Path(self.config.local_fallback_path)
        fallback_dir.mkdir(parents=True, exist_ok=True)
        
        batch_id = str(uuid.uuid4())[:8]
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        local_path = fallback_dir / f"batch_{ts_str}_{batch_id}.parquet"
        
        ParquetWriter.records_to_parquet(records, str(local_path))
        logger.warning(f"Wrote {len(records)} records to local fallback: {local_path}")
        
        return str(local_path)


# =============================================================================
# Inference Logger
# =============================================================================

class InferenceLogger:
    """
    Async inference logger for model monitoring.
    
    Features:
    - Non-blocking async logging
    - Batched writes for efficiency
    - Auto-flush on buffer full or time interval
    - Azure Blob Storage with local fallback
    
    Usage:
        logger = InferenceLogger(
            storage_account_name="...",
            container_name="data",
        )
        await logger.start()
        
        # Log predictions
        await logger.log_prediction(...)
        
        # Shutdown
        await logger.stop()
    """
    
    def __init__(
        self,
        storage_account_name: Optional[str] = None,
        storage_account_key: Optional[str] = None,
        container_name: str = "data",
        model_name: str = "spam-detector",
        buffer_size: int = 100,
        flush_interval_seconds: int = 60,
        enabled: bool = True,
        log_path_prefix: str = "inference-logs",
    ):
        self.config = LoggerConfig(
            azure_storage_account_name=storage_account_name or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", ""),
            azure_storage_account_key=storage_account_key or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY"),
            azure_storage_container=container_name,
            model_name=model_name,
            buffer_size=buffer_size,
            flush_interval_seconds=flush_interval_seconds,
            enabled=enabled,
            log_path_prefix=log_path_prefix,
        )
        self._buffer = AsyncBuffer(self.config.buffer_size)
        self._writer = AsyncBlobWriter(self.config)
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = {
            "records_logged": 0,
            "batches_written": 0,
            "errors": 0,
        }
    
    async def start(self):
        """Start the inference logger."""
        if not self.config.enabled:
            logger.info("Inference logging is disabled")
            return
        
        logger.info("Starting inference logger")
        
        try:
            await self._writer.connect()
        except Exception as e:
            logger.error(f"Failed to connect to blob storage: {e}")
            if not self.config.local_fallback_enabled:
                raise
            logger.warning("Will use local fallback for logging")
        
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
        logger.info(
            f"Inference logger started (buffer_size={self.config.buffer_size}, "
            f"flush_interval={self.config.flush_interval_seconds}s)"
        )
    
    async def stop(self):
        """Stop the inference logger and flush remaining records."""
        logger.info("Stopping inference logger")
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush()
        
        # Close writer
        await self._writer.close()
        
        logger.info(
            f"Inference logger stopped. Stats: {self._stats}"
        )
    
    async def log(
        self,
        record: InferenceRecord,
    ):
        """
        Log an inference record (non-blocking).
        
        This method returns immediately - the actual write
        happens asynchronously in batches.
        """
        if not self.config.enabled:
            return
        
        # Add to buffer
        should_flush = await self._buffer.add(record)
        self._stats["records_logged"] += 1
        
        # Flush if buffer is full
        if should_flush:
            asyncio.create_task(self._flush())
    
    async def log_prediction(
        self,
        inference_id: str,
        model_name: str,
        model_version: str,
        features: Dict[str, float],
        prediction: str,
        spam_probability: float,
        latency_ms: float,
        email_id: Optional[str] = None,
        sender_domain: Optional[str] = None,
        api_version: str = "1.0.0",
        environment: str = "production",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Convenience method to log a prediction.
        
        Constructs an InferenceRecord and logs it.
        """
        record = InferenceRecord(
            inference_id=inference_id,
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            model_version=model_version,
            features=features,
            prediction=prediction,
            spam_probability=spam_probability,
            latency_ms=latency_ms,
            api_version=api_version,
            environment=environment,
            email_id=email_id,
            sender_domain=sender_domain,
            metadata=metadata,
        )
        await self.log(record)
    
    async def _flush(self):
        """Flush buffer to storage."""
        records = await self._buffer.drain()
        
        if not records:
            return
        
        try:
            await self._writer.write_batch(records)
            self._stats["batches_written"] += 1
        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
            self._stats["errors"] += 1
            # Records are lost if all retries and fallback fail
    
    async def _periodic_flush(self):
        """Background task for periodic flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                
                # Check if we have records to flush
                if await self._buffer.size() > 0:
                    await self._flush()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get logging statistics."""
        return self._stats.copy()
    
    async def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return await self._buffer.size()
