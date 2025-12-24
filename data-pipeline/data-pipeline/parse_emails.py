"""
Parse raw emails from SpamAssassin corpus into structured Parquet format.

This script:
1. Reads raw email files from Azure Blob Storage
2. Parses email structure (headers, body, metadata)
3. Writes structured data as Parquet to processed location

Input:  abfss://datasets@<storage_account>.dfs.core.windows.net/raw/spamassassin/
Output: abfss://datasets@<storage_account>.dfs.core.windows.net/processed/emails/
"""

import os
import re
import email
import hashlib
import logging
from email.utils import parsedate_to_datetime, parseaddr
from datetime import datetime
from typing import Optional

import ray
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from adlfs import AzureBlobFileSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Azure configuration
STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

INPUT_CONTAINER = "datasets"
INPUT_PATH = "raw/spamassassin"
OUTPUT_CONTAINER = "datasets"
OUTPUT_PATH = "processed/emails"


def get_azure_fs() -> AzureBlobFileSystem:
    """Create Azure Blob FileSystem client."""
    return AzureBlobFileSystem(
        account_name=STORAGE_ACCOUNT_NAME,
        account_key=STORAGE_ACCOUNT_KEY
    )


def parse_email_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse email date header into datetime."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return None


def extract_email_body(msg: email.message.Message) -> tuple[str, Optional[str]]:
    """Extract text and HTML body from email message."""
    text_body = ""
    html_body = None
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    # Try to decode as text
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        decoded = payload.decode(charset, errors='replace')
                    except (LookupError, UnicodeDecodeError):
                        decoded = payload.decode('utf-8', errors='replace')
                    
                    if content_type == "text/plain":
                        text_body += decoded
                    elif content_type == "text/html":
                        html_body = decoded
            except Exception:
                continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    text_body = payload.decode(charset, errors='replace')
                except (LookupError, UnicodeDecodeError):
                    text_body = payload.decode('utf-8', errors='replace')
        except Exception:
            text_body = str(msg.get_payload())
    
    return text_body, html_body


def extract_received_hops(msg: email.message.Message) -> list[str]:
    """Extract all Received headers from email."""
    received = msg.get_all("Received", [])
    # Convert Header objects to strings
    return [str(r) for r in received] if received else []


def parse_single_email(content: bytes, filename: str, label: str) -> Optional[dict]:
    """Parse a single email into structured format."""
    try:
        # Generate unique ID from content hash
        email_id = hashlib.sha256(content).hexdigest()[:16]
        
        # Parse email
        msg = email.message_from_bytes(content)
        
        # Helper to safely convert header to string
        def safe_header(value):
            if value is None:
                return ""
            return str(value)
        
        # Extract sender info
        sender_raw = safe_header(msg.get("From", ""))
        sender_name, sender_email = parseaddr(sender_raw)
        sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""
        
        # Extract recipient
        recipient_raw = safe_header(msg.get("To", ""))
        _, recipient_email = parseaddr(recipient_raw)
        
        # Extract body
        text_body, html_body = extract_email_body(msg)
        
        # Extract headers
        received_hops = extract_received_hops(msg)
        
        return {
            "email_id": email_id,
            "filename": filename,
            "label": label,
            
            # Basic headers
            "subject": safe_header(msg.get("Subject", "")),
            "sender_raw": sender_raw,
            "sender_email": sender_email,
            "sender_domain": sender_domain,
            "recipient": recipient_email,
            "date": parse_email_date(safe_header(msg.get("Date"))),
            
            # Body content
            "body_text": text_body[:50000] if text_body else "",  # Truncate very long bodies
            "body_html": html_body[:50000] if html_body else None,
            
            # Header metadata
            "content_type": msg.get_content_type(),
            "x_mailer": safe_header(msg.get("X-Mailer")),
            "received_hop_count": len(received_hops),
            "received_hops": received_hops[:10],  # Keep first 10 hops
            
            # Additional headers useful for spam detection
            "return_path": safe_header(msg.get("Return-Path")),
            "message_id": safe_header(msg.get("Message-ID")),
            "x_spam_status": safe_header(msg.get("X-Spam-Status")),
            "x_spam_score": safe_header(msg.get("X-Spam-Score")),
        }
        
    except Exception as e:
        logger.warning(f"Failed to parse {filename}: {e}")
        return None


@ray.remote
def process_email_batch(file_paths: list[str], label: str) -> list[dict]:
    """Process a batch of emails and return parsed records."""
    fs = get_azure_fs()
    records = []
    
    for file_path in file_paths:
        try:
            with fs.open(file_path, "rb") as f:
                content = f.read()
            
            filename = file_path.split("/")[-1]
            record = parse_single_email(content, filename, label)
            
            if record:
                records.append(record)
                
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
    
    return records


def list_emails_by_label(fs: AzureBlobFileSystem, label: str) -> list[str]:
    """List all email files for a given label."""
    path = f"{INPUT_CONTAINER}/{INPUT_PATH}/{label}"
    try:
        files = fs.ls(path, detail=False)
        # Filter to only files (not directories)
        return [f for f in files if not f.endswith("/")]
    except Exception as e:
        logger.error(f"Error listing {path}: {e}")
        return []


def main():
    """Main entry point."""
    logger.info("Starting email parsing job")
    
    if not STORAGE_ACCOUNT_NAME or not STORAGE_ACCOUNT_KEY:
        raise ValueError("Azure storage credentials not set")
    
    # Initialize Ray
    ray.init(address="auto")
    logger.info(f"Connected to Ray cluster: {ray.cluster_resources()}")
    
    fs = get_azure_fs()
    
    # Collect all email files
    all_files = []
    for label in ["spam", "ham"]:
        files = list_emails_by_label(fs, label)
        all_files.extend([(f, label) for f in files])
        logger.info(f"Found {len(files)} {label} emails")
    
    if not all_files:
        logger.error("No email files found!")
        return
    
    # Batch files for parallel processing
    batch_size = 100
    batches = []
    
    # Group by label for batching
    spam_files = [f for f, l in all_files if l == "spam"]
    ham_files = [f for f, l in all_files if l == "ham"]
    
    for i in range(0, len(spam_files), batch_size):
        batches.append((spam_files[i:i+batch_size], "spam"))
    
    for i in range(0, len(ham_files), batch_size):
        batches.append((ham_files[i:i+batch_size], "ham"))
    
    logger.info(f"Processing {len(all_files)} emails in {len(batches)} batches")
    
    # Submit batch processing tasks
    futures = [
        process_email_batch.remote(files, label)
        for files, label in batches
    ]
    
    # Collect results
    all_records = []
    for result in ray.get(futures):
        all_records.extend(result)
    
    logger.info(f"Parsed {len(all_records)} emails successfully")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Convert date column to proper datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    
    # Convert received_hops list to string for Parquet compatibility
    # Ensure all items are strings (some may be Header objects)
    df["received_hops"] = df["received_hops"].apply(
        lambda x: "|||".join(str(item) for item in x) if x else ""
    )
    
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Write to Parquet, partitioned by label
    output_path = f"{OUTPUT_CONTAINER}/{OUTPUT_PATH}"
    
    for label in ["spam", "ham"]:
        label_df = df[df["label"] == label]
        label_path = f"{output_path}/{label}/emails.parquet"
        
        logger.info(f"Writing {len(label_df)} {label} records to {label_path}")
        
        table = pa.Table.from_pandas(label_df, preserve_index=False)
        
        with fs.open(label_path, "wb") as f:
            pq.write_table(table, f)
    
    # Also write a combined file for convenience
    combined_path = f"{output_path}/all_emails.parquet"
    logger.info(f"Writing combined dataset to {combined_path}")
    
    table = pa.Table.from_pandas(df, preserve_index=False)
    with fs.open(combined_path, "wb") as f:
        pq.write_table(table, f)
    
    logger.info("Email parsing complete!")
    
    return {
        "total_emails": len(all_records),
        "spam_count": len(df[df["label"] == "spam"]),
        "ham_count": len(df[df["label"] == "ham"]),
    }


if __name__ == "__main__":
    main()
