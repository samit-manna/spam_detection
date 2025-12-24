"""
Download SpamAssassin Public Corpus and upload to Azure Blob Storage.

This script:
1. Downloads the SpamAssassin public corpus (spam + ham)
2. Extracts the archives
3. Uploads raw email files to Azure Blob Storage

Target path: abfss://datasets@<storage_account>.dfs.core.windows.net/raw/spamassassin/
"""

import os
import tarfile
import tempfile
import logging
from pathlib import Path
from urllib.request import urlretrieve

import ray
from adlfs import AzureBlobFileSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SpamAssassin corpus URLs
CORPUS_URLS = {
    "spam": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "spam_2": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
    "ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "ham_2": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
}

# Azure configuration from environment
STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = "datasets"
TARGET_PATH = "raw/spamassassin"


def get_azure_fs() -> AzureBlobFileSystem:
    """Create Azure Blob FileSystem client."""
    return AzureBlobFileSystem(
        account_name=STORAGE_ACCOUNT_NAME,
        account_key=STORAGE_ACCOUNT_KEY
    )


def download_and_extract(url: str, extract_dir: Path) -> Path:
    """Download and extract a tar.bz2 archive."""
    logger.info(f"Downloading {url}")
    
    archive_path = extract_dir / "archive.tar.bz2"
    urlretrieve(url, archive_path)
    
    logger.info(f"Extracting to {extract_dir}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(extract_dir)
    
    # Remove the archive after extraction
    archive_path.unlink()
    
    # Find the extracted directory (it's usually the only directory)
    extracted_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    return extracted_dirs[0] if extracted_dirs else extract_dir


@ray.remote
def process_corpus(corpus_name: str, url: str) -> dict:
    """
    Download, extract, and upload a corpus to Azure Blob Storage.
    Returns statistics about the processed corpus.
    """
    fs = get_azure_fs()
    stats = {"corpus": corpus_name, "files_uploaded": 0, "errors": 0}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Download and extract
        try:
            extracted_dir = download_and_extract(url, tmpdir_path)
        except Exception as e:
            logger.error(f"Failed to download/extract {corpus_name}: {e}")
            stats["errors"] += 1
            return stats
        
        # Determine label from corpus name
        label = "spam" if "spam" in corpus_name else "ham"
        
        # Upload each email file
        for email_file in extracted_dir.rglob("*"):
            if email_file.is_file() and not email_file.name.startswith("."):
                try:
                    # Read file content
                    content = email_file.read_bytes()
                    
                    # Construct target path
                    target_blob = f"{CONTAINER_NAME}/{TARGET_PATH}/{label}/{corpus_name}_{email_file.name}"
                    
                    # Upload to Azure
                    with fs.open(target_blob, "wb") as f:
                        f.write(content)
                    
                    stats["files_uploaded"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to upload {email_file.name}: {e}")
                    stats["errors"] += 1
        
        logger.info(f"Completed {corpus_name}: {stats['files_uploaded']} files uploaded")
    
    return stats


def main():
    """Main entry point."""
    logger.info("Starting SpamAssassin corpus download")
    
    # Validate environment
    if not STORAGE_ACCOUNT_NAME or not STORAGE_ACCOUNT_KEY:
        raise ValueError("AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set")
    
    # Initialize Ray (connects to existing cluster)
    ray.init(address="auto")
    
    logger.info(f"Connected to Ray cluster: {ray.cluster_resources()}")
    
    # Submit download tasks in parallel
    futures = [
        process_corpus.remote(name, url) 
        for name, url in CORPUS_URLS.items()
    ]
    
    # Wait for all downloads to complete
    results = ray.get(futures)
    
    # Summarize results
    total_files = sum(r["files_uploaded"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    
    logger.info(f"Download complete: {total_files} files uploaded, {total_errors} errors")
    
    for result in results:
        logger.info(f"  {result['corpus']}: {result['files_uploaded']} files")
    
    return {"total_files": total_files, "total_errors": total_errors}


if __name__ == "__main__":
    main()
