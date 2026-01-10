# Data Pipeline

Ray-based data preprocessing pipeline for spam detection. Downloads, parses, and extracts features from raw email data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Download   │───▶│    Parse     │───▶│   Extract    │                   │
│  │   Dataset    │    │   Emails     │    │   Features   │                   │
│  │   (RayJob)   │    │   (RayJob)   │    │   (RayJob)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│   datasets/raw/      datasets/processed/   feast/features/                  │
│   spamassassin/      emails/*.parquet      email_features/                  │
│                                            sender_features/                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Step 1: Start port-forward to Ray cluster (in a separate terminal)
make port-forward

# Step 2: Run complete pipeline (in main terminal)
make run-pipeline

# Or run individual stages
make download-dataset    # Stage 1: Download SpamAssassin corpus (~5-10 min)
make parse-emails        # Stage 2: Parse raw emails to Parquet (~10-15 min)  
make extract-features    # Stage 3: Extract ML features (~15-20 min)

# Monitor progress
make status              # Check Ray cluster status
make logs                # Open Ray dashboard in browser
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `port-forward` | Start port-forward to Ray cluster (run in separate terminal) |
| `check-ray-connection` | Verify Ray cluster is accessible |
| `run-pipeline` | Run all 3 stages sequentially |
| `download-dataset` | Stage 1: Download SpamAssassin corpus |
| `parse-emails` | Stage 2: Parse raw emails to Parquet |
| `extract-features` | Stage 3: Extract ML features |
| `status` | Check status of Ray cluster |
| `logs` | Open Ray dashboard in browser |
| `clean` | Delete old RayJob resources |
| `verify-data` | Verify outputs in Azure Blob |

## Directory Structure

```
data-pipeline/
├── data-pipeline/
│   ├── download_dataset.py         # Download SpamAssassin corpus
│   ├── parse_emails.py             # Parse raw emails to Parquet
│   ├── extract_features.py         # Extract ML features
│   ├── rayjob-download-dataset.yaml
│   ├── rayjob-parse-emails.yaml
│   └── rayjob-extract-features.yaml
├── feast/
│   └── feature_repo/
│       ├── feature_store.yaml      # Feast config
│       ├── entities.py             # Entity definitions
│       └── features.py             # Feature views
└── README.md
```

## Output Data

| Step | Output Path | Description |
|------|-------------|-------------|
| Download | `datasets/raw/spamassassin/` | ~6,000 raw email files |
| Parse | `datasets/processed/emails/` | Structured Parquet files |
| Extract | `feast/features/email_features/` | 524 ML features |
| Extract | `feast/features/sender_features/` | Sender domain stats |
| Extract | `feast/features/artifacts/` | TF-IDF vectorizer |

## Feature Schema

| Feature Group | Count | Examples |
|---------------|-------|----------|
| Text | 8 | url_count, word_count, uppercase_ratio |
| Structural | 8 | has_html, subject_length, received_hop_count |
| Temporal | 4 | hour_of_day, is_weekend, is_night_hour |
| TF-IDF | 500 | tfidf_0 to tfidf_499 |
| Sender | 4 | spam_ratio, email_count |
| **Total** | **524** | |

## Verification

```bash
# Check outputs in Azure Blob
az storage blob list --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container-name datasets --prefix processed/emails/ --query "[].name"

az storage blob list --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container-name feast --prefix features/ --query "[].name"
```

## Next Steps

After data pipeline completes, proceed to **Training Pipeline**:
1. Features are loaded from Feast
2. Model is trained with Ray HPO
3. Best model is registered in MLflow
