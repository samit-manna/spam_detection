# Feast Feature Store

Serverless Feast feature registry for spam detection. Features are stored in Azure Blob Storage and accessed directly via the Feast SDK - **no Feast server required**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEAST (SERVERLESS MODE)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    feast apply    ┌─────────────────────────────────┐ │
│  │  Feature Defs    │──────────────────▶│      Azure Blob Storage         │ │
│  │  (entities.py,   │                   │  ┌────────────────────────────┐ │ │
│  │   features.py)   │                   │  │ feast/registry.db          │ │ │
│  └──────────────────┘                   │  │ feast/features/*.parquet   │ │ │
│                                         │  └────────────────────────────┘ │ │
│                                         └─────────────────────────────────┘ │
│                                                        │                     │
│                                         ┌──────────────┴──────────────┐     │
│                                         ▼                             ▼     │
│                                 ┌──────────────┐              ┌──────────┐  │
│                                 │ Training Job │              │ Inference│  │
│                                 │ (Ray/K8s)    │              │ Service  │  │
│                                 └──────────────┘              └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# From data-pipeline directory
make feast-apply    # Register features in AKS

# Or run directly
cd feast/feature_repo
./apply_to_aks.sh
```

## Features Summary

| Feature View | Count | Description |
|-------------|-------|-------------|
| email_text_features | 8 | url_count, word_count, spam_keyword_count |
| email_structural_features | 8 | has_html, subject_length, received_hop_count |
| email_temporal_features | 4 | hour_of_day, day_of_week, is_weekend |
| email_tfidf_features | 500 | TF-IDF vectors |
| sender_domain_features | 4 | spam_ratio, email_count |
| **Total** | **524** | |

## Files

```
feature_repo/
├── feature_store.yaml    # Feast configuration (file-based, no server)
├── entities.py          # Entity definitions (email, sender_domain)
├── features.py          # Feature view definitions
├── apply_to_aks.sh      # Register features in AKS
└── setup_env.sh         # Environment setup helper
```

## Configuration

feature_store.yaml uses serverless mode:

```yaml
project: spam_detection
provider: local
registry: /data/registry.db    # File-based registry
offline_store:
  type: file                    # No external database
# No online_store configured - direct file access
```

## Using Features

### In Training Jobs (Python)

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

# Point to feature_repo directory (mounted as ConfigMap in K8s)
store = FeatureStore(repo_path="/feast-repo")

# Create entity dataframe
entity_df = pd.DataFrame({
    'email_id': ['email_001', 'email_002'],
    'event_timestamp': [datetime.now()] * 2
})

# Get historical features
features = store.get_historical_features(
    entity_df=entity_df,
    features=[
        'email_text_features:url_count',
        'email_text_features:spam_keyword_count',
        'email_structural_features:has_html',
    ],
).to_df()
```

### In Kubernetes Jobs

Mount the ConfigMap with feature definitions:

```yaml
volumes:
- name: feast-defs
  configMap:
    name: feast-feature-defs
containers:
- volumeMounts:
  - name: feast-defs
    mountPath: /feast-repo
```

## Verification

```bash
# Check registered features
kubectl logs -n feast -l app=feast-apply --tail=50

# List feature files in Azure
az storage blob list --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container-name feast --prefix features/ --query "[].name" -o tsv
```

## Why Serverless?

| Aspect | With Server | Serverless (Current) |
|--------|-------------|---------------------|
| Infrastructure | Feast server pod | None |
| Cost | Always running | Pay per use |
| Complexity | Service + PVC | ConfigMap only |
| Access | HTTP API | Direct SDK |
| Best for | Online serving | Batch ML pipelines |

For this spam detection pipeline (batch training, feature engineering), serverless Feast is simpler and more cost-effective.
