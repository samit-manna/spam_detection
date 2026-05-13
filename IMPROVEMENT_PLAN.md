# E2E ML Platform — Improvement Roadmap

> Generated: 2026-02-26
> Based on analysis of the spam detection MLOps platform (3-day take-home assessment).

---

## Section 1: Data Pipeline

### 1.1 Fix Feature Count Inconsistency (Critical Bug)
`extract_features.py` produces 524 features, but `feature-transformer/transformer.py` and integration tests expect 528. If a model trained on 524 is served against 528-feature vectors, inference silently breaks. Audit which count is correct, align all components, and add an assertion in the transformer's `/ready` check that validates `n_features_in_` matches the served model.

### 1.2 Add Data Validation Layer
Currently there is no schema enforcement on raw ingested data. Add a validation step between parse and feature extraction using **Great Expectations** or **Pandera**:
- Assert expected column presence and dtypes in the parsed Parquet
- Assert feature vector shape and value ranges post-extraction
- Fail fast with a structured error rather than silently producing garbage features

### 1.3 Incremental / Streaming Ingestion
The current pipeline is a full-corpus batch job on SpamAssassin. A production system should support incremental ingestion:
- Track which emails have been processed (watermark or manifest file in Azure Blob)
- Only re-run extraction for new data
- Consider a streaming path using **Azure Event Hubs → Spark Structured Streaming** for real-time corpus updates

### 1.4 Data Lineage & Cataloging
Add lineage tracking so every model version can be traced back to the exact dataset version:
- Tag Parquet files with pipeline run ID and commit SHA
- Register datasets in **Azure Purview** or emit OpenLineage events to MLflow
- Store dataset hash in MLflow `run.data.tags` alongside the model

### 1.5 Sender Domain Feature Fallback Hardening
`transformer.py` falls back to `spam_ratio=0.15` for unknown sender domains. This is a fixed prior that may not match training-time behavior. Options:
- Store a `[UNKNOWN]` bucket in Feast trained on the distribution of unknown domains
- Use Laplace smoothing during feature extraction so inference matches training exactly
- Log every fallback occurrence as a metric for drift monitoring

### 1.6 Unit Tests for Data Pipeline
There are zero pytest files under `data-pipeline/`. Add:
- Unit tests for email parsing (edge cases: multipart MIME, missing headers, non-ASCII)
- Unit tests for each feature extraction function
- Parametrized tests against known spam/ham samples with expected feature values

---

## Section 2: Model Training

### 2.1 Fix Ray Tune API Mixed Usage (Compatibility Risk)
`hpo_train.py` mixes the legacy `tune.run()` API with the new `train.report()`. In Ray 2.x, the correct path is `Tuner` + `TuneConfig` + `ResultGrid`. The current code may break silently on Ray upgrade. Migrate fully to the new API.

### 2.2 Fix Deprecated XGBoost Parameter
`use_label_encoder=False` was removed in XGBoost 2.0. Remove it from both `baseline_training.py` and `hpo_train.py` to prevent runtime errors on library upgrades.

### 2.3 Fix Fragile Image URI Injection
All 8 KFP component decorators use `base_image="placeholder"` and the pipeline compiler patches the YAML via regex. A single regex mistake leaves `placeholder` in the final YAML and the pipeline silently fails at submission. Instead:
- Pass image URIs as environment variables read at compile time (`os.getenv("COMPONENT_IMAGE")`)
- Or use KFP's `set_default_base_image()` call before pipeline compilation
- Add a post-compile validation step that greps the YAML for `placeholder` and fails the `make compile-pipeline` target if found

### 2.4 Expand HPO Search Space
The current HPO covers 9 XGBoost hyperparameters over 20 trials. Improvements:
- Add `scale_pos_weight` (critical for class-imbalanced email corpora — SpamAssassin is ~31% spam)
- Add `subsample_freq` for better regularization
- Increase to 50+ trials and use **Optuna** search instead of random for sample efficiency
- Log Pareto-optimal models (F1 vs. latency) rather than single best

### 2.5 Add Offline Bias & Fairness Evaluation
The training pipeline only tracks F1, AUC, precision, recall. Add a dedicated fairness component:
- Slice metrics by sender domain, email client, language
- Flag if spam rate disparity across groups exceeds a threshold
- Log sliced metrics to MLflow for longitudinal comparison

### 2.6 Reproducibility Guarantees
- Pin random seeds in every component (`numpy`, `sklearn`, `xgboost`, `ray`)
- Log Python + library versions as MLflow tags
- Store the exact `pip freeze` output as an artifact per run
- Use Kubeflow's artifact caching to avoid re-running unchanged upstream components

### 2.7 Add Cross-Validation to Baseline Training
`baseline_training.py` does a single train/test split. Add stratified k-fold CV (k=5) and report mean ± std of all metrics. This gives a more reliable signal before committing to HPO, especially with ~6,000 samples.

### 2.8 Unit Tests for Training Components
Currently there are no unit tests for `training/pipeline/components/`. Add:
- Mocked MLflow + Azure Blob tests for each component
- Schema validation tests (input/output artifact shapes)
- Quality gate logic tests (assert that a model with F1=0.84 is correctly rejected)

---

## Section 3: Model Serving

### 3.1 Implement Canary / Progressive Rollout
The production deploy in `model-serving.yaml` sets `canaryTrafficPercent: 0` — meaning 100% traffic switches immediately to the new model. Replace this with a proper canary ramp:
```
10% → monitor 15 min → 50% → monitor 15 min → 100%
```
Use KServe's native `canaryTrafficPercent` field and a GitHub Actions job that polls error rate + latency before advancing each stage.

### 3.2 Add Shadow Mode / Traffic Mirroring
Before canary promotion, run the new model in shadow mode — it receives mirrored traffic and logs predictions but does not serve responses. Compare shadow predictions against the production model to detect silent regressions before any user is affected.

### 3.3 Fix Rollback Workflow Hardcoded ACR Name
`rollback.yaml` line 57 has a literal ACR hostname instead of `${{ secrets.ACR_NAME }}`. This means rollbacks always target the dev registry even in production. Fix immediately.

### 3.4 Fix Unreachable MLflow Code in `model_evaluation.py`
Line 332 calls `mlflow.log_param()` inside the `except` block for a failed `mlflow.start_run()`. Since the run never started, this raises a secondary exception that is silently swallowed. The fix is to log to stderr/stdout instead and re-raise the original exception.

### 3.5 Async Inference & Request Coalescing
The single `/predict` endpoint is synchronous. For high-throughput scenarios:
- Add request batching at the gateway (collect requests for 10ms, batch to Triton)
- Use Triton's dynamic batching instead of re-implementing at the gateway level
- Consider `asyncio` streaming responses for very large batch jobs

### 3.6 Add Unit Tests for API Gateway
`model-serving/tests/` only has integration tests. Add unit tests with mocked services:
- Test auth middleware (missing key, wrong role, correct key)
- Test routing logic (`X-Environment` header → staging vs prod)
- Test spam threshold behavior (borderline scores around 0.7)
- Test error handler response format

### 3.7 Request & Response Schema Validation
Use **Pydantic v2** strict mode on all request/response models to catch malformed inputs at the boundary rather than propagating errors into downstream services.

### 3.8 Add Rate Limiting
The API gateway has auth middleware but no rate limiting. Add per-API-key rate limits (e.g., 1000 req/min for `operator`, 100 req/min for `viewer`) using `slowapi` or a Redis-backed token bucket.

### 3.9 Inference Schema Versioning
When the feature vector shape changes (e.g., 524→528), there is no mechanism to detect that the deployed model and the feature transformer are mismatched. Add:
- A schema version tag in the MLflow model registry
- A startup check in the API gateway that compares schema versions between feature transformer and the active model
- Fail `/ready` if versions do not match

---

## Section 4: Monitoring & Observability

### 4.1 Add Real-Time Alerting Integration
`alerts.py` sends webhook POST alerts but there is no integration with PagerDuty, Opsgenie, or Slack out of the box. Add:
- **Slack** webhook for non-critical drift (minor PSI 0.1–0.2)
- **PagerDuty** for critical drift (PSI > 0.2, prediction drift > threshold)
- Alert de-duplication (do not re-fire the same alert every hour)

### 4.2 Model Performance Monitoring (Beyond Drift)
Current monitoring only tracks feature distribution drift. Add:
- **Label efficiency**: Track spam/ham ratio in serving traffic over time
- **Confidence calibration**: Monitor prediction score distribution (are scores clustering near 0.5?)
- **Business metrics**: False positive rate (legitimate email marked spam) — more impactful than raw accuracy
- Integration with Evidently's `TestSuite` for automated pass/fail rather than just reports

### 4.3 Inference Logging Schema Validation
`inference_logger.py` writes Parquet without schema enforcement. Add:
- Arrow schema pinning to inference log writes
- Schema compatibility check at drift detector startup (bail out if logs do not match baseline schema)
- Version the log format alongside the model schema version

### 4.4 Add Grafana Dashboards
Prometheus metrics are exposed at `/metrics` but there are no pre-built Grafana dashboards. Provide dashboard JSON for:
- Request rate / error rate / latency (p50, p95, p99) per environment
- Drift score over time per feature group
- Model score distribution over time
- HPO trial outcomes

### 4.5 Feedback Loop / Active Learning
The system has no mechanism for users to flag false positives/negatives. Add:
- A `/feedback` endpoint on the API gateway that accepts `{email_id, label, confidence}`
- Store feedback in Azure Blob partitioned by date
- Incorporate high-confidence feedback labels into the next retraining run as pseudo-labels

---

## Section 5: Infrastructure & DevOps

### 5.1 Add Terraform State Backend
The Terraform modules do not configure a remote state backend. In a team environment this causes state conflicts. Add an Azure Storage Account + blob container as the Terraform backend with state locking.

### 5.2 Separate Staging and Production Terraform Workspaces
Currently there is a single set of Terraform modules. Use **Terraform workspaces** or separate `tfvars` files to manage staging and production infrastructure independently with environment-specific sizing (e.g., prod AKS uses larger node pools).

### 5.3 Add Cost Monitoring
AKS + Ray clusters are expensive. Add:
- Azure Cost Management budget alerts
- Tag all resources with `environment`, `team`, `pipeline-run-id`
- Auto-shutdown of HPO Ray clusters after job completion (the `hpo_tuning.py` component polls for completion but does not verify teardown)

### 5.4 Secret Rotation
All secrets are stored as Kubernetes secrets and GitHub Actions secrets. Add:
- Azure Key Vault integration for secret rotation without redeployment
- Reference secrets from Key Vault in pod specs using the **Secrets Store CSI driver**

### 5.5 Add Dependency Vulnerability Scanning
No `requirements.txt` auditing in the CI pipeline. Add:
- `pip-audit` or `safety` check on every PR
- Dependabot for automated dependency PRs
- Container image scanning with **Trivy** in the build step of every workflow

---

## Section 6: Testing & Quality

### 6.1 Add End-to-End Integration Test Suite
The current `integration_test.py` covers basic paths but not failure modes. Add:
- Adversarial email tests (obfuscated spam, image-only spam)
- Load tests (`locust` with realistic traffic shapes)
- Chaos tests (kill the feature transformer mid-request, expect graceful degradation)

### 6.2 Contract Testing Between Services
The API gateway, feature transformer, and Triton inference server have implicit interface contracts. Add **Pact** consumer-driven contract tests so changes to the feature transformer's `/transform` response format are caught before deployment.

### 6.3 Data Pipeline Testing
- Add `pytest` + `pytest-mock` tests for all three Ray job scripts
- Test email parsing against malformed/adversarial MIME inputs
- Test feature extraction reproducibility (same input → same output, deterministic)

### 6.4 Model Card & Documentation
Generate a **model card** as a CI artifact on every training run:
- Training data description and known biases
- Performance metrics across slices
- Intended use cases and limitations
- Out-of-scope use cases

---

## Priority Matrix

| Priority | Item | Location |
|----------|------|----------|
| **P0 — Fix now** | Feature count mismatch (524 vs 528) | `extract_features.py`, `transformer.py` |
| **P0 — Fix now** | Hardcoded ACR name in rollback workflow | `rollback.yaml` line 57 |
| **P0 — Fix now** | Deprecated `use_label_encoder=False` | `baseline_training.py`, `hpo_train.py` |
| **P0 — Fix now** | Unreachable MLflow code in exception handler | `model_evaluation.py` line 332 |
| **P1 — Next sprint** | Canary / progressive rollout | `model-serving.yaml`, KServe ISVCs |
| **P1 — Next sprint** | Unit tests for API gateway | `model-serving/tests/` |
| **P1 — Next sprint** | Unit tests for training components | `training/pipeline/components/` |
| **P1 — Next sprint** | Data validation layer (Great Expectations / Pandera) | `data-pipeline/` |
| **P1 — Next sprint** | Ray Tune API migration to `Tuner` + `ResultGrid` | `docker/ray-train/scripts/hpo_train.py` |
| **P1 — Next sprint** | Fix fragile `placeholder` image injection | `spam_detection_pipeline.py` |
| **P2 — Next quarter** | Shadow mode / traffic mirroring | Model serving |
| **P2 — Next quarter** | Inference schema versioning | API gateway, MLflow registry |
| **P2 — Next quarter** | Feedback loop `/feedback` endpoint | API gateway |
| **P2 — Next quarter** | Grafana dashboards | Monitoring |
| **P2 — Next quarter** | Contract testing (Pact) | `model-serving/tests/` |
| **P2 — Next quarter** | Terraform remote state backend | `terraform/` |
| **P3 — Long term** | Active learning integration | Training pipeline |
| **P3 — Long term** | Streaming ingestion (Event Hubs) | Data pipeline |
| **P3 — Long term** | Cost monitoring & auto-shutdown | Infrastructure |
| **P3 — Long term** | Model cards as CI artifacts | Training pipeline |
| **P3 — Long term** | Full Terraform workspace separation | `terraform/` |
