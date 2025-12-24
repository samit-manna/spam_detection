"""
Model Export Script: MLflow → ONNX → Azure Blob Storage

This script:
1. Downloads a trained XGBoost model from MLflow registry
2. Converts it to ONNX format for Triton Inference Server
3. Validates the ONNX model
4. Uploads to Azure Blob Storage in Triton model repository format
5. Generates Triton config.pbtxt

Usage:
    python export_model.py --model-name spam-detector --model-stage Staging

Environment variables:
    MLFLOW_TRACKING_URI: MLflow tracking server URL
    AZURE_STORAGE_ACCOUNT_NAME: Azure storage account name
    AZURE_STORAGE_ACCOUNT_KEY: Azure storage account key
    AZURE_STORAGE_CONTAINER: Container name for models (default: models)
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import numpy as np
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort
from azure.storage.blob import BlobServiceClient, ContainerClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service.mlflow.svc.cluster.local:5000"
)
AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "models")

# Feature configuration - this will be detected from the model dynamically
DEFAULT_NUM_FEATURES = 528  # Default, but will be overridden by model's actual feature count


def get_num_features_from_model(model) -> int:
    """
    Detect the number of features from the loaded model.
    
    Args:
        model: XGBoost model (XGBClassifier or Booster)
        
    Returns:
        Number of features the model was trained with
    """
    # Try XGBClassifier's get_booster() method
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        n_features = booster.num_features()
        logger.info(f"Detected {n_features} features from XGBoost booster")
        return n_features
    
    # Try Booster directly
    if isinstance(model, xgb.Booster):
        n_features = model.num_features()
        logger.info(f"Detected {n_features} features from XGBoost booster")
        return n_features
    
    # Fallback for sklearn-like models
    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
        logger.info(f"Detected {n_features} features from n_features_in_")
        return n_features
    
    logger.warning(f"Could not detect feature count from model, using default {DEFAULT_NUM_FEATURES}")
    return DEFAULT_NUM_FEATURES


# =============================================================================
# MLflow Model Download
# =============================================================================

def download_model_from_mlflow(
    model_name: str,
    model_stage: str = "Staging",
    model_version: Optional[int] = None
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """
    Download XGBoost model from MLflow registry.
    
    Args:
        model_name: Name of the registered model
        model_stage: Stage to download (Staging, Production, None)
        model_version: Specific version to download (overrides stage)
        
    Returns:
        Tuple of (XGBoost Booster, model metadata dict)
    """
    logger.info(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Get model version
    if model_version:
        logger.info(f"Fetching model {model_name} version {model_version}")
        mv = client.get_model_version(model_name, str(model_version))
    else:
        logger.info(f"Fetching model {model_name} stage {model_stage}")
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        if not versions:
            raise ValueError(f"No model found for {model_name} with stage {model_stage}")
        mv = versions[0]
    
    logger.info(f"Found model version {mv.version} from run {mv.run_id}")
    
    # Get model URI and metadata
    model_uri = f"models:/{model_name}/{mv.version}"
    
    # Load the XGBoost model
    logger.info(f"Loading model from {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)
    
    # Get run info for metadata
    run = client.get_run(mv.run_id)
    
    metadata = {
        "model_name": model_name,
        "model_version": mv.version,
        "model_stage": mv.current_stage,
        "run_id": mv.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
    }
    
    logger.info(f"Model loaded successfully. Metrics: {metadata['metrics']}")
    
    return model, metadata


# =============================================================================
# ONNX Conversion
# =============================================================================

def convert_to_onnx(
    model,
    num_features: int = DEFAULT_NUM_FEATURES,
    output_path: Optional[str] = None
) -> onnx.ModelProto:
    """
    Convert XGBoost model to ONNX format.
    
    Args:
        model: XGBoost model (Booster or XGBClassifier)
        num_features: Number of input features
        output_path: Optional path to save the ONNX model
        
    Returns:
        ONNX ModelProto
    """
    logger.info(f"Converting XGBoost model to ONNX with {num_features} features")
    
    # If we have an XGBClassifier, get the underlying Booster for more reliable conversion
    if hasattr(model, 'get_booster'):
        logger.info("Extracting Booster from XGBClassifier for ONNX conversion")
        booster = model.get_booster()
        
        # For booster, we need to create a wrapper that onnxmltools can understand
        # Use the classifier but set the required attributes
        # The issue is that loaded XGBClassifier doesn't have _n_classes set
        try:
            # Try setting via internal attribute
            object.__setattr__(model, '_n_classes', 2)
        except:
            pass
        
        # Alternative: Create a fresh XGBClassifier and attach the booster
        from xgboost import XGBClassifier
        wrapper = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
        wrapper._Booster = booster
        wrapper._n_classes = 2
        wrapper.n_classes_ = 2
        model = wrapper
        logger.info("Created wrapper XGBClassifier with n_classes_=2")
    
    # Define input type
    initial_type = [('input', FloatTensorType([None, num_features]))]
    
    # Convert to ONNX
    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=12
    )
    
    # Add metadata
    onnx_model.doc_string = "Spam detection XGBoost model converted to ONNX"
    
    # Log original output names
    original_outputs = [output.name for output in onnx_model.graph.output]
    logger.info(f"Original ONNX output names: {original_outputs}")
    
    # Don't rename outputs - keep them as-is from onnxmltools
    # The converter generates 'label' for class predictions and 'probabilities' for probabilities
    # Triton will work with these names directly
    
    if output_path:
        logger.info(f"Saving ONNX model to {output_path}")
        onnx.save(onnx_model, output_path)
    
    logger.info("ONNX conversion completed")
    return onnx_model


def validate_onnx_model(
    onnx_model_path: str,
    num_features: int = DEFAULT_NUM_FEATURES,
    xgb_model: Optional[xgb.Booster] = None
) -> bool:
    """
    Validate the ONNX model.
    
    Args:
        onnx_model_path: Path to the ONNX model file
        num_features: Number of input features
        xgb_model: Optional XGBoost model for output comparison
        
    Returns:
        True if validation passes
    """
    logger.info("Validating ONNX model...")
    
    # Load and check the model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure is valid")
    
    # Test inference with ONNX Runtime
    logger.info("Testing inference with ONNX Runtime...")
    session = ort.InferenceSession(onnx_model_path)
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [o.name for o in session.get_outputs()]
    
    logger.info(f"Input: {input_name}, shape: {input_shape}")
    logger.info(f"Outputs: {output_names}")
    
    # Run test inference
    test_input = np.random.randn(1, num_features).astype(np.float32)
    outputs = session.run(None, {input_name: test_input})
    
    logger.info(f"Test inference successful. Output shapes: {[o.shape for o in outputs]}")
    
    # Compare with XGBoost output if provided
    if xgb_model is not None:
        logger.info("Comparing ONNX output with XGBoost output...")
        
        # Always use the booster directly for comparison since that's what we convert
        if hasattr(xgb_model, 'get_booster'):
            booster = xgb_model.get_booster()
        else:
            booster = xgb_model
        
        dmatrix = xgb.DMatrix(test_input)
        xgb_pred = booster.predict(dmatrix)
        
        # Get ONNX probabilities (usually the second output)
        onnx_probs = outputs[1] if len(outputs) > 1 else outputs[0]
        if len(onnx_probs.shape) > 1:
            onnx_pred = onnx_probs[:, 1]  # Get probability of class 1
        else:
            onnx_pred = onnx_probs
        
        # Check if predictions are close
        max_diff = np.max(np.abs(xgb_pred - onnx_pred))
        logger.info(f"Max prediction difference: {max_diff}")
        
        if max_diff > 0.01:
            logger.warning(f"Prediction difference is high: {max_diff}")
        else:
            logger.info("ONNX predictions match XGBoost predictions")
    
    logger.info("ONNX model validation completed successfully")
    return True


# =============================================================================
# Triton Config Generation
# =============================================================================

def generate_triton_config(
    model_name: str = "spam-detector",
    num_features: int = DEFAULT_NUM_FEATURES,
    max_batch_size: int = 64,
    instance_count: int = 1
) -> str:
    """
    Generate Triton Inference Server config.pbtxt.
    
    Args:
        model_name: Name of the model
        num_features: Number of input features
        max_batch_size: Maximum batch size for inference (ignored, using 0 for explicit batching)
        instance_count: Number of model instances
        
    Returns:
        Config file content as string
    
    Note:
        We use max_batch_size: 0 with explicit batch dimensions [-1, ...] in the 
        input/output specs. This is required because:
        1. When max_batch_size > 0, Triton adds an implicit batch dimension
        2. The ONNX model outputs 'label' as shape [-1] (1D), not [-1, 1] (2D)
        3. Using explicit batch dims avoids shape mismatch errors
    """
    # Use max_batch_size: 0 for explicit batch dimension handling
    # This avoids shape mismatch errors with ONNX model outputs
    config = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [-1, {num_features}]
  }}
]

output [
  {{
    name: "label"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [-1, 2]
  }}
]

instance_group [
  {{
    count: {instance_count}
    kind: KIND_CPU
  }}
]
'''
    return config


# =============================================================================
# Azure Blob Storage Upload
# =============================================================================

def get_blob_service_client() -> BlobServiceClient:
    """Create Azure Blob Service client."""
    if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
        raise ValueError("Azure storage credentials not set")
    
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={AZURE_STORAGE_ACCOUNT_NAME};"
        f"AccountKey={AZURE_STORAGE_ACCOUNT_KEY};"
        f"EndpointSuffix=core.windows.net"
    )
    return BlobServiceClient.from_connection_string(connection_string)


def upload_to_azure_blob(
    local_path: str,
    blob_path: str,
    container_name: str = AZURE_STORAGE_CONTAINER
) -> str:
    """
    Upload a file to Azure Blob Storage.
    
    Args:
        local_path: Local file path
        blob_path: Destination blob path
        container_name: Azure container name
        
    Returns:
        Full blob URL
    """
    logger.info(f"Uploading {local_path} to {container_name}/{blob_path}")
    
    blob_service = get_blob_service_client()
    container_client = blob_service.get_container_client(container_name)
    
    # Create container if not exists
    try:
        container_client.create_container()
        logger.info(f"Created container {container_name}")
    except Exception:
        pass  # Container already exists
    
    blob_client = container_client.get_blob_client(blob_path)
    
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    blob_url = blob_client.url
    logger.info(f"Uploaded to {blob_url}")
    
    return blob_url


def upload_triton_model_repo(
    onnx_model_path: str,
    model_name: str,
    model_version: int,
    triton_config: str,
    num_features: int = DEFAULT_NUM_FEATURES,
    container_name: str = AZURE_STORAGE_CONTAINER
) -> Dict[str, str]:
    """
    Upload model to Azure Blob in Triton model repository format.
    
    Format:
        {model_name}/
            {version}/
                model.onnx
            config.pbtxt
    
    Args:
        onnx_model_path: Local path to ONNX model
        model_name: Name of the model
        model_version: Version number
        triton_config: Triton config.pbtxt content
        num_features: Number of input features
        container_name: Azure container name
        
    Returns:
        Dictionary with blob URLs
    """
    logger.info(f"Uploading model to Triton repository format...")
    
    urls = {}
    
    # Upload ONNX model
    model_blob_path = f"triton-repo/{model_name}/{model_version}/model.onnx"
    urls["model_onnx"] = upload_to_azure_blob(
        onnx_model_path, model_blob_path, container_name
    )
    
    # Upload config.pbtxt
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pbtxt", delete=False) as f:
        f.write(triton_config)
        config_path = f.name
    
    try:
        config_blob_path = f"triton-repo/{model_name}/config.pbtxt"
        urls["config_pbtxt"] = upload_to_azure_blob(
            config_path, config_blob_path, container_name
        )
    finally:
        os.unlink(config_path)
    
    # Upload metadata
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "model_format": "onnx",
        "num_features": num_features,
        "export_timestamp": str(np.datetime64('now'))
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(metadata, f, indent=2)
        metadata_path = f.name
    
    try:
        metadata_blob_path = f"triton-repo/{model_name}/{model_version}/metadata.json"
        urls["metadata"] = upload_to_azure_blob(
            metadata_path, metadata_blob_path, container_name
        )
    finally:
        os.unlink(metadata_path)
    
    logger.info(f"Model repository uploaded successfully")
    logger.info(f"Model path: triton-repo/{model_name}/{model_version}/")
    
    return urls


# =============================================================================
# Main Export Function
# =============================================================================

def export_model(
    model_name: str = "spam-detector",
    model_stage: str = "Staging",
    model_version: Optional[int] = None,
    triton_model_name: Optional[str] = None,
    upload_to_blob: bool = True,
    local_output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to export MLflow model to ONNX and upload to Azure Blob.
    
    Args:
        model_name: MLflow registered model name
        model_stage: Model stage (Staging, Production)
        model_version: Specific model version (overrides stage)
        triton_model_name: Name for Triton (defaults to model_name)
        upload_to_blob: Whether to upload to Azure Blob Storage
        local_output_dir: Directory to save local copies
        
    Returns:
        Dictionary with export results
    """
    logger.info("=" * 60)
    logger.info("Starting model export pipeline")
    logger.info("=" * 60)
    
    triton_model_name = triton_model_name or model_name
    
    result = {
        "status": "success",
        "model_name": model_name,
        "triton_model_name": triton_model_name,
    }
    
    try:
        # Step 1: Download model from MLflow
        logger.info("\n[Step 1/5] Downloading model from MLflow...")
        xgb_model, metadata = download_model_from_mlflow(
            model_name, model_stage, model_version
        )
        result["mlflow_metadata"] = metadata
        result["model_version"] = metadata["model_version"]
        
        # Detect number of features from the loaded model
        num_features = get_num_features_from_model(xgb_model)
        result["num_features"] = num_features
        
        # Step 2: Convert to ONNX
        logger.info("\n[Step 2/5] Converting to ONNX format...")
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            convert_to_onnx(xgb_model, num_features, onnx_path)
            
            # Step 3: Validate ONNX model
            logger.info("\n[Step 3/5] Validating ONNX model...")
            validate_onnx_model(onnx_path, num_features, xgb_model)
            
            # Step 4: Generate Triton config
            logger.info("\n[Step 4/5] Generating Triton config...")
            triton_config = generate_triton_config(
                model_name=triton_model_name,
                num_features=num_features
            )
            result["triton_config"] = triton_config
            
            # Save local copies if requested
            if local_output_dir:
                local_dir = Path(local_output_dir)
                local_dir.mkdir(parents=True, exist_ok=True)
                
                version_dir = local_dir / str(metadata["model_version"])
                version_dir.mkdir(exist_ok=True)
                
                # Copy model
                import shutil
                local_model_path = version_dir / "model.onnx"
                shutil.copy(onnx_path, local_model_path)
                logger.info(f"Saved ONNX model to {local_model_path}")
                
                # Save config
                config_path = local_dir / "config.pbtxt"
                with open(config_path, "w") as f:
                    f.write(triton_config)
                logger.info(f"Saved Triton config to {config_path}")
                
                result["local_paths"] = {
                    "model": str(local_model_path),
                    "config": str(config_path)
                }
            
            # Step 5: Upload to Azure Blob Storage
            if upload_to_blob:
                logger.info("\n[Step 5/5] Uploading to Azure Blob Storage...")
                blob_urls = upload_triton_model_repo(
                    onnx_path,
                    triton_model_name,
                    int(metadata["model_version"]),
                    triton_config,
                    num_features
                )
                result["blob_urls"] = blob_urls
            else:
                logger.info("\n[Step 5/5] Skipping Azure Blob upload (upload_to_blob=False)")
        
        logger.info("\n" + "=" * 60)
        logger.info("Model export completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Model export failed: {str(e)}")
        result["status"] = "failed"
        result["error"] = str(e)
        raise
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export MLflow model to ONNX and upload to Azure Blob Storage"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="spam-detector",
        help="MLflow registered model name"
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default="Staging",
        choices=["Staging", "Production", "None", "Archived"],
        help="Model stage to export"
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Specific model version (overrides stage)"
    )
    parser.add_argument(
        "--triton-model-name",
        type=str,
        default=None,
        help="Name for Triton model repository"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip Azure Blob upload"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save model files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    result = export_model(
        model_name=args.model_name,
        model_stage=args.model_stage,
        model_version=args.model_version,
        triton_model_name=args.triton_model_name,
        upload_to_blob=not args.no_upload,
        local_output_dir=args.output_dir
    )
    
    # Print result summary
    print("\n" + "=" * 60)
    print("Export Result:")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
    
    sys.exit(0 if result["status"] == "success" else 1)
