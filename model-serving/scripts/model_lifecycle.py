#!/usr/bin/env python3
"""
Model Lifecycle Management Script

Provides commands for:
- Listing model versions and their stages
- Promoting models between stages (None -> Staging -> Production)
- Rolling back to previous model versions
- Deploying models to KServe after promotion

Usage:
    python model_lifecycle.py list --model-name spam-detector
    python model_lifecycle.py promote --model-name spam-detector --version 2 --stage Staging
    python model_lifecycle.py rollback --model-name spam-detector --stage Staging
    python model_lifecycle.py deploy --model-name spam-detector --stage Staging
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", 
    "http://localhost:5000"
)

VALID_STAGES = ["None", "Staging", "Production", "Archived"]
PROMOTION_ORDER = ["None", "Staging", "Production"]


class MLflowClient:
    """Simple MLflow REST API client."""
    
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri.rstrip("/")
        self.api_base = f"{self.tracking_uri}/api/2.0/mlflow"
    
    def get_registered_model(self, name: str) -> Dict[str, Any]:
        """Get registered model details."""
        response = requests.get(
            f"{self.api_base}/registered-models/get",
            params={"name": name}
        )
        response.raise_for_status()
        return response.json().get("registered_model", {})
    
    def get_model_version(self, name: str, version: str) -> Dict[str, Any]:
        """Get specific model version details."""
        response = requests.get(
            f"{self.api_base}/model-versions/get",
            params={"name": name, "version": version}
        )
        response.raise_for_status()
        return response.json().get("model_version", {})
    
    def search_model_versions(self, name: str) -> List[Dict[str, Any]]:
        """Search all versions of a model."""
        response = requests.get(
            f"{self.api_base}/model-versions/search",
            params={"filter": f"name='{name}'"}
        )
        response.raise_for_status()
        return response.json().get("model_versions", [])
    
    def transition_model_version_stage(
        self, 
        name: str, 
        version: str, 
        stage: str,
        archive_existing: bool = True
    ) -> Dict[str, Any]:
        """Transition a model version to a new stage."""
        response = requests.post(
            f"{self.api_base}/model-versions/transition-stage",
            json={
                "name": name,
                "version": version,
                "stage": stage,
                "archive_existing_versions": archive_existing
            }
        )
        response.raise_for_status()
        return response.json().get("model_version", {})
    
    def set_model_version_tag(
        self, 
        name: str, 
        version: str, 
        key: str, 
        value: str
    ):
        """Set a tag on a model version."""
        response = requests.post(
            f"{self.api_base}/model-versions/set-tag",
            json={
                "name": name,
                "version": version,
                "key": key,
                "value": value
            }
        )
        response.raise_for_status()


def list_models(client: MLflowClient, model_name: str, show_all: bool = False):
    """List all versions of a model with their stages."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        versions = client.search_model_versions(model_name)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch model versions: {e}")
        sys.exit(1)
    
    if not versions:
        print("No model versions found.")
        return
    
    # Sort by version number descending
    versions.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
    
    # Filter if not showing all
    if not show_all:
        versions = [v for v in versions if v.get("current_stage") != "Archived"]
    
    # Print header
    print(f"{'Version':<10} {'Stage':<15} {'Created':<20} {'Tags'}")
    print("-" * 70)
    
    for v in versions:
        version = v.get("version", "?")
        stage = v.get("current_stage", "None")
        
        # Format timestamp
        ts = v.get("creation_timestamp", 0)
        created = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "Unknown"
        
        # Get relevant tags
        tags = v.get("tags", [])
        tag_str = ", ".join([f"{t['key']}={t['value']}" for t in tags[:3]])
        if len(tags) > 3:
            tag_str += "..."
        
        # Highlight current staging/production
        stage_display = stage
        if stage == "Staging":
            stage_display = f"\033[93m{stage}\033[0m"  # Yellow
        elif stage == "Production":
            stage_display = f"\033[92m{stage}\033[0m"  # Green
        elif stage == "Archived":
            stage_display = f"\033[90m{stage}\033[0m"  # Gray
        
        print(f"v{version:<9} {stage_display:<24} {created:<20} {tag_str}")
    
    print()
    
    # Show current active versions
    staging = [v for v in versions if v.get("current_stage") == "Staging"]
    production = [v for v in versions if v.get("current_stage") == "Production"]
    
    staging_str = f"v{staging[0]['version']}" if staging else "None"
    production_str = f"v{production[0]['version']}" if production else "None"
    
    print("Current Active Versions:")
    print(f"  Staging:    {staging_str}")
    print(f"  Production: {production_str}")
    print()


def promote_model(
    client: MLflowClient, 
    model_name: str, 
    version: str, 
    target_stage: str,
    skip_confirmation: bool = False
):
    """Promote a model version to a target stage."""
    if target_stage not in VALID_STAGES:
        logger.error(f"Invalid stage: {target_stage}. Must be one of {VALID_STAGES}")
        sys.exit(1)
    
    # Get current version info
    try:
        model_version = client.get_model_version(model_name, version)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get model version: {e}")
        sys.exit(1)
    
    current_stage = model_version.get("current_stage", "None")
    
    print(f"\n{'='*60}")
    print(f"Model Promotion")
    print(f"{'='*60}")
    print(f"  Model:         {model_name}")
    print(f"  Version:       v{version}")
    print(f"  Current Stage: {current_stage}")
    print(f"  Target Stage:  {target_stage}")
    print()
    
    # Check if this is a valid promotion
    if current_stage == target_stage:
        logger.warning(f"Model is already in {target_stage} stage")
        return
    
    # Confirm promotion
    if not skip_confirmation:
        confirm = input(f"Promote v{version} to {target_stage}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Promotion cancelled.")
            return
    
    # Perform promotion
    try:
        result = client.transition_model_version_stage(
            model_name, 
            version, 
            target_stage,
            archive_existing=True
        )
        
        # Add promotion metadata
        timestamp = datetime.utcnow().isoformat()
        client.set_model_version_tag(model_name, version, "promoted_at", timestamp)
        client.set_model_version_tag(model_name, version, "promoted_from", current_stage)
        
        print(f"\n✓ Successfully promoted v{version} to {target_stage}")
        
        # Show next steps
        print(f"\nNext steps:")
        print(f"  1. Export model:  make export-model MODEL_STAGE={target_stage}")
        print(f"  2. Deploy:        make deploy-{'staging' if target_stage == 'Staging' else 'prod'}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to promote model: {e}")
        sys.exit(1)


def rollback_model(
    client: MLflowClient, 
    model_name: str, 
    stage: str,
    skip_confirmation: bool = False
):
    """Rollback to the previous model version in a stage."""
    if stage not in ["Staging", "Production"]:
        logger.error("Rollback only supported for Staging and Production stages")
        sys.exit(1)
    
    # Get all versions
    try:
        versions = client.search_model_versions(model_name)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch model versions: {e}")
        sys.exit(1)
    
    # Find current version in target stage
    current = [v for v in versions if v.get("current_stage") == stage]
    if not current:
        logger.error(f"No model currently in {stage} stage")
        sys.exit(1)
    
    current_version = current[0]
    
    # Find previous version (archived from this stage, or next lower version)
    archived = [
        v for v in versions 
        if v.get("current_stage") == "Archived"
        and int(v.get("version", 0)) < int(current_version.get("version", 0))
    ]
    
    # Sort by version descending to get most recent archived
    archived.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
    
    if not archived:
        # Try to find any lower version that's not the current one
        lower_versions = [
            v for v in versions
            if int(v.get("version", 0)) < int(current_version.get("version", 0))
            and v.get("current_stage") != stage
        ]
        lower_versions.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
        
        if not lower_versions:
            logger.error(f"No previous version available for rollback")
            sys.exit(1)
        
        rollback_to = lower_versions[0]
    else:
        rollback_to = archived[0]
    
    print(f"\n{'='*60}")
    print(f"Model Rollback")
    print(f"{'='*60}")
    print(f"  Model:           {model_name}")
    print(f"  Stage:           {stage}")
    print(f"  Current Version: v{current_version.get('version')}")
    print(f"  Rollback To:     v{rollback_to.get('version')}")
    print()
    
    # Confirm rollback
    if not skip_confirmation:
        confirm = input(f"Rollback {stage} from v{current_version.get('version')} to v{rollback_to.get('version')}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Rollback cancelled.")
            return
    
    # Perform rollback
    try:
        result = client.transition_model_version_stage(
            model_name,
            rollback_to.get("version"),
            stage,
            archive_existing=True
        )
        
        # Add rollback metadata
        timestamp = datetime.utcnow().isoformat()
        client.set_model_version_tag(
            model_name, 
            rollback_to.get("version"), 
            "rollback_at", 
            timestamp
        )
        client.set_model_version_tag(
            model_name,
            rollback_to.get("version"),
            "rollback_from",
            current_version.get("version")
        )
        
        print(f"\n✓ Successfully rolled back {stage} to v{rollback_to.get('version')}")
        
        # Show next steps
        print(f"\nNext steps:")
        print(f"  1. Export model:  make export-model MODEL_STAGE={stage}")
        print(f"  2. Deploy:        make deploy-{'staging' if stage == 'Staging' else 'prod'}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to rollback model: {e}")
        sys.exit(1)


def deploy_model(
    model_name: str,
    stage: str,
    image_tag: str = "v0.35",
    skip_export: bool = False
):
    """Export and deploy model to KServe."""
    if stage not in ["Staging", "Production"]:
        logger.error("Deploy only supported for Staging and Production stages")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Model Deployment")
    print(f"{'='*60}")
    print(f"  Model:     {model_name}")
    print(f"  Stage:     {stage}")
    print(f"  Image Tag: {image_tag}")
    print()
    
    # Change to model-serving directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    serving_dir = os.path.dirname(script_dir)
    os.chdir(serving_dir)
    
    # Step 1: Export model to ONNX
    if not skip_export:
        print("Step 1: Exporting model to ONNX format...")
        result = subprocess.run(
            ["make", "export-model", f"MODEL_STAGE={stage}", f"IMAGE_TAG={image_tag}"],
            capture_output=False
        )
        if result.returncode != 0:
            logger.error("Model export failed")
            sys.exit(1)
        print("✓ Model exported successfully\n")
    
    # Step 2: Deploy to KServe
    deploy_target = "deploy-staging" if stage == "Staging" else "deploy-prod"
    print(f"Step 2: Deploying to KServe ({stage})...")
    result = subprocess.run(
        ["make", deploy_target],
        capture_output=False
    )
    if result.returncode != 0:
        logger.error("Deployment failed")
        sys.exit(1)
    print(f"✓ Deployed to {stage} successfully\n")
    
    # Step 3: Restart pods to load new model
    print("Step 3: Restarting inference service pods...")
    isvc_name = f"{model_name}-staging" if stage == "Staging" else model_name
    subprocess.run([
        "kubectl", "delete", "pod", "-n", "kserve",
        "-l", f"serving.kserve.io/inferenceservice={isvc_name}"
    ])
    
    # Wait for pod to be ready
    subprocess.run([
        "kubectl", "wait", "--for=condition=Ready",
        "pod", "-l", f"serving.kserve.io/inferenceservice={isvc_name}",
        "-n", "kserve", "--timeout=120s"
    ])
    print(f"✓ Inference service restarted\n")
    
    print(f"{'='*60}")
    print(f"✓ Deployment complete!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Model Lifecycle Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all model versions
  python model_lifecycle.py list --model-name spam-detector

  # Promote version 2 to Staging
  python model_lifecycle.py promote --model-name spam-detector --version 2 --stage Staging

  # Rollback Staging to previous version
  python model_lifecycle.py rollback --model-name spam-detector --stage Staging

  # Export and deploy model
  python model_lifecycle.py deploy --model-name spam-detector --stage Staging
        """
    )
    
    # Parent parser for common arguments (inherited by subparsers)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--mlflow-uri", 
        default=MLFLOW_TRACKING_URI,
        help="MLflow tracking URI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List model versions", parents=[parent_parser])
    list_parser.add_argument("--model-name", required=True, help="Model name")
    list_parser.add_argument("--all", action="store_true", help="Show archived versions")
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to stage", parents=[parent_parser])
    promote_parser.add_argument("--model-name", required=True, help="Model name")
    promote_parser.add_argument("--version", required=True, help="Version to promote")
    promote_parser.add_argument("--stage", required=True, choices=VALID_STAGES, help="Target stage")
    promote_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version", parents=[parent_parser])
    rollback_parser.add_argument("--model-name", required=True, help="Model name")
    rollback_parser.add_argument("--stage", required=True, choices=["Staging", "Production"], help="Stage to rollback")
    rollback_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Export and deploy model", parents=[parent_parser])
    deploy_parser.add_argument("--model-name", required=True, help="Model name")
    deploy_parser.add_argument("--stage", required=True, choices=["Staging", "Production"], help="Stage to deploy")
    deploy_parser.add_argument("--image-tag", default="v0.35", help="Docker image tag")
    deploy_parser.add_argument("--skip-export", action="store_true", help="Skip model export")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize client
    client = MLflowClient(args.mlflow_uri)
    
    # Execute command
    if args.command == "list":
        list_models(client, args.model_name, args.all)
    elif args.command == "promote":
        promote_model(client, args.model_name, args.version, args.stage, args.yes)
    elif args.command == "rollback":
        rollback_model(client, args.model_name, args.stage, args.yes)
    elif args.command == "deploy":
        deploy_model(args.model_name, args.stage, args.image_tag, args.skip_export)


if __name__ == "__main__":
    main()
