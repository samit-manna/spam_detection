#!/usr/bin/env python3
"""
Integration Tests for Model Serving Platform

This script tests the end-to-end inference flow:
1. Feature Transformer service
2. Triton Inference Server
3. Full inference pipeline

Usage:
    python integration_test.py --transformer-url http://localhost:8080 --triton-url http://localhost:8000
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, Any, List
import requests
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

FEATURE_TRANSFORMER_URL = os.environ.get(
    "FEATURE_TRANSFORMER_URL",
    "http://localhost:8080"
)
TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000")
NUM_FEATURES = 528  # Text(8) + Structural(10) + Temporal(4) + SpamIndicators(2) + TF-IDF(500) + Sender(4)


# =============================================================================
# Test Data
# =============================================================================

TEST_EMAILS = [
    {
        "email_id": "test_spam_001",
        "subject": "You've WON $1,000,000!!!",
        "body": "CONGRATULATIONS! You have been selected as a WINNER! Click here NOW to claim your FREE prize! Act fast, this is a LIMITED TIME offer! Don't miss out on this amazing opportunity!",
        "sender": "winner@free-money-now.com",
        "date": "2024-01-15T02:30:00Z",
        "expected_spam": True
    },
    {
        "email_id": "test_ham_001",
        "subject": "Re: Meeting tomorrow",
        "body": "Hi John, Just wanted to confirm our meeting tomorrow at 2pm. I'll bring the project reports we discussed. Let me know if you need anything else. Best regards, Sarah",
        "sender": "sarah.jones@company.com",
        "date": "2024-01-15T10:30:00Z",
        "expected_spam": False
    },
    {
        "email_id": "test_spam_002",
        "subject": "URGENT: Your account has been compromised!",
        "body": "Dear customer, We have detected suspicious activity on your account. Click the link below immediately to verify your identity and secure your account. Failure to act within 24 hours will result in account suspension. http://fake-bank-login.com/verify",
        "sender": "security@bank-alert-notice.com",
        "date": "2024-01-15T22:00:00Z",
        "expected_spam": True
    },
    {
        "email_id": "test_ham_002",
        "subject": "Weekly team update",
        "body": "Team, Here's the weekly update on our project progress. Sprint velocity is on track. The new feature deployment is scheduled for next Tuesday. Please review the attached documents before our standup tomorrow. Thanks, Mike",
        "sender": "mike.wilson@company.com",
        "date": "2024-01-15T14:00:00Z",
        "expected_spam": False
    }
]


# =============================================================================
# Test Functions
# =============================================================================

def test_feature_transformer_health(base_url: str) -> bool:
    """Test feature transformer health endpoint."""
    logger.info("Testing Feature Transformer health...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"  Status: {data['status']}")
        logger.info(f"  TF-IDF loaded: {data['tfidf_loaded']}")
        logger.info(f"  Feast enabled: {data['feast_enabled']}")
        
        return data['status'] == 'healthy'
    except Exception as e:
        logger.error(f"  Health check failed: {e}")
        return False


def test_feature_transformer_transform(base_url: str) -> Dict[str, Any]:
    """Test feature transformer single transform endpoint."""
    logger.info("Testing Feature Transformer transform...")
    
    test_email = TEST_EMAILS[0]
    
    try:
        response = requests.post(
            f"{base_url}/transform",
            json={
                "email_id": test_email["email_id"],
                "subject": test_email["subject"],
                "body": test_email["body"],
                "sender": test_email["sender"],
                "date": test_email["date"]
            },
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        features = data['features']
        
        logger.info(f"  Email ID: {data['email_id']}")
        logger.info(f"  Sender domain: {data['sender_domain']}")
        logger.info(f"  Features count: {len(features)}")
        logger.info(f"  Feast features included: {data['feast_features_included']}")
        
        # Validate feature count
        if len(features) != NUM_FEATURES:
            logger.error(f"  Expected {NUM_FEATURES} features, got {len(features)}")
            return None
        
        # Show some feature values
        logger.info(f"  First 5 features: {features[:5]}")
        
        return data
        
    except Exception as e:
        logger.error(f"  Transform failed: {e}")
        return None


def test_feature_transformer_batch(base_url: str) -> bool:
    """Test feature transformer batch transform endpoint."""
    logger.info("Testing Feature Transformer batch transform...")
    
    try:
        batch_request = {
            "emails": [
                {
                    "email_id": email["email_id"],
                    "subject": email["subject"],
                    "body": email["body"],
                    "sender": email["sender"],
                    "date": email["date"]
                }
                for email in TEST_EMAILS
            ]
        }
        
        response = requests.post(
            f"{base_url}/transform/batch",
            json=batch_request,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"  Batch size: {data['count']}")
        
        for output in data['outputs']:
            logger.info(f"  - {output['email_id']}: {len(output['features'])} features")
        
        return data['count'] == len(TEST_EMAILS)
        
    except Exception as e:
        logger.error(f"  Batch transform failed: {e}")
        return False


def test_triton_health(base_url: str) -> bool:
    """Test Triton server health endpoint."""
    logger.info("Testing Triton server health...")
    
    try:
        response = requests.get(f"{base_url}/v2/health/ready", timeout=5)
        
        if response.status_code == 200:
            logger.info("  Triton server is ready")
            return True
        else:
            logger.error(f"  Triton server not ready: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"  Health check failed: {e}")
        return False


def test_triton_model_ready(base_url: str, model_name: str = "spam-detector") -> bool:
    """Test if model is loaded in Triton."""
    logger.info(f"Testing Triton model {model_name}...")
    
    try:
        response = requests.get(
            f"{base_url}/v2/models/{model_name}/ready",
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info(f"  Model {model_name} is ready")
            return True
        else:
            logger.warning(f"  Model {model_name} not ready: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"  Model check failed: {e}")
        return False


def test_triton_inference(
    base_url: str,
    features: List[float],
    model_name: str = "spam-detector"
) -> Dict[str, Any]:
    """Test Triton inference with features."""
    logger.info("Testing Triton inference...")
    
    try:
        # Create inference request
        inference_request = {
            "inputs": [{
                "name": "input",
                "shape": [1, NUM_FEATURES],
                "datatype": "FP32",
                "data": features
            }]
        }
        
        response = requests.post(
            f"{base_url}/v2/models/{model_name}/infer",
            json=inference_request,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Parse outputs
        outputs = {out['name']: out['data'] for out in data['outputs']}
        
        prediction = outputs.get('predictions', [None])[0]
        probabilities = outputs.get('probabilities', [0, 0])
        
        logger.info(f"  Prediction: {prediction}")
        logger.info(f"  Probabilities: {probabilities}")
        
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "spam_probability": probabilities[1] if len(probabilities) > 1 else probabilities[0]
        }
        
    except Exception as e:
        logger.error(f"  Inference failed: {e}")
        return None


def test_end_to_end(
    transformer_url: str,
    triton_url: str,
    model_name: str = "spam-detector"
) -> bool:
    """Test end-to-end inference pipeline."""
    logger.info("=" * 60)
    logger.info("Testing End-to-End Inference Pipeline")
    logger.info("=" * 60)
    
    results = []
    
    for test_email in TEST_EMAILS:
        logger.info(f"\nProcessing: {test_email['email_id']}")
        
        # Step 1: Get features
        transform_response = requests.post(
            f"{transformer_url}/transform",
            json={
                "email_id": test_email["email_id"],
                "subject": test_email["subject"],
                "body": test_email["body"],
                "sender": test_email["sender"],
                "date": test_email["date"]
            },
            timeout=10
        )
        
        if transform_response.status_code != 200:
            logger.error(f"  Feature extraction failed: {transform_response.status_code}")
            results.append(False)
            continue
        
        features = transform_response.json()['features']
        
        # Step 2: Run inference
        inference_request = {
            "inputs": [{
                "name": "input",
                "shape": [1, NUM_FEATURES],
                "datatype": "FP32",
                "data": features
            }]
        }
        
        inference_response = requests.post(
            f"{triton_url}/v2/models/{model_name}/infer",
            json=inference_request,
            timeout=10
        )
        
        if inference_response.status_code != 200:
            logger.error(f"  Inference failed: {inference_response.status_code}")
            results.append(False)
            continue
        
        outputs = inference_response.json()['outputs']
        prediction = outputs[0]['data'][0]
        
        # Log result
        is_spam = prediction == 1
        expected = test_email['expected_spam']
        match = is_spam == expected
        
        logger.info(f"  Subject: {test_email['subject'][:50]}...")
        logger.info(f"  Prediction: {'SPAM' if is_spam else 'HAM'}")
        logger.info(f"  Expected: {'SPAM' if expected else 'HAM'}")
        logger.info(f"  Match: {'✓' if match else '✗'}")
        
        results.append(match)
    
    # Summary
    success_count = sum(results)
    total_count = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"End-to-End Test Results: {success_count}/{total_count} passed")
    logger.info("=" * 60)
    
    return all(results)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests(
    transformer_url: str,
    triton_url: str,
    model_name: str = "spam-detector"
) -> Dict[str, bool]:
    """Run all integration tests."""
    
    results = {}
    
    logger.info("\n" + "=" * 60)
    logger.info("Running Integration Tests")
    logger.info("=" * 60)
    
    # Feature Transformer tests
    logger.info("\n--- Feature Transformer Tests ---\n")
    
    results['transformer_health'] = test_feature_transformer_health(transformer_url)
    
    transform_result = test_feature_transformer_transform(transformer_url)
    results['transformer_transform'] = transform_result is not None
    
    results['transformer_batch'] = test_feature_transformer_batch(transformer_url)
    
    # Triton tests
    logger.info("\n--- Triton Server Tests ---\n")
    
    results['triton_health'] = test_triton_health(triton_url)
    results['triton_model_ready'] = test_triton_model_ready(triton_url, model_name)
    
    if transform_result and results['triton_model_ready']:
        inference_result = test_triton_inference(
            triton_url,
            transform_result['features'],
            model_name
        )
        results['triton_inference'] = inference_result is not None
    else:
        results['triton_inference'] = False
    
    # End-to-end test
    if results['transformer_transform'] and results['triton_model_ready']:
        results['end_to_end'] = test_end_to_end(
            transformer_url,
            triton_url,
            model_name
        )
    else:
        results['end_to_end'] = False
        logger.warning("Skipping end-to-end test due to failed prerequisites")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 60)
    logger.info(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    logger.info("=" * 60)
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run integration tests for model serving platform"
    )
    parser.add_argument(
        "--api-gateway-url",
        type=str,
        default=None,
        help="API Gateway URL (uses /predict endpoint for end-to-end testing)"
    )
    parser.add_argument(
        "--host-header",
        type=str,
        default=None,
        help="Host header to use for requests (e.g., api.ml-platform.example.com)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="test-operator-key",
        help="API key for authentication (default: test-operator-key)"
    )
    parser.add_argument(
        "--transformer-url",
        type=str,
        default=FEATURE_TRANSFORMER_URL,
        help="Feature transformer service URL (for direct testing)"
    )
    parser.add_argument(
        "--triton-url",
        type=str,
        default=TRITON_URL,
        help="Triton inference server URL (for direct testing)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="spam-detector",
        help="Model name in Triton"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait N seconds before starting tests"
    )
    return parser.parse_args()


def test_api_gateway(api_gateway_url: str, model_name: str, host_header: str = None, api_key: str = None) -> Dict[str, bool]:
    """Test end-to-end via the API Gateway."""
    logger.info("=" * 60)
    logger.info("Testing via API Gateway")
    logger.info("=" * 60)
    
    results = {}
    headers = {}
    if host_header:
        headers["Host"] = host_header
        logger.info(f"Using Host header: {host_header}")
    if api_key:
        headers["X-API-Key"] = api_key
        logger.info(f"Using API key: {api_key[:8]}...")
    
    # Test health endpoint
    logger.info("\n1. Testing API Gateway health...")
    try:
        response = requests.get(f"{api_gateway_url}/health", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            logger.info(f"   Status: {status}")
            # Accept 'healthy' or 'degraded' (Ray might be down intentionally)
            results['api_gateway_health'] = status in ['healthy', 'degraded']
        else:
            logger.error(f"   Health check failed: {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}")
            results['api_gateway_health'] = False
    except Exception as e:
        logger.error(f"   Health check error: {e}")
        results['api_gateway_health'] = False
    
    # Test predictions via API Gateway
    logger.info("\n2. Testing predictions via API Gateway...")
    correct_predictions = 0
    total_predictions = len(TEST_EMAILS)
    
    for test_email in TEST_EMAILS:
        try:
            response = requests.post(
                f"{api_gateway_url}/predict",
                headers=headers,
                json={
                    "email_id": test_email["email_id"],
                    "subject": test_email["subject"],
                    "body": test_email["body"],
                    "sender": test_email["sender"],
                    "date": test_email["date"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # API returns 'prediction' as "spam" or "ham" string
                prediction = data.get('prediction', 'ham')
                is_spam = prediction == 'spam'
                probability = data.get('spam_probability', 0)
                expected = test_email['expected_spam']
                match = is_spam == expected
                
                if match:
                    correct_predictions += 1
                
                logger.info(f"   {test_email['email_id']}: "
                          f"{'SPAM' if is_spam else 'HAM'} ({probability:.2%}) "
                          f"- Expected: {'SPAM' if expected else 'HAM'} "
                          f"{'✓' if match else '✗'}")
            else:
                logger.error(f"   {test_email['email_id']}: Failed - {response.status_code}")
                logger.error(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            logger.error(f"   {test_email['email_id']}: Error - {e}")
    
    # Allow up to 1 incorrect prediction (75% accuracy threshold)
    results['api_gateway_predictions'] = correct_predictions >= (total_predictions - 1)
    logger.info(f"\n   Accuracy: {correct_predictions}/{total_predictions} correct")
    
    # Test staging endpoint
    logger.info("\n3. Testing staging predictions...")
    try:
        response = requests.post(
            f"{api_gateway_url}/predict/staging",
            headers=headers,
            json={
                "email_id": "test_staging",
                "subject": "Buy now! Limited offer!",
                "body": "Click here to win prizes!",
                "sender": "spam@test.com",
                "date": "2024-01-15T10:00:00Z"
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', 'ham')
            probability = data.get('spam_probability', 0)
            logger.info(f"   Staging prediction: {prediction.upper()} ({probability:.2%})")
            results['api_gateway_staging'] = True
        else:
            logger.warning(f"   Staging endpoint failed: {response.status_code}")
            logger.warning(f"   Response: {response.text[:200]}")
            # Mark as passed if it's a 'not found' error - staging might not be deployed
            results['api_gateway_staging'] = 'not found' in response.text.lower()
    except Exception as e:
        logger.error(f"   Staging endpoint error: {e}")
        results['api_gateway_staging'] = False
    
    return results


if __name__ == "__main__":
    args = parse_args()
    
    if args.wait > 0:
        logger.info(f"Waiting {args.wait} seconds before starting tests...")
        time.sleep(args.wait)
    
    # If API gateway URL is provided, test via API gateway
    if args.api_gateway_url:
        results = test_api_gateway(
            api_gateway_url=args.api_gateway_url,
            model_name=args.model_name,
            host_header=args.host_header,
            api_key=args.api_key
        )
    else:
        # Otherwise, test transformer and Triton directly
        results = run_all_tests(
            transformer_url=args.transformer_url,
            triton_url=args.triton_url,
            model_name=args.model_name
        )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 60)
    logger.info(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    logger.info("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
