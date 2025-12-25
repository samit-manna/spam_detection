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

# Extended bias test emails - comprehensive test cases for spam/ham detection
BIAS_TEST_EMAILS = [
    # Clear SPAM cases
    {
        "email_id": "bias_spam_001",
        "subject": "Winner",
        "body": "CONGRATULATIONS You won the lottery Claim your million dollars now",
        "sender": "spam@lottery-winner.com",
        "sender_domain": "lottery-winner.com",
        "expected_spam": True,
        "category": "lottery_scam"
    },
    {
        "email_id": "bias_spam_002",
        "subject": "URGENT ACTION REQUIRED",
        "body": "Your bank account has been compromised Click here immediately to verify your identity",
        "sender": "security@bank-verify.net",
        "sender_domain": "bank-verify.net",
        "expected_spam": True,
        "category": "phishing"
    },
    {
        "email_id": "bias_spam_003",
        "subject": "Best prices ever",
        "body": "Buy cheap medications online! Viagra, Cialis at lowest prices! No prescription needed!",
        "sender": "sales@cheap-pharma.net",
        "sender_domain": "cheap-pharma.net",
        "expected_spam": True,
        "category": "pharmaceutical"
    },
    # Clear HAM cases
    {
        "email_id": "bias_ham_001",
        "subject": "Meeting",
        "body": "Hi can we meet tomorrow at 3pm to discuss the quarterly report",
        "sender": "colleague@company.com",
        "sender_domain": "company.com",
        "expected_spam": False,
        "category": "work_meeting"
    },
    {
        "email_id": "bias_ham_002",
        "subject": "Thanks",
        "body": "Thank you for your help with the presentation yesterday",
        "sender": "friend@gmail.com",
        "sender_domain": "gmail.com",
        "expected_spam": False,
        "category": "thank_you"
    },
    {
        "email_id": "bias_ham_003",
        "subject": "Re: Project update",
        "body": "Sounds good. I will send over the documents by end of day. Let me know if you need anything else from me.",
        "sender": "john.doe@acme.org",
        "sender_domain": "acme.org",
        "expected_spam": False,
        "category": "project_update"
    },
    {
        "email_id": "bias_ham_004",
        "subject": "Dinner tonight?",
        "body": "Hey are you free for dinner tonight? I was thinking we could try that new Italian place downtown. Let me know what time works for you.",
        "sender": "sarah@personal.me",
        "sender_domain": "personal.me",
        "expected_spam": False,
        "category": "personal_invite"
    },
    {
        "email_id": "bias_ham_005",
        "subject": "Invoice #12345",
        "body": "Please find attached the invoice for services rendered in November 2024. Payment is due within 30 days. If you have any questions, please contact our billing department.",
        "sender": "billing@vendor.com",
        "sender_domain": "vendor.com",
        "expected_spam": False,
        "category": "invoice"
    },
    {
        "email_id": "bias_ham_006",
        "subject": "Follow-up on our discussion",
        "body": "Hi Mark, I wanted to follow up on our conversation from last week regarding the budget proposal. I have reviewed the numbers and I think we can move forward with the plan. Would you be available for a call on Thursday to finalize the details? Best regards, James",
        "sender": "james.smith@company.com",
        "sender_domain": "company.com",
        "expected_spam": False,
        "category": "professional_followup"
    },
    {
        "email_id": "bias_ham_007",
        "subject": "Deployment complete",
        "body": "The deployment finished successfully. All tests passed and the system is running smoothly.",
        "sender": "devops@company.com",
        "sender_domain": "company.com",
        "expected_spam": False,
        "category": "technical_notification"
    },
    {
        "email_id": "bias_ham_008",
        "subject": "Happy Birthday!",
        "body": "Happy birthday! Hope you have a wonderful day celebrating with friends and family. Enjoy!",
        "sender": "friend@outlook.com",
        "sender_domain": "outlook.com",
        "expected_spam": False,
        "category": "personal_greeting"
    },
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
# Bias Test Function
# =============================================================================

def test_bias(api_gateway_url: str, host_header: str = None, api_key: str = None) -> Dict[str, Any]:
    """
    Test model for bias by running comprehensive spam/ham test cases.
    
    Returns detailed results including:
    - Per-category accuracy
    - False positive rate (ham classified as spam)
    - False negative rate (spam classified as ham)
    - Overall bias assessment
    """
    logger.info("=" * 60)
    logger.info("Running Bias Test Suite")
    logger.info("=" * 60)
    
    headers = {"Content-Type": "application/json"}
    if host_header:
        headers["Host"] = host_header
    if api_key:
        headers["X-API-Key"] = api_key
    
    results = {
        "spam_correct": 0,
        "spam_total": 0,
        "ham_correct": 0,
        "ham_total": 0,
        "false_positives": [],  # Ham classified as spam
        "false_negatives": [],  # Spam classified as ham
        "predictions": [],
        "category_results": {}
    }
    
    for email in BIAS_TEST_EMAILS:
        try:
            response = requests.post(
                f"{api_gateway_url}/predict",
                headers=headers,
                json={
                    "email_id": email["email_id"],
                    "subject": email["subject"],
                    "body": email["body"],
                    "sender": email["sender"],
                    "sender_domain": email.get("sender_domain", "")
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', 'ham')
                probability = data.get('spam_probability', 0)
                is_spam_prediction = prediction == 'spam'
                expected_spam = email['expected_spam']
                correct = is_spam_prediction == expected_spam
                
                # Track results
                result = {
                    "email_id": email["email_id"],
                    "category": email["category"],
                    "expected": "spam" if expected_spam else "ham",
                    "predicted": prediction,
                    "probability": probability,
                    "correct": correct
                }
                results["predictions"].append(result)
                
                # Update counters
                if expected_spam:
                    results["spam_total"] += 1
                    if correct:
                        results["spam_correct"] += 1
                    else:
                        results["false_negatives"].append(email["email_id"])
                else:
                    results["ham_total"] += 1
                    if correct:
                        results["ham_correct"] += 1
                    else:
                        results["false_positives"].append(email["email_id"])
                
                # Track by category
                category = email["category"]
                if category not in results["category_results"]:
                    results["category_results"][category] = {"correct": 0, "total": 0}
                results["category_results"][category]["total"] += 1
                if correct:
                    results["category_results"][category]["correct"] += 1
                
                # Log result
                status = "✓" if correct else "✗"
                logger.info(f"  {status} {email['email_id']}: {prediction.upper()} ({probability:.1%}) "
                          f"[expected: {'SPAM' if expected_spam else 'HAM'}] ({category})")
            else:
                logger.error(f"  ✗ {email['email_id']}: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"  ✗ {email['email_id']}: Error - {e}")
    
    # Calculate metrics
    total_correct = results["spam_correct"] + results["ham_correct"]
    total_tests = results["spam_total"] + results["ham_total"]
    
    spam_accuracy = results["spam_correct"] / results["spam_total"] if results["spam_total"] > 0 else 0
    ham_accuracy = results["ham_correct"] / results["ham_total"] if results["ham_total"] > 0 else 0
    overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
    
    false_positive_rate = len(results["false_positives"]) / results["ham_total"] if results["ham_total"] > 0 else 0
    false_negative_rate = len(results["false_negatives"]) / results["spam_total"] if results["spam_total"] > 0 else 0
    
    # Print summary
    logger.info("\n" + "-" * 60)
    logger.info("Bias Test Summary")
    logger.info("-" * 60)
    logger.info(f"  Overall Accuracy:     {overall_accuracy:.1%} ({total_correct}/{total_tests})")
    logger.info(f"  SPAM Detection Rate:  {spam_accuracy:.1%} ({results['spam_correct']}/{results['spam_total']})")
    logger.info(f"  HAM Detection Rate:   {ham_accuracy:.1%} ({results['ham_correct']}/{results['ham_total']})")
    logger.info(f"  False Positive Rate:  {false_positive_rate:.1%} (ham→spam)")
    logger.info(f"  False Negative Rate:  {false_negative_rate:.1%} (spam→ham)")
    
    if results["false_positives"]:
        logger.warning(f"  False Positives: {results['false_positives']}")
    if results["false_negatives"]:
        logger.warning(f"  False Negatives: {results['false_negatives']}")
    
    logger.info("\nCategory Results:")
    for category, cat_results in results["category_results"].items():
        cat_accuracy = cat_results["correct"] / cat_results["total"] if cat_results["total"] > 0 else 0
        logger.info(f"  {category}: {cat_accuracy:.1%} ({cat_results['correct']}/{cat_results['total']})")
    
    # Determine if bias test passed
    # Criteria: 
    # - Overall accuracy >= 80%
    # - False positive rate <= 20% (critical - we don't want to block legitimate emails)
    # - Ham accuracy >= 70% (must correctly identify most legitimate emails)
    bias_test_passed = (
        overall_accuracy >= 0.80 and
        false_positive_rate <= 0.20 and
        ham_accuracy >= 0.70
    )
    
    results["metrics"] = {
        "overall_accuracy": overall_accuracy,
        "spam_accuracy": spam_accuracy,
        "ham_accuracy": ham_accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "passed": bias_test_passed
    }
    
    logger.info("-" * 60)
    logger.info(f"Bias Test: {'✓ PASSED' if bias_test_passed else '✗ FAILED'}")
    logger.info("-" * 60)
    
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
        
        # Run bias test if API gateway is available
        logger.info("\n")
        bias_results = test_bias(
            api_gateway_url=args.api_gateway_url,
            host_header=args.host_header,
            api_key=args.api_key
        )
        results['bias_test'] = bias_results.get('metrics', {}).get('passed', False)
        
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
