"""
Create test data for batch prediction.

This script generates sample email data and uploads it to Azure Blob Storage
for testing the batch prediction pipeline.

Usage:
    python create_test_data.py --output-path batch/emails_to_predict.parquet --num-samples 100

Environment variables:
    AZURE_STORAGE_ACCOUNT_NAME: Azure storage account name
    AZURE_STORAGE_ACCOUNT_KEY: Azure storage account key
"""

import os
import argparse
import random
from datetime import datetime, timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from adlfs import AzureBlobFileSystem

# Azure config
AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "datasets")

# Sample data for generating test emails
SPAM_SUBJECTS = [
    "CONGRATULATIONS! You've WON $1,000,000!!!",
    "URGENT: Your account has been compromised!",
    "FREE V1AGRA - 90% OFF!!!",
    "Make $5000/week from home - GUARANTEED!",
    "Urgent Business Proposal - $10,000,000 Transfer",
    "Your PayPal account is limited - Act Now!",
    "Hot singles in your area want to meet!",
    "Limited time offer - FREE iPhone 15!",
    "You've been selected for a cash prize!",
    "WORK FROM HOME - Easy money!!!",
]

SPAM_BODIES = [
    "Dear Winner, You have been selected as the winner of our MEGA LOTTERY! Click here immediately to claim your prize. Send us your bank details to receive your winnings. Act NOW!",
    "We detected suspicious activity on your account. Your account will be suspended unless you verify your identity within 24 hours. Click here: http://fake-bank.com/verify",
    "Buy cheap medications online! No prescription needed! V1agra, C1alis at unbeatable prices. Order now and get free shipping! Click here: http://cheap-pills.xyz",
    "Make money from home! No experience needed! Earn $5000 per week working just 2 hours a day! Send $50 processing fee to get started!",
    "Dear Sir/Madam, I am Prince Abubakar. I need your help to transfer $10,000,000. You will receive 40% commission. Send your bank details immediately.",
    "ALERT: Your PayPal account has been limited. We need you to confirm your identity. Click here to unlock your account: http://paypa1-verify.com",
    "Hot singles are waiting for you! Sign up now for FREE and meet attractive people in your area. No credit card required! Visit: http://dating-scam.com",
    "Congratulations! You've been selected to receive a FREE iPhone 15! Just pay shipping ($4.99). Claim here: http://free-phone-scam.net",
    "You've won a $1000 gift card! This is NOT a joke! Click to claim before it expires: http://gift-card-scam.com",
    "URGENT: Work from home opportunity! Make easy money with our proven system! No skills required! Start earning TODAY!",
]

HAM_SUBJECTS = [
    "Team meeting tomorrow at 2pm",
    "Re: Project update - Q4 review",
    "Lunch on Saturday?",
    "Your order has been shipped",
    "Weekly newsletter - Tech Digest",
    "Invoice #12345 attached",
    "Quick question about the report",
    "Happy Birthday!",
    "Meeting notes from yesterday",
    "Password reset request",
]

HAM_BODIES = [
    "Hi team, Just a reminder that we have our weekly sync meeting tomorrow at 2pm in Conference Room B. Please come prepared with your status updates. Thanks!",
    "Hi John, Here's the Q4 project update as discussed. The metrics look good - we're on track to meet our targets. Let me know if you have questions.",
    "Hey! Saturday at noon works great for me. How about we try that new Italian place downtown? I heard they have amazing pasta. See you then!",
    "Thank you for your order! Order #12345 has been shipped. Items: Wireless Headphones ($79.99). Expected delivery: Jan 20, 2024. Track at our website.",
    "Hello subscriber, Here's your weekly tech digest. Top stories: 1. New AI developments. 2. Cloud computing trends. 3. Cybersecurity best practices.",
    "Please find attached invoice #12345 for services rendered in December. Payment is due within 30 days. Contact us if you have any questions.",
    "Hi Sarah, Quick question - did you get a chance to review the quarterly report? I need to submit it by Friday. Thanks!",
    "Happy Birthday! Wishing you a wonderful day filled with joy and happiness. Hope this year brings you everything you've been hoping for!",
    "Hi all, Here are the notes from yesterday's meeting. Action items: 1. John - update docs. 2. Sarah - review PRs. 3. Mike - deploy staging.",
    "We received a request to reset your password. If you made this request, click the link below. If not, please ignore this email.",
]

SPAM_SENDERS = [
    "winner@lottery-scam.com",
    "security@fake-bank.net",
    "deals@pharmacy-spam.xyz",
    "jobs@work-from-home.biz",
    "prince@nigeria-royal.com",
    "alert@paypa1-fake.com",
    "singles@dating-scam.org",
    "promo@free-iphone.net",
    "winner@gift-card.com",
    "money@easy-cash.biz",
]

HAM_SENDERS = [
    "manager@company.com",
    "john.smith@work.com",
    "friend@gmail.com",
    "orders@amazon.com",
    "newsletter@techdigest.com",
    "billing@vendor.com",
    "sarah@team.com",
    "family@email.com",
    "notes@company.com",
    "noreply@service.com",
]


def generate_random_date(days_back: int = 30) -> str:
    """Generate a random datetime string within the last N days."""
    base = datetime.now()
    random_days = random.randint(0, days_back)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    dt = base - timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    return dt.isoformat() + "Z"


def generate_test_emails(num_samples: int, spam_ratio: float = 0.5) -> pd.DataFrame:
    """
    Generate test email data.
    
    Args:
        num_samples: Number of emails to generate
        spam_ratio: Proportion of spam emails (0.0 to 1.0)
    
    Returns:
        DataFrame with email data
    """
    emails = []
    num_spam = int(num_samples * spam_ratio)
    num_ham = num_samples - num_spam
    
    # Generate spam emails
    for i in range(num_spam):
        emails.append({
            "message_id": f"spam_{i:04d}@test.com",
            "subject": random.choice(SPAM_SUBJECTS),
            "sender_name": "Spammer",
            "sender_email": random.choice(SPAM_SENDERS),
            "date": generate_random_date(),
            "body_text": random.choice(SPAM_BODIES),
            "body_html": f"<html><body>{random.choice(SPAM_BODIES)}</body></html>" if random.random() > 0.3 else None,
            "x_mailer": None,
            "received_hop_count": random.randint(3, 10),
            # Feast features (will be enriched, but provide defaults)
            "email_count": 0,
            "spam_count": 0,
            "ham_count": 0,
            "spam_ratio": 0.5,
            # Ground truth for validation (not used in prediction)
            "_expected_label": "spam",
        })
    
    # Generate ham emails
    for i in range(num_ham):
        emails.append({
            "message_id": f"ham_{i:04d}@test.com",
            "subject": random.choice(HAM_SUBJECTS),
            "sender_name": "Legitimate Sender",
            "sender_email": random.choice(HAM_SENDERS),
            "date": generate_random_date(),
            "body_text": random.choice(HAM_BODIES),
            "body_html": None,
            "x_mailer": "Microsoft Outlook" if random.random() > 0.5 else None,
            "received_hop_count": random.randint(1, 3),
            # Feast features (will be enriched, but provide defaults)
            "email_count": random.randint(10, 100),
            "spam_count": random.randint(0, 5),
            "ham_count": random.randint(5, 95),
            "spam_ratio": random.uniform(0.0, 0.2),
            # Ground truth for validation (not used in prediction)
            "_expected_label": "ham",
        })
    
    # Shuffle
    random.shuffle(emails)
    
    return pd.DataFrame(emails)


def upload_to_azure(df: pd.DataFrame, output_path: str):
    """Upload DataFrame to Azure Blob Storage as parquet."""
    if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
        raise ValueError("Azure storage credentials not set. Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY")
    
    fs = AzureBlobFileSystem(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        account_key=AZURE_STORAGE_ACCOUNT_KEY
    )
    
    blob_path = f"{AZURE_STORAGE_CONTAINER}/{output_path}"
    print(f"Uploading to: {blob_path}")
    
    # Convert to PyArrow table and write
    table = pa.Table.from_pandas(df)
    
    with fs.open(blob_path, "wb") as f:
        pq.write_table(table, f)
    
    print(f"✓ Uploaded {len(df)} emails to {blob_path}")


def save_local(df: pd.DataFrame, output_path: str):
    """Save DataFrame locally as parquet."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✓ Saved {len(df)} emails to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for batch prediction")
    parser.add_argument(
        "--output-path",
        type=str,
        default="batch/emails_to_predict.parquet",
        help="Output path in Azure Blob (relative to container)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test emails to generate"
    )
    parser.add_argument(
        "--spam-ratio",
        type=float,
        default=0.5,
        help="Proportion of spam emails (0.0 to 1.0)"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Save locally instead of uploading to Azure"
    )
    args = parser.parse_args()
    
    print(f"Generating {args.num_samples} test emails ({args.spam_ratio*100:.0f}% spam)...")
    df = generate_test_emails(args.num_samples, args.spam_ratio)
    
    print(f"\nDataset summary:")
    print(f"  Total emails: {len(df)}")
    print(f"  Spam: {(df['_expected_label'] == 'spam').sum()}")
    print(f"  Ham: {(df['_expected_label'] == 'ham').sum()}")
    print(f"  Columns: {list(df.columns)}")
    
    if args.local_only:
        save_local(df, args.output_path)
    else:
        upload_to_azure(df, args.output_path)
    
    print("\n✓ Test data created successfully!")
    print(f"Run batch prediction with: make batch-predict")


if __name__ == "__main__":
    main()
