"""
Feast Entity Definitions for Spam Detection

Entities define the primary keys for feature lookups.
"""

from feast import Entity, ValueType

# Email entity - primary key for per-email features
email = Entity(
    name="email",
    description="Unique identifier for an email message",
    join_keys=["email_id"],
    value_type=ValueType.STRING,
)

# Sender domain entity - for aggregated sender features
sender_domain = Entity(
    name="sender_domain",
    description="Email sender domain (e.g., gmail.com, company.com)",
    join_keys=["sender_domain"],
    value_type=ValueType.STRING,
)
