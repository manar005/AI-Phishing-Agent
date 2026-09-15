# Explicit feature lists for the initial phishing models. No training happens here.

TARGET_COLUMN = "label"

# MeAJOR -----------------------------------------------------------------

MEAJOR_TEXT_FEATURES = ["subject", "body"]

MEAJOR_STRUCTURAL_FEATURES = [
    "content_types",
    "url_count",
    "url_length_max",
    "url_length_avg",
    "url_subdom_max",
    "url_subdom_avg",
    "attachment_count",
    "has_attachments",
]

# Identities (sender/receiver and domains) are held out so the model learns
# content and structure instead of memorizing specific people or mailboxes.
# `source` is excluded because corpus origin (trec5/trec6/trec7) can leak the label.
# Raw `urls` are excluded from the initial model; URL *statistics* are kept.
MEAJOR_EXCLUDED_FEATURES = [
    "sender",
    "sender_domain",
    "receiver",
    "receiver_domain",
    "date",
    "source",
    "urls",
]

MEAJOR_MODEL_FEATURES = MEAJOR_TEXT_FEATURES + MEAJOR_STRUCTURAL_FEATURES

# Kaggle -----------------------------------------------------------------

KAGGLE_TEXT_FEATURES = ["subject", "body_plain"]

KAGGLE_SECURITY_FEATURES = [
    "num_received_headers",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "has_attachments",
    "has_html",
    "num_urls",
    "num_emails_in_body",
    "num_phone_numbers",
    "contains_tracking_token",
]

# Exact addresses, domains, IPs, timestamps, and message IDs are excluded so
# the model generalizes beyond this dataset's identities and collection artifacts.
# `x_spam_score` is excluded because it is an existing filter score and would leak
# the label rather than teaching the model to detect phishing itself.
KAGGLE_EXCLUDED_FEATURES = [
    "from_address",
    "from_domain",
    "reply_to",
    "to_addresses",
    "cc_addresses",
    "date",
    "hour_of_day",
    "message_id",
    "in_reply_to",
    "received_origin_ip",
    "user_agent",
    "x_spam_score",
    "list_unsubscribe",
    "body_html",
    "raw_text",
]

KAGGLE_MODEL_FEATURES = KAGGLE_TEXT_FEATURES + KAGGLE_SECURITY_FEATURES
