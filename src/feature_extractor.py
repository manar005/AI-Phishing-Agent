# Convert ParsedEmail + SecurityEvidence into model features.
# Transformation only: no phishing score, classification, or identity fields.

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.email_parser import ParsedEmail
from src.security_checks import AUTH_RESULT_VALUES, SecurityEvidence

UNKNOWN = "unknown"


@dataclass
class EmailFeatures:
    subject: str
    body_plain: str
    header_spf_result: str
    header_dkim_result: str
    header_dmarc_result: str
    has_reply_to: bool
    from_reply_to_address_mismatch: str
    from_reply_to_domain_mismatch: str
    from_return_path_domain_mismatch: str
    num_received_headers: int
    url_count: int
    unique_url_count: int
    url_length_max: float | None
    ip_url_count: int
    punycode_url_count: int
    attachment_count: int
    has_risky_attachment: bool
    has_html: bool

    def to_dict(self) -> dict:
        return asdict(self)


def extract_features(parsed: ParsedEmail, evidence: SecurityEvidence) -> EmailFeatures:
    """Map parser output and security evidence into detector features."""
    return EmailFeatures(
        subject=parsed.subject or "",
        body_plain=parsed.body_plain or "",
        header_spf_result=_auth_category(evidence.header_spf_result),
        header_dkim_result=_auth_category(evidence.header_dkim_result),
        header_dmarc_result=_auth_category(evidence.header_dmarc_result),
        has_reply_to=evidence.has_reply_to,
        from_reply_to_address_mismatch=_bool_category(evidence.from_reply_to_address_mismatch),
        from_reply_to_domain_mismatch=_bool_category(evidence.from_reply_to_domain_mismatch),
        from_return_path_domain_mismatch=_bool_category(evidence.from_return_path_domain_mismatch),
        num_received_headers=evidence.num_received_headers,
        url_count=evidence.url_count,
        unique_url_count=evidence.unique_url_count,
        url_length_max=evidence.url_length_max,
        ip_url_count=evidence.ip_url_count,
        punycode_url_count=evidence.punycode_url_count,
        attachment_count=evidence.attachment_count,
        has_risky_attachment=_has_risky_attachment(evidence),
        has_html=evidence.has_html,
    )


def _auth_category(value: str | None) -> str:
    if value in AUTH_RESULT_VALUES:
        return value
    return UNKNOWN


def _bool_category(value: bool | None) -> str:
    if value is None:
        return UNKNOWN
    return "true" if value else "false"


def _has_risky_attachment(evidence: SecurityEvidence) -> bool:
    return bool(evidence.risky_attachment_extensions or evidence.risky_attachment_mime_types)
