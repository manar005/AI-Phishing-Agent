# Synthetic feature-extractor tests. No model inference or phishing labels.

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.email_parser import ParsedEmail
from src.feature_extractor import EmailFeatures, extract_features
from src.security_checks import SecurityEvidence

IDENTITY_LIKE_FIELDS = {
    "from_address",
    "reply_to",
    "reply_to_address",
    "from_domain",
    "reply_to_domain",
    "to_addresses",
    "cc_addresses",
    "date",
    "message_id",
    "received_headers",
    "body_html",
    "attachment_filenames",
    "urls",
    "url_hosts",
    "url_domains",
    "source",
    "score",
    "label",
    "is_phishing",
}


def _sample_parsed(**overrides) -> ParsedEmail:
    parsed = ParsedEmail(
        subject="Meeting notes",
        body_plain="Agenda is here.",
        body_html="<p>Agenda is here.</p>",
        from_address="alice@example.com",
        from_domain="example.com",
        reply_to="alice@example.com",
        to_addresses=["bob@example.com"],
        cc_addresses=["carol@example.com"],
        date="Mon, 14 Sep 2026 10:00:00 +0000",
        message_id="<notes@example.com>",
        received_headers=["from mail.example.com by mx.example.com"],
        content_type="multipart/alternative",
        has_attachments=True,
        attachment_count=1,
        attachment_filenames=["agenda.pdf"],
        attachment_mime_types=["application/pdf"],
    )
    for key, value in overrides.items():
        setattr(parsed, key, value)
    return parsed


def _sample_evidence(**overrides) -> SecurityEvidence:
    evidence = SecurityEvidence(
        has_reply_to=True,
        from_address="alice@example.com",
        reply_to_address="alice@example.com",
        from_domain="example.com",
        reply_to_domain="example.com",
        from_reply_to_address_mismatch=False,
        from_reply_to_domain_mismatch=False,
        header_spf_result="pass",
        header_dkim_result="pass",
        header_dmarc_result="pass",
        num_to_recipients=1,
        num_cc_recipients=1,
        num_received_headers=1,
        has_html=True,
        attachment_count=1,
        attachment_extensions=[".pdf"],
        attachment_mime_types=["application/pdf"],
        url_count=1,
        unique_url_count=1,
        url_length_max=40.0,
        url_length_avg=40.0,
        ip_url_count=0,
        urls=["https://intranet.example.com/agenda"],
        url_hosts=["intranet.example.com"],
    )
    for key, value in overrides.items():
        setattr(evidence, key, value)
    return evidence


class FeatureExtractorTests(unittest.TestCase):
    def test_maps_approved_behavioral_features(self) -> None:
        features = extract_features(_sample_parsed(), _sample_evidence())
        self.assertIsInstance(features, EmailFeatures)
        self.assertEqual(features.subject, "Meeting notes")
        self.assertEqual(features.body_plain, "Agenda is here.")
        self.assertEqual(features.header_spf_result, "pass")
        self.assertEqual(features.header_dkim_result, "pass")
        self.assertEqual(features.header_dmarc_result, "pass")
        self.assertTrue(features.has_reply_to)
        self.assertEqual(features.from_reply_to_address_mismatch, "false")
        self.assertEqual(features.from_reply_to_domain_mismatch, "false")
        self.assertEqual(features.num_to_recipients, 1)
        self.assertEqual(features.num_cc_recipients, 1)
        self.assertEqual(features.num_received_headers, 1)
        self.assertEqual(features.url_count, 1)
        self.assertEqual(features.unique_url_count, 1)
        self.assertEqual(features.url_length_max, 40.0)
        self.assertEqual(features.url_length_avg, 40.0)
        self.assertEqual(features.ip_url_count, 0)
        self.assertEqual(features.attachment_count, 1)
        self.assertFalse(features.has_risky_attachment)
        self.assertTrue(features.has_html)

    def test_missing_auth_and_reply_to_become_unknown_not_fail(self) -> None:
        evidence = _sample_evidence(
            has_reply_to=False,
            reply_to_address=None,
            reply_to_domain=None,
            from_reply_to_address_mismatch=None,
            from_reply_to_domain_mismatch=None,
            header_spf_result=None,
            header_dkim_result=None,
            header_dmarc_result=None,
        )
        features = extract_features(_sample_parsed(reply_to=None), evidence)
        self.assertEqual(features.header_spf_result, "unknown")
        self.assertEqual(features.header_dkim_result, "unknown")
        self.assertEqual(features.header_dmarc_result, "unknown")
        self.assertEqual(features.from_reply_to_address_mismatch, "unknown")
        self.assertEqual(features.from_reply_to_domain_mismatch, "unknown")
        self.assertFalse(features.has_reply_to)
        self.assertNotEqual(features.header_spf_result, "fail")

    def test_header_none_is_preserved_separately_from_unknown(self) -> None:
        features = extract_features(
            _sample_parsed(),
            _sample_evidence(header_dmarc_result="none"),
        )
        self.assertEqual(features.header_dmarc_result, "none")
        self.assertNotEqual(features.header_dmarc_result, "unknown")

    def test_identity_like_fields_are_not_exposed(self) -> None:
        features = extract_features(_sample_parsed(), _sample_evidence())
        exported = features.to_dict()
        for field_name in IDENTITY_LIKE_FIELDS:
            self.assertNotIn(field_name, exported)
        self.assertNotIn("risky_attachment_count", exported)

    def test_url_length_stats_stay_none_when_there_are_no_urls(self) -> None:
        evidence = _sample_evidence(
            url_count=0,
            unique_url_count=0,
            url_length_max=None,
            url_length_avg=None,
            ip_url_count=0,
            urls=[],
            url_hosts=[],
        )
        features = extract_features(_sample_parsed(), evidence)
        self.assertEqual(features.url_count, 0)
        self.assertIsNone(features.url_length_max)
        self.assertIsNone(features.url_length_avg)

    def test_has_risky_attachment_from_extension_or_mime_without_counting(self) -> None:
        both = extract_features(
            _sample_parsed(),
            _sample_evidence(
                risky_attachment_extensions=[".js"],
                risky_attachment_mime_types=["application/javascript"],
            ),
        )
        self.assertTrue(both.has_risky_attachment)

        extension_only = extract_features(
            _sample_parsed(),
            _sample_evidence(
                risky_attachment_extensions=[".exe"],
                risky_attachment_mime_types=[],
            ),
        )
        self.assertTrue(extension_only.has_risky_attachment)

        mime_only = extract_features(
            _sample_parsed(),
            _sample_evidence(
                risky_attachment_extensions=[],
                risky_attachment_mime_types=["application/x-msdownload"],
            ),
        )
        self.assertTrue(mime_only.has_risky_attachment)

        none_risky = extract_features(
            _sample_parsed(),
            _sample_evidence(
                risky_attachment_extensions=[],
                risky_attachment_mime_types=[],
            ),
        )
        self.assertFalse(none_risky.has_risky_attachment)

    def test_empty_text_is_safe_and_no_score_is_created(self) -> None:
        features = extract_features(
            _sample_parsed(subject=None, body_plain=None),
            _sample_evidence(),
        )
        self.assertEqual(features.subject, "")
        self.assertEqual(features.body_plain, "")
        self.assertNotIn("score", features.to_dict())
        self.assertNotIn("phishing_score", features.to_dict())


if __name__ == "__main__":
    unittest.main()
