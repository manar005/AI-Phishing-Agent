# Synthetic security-evidence tests only. No live mail, URL visits, or phishing labels.

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.email_parser import ParsedEmail
from src.security_checks import SecurityEvidence, run_security_checks


def _normal_email() -> ParsedEmail:
    return ParsedEmail(
        subject="Meeting notes",
        body_plain="Agenda is here: https://intranet.example.com/agenda",
        body_html='<p>Agenda is here: <a href="https://intranet.example.com/agenda">agenda</a></p>',
        from_address="alice@example.com",
        from_domain="example.com",
        reply_to="alice@example.com",
        to_addresses=["bob@example.com"],
        cc_addresses=["carol@example.com"],
        received_headers=["from mail.example.com by mx.example.com"],
        content_type="multipart/alternative",
        has_attachments=True,
        attachment_count=1,
        attachment_filenames=["agenda.pdf"],
        attachment_mime_types=["application/pdf"],
        headers={
            "Authentication-Results": [
                "mx.example.com; spf=pass smtp.mailfrom=example.com; "
                "dkim=pass header.d=example.com; dmarc=pass action=none"
            ],
            "From": ["Alice <alice@example.com>"],
        },
    )


def _suspicious_looking_email() -> ParsedEmail:
    return ParsedEmail(
        subject="Urgent account notice",
        body_plain=(
            "Reset now http://192.0.2.10/login and also "
            "https://help-desk.other-site.biz/reset"
        ),
        body_html='<p><a href="http://192.0.2.10/login">reset</a></p>',
        from_address="billing@example.com",
        from_domain="example.com",
        reply_to="support@unrelated.biz",
        to_addresses=["one@example.com", "two@example.com"],
        cc_addresses=[],
        received_headers=[
            "from unknown.example.net by mx.example.com",
            "from 192.0.2.10 by unknown.example.net",
        ],
        content_type="multipart/mixed",
        has_attachments=True,
        attachment_count=2,
        attachment_filenames=["invoice.pdf", "update.js"],
        attachment_mime_types=["application/pdf", "application/javascript"],
        headers={
            "Authentication-Results": [
                "mx.example.com; spf=fail smtp.mailfrom=example.com; "
                "dkim=fail header.d=example.com; dmarc=fail action=reject"
            ],
            "Reply-To": ["support@unrelated.biz"],
        },
    )


class SecurityChecksTests(unittest.TestCase):
    def test_normal_email_evidence(self) -> None:
        evidence = run_security_checks(_normal_email())

        self.assertIsInstance(evidence, SecurityEvidence)
        self.assertTrue(evidence.has_reply_to)
        self.assertEqual(evidence.from_address, "alice@example.com")
        self.assertEqual(evidence.reply_to_address, "alice@example.com")
        self.assertEqual(evidence.from_domain, "example.com")
        self.assertEqual(evidence.reply_to_domain, "example.com")
        self.assertFalse(evidence.from_reply_to_address_mismatch)
        self.assertFalse(evidence.from_reply_to_domain_mismatch)
        self.assertEqual(evidence.header_spf_result, "pass")
        self.assertEqual(evidence.header_dkim_result, "pass")
        self.assertEqual(evidence.header_dmarc_result, "pass")
        self.assertEqual(evidence.num_to_recipients, 1)
        self.assertEqual(evidence.num_cc_recipients, 1)
        self.assertEqual(evidence.num_received_headers, 1)
        self.assertTrue(evidence.has_html)
        self.assertEqual(evidence.attachment_count, 1)
        self.assertEqual(evidence.attachment_extensions, [".pdf"])
        self.assertEqual(evidence.risky_attachment_extensions, [])
        self.assertEqual(evidence.url_count, 2)
        self.assertEqual(evidence.unique_url_count, 1)
        self.assertGreater(evidence.url_length_max or 0, 0)
        self.assertFalse(evidence.has_ip_url)
        self.assertEqual(evidence.url_hosts, ["intranet.example.com"])
        self.assertNotIn("is_phishing", evidence.to_dict())
        self.assertNotIn("score", evidence.to_dict())
        self.assertNotIn("from_domain_url_domain_mismatch", evidence.to_dict())
        self.assertNotIn("url_domains", evidence.to_dict())

    def test_suspicious_looking_combinations_are_recorded_not_labeled(self) -> None:
        evidence = run_security_checks(_suspicious_looking_email())

        self.assertTrue(evidence.from_reply_to_address_mismatch)
        self.assertTrue(evidence.from_reply_to_domain_mismatch)
        self.assertEqual(evidence.header_spf_result, "fail")
        self.assertEqual(evidence.header_dkim_result, "fail")
        self.assertEqual(evidence.header_dmarc_result, "fail")
        self.assertEqual(evidence.num_to_recipients, 2)
        self.assertEqual(evidence.num_received_headers, 2)
        self.assertIn(".js", evidence.risky_attachment_extensions)
        self.assertIn("application/javascript", evidence.risky_attachment_mime_types)
        self.assertNotIn(".pdf", evidence.risky_attachment_extensions)
        self.assertTrue(evidence.has_ip_url)
        self.assertEqual(evidence.ip_url_count, 2)
        self.assertIn("192.0.2.10", evidence.url_hosts)
        self.assertIn("help-desk.other-site.biz", evidence.url_hosts)
        self.assertEqual(evidence.url_count, 3)
        self.assertEqual(evidence.unique_url_count, 2)
        self.assertNotIn("url_domains", evidence.to_dict())
        self.assertNotIn("label", evidence.to_dict())
        self.assertNotIn("phishing", evidence.to_dict())

    def test_missing_reply_to_and_auth_are_unknown_not_malicious(self) -> None:
        parsed = ParsedEmail(
            subject="Hello",
            body_plain="No links here.",
            from_address="alice@example.com",
            from_domain="example.com",
            to_addresses=["bob@example.com"],
            content_type="text/plain",
        )
        evidence = run_security_checks(parsed)

        self.assertFalse(evidence.has_reply_to)
        self.assertIsNone(evidence.reply_to_address)
        self.assertIsNone(evidence.from_reply_to_address_mismatch)
        self.assertIsNone(evidence.from_reply_to_domain_mismatch)
        self.assertIsNone(evidence.header_spf_result)
        self.assertIsNone(evidence.header_dkim_result)
        self.assertIsNone(evidence.header_dmarc_result)
        self.assertEqual(evidence.url_count, 0)
        self.assertIsNone(evidence.url_length_max)
        self.assertIsNone(evidence.url_length_avg)
        self.assertFalse(evidence.has_html)
        self.assertEqual(evidence.risky_attachment_extensions, [])

    def test_received_spf_header_is_used_when_authentication_results_missing(self) -> None:
        parsed = ParsedEmail(
            from_address="alice@example.com",
            from_domain="example.com",
            headers={"Received-SPF": ["softfail (example.com: sender SPF not aligned)"]},
        )
        evidence = run_security_checks(parsed)
        self.assertEqual(evidence.header_spf_result, "softfail")
        self.assertIsNone(evidence.header_dkim_result)
        self.assertIsNone(evidence.header_dmarc_result)

    def test_html_href_is_not_double_counted_by_raw_url_regex(self) -> None:
        parsed = ParsedEmail(
            body_html='<p>See <a href="https://intranet.example.com/agenda">agenda</a></p>',
        )
        evidence = run_security_checks(parsed)
        self.assertEqual(evidence.url_count, 1)
        self.assertEqual(evidence.unique_url_count, 1)
        self.assertEqual(evidence.urls, ["https://intranet.example.com/agenda"])
        self.assertEqual(evidence.url_hosts, ["intranet.example.com"])

    def test_html_visible_url_and_different_href_are_both_counted(self) -> None:
        parsed = ParsedEmail(
            body_html=(
                '<p>Also visit https://visible.example.com/help '
                'and <a href="https://href.example.com/y">y</a></p>'
            ),
        )
        evidence = run_security_checks(parsed)
        self.assertEqual(evidence.url_count, 2)
        self.assertEqual(evidence.unique_url_count, 2)
        self.assertCountEqual(
            evidence.url_hosts,
            ["visible.example.com", "href.example.com"],
        )

    def test_url_hosts_keep_full_hostname_including_public_suffixes(self) -> None:
        parsed = ParsedEmail(
            body_plain="Read https://www.news.example.co.uk/article",
        )
        evidence = run_security_checks(parsed)
        self.assertEqual(evidence.url_count, 1)
        self.assertEqual(evidence.unique_url_count, 1)
        self.assertEqual(evidence.url_hosts, ["www.news.example.co.uk"])
        self.assertNotIn("url_domains", evidence.to_dict())
        self.assertNotEqual(evidence.url_hosts, ["example.co.uk"])
        self.assertNotEqual(evidence.url_hosts, ["co.uk"])

    def test_text_plain_body_urls_and_html_hrefs_are_extracted_without_visiting(self) -> None:
        parsed = ParsedEmail(
            body_plain="See www.docs.example.com/help",
            body_html='Click <a href="https://files.cdn.net/img.png">image</a>',
            from_domain="example.com",
        )
        evidence = run_security_checks(parsed)
        self.assertEqual(evidence.url_count, 2)
        self.assertEqual(evidence.unique_url_count, 2)
        self.assertIn("www.docs.example.com", evidence.url_hosts)
        self.assertIn("files.cdn.net", evidence.url_hosts)
        self.assertNotIn("url_domains", evidence.to_dict())


if __name__ == "__main__":
    unittest.main()
