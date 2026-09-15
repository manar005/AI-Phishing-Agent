# Synthetic parser tests only. No live mail, attachments, or network lookups.

from __future__ import annotations

import sys
import tempfile
import unittest
from email.message import EmailMessage, Message
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.email_parser import ParsedEmail, parse_email

PLAIN_BODY_1 = "Please review the attached invoice."
PLAIN_BODY_2 = "This is a follow-up paragraph."
HTML_BODY = "<p>Please review the attached invoice.</p>"
NOTES_ATTACHMENT_TEXT = "UNIQUE_NOTES_ATTACHMENT_CONTENT"


def _sample_message() -> Message:
    message = MIMEMultipart("mixed")
    message["Subject"] = "Invoice attached"
    message["From"] = "Alice Example <alice@example.com>"
    message["To"] = "Bob <bob@company.com>, carol@company.com"
    message["Cc"] = "Dave <dave@company.com>"
    message["Reply-To"] = "Billing <noreply@example.com>"
    message["Date"] = "Mon, 14 Sep 2026 10:00:00 +0000"
    message["Message-ID"] = "<invoice-123@example.com>"
    message["Received"] = (
        "from mail.example.com (mail.example.com [192.0.2.10]) "
        "by mx.company.com with ESMTPS; Mon, 14 Sep 2026 10:00:00 +0000"
    )

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(PLAIN_BODY_1, "plain", "utf-8"))
    alternative.attach(MIMEText(HTML_BODY, "html", "utf-8"))
    message.attach(alternative)
    message.attach(MIMEText(PLAIN_BODY_2, "plain", "utf-8"))

    notes = MIMEText(NOTES_ATTACHMENT_TEXT, "plain", "utf-8")
    notes.add_header("Content-Disposition", "attachment", filename="notes.txt")
    message.attach(notes)

    pdf = MIMEApplication(b"%PDF-fake-content", _subtype="pdf")
    pdf.add_header("Content-Disposition", "attachment", filename="invoice.pdf")
    message.attach(pdf)
    return message


class EmailParserTests(unittest.TestCase):
    def test_parse_multipart_sample_from_bytes(self) -> None:
        raw = _sample_message().as_bytes()
        parsed = parse_email(raw)

        self.assertIsInstance(parsed, ParsedEmail)
        self.assertEqual(parsed.subject, "Invoice attached")
        self.assertIn(PLAIN_BODY_1, parsed.body_plain or "")
        self.assertIn(PLAIN_BODY_2, parsed.body_plain or "")
        self.assertNotIn(NOTES_ATTACHMENT_TEXT, parsed.body_plain or "")
        self.assertIn(HTML_BODY, parsed.body_html or "")
        self.assertEqual(parsed.from_address, "alice@example.com")
        self.assertEqual(parsed.from_domain, "example.com")
        self.assertEqual(parsed.reply_to, "noreply@example.com")
        self.assertEqual(parsed.to_addresses, ["bob@company.com", "carol@company.com"])
        self.assertEqual(parsed.cc_addresses, ["dave@company.com"])
        self.assertEqual(parsed.date, "Mon, 14 Sep 2026 10:00:00 +0000")
        self.assertEqual(parsed.message_id, "<invoice-123@example.com>")
        self.assertEqual(len(parsed.received_headers), 1)
        self.assertIn("mail.example.com", parsed.received_headers[0])
        self.assertTrue(parsed.content_type.startswith("multipart/"))
        self.assertTrue(parsed.has_attachments)
        self.assertEqual(parsed.attachment_count, 2)
        self.assertCountEqual(parsed.attachment_filenames, ["notes.txt", "invoice.pdf"])
        self.assertIn("text/plain", parsed.attachment_mime_types)
        self.assertIn("application/pdf", parsed.attachment_mime_types)
        self.assertIn("From", parsed.headers)
        self.assertIn("Received", parsed.headers)
        self.assertNotIn("label", parsed.to_dict())
        self.assertNotIn("score", parsed.to_dict())

    def test_parse_eml_file_and_raw_text(self) -> None:
        raw_bytes = _sample_message().as_bytes()
        raw_text = _sample_message().as_string()

        with tempfile.TemporaryDirectory() as tmp:
            eml_path = Path(tmp) / "sample.eml"
            eml_path.write_bytes(raw_bytes)
            from_file = parse_email(eml_path)
            from_text = parse_email(raw_text)

        self.assertEqual(from_file.subject, "Invoice attached")
        self.assertEqual(from_file.from_address, "alice@example.com")
        self.assertIn(PLAIN_BODY_1, from_file.body_plain or "")
        self.assertIn(PLAIN_BODY_2, from_file.body_plain or "")
        self.assertNotIn(NOTES_ATTACHMENT_TEXT, from_file.body_plain or "")
        self.assertIn("notes.txt", from_file.attachment_filenames)
        self.assertEqual(from_text.subject, "Invoice attached")
        self.assertEqual(from_text.from_domain, "example.com")
        self.assertIn(PLAIN_BODY_1, from_text.body_plain or "")
        self.assertIn(PLAIN_BODY_2, from_text.body_plain or "")
        self.assertNotIn(NOTES_ATTACHMENT_TEXT, from_text.body_plain or "")
        self.assertIn("invoice.pdf", from_text.attachment_filenames)

    def test_missing_fields_are_safe(self) -> None:
        message = EmailMessage()
        message.set_content("Hello without headers.")
        parsed = parse_email(message.as_bytes())

        self.assertIsNone(parsed.subject)
        self.assertIsNone(parsed.from_address)
        self.assertIsNone(parsed.from_domain)
        self.assertIsNone(parsed.reply_to)
        self.assertEqual(parsed.to_addresses, [])
        self.assertEqual(parsed.cc_addresses, [])
        self.assertFalse(parsed.has_attachments)
        self.assertEqual(parsed.attachment_count, 0)
        self.assertEqual(parsed.body_plain, "Hello without headers.")


if __name__ == "__main__":
    unittest.main()
