# Parse incoming emails into structured fields for phishing analysis.
# Extraction only: this module does not classify, score, or verify authentication.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from email import policy
from email.message import Message
from email.parser import BytesParser, Parser
from email.utils import getaddresses, parseaddr
from pathlib import Path

BODY_CONTENT_TYPES = {"text/plain", "text/html"}


@dataclass
class ParsedEmail:
    subject: str | None = None
    body_plain: str | None = None
    body_html: str | None = None
    from_address: str | None = None
    from_domain: str | None = None
    reply_to: str | None = None
    to_addresses: list[str] = field(default_factory=list)
    cc_addresses: list[str] = field(default_factory=list)
    date: str | None = None
    message_id: str | None = None
    received_headers: list[str] = field(default_factory=list)
    content_type: str | None = None
    has_attachments: bool = False
    attachment_count: int = 0
    attachment_filenames: list[str] = field(default_factory=list)
    attachment_mime_types: list[str] = field(default_factory=list)
    headers: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_email(source: str | Path | bytes) -> ParsedEmail:
    """Parse an .eml path, raw email bytes, or raw email text into structured fields."""
    if isinstance(source, bytes):
        return parse_bytes(source)
    if isinstance(source, Path) or _is_existing_file(source):
        return parse_eml_file(Path(source))
    if isinstance(source, str):
        return parse_text(source)
    raise TypeError(f"Unsupported email source type: {type(source)!r}")


def parse_eml_file(path: Path) -> ParsedEmail:
    data = Path(path).read_bytes()
    return parse_bytes(data)


def parse_bytes(data: bytes) -> ParsedEmail:
    message = _parse_bytes_message(data)
    return _extract(message)


def parse_text(text: str) -> ParsedEmail:
    message = _parse_text_message(text)
    return _extract(message)


def _is_existing_file(source: str) -> bool:
    try:
        return Path(source).is_file()
    except OSError:
        return False


def _parse_bytes_message(data: bytes) -> Message:
    try:
        return BytesParser(policy=policy.default).parsebytes(data)
    except Exception:
        return BytesParser(policy=policy.compat32).parsebytes(data)


def _parse_text_message(text: str) -> Message:
    try:
        return Parser(policy=policy.default).parsestr(text)
    except Exception:
        return Parser(policy=policy.compat32).parsestr(text)


def _extract(message: Message) -> ParsedEmail:
    from_address = _first_address(message.get("From"))
    reply_to = _first_address(message.get("Reply-To"))
    attachments = _collect_attachments(message)
    body_plain, body_html = _extract_bodies(message)

    return ParsedEmail(
        subject=_header(message, "Subject"),
        body_plain=body_plain,
        body_html=body_html,
        from_address=from_address,
        from_domain=_domain_from_address(from_address),
        reply_to=reply_to,
        to_addresses=_address_list(message, "To"),
        cc_addresses=_address_list(message, "Cc"),
        date=_header(message, "Date"),
        message_id=_header(message, "Message-ID"),
        received_headers=_all_headers(message, "Received"),
        content_type=_content_type(message),
        has_attachments=bool(attachments),
        attachment_count=len(attachments),
        attachment_filenames=[item["filename"] for item in attachments],
        attachment_mime_types=[item["mime_type"] for item in attachments],
        headers=_structured_headers(message),
    )


def _header(message: Message, name: str) -> str | None:
    value = message.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _all_headers(message: Message, name: str) -> list[str]:
    values = message.get_all(name, [])
    return [str(value).strip() for value in values if str(value).strip()]


def _structured_headers(message: Message) -> dict[str, list[str]]:
    headers: dict[str, list[str]] = {}
    for key, value in message.items():
        text = str(value).strip()
        if not text:
            continue
        headers.setdefault(key, []).append(text)
    return headers


def _address_list(message: Message, name: str) -> list[str]:
    values = message.get_all(name, [])
    addresses = []
    for _, address in getaddresses(str(value) for value in values):
        cleaned = address.strip()
        if cleaned:
            addresses.append(cleaned)
    return addresses


def _first_address(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    _, address = parseaddr(str(raw_value))
    cleaned = address.strip()
    return cleaned or None


def _domain_from_address(address: str | None) -> str | None:
    if not address or "@" not in address:
        return None
    domain = address.rsplit("@", 1)[1].strip().lower()
    return domain or None


def _content_type(part: Message) -> str | None:
    getter = getattr(part, "get_content_type", None)
    if getter is not None:
        return getter()
    raw = part.get("Content-Type")
    if not raw:
        return None
    return str(raw).split(";", 1)[0].strip().lower() or None


def _disposition(part: Message) -> str:
    getter = getattr(part, "get_content_disposition", None)
    if getter is not None:
        return (getter() or "").lower()
    raw = part.get("Content-Disposition", "")
    return str(raw).split(";", 1)[0].strip().lower()


def _iter_leaf_parts(message: Message):
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                continue
            yield part
        return
    yield message


def _is_attachment(part: Message) -> bool:
    disposition = _disposition(part)
    filename = part.get_filename()
    if disposition == "attachment":
        return True
    if filename:
        return True
    return False


def _decode_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return str(raw) if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_bodies(message: Message) -> tuple[str | None, str | None]:
    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in _iter_leaf_parts(message):
        if _is_attachment(part):
            continue
        content_type = (_content_type(part) or "").lower()
        text = _decode_text(part).strip()
        if not text:
            continue
        if content_type == "text/plain":
            plain_parts.append(text)
        elif content_type == "text/html":
            html_parts.append(text)
    plain = "\n\n".join(plain_parts) or None
    html = "\n\n".join(html_parts) or None
    return plain, html


def _collect_attachments(message: Message) -> list[dict[str, str]]:
    # Metadata only: payloads are never written to disk or executed.
    attachments: list[dict[str, str]] = []
    for part in _iter_leaf_parts(message):
        if not _is_attachment(part):
            continue
        filename = part.get_filename() or ""
        mime_type = _content_type(part) or "application/octet-stream"
        attachments.append({"filename": filename, "mime_type": mime_type})
    return attachments
