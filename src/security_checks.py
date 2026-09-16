# Derive behavioral/security evidence from a parsed email.
# This module does not classify phishing or compute a final score.
# SPF/DKIM/DMARC values are copied from message headers; they are not
# independently verified by this application.

from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.email_parser import ParsedEmail

AUTH_RESULT_VALUES = {
    "pass",
    "fail",
    "softfail",
    "neutral",
    "none",
    "temperror",
    "permerror",
}

# Extension classes that are commonly abused, returned as evidence only.
RISKY_EXTENSIONS = {
    ".exe",
    ".scr",
    ".bat",
    ".cmd",
    ".com",
    ".pif",
    ".msi",
    ".dll",
    ".js",
    ".jse",
    ".vbs",
    ".vbe",
    ".ps1",
    ".wsf",
    ".hta",
    ".lnk",
    ".jar",
    ".iso",
    ".img",
    ".html",
    ".htm",
    ".docm",
    ".xlsm",
    ".pptm",
    ".dotm",
    ".xltm",
}

RISKY_MIME_TYPES = {
    "application/x-msdownload",
    "application/x-msdos-program",
    "application/x-executable",
    "application/javascript",
    "text/javascript",
    "application/x-javascript",
    "application/vnd.ms-htmlhelp",
}

URL_RE = re.compile(r"""https?://[^\s<>"'\\]+|www\.[^\s<>"'\\]+""", re.IGNORECASE)
HREF_RE = re.compile(r"""href\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE)
HREF_ATTR_RE = re.compile(r"""\shref\s*=\s*['"][^'"]*['"]""", re.IGNORECASE)
AUTH_TOKEN_RE = re.compile(r"\b(spf|dkim|dmarc)\s*=\s*([a-z0-9]+)", re.IGNORECASE)
TRAILING_URL_PUNCT_RE = re.compile(r"[),.;:!?)]+$")


@dataclass
class SecurityEvidence:
    has_reply_to: bool = False
    from_address: str | None = None
    reply_to_address: str | None = None
    return_path_address: str | None = None
    from_domain: str | None = None
    reply_to_domain: str | None = None
    return_path_domain: str | None = None
    from_reply_to_address_mismatch: bool | None = None
    from_reply_to_domain_mismatch: bool | None = None
    from_return_path_domain_mismatch: bool | None = None
    # Header-reported results only; this app does not verify SPF/DKIM/DMARC.
    header_spf_result: str | None = None
    header_dkim_result: str | None = None
    header_dmarc_result: str | None = None
    num_to_recipients: int = 0
    num_cc_recipients: int = 0
    num_received_headers: int = 0
    has_html: bool = False
    attachment_count: int = 0
    attachment_extensions: list[str] = field(default_factory=list)
    attachment_mime_types: list[str] = field(default_factory=list)
    risky_attachment_extensions: list[str] = field(default_factory=list)
    risky_attachment_mime_types: list[str] = field(default_factory=list)
    url_count: int = 0
    unique_url_count: int = 0
    url_length_max: float | None = None
    url_length_avg: float | None = None
    ip_url_count: int = 0
    punycode_url_count: int = 0
    has_ip_url: bool = False
    url_hosts: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_security_checks(parsed: ParsedEmail) -> SecurityEvidence:
    """Return behavioral evidence from a parsed email. No phishing label or score."""
    from_address = _normalize_address(parsed.from_address)
    reply_to_address = _normalize_address(parsed.reply_to)
    return_path_address = _normalize_address(parsed.return_path)
    from_domain = parsed.from_domain.lower().strip() if parsed.from_domain else _domain_from_address(from_address)
    reply_to_domain = _domain_from_address(reply_to_address)
    return_path_domain = (
        parsed.return_path_domain.lower().strip()
        if parsed.return_path_domain
        else _domain_from_address(return_path_address)
    )

    urls = _extract_urls(parsed.body_plain, parsed.body_html)
    hosts = [_url_host(url) for url in urls]
    hosts = [host for host in hosts if host]
    lengths = [len(url) for url in urls]
    ip_count = sum(1 for host in hosts if _is_ip_host(host))
    punycode_count = sum(1 for host in hosts if _host_has_punycode_label(host))

    extensions = _attachment_extensions(parsed.attachment_filenames)
    mime_types = [mime.lower() for mime in parsed.attachment_mime_types if mime]
    auth = _header_auth_results(parsed.headers)

    return SecurityEvidence(
        has_reply_to=bool(reply_to_address),
        from_address=from_address,
        reply_to_address=reply_to_address,
        return_path_address=return_path_address,
        from_domain=from_domain,
        reply_to_domain=reply_to_domain,
        return_path_domain=return_path_domain,
        from_reply_to_address_mismatch=_mismatch(from_address, reply_to_address),
        from_reply_to_domain_mismatch=_mismatch(from_domain, reply_to_domain),
        from_return_path_domain_mismatch=_mismatch(from_domain, return_path_domain),
        header_spf_result=auth["spf"],
        header_dkim_result=auth["dkim"],
        header_dmarc_result=auth["dmarc"],
        num_to_recipients=len(parsed.to_addresses),
        num_cc_recipients=len(parsed.cc_addresses),
        num_received_headers=len(parsed.received_headers),
        has_html=_has_html(parsed),
        attachment_count=parsed.attachment_count,
        attachment_extensions=extensions,
        attachment_mime_types=mime_types,
        risky_attachment_extensions=[ext for ext in extensions if ext in RISKY_EXTENSIONS],
        risky_attachment_mime_types=[mime for mime in mime_types if mime in RISKY_MIME_TYPES],
        url_count=len(urls),
        unique_url_count=len(set(urls)),
        url_length_max=float(max(lengths)) if lengths else None,
        url_length_avg=round(sum(lengths) / len(lengths), 4) if lengths else None,
        ip_url_count=ip_count,
        punycode_url_count=punycode_count,
        has_ip_url=ip_count > 0,
        url_hosts=_unique_keep_order(hosts),
        urls=urls,
    )


def _normalize_address(address: str | None) -> str | None:
    if not address:
        return None
    cleaned = address.strip().lower()
    return cleaned or None


def _domain_from_address(address: str | None) -> str | None:
    if not address or "@" not in address:
        return None
    domain = address.rsplit("@", 1)[1].strip().lower()
    return domain or None


def _mismatch(left: str | None, right: str | None) -> bool | None:
    if not left or not right:
        return None
    return left != right


def _has_html(parsed: ParsedEmail) -> bool:
    if parsed.body_html and parsed.body_html.strip():
        return True
    content_type = (parsed.content_type or "").lower()
    return "text/html" in content_type


def _header_values(headers: dict[str, list[str]], name: str) -> list[str]:
    wanted = name.lower()
    values: list[str] = []
    for key, items in headers.items():
        if key.lower() == wanted:
            values.extend(items)
    return values


def _normalize_auth_result(value: str | None) -> str | None:
    if not value:
        return None
    token = value.strip().lower()
    if token in AUTH_RESULT_VALUES:
        return token
    return None


def _parse_authentication_results(values: list[str]) -> dict[str, str | None]:
    found = {"spf": None, "dkim": None, "dmarc": None}
    for raw in values:
        for method, result in AUTH_TOKEN_RE.findall(raw):
            method_key = method.lower()
            if method_key in found and found[method_key] is None:
                found[method_key] = _normalize_auth_result(result)
    return found


def _parse_received_spf(values: list[str]) -> str | None:
    for raw in values:
        first = raw.strip().split(None, 1)[0] if raw.strip() else ""
        result = _normalize_auth_result(first)
        if result:
            return result
    return None


def _header_auth_results(headers: dict[str, list[str]]) -> dict[str, str | None]:
    """Read SPF/DKIM/DMARC tokens from headers. No cryptographic or DNS check is performed."""
    auth_headers = _header_values(headers, "Authentication-Results")
    auth_headers.extend(_header_values(headers, "ARC-Authentication-Results"))
    parsed = _parse_authentication_results(auth_headers)
    if parsed["spf"] is None:
        parsed["spf"] = _parse_received_spf(_header_values(headers, "Received-SPF"))
    return parsed


def _clean_url(raw: str) -> str | None:
    url = TRAILING_URL_PUNCT_RE.sub("", raw.strip())
    if url.lower().startswith("www."):
        url = "http://" + url
    if not url.lower().startswith(("http://", "https://")):
        return None
    return url


def _extract_urls(body_plain: str | None, body_html: str | None) -> list[str]:
    found: list[str] = []
    for match in URL_RE.findall(body_plain or ""):
        url = _clean_url(match)
        if url:
            found.append(url)

    html = body_html or ""
    for match in HREF_RE.findall(html):
        url = _clean_url(match)
        if url:
            found.append(url)
    # Strip href attributes first so the same HTML link is not counted again
    # by the raw URL regex matching inside href="...".
    html_without_hrefs = HREF_ATTR_RE.sub(" ", html)
    for match in URL_RE.findall(html_without_hrefs):
        url = _clean_url(match)
        if url:
            found.append(url)
    return found


def _url_host(url: str) -> str | None:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    return host.strip().lower().rstrip(".") or None


def _is_ip_host(host: str) -> bool:
    candidate = host.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def _host_has_punycode_label(host: str) -> bool:
    return any(label.startswith("xn--") for label in host.lower().split("."))


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _attachment_extensions(filenames: list[str]) -> list[str]:
    extensions: list[str] = []
    for name in filenames:
        suffix = Path(name).suffix.lower()
        if suffix:
            extensions.append(suffix)
    return extensions
