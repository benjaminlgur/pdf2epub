#!/usr/bin/env python3
"""OCR for scanned PDFs through Google Cloud Vision - no GPU needed.

marker's own OCR runs a vision model on an NVIDIA GPU (served by llama.cpp).
This module is the alternative for machines without one: every page is rendered
to a greyscale JPEG, sent to Cloud Vision's DOCUMENT_TEXT_DETECTION, and the text
that comes back is rebuilt into markdown that pdf2epub.py then treats exactly
like marker's output (cleanup, chapter split, pandoc).

Rebuilding the markdown is where the work is, because Vision returns text and
boxes, not a document:

* paragraphs come from Vision's own paragraph detection, and words hyphenated at
  a line end are joined again;
* running heads, running feet and page numbers are recognised by where they sit
  (the top or bottom edge of the page) and by repeating on several pages, and
  are dropped;
* headings are paragraphs set noticeably larger than the body text. Every
  distinct size becomes one heading level, so pdf2epub's split-level detection
  can pick the chapter level just as it does for marker's output;
* paragraphs that a page break (or a footnote) cut in two are joined again.

Figures are not extracted: Vision reads text only.

Talking to the API needs nothing beyond the standard library - an API key in
GOOGLE_VISION_API_KEY is enough. Service-account or gcloud credentials work too
when the optional google-auth package is installed.

Cloud Vision bills per page. The first 1,000 pages a month are free, then it is
US$1.50 per 1,000 (Google's price list in 2026). Two things keep that predictable:

* every page's result is cached in the work folder, so a re-run, a crash or a
  cancelled job never pays for the same page twice;
* pages sent are counted per calendar month on this computer, and a run that
  would go over the monthly limit (by default the 1,000 free pages) stops before
  it sends anything.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

LogFn = Callable[[str], None]
StopFn = Callable[[], bool]

API_URL = "https://vision.googleapis.com/v1/images:annotate"
API_KEY_ENV = "GOOGLE_VISION_API_KEY"
VISION_SCOPE = "https://www.googleapis.com/auth/cloud-vision"

# Google's free tier: the first 1,000 units (pages) of text detection each month.
FREE_PAGES_PER_MONTH = 1000
# Vision asks for at least 1024x768 pixels for dense text; 200 dpi greyscale is
# comfortably above that and keeps one page upload around 200-400 KB.
RENDER_DPI = 200
# Huge pages (posters, fold-outs) are scaled down to this long edge in pixels.
MAX_IMAGE_EDGE = 4000
JPEG_QUALITY = 85
# Pages in flight at once. Vision's default quota is 1,800 requests a minute.
PARALLEL_REQUESTS = 4
REQUEST_TIMEOUT = 120
ATTEMPTS = 4
# Rate limits and transient failures are retried; everything else is reported.
RETRY_HTTP = {408, 429, 500, 502, 503, 504}
RETRY_RPC = {4, 8, 13, 14}  # DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, INTERNAL, UNAVAILABLE
# Bump when the cached page format, or the way text is rebuilt into it, changes.
CACHE_VERSION = 1

# A short paragraph this much larger than the body text is a heading (a 14 pt
# heading over 11 pt text measures 1.17-1.25) ...
HEADING_RATIO = 1.15
# ... and heading sizes within 10 % of each other share a level.
LEVEL_STEP = 0.9
MAX_HEADING_CHARS = 150
MAX_HEADING_LINES = 3
# Running heads, running feet and page numbers sit in this share of the page
# height at the top or the bottom.
MARGIN_BAND = 0.15
# Small print at the foot of a page (footnotes) is at most this size.
NOTE_RATIO = 0.85
# Short paragraphs Vision itself doubts are usually specks or bits of a picture.
NOISE_CONFIDENCE = 0.5
NOISE_MAX_CHARS = 30

SKIPPED_BLOCKS = {"PICTURE", "RULER", "BARCODE"}
TALL_LOWERCASE = set("bdfhklt")
HYPHENS = "-‐‑­"
SOFT_WRAP = "\x00"  # marks "the line ended inside a word" while text is rebuilt

ROMAN = r"(?=[MDCLXVI])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
PAGE_NUMBER_RE = re.compile(
    rf"[\s\-–—]*(?i:page\s+|p\.\s*)?(?:\d{{1,4}}|(?i:{ROMAN}))[\s\-–—.]*"
)
# "Chapter 3", "PART TWO", "IV", "12." - a label that belongs to the title after it.
CHAPTER_LABEL_RE = re.compile(
    r"(?i:chapter|part|book|section|kapitel|teil|chapitre|partie|cap[ií]tulo|parte)"
    r"\s+[\w-]+(?:\s+[\w-]+)?\.?"
    rf"|\d{{1,3}}\.?|{ROMAN}\.?"
)
NUMERIC_LABEL_RE = re.compile(rf"(?:\d{{1,3}}|{ROMAN})\.?")
SENTENCE_END_RE = re.compile(r"[.!?…][\"'”’»)\]]*$")
LINE_END_HYPHEN_RE = re.compile(rf"([{HYPHENS}])\n(\w)")
MARKDOWN_SPECIAL_RE = re.compile(r"([\\`*_\[\]<>#$^~|])")
ORDERED_LIST_START_RE = re.compile(r"^(\d+)([.)])(?=\s|$)")
OTHER_BLOCK_START_RE = re.compile(r"^([-+:])(?=\s|$)|^(%)")


class VisionOcrError(RuntimeError):
    """Something the user has to act on: credentials, quota, the API refusing."""


SETUP_HINT = (
    f"Set the {API_KEY_ENV} environment variable to an API key of a Google Cloud "
    "project that has the Cloud Vision API enabled - MANUAL.md, \"OCR without a "
    "GPU\", walks through it."
)


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


class Credentials:
    """How requests are authorised: an API key, or google-auth credentials."""

    def __init__(self, source: str, api_key: str = "", google: Any = None) -> None:
        self.source = source  # where they came from, for the log
        self.api_key = api_key
        self.google = google
        self._lock = threading.Lock()

    def headers(self) -> dict[str, str]:
        if self.api_key:
            # A header rather than ?key=..., so the key never ends up in a URL.
            return {"x-goog-api-key": self.api_key}
        with self._lock:
            if not self.google.valid:
                try:
                    from google.auth.transport.requests import Request
                    self.google.refresh(Request())
                except ImportError as error:
                    raise VisionOcrError(
                        "google-auth needs the requests package: pip install requests"
                    ) from error
                except Exception as error:
                    raise VisionOcrError(f"could not get a Google access token: {error}") from error
            headers = {"Authorization": f"Bearer {self.google.token}"}
        quota_project = getattr(self.google, "quota_project_id", None)
        if quota_project:
            headers["x-goog-user-project"] = quota_project
        return headers


def _adc_file() -> Path | None:
    """The credentials file google-auth would use, if there is one.

    Looked up by hand because google.auth.default() without a file goes on to
    probe for a cloud metadata server, which takes seconds on a home PC.
    """
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        return Path(explicit)
    config = os.environ.get("CLOUDSDK_CONFIG")
    if config:
        folder = Path(config)
    elif os.name == "nt":
        folder = Path(os.environ.get("APPDATA", "")) / "gcloud"
    else:
        folder = Path.home() / ".config" / "gcloud"
    candidate = folder / "application_default_credentials.json"
    return candidate if candidate.is_file() else None


def find_credentials() -> Credentials | None:
    """An API key from the environment, else Application Default Credentials."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        return Credentials(f"API key from {API_KEY_ENV}", api_key=key)
    adc = _adc_file()
    if adc is None:
        return None
    if not adc.is_file():
        raise VisionOcrError(f"GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {adc}")
    try:
        import google.auth
    except ImportError as error:
        raise VisionOcrError(
            f"found Google credentials ({adc}), but the google-auth package that reads "
            "them is not installed: pip install google-auth"
        ) from error
    try:
        google_credentials, _ = google.auth.default(scopes=[VISION_SCOPE])
    except Exception as error:
        raise VisionOcrError(f"could not load the Google credentials in {adc}: {error}") from error
    return Credentials(f"Google credentials from {adc}", google=google_credentials)


def credentials_status() -> tuple[bool, str]:
    """(usable, description) for the UI and the start-up banner."""
    try:
        credentials = find_credentials()
    except VisionOcrError as error:
        return False, str(error)
    if credentials is None:
        return False, f"{API_KEY_ENV} is not set"
    return True, credentials.source


# --------------------------------------------------------------------------- #
# Pages sent this month
# --------------------------------------------------------------------------- #

_USAGE_LOCK = threading.Lock()


def usage_file() -> Path:
    """Outside the work folder, so clearing the scratch files keeps the count."""
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "pdf2epub" / "vision-usage.json"


def _this_month() -> str:
    return date.today().strftime("%Y-%m")


def _read_usage(path: Path) -> dict[str, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(month): count for month, count in data.items() if isinstance(count, int)}


def pages_used_this_month() -> int:
    return _read_usage(usage_file()).get(_this_month(), 0)


def record_usage(pages: int) -> None:
    if pages <= 0:
        return
    path = usage_file()
    with _USAGE_LOCK:
        usage = _read_usage(path)
        month = _this_month()
        usage[month] = usage.get(month, 0) + pages
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(usage, indent=1, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# The page model (what is cached per page)
# --------------------------------------------------------------------------- #


@dataclass
class Paragraph:
    """One paragraph as Vision found it.

    Geometry is in PDF points, measured on the page the way it is read (text
    running left to right, lines top to bottom), so pages scanned sideways or at
    different resolutions compare like any other.
    """

    text: str
    left: float
    top: float
    right: float
    bottom: float
    size: float  # median height of the tall glyphs: capitals, digits, ascenders
    weight: int  # how many glyphs that size was measured on
    lines: int
    confidence: float = 1.0


@dataclass
class Page:
    index: int  # 0-based page number in the PDF
    width: float
    height: float
    paragraphs: list[Paragraph] = field(default_factory=list)


def _cache_path(folder: Path, index: int) -> Path:
    return folder / f"page-{index + 1:05d}.json"


def _load_cached(folder: Path, index: int) -> Page | None:
    try:
        data = json.loads(_cache_path(folder, index).read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            return None
        return Page(
            index=index,
            width=data["width"],
            height=data["height"],
            paragraphs=[Paragraph(**item) for item in data["paragraphs"]],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_cached(folder: Path, page: Page) -> None:
    target = _cache_path(folder, page.index)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": CACHE_VERSION, **asdict(page)}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(target)  # a crash never leaves half a page behind


def _fingerprint(pdf: Path) -> str:
    """Cache key of a PDF: its content, so renaming or re-uploading it still hits."""
    digest = hashlib.sha256()
    with pdf.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Rendering and the API call
# --------------------------------------------------------------------------- #


def parse_page_range(spec: str | None, page_count: int) -> list[int]:
    """Page indices to read, in marker's --page_range syntax (0-based: "0,5-10,20")."""
    if not spec or not spec.strip():
        return list(range(page_count))
    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*(?:-\s*(\d+))?", part)
        if not match:
            raise VisionOcrError(f"page range not understood: {part!r} (use e.g. 0,5-10,20)")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        chosen.update(range(first, min(last, page_count - 1) + 1))
    pages = sorted(chosen)
    if not pages:
        raise VisionOcrError(f"page range {spec!r} selects none of the {page_count} pages")
    return pages


def render_page(page: Any, dpi: int = RENDER_DPI) -> tuple[bytes, tuple[int, int], float]:
    """A PDF page as a greyscale JPEG: (data, size in pixels, points per pixel)."""
    width, height = page.get_size()
    scale = dpi / 72
    longest = max(width, height) * scale
    if longest > MAX_IMAGE_EDGE:
        scale *= MAX_IMAGE_EDGE / longest
    # render() accepts a float scale (the stub claims int).
    image = page.render(scale=scale, grayscale=True).to_pil()  # type: ignore
    if image.mode != "L":
        image = image.convert("L")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue(), image.size, 1 / scale


def _describe_http_error(error: urllib.error.HTTPError) -> str:
    detail = ""
    try:
        payload = json.loads(error.read().decode("utf-8", errors="replace"))
        detail = str(payload["error"].get("message", ""))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    message = f"Google Cloud Vision refused the request (HTTP {error.code}): {detail or error.reason}"
    if "billing" in detail.lower():
        message += (
            "\nCloud Vision only works in a project with a billing account attached - "
            "even when you stay inside the free pages."
        )
    elif error.code in (400, 401, 403):
        message += "\nSee MANUAL.md, \"OCR without a GPU\", for the setup steps."
    return message


def annotate(image: bytes, credentials: Credentials) -> dict:
    """Send one page image to Cloud Vision and return its AnnotateImageResponse."""
    body = json.dumps({
        "requests": [{
            "image": {"content": base64.b64encode(image).decode("ascii")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }).encode("utf-8")
    delay = 2.0
    for attempt in range(1, ATTEMPTS + 1):
        last_try = attempt == ATTEMPTS
        request = urllib.request.Request(
            API_URL,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8", **credentials.headers()},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as reply:
                payload = json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in RETRY_HTTP and not last_try:
                time.sleep(delay)
                delay *= 2
                continue
            raise VisionOcrError(_describe_http_error(error)) from error
        except (OSError, ValueError, http.client.HTTPException) as error:
            # No connection, a timeout, or a reply cut off half-way.
            if not last_try:
                time.sleep(delay)
                delay *= 2
                continue
            raise VisionOcrError(f"could not reach Google Cloud Vision: {error}") from error
        result = (payload.get("responses") or [{}])[0]
        problem = result.get("error")
        if problem:
            if problem.get("code") in RETRY_RPC and not last_try:
                time.sleep(delay)
                delay *= 2
                continue
            raise VisionOcrError(
                f"Google Cloud Vision could not read a page: {problem.get('message') or problem}"
            )
        return result
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Vision's answer -> the page model
# --------------------------------------------------------------------------- #


def _vertices(box: dict | None) -> list[tuple[float, float]]:
    # proto3 JSON leaves zero coordinates out, hence the defaults.
    return [(float(point.get("x", 0)), float(point.get("y", 0)))
            for point in (box or {}).get("vertices") or []]


def _text_direction(paragraphs: list[dict]) -> tuple[int, int]:
    """Which way the text runs in the image: (1, 0) for an upright page.

    Vision orders a word's corners as the text is read (top-left, top-right,
    ...), whatever the rotation, so the first edge points along the line.
    """
    votes: dict[tuple[int, int], int] = {}
    for paragraph in paragraphs:
        for word in paragraph.get("words") or []:
            points = _vertices(word.get("boundingBox"))
            if len(points) < 2:
                continue
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            if abs(dx) >= abs(dy):
                key = (1 if dx >= 0 else -1, 0)
            else:
                key = (0, 1 if dy > 0 else -1)
            votes[key] = votes.get(key, 0) + len(word.get("symbols") or [])
    return max(votes, key=lambda key: votes[key]) if votes else (1, 0)


class _Frame:
    """Maps image pixels onto the page as it is read: lines run left to right
    and follow each other top to bottom, even when the scan lies sideways."""

    def __init__(self, direction: tuple[int, int], width: float, height: float) -> None:
        self.dx, self.dy = direction
        corners = [self._turn(x, y) for x, y in ((0, 0), (width, 0), (width, height), (0, height))]
        self.u0 = min(u for u, _ in corners)
        self.v0 = min(v for _, v in corners)
        self.width = max(u for u, _ in corners) - self.u0
        self.height = max(v for _, v in corners) - self.v0

    def _turn(self, x: float, y: float) -> tuple[float, float]:
        return x * self.dx + y * self.dy, y * self.dx - x * self.dy

    def box(self, bounding: dict | None) -> tuple[float, float, float, float] | None:
        """(left, top, right, bottom) of a Vision bounding box, or None."""
        points = [self._turn(x, y) for x, y in _vertices(bounding)]
        if not points:
            return None
        us = [u - self.u0 for u, _ in points]
        vs = [v - self.v0 for _, v in points]
        return min(us), min(vs), max(us), max(vs)


def _rejoin(match: re.Match) -> str:
    hyphen, following = match.group(1), match.group(2)
    if hyphen == "­" or following.islower():
        return following  # a word split to fit the line
    return hyphen + following  # a real hyphen: "Anglo-Saxon", "1914-18"


def _paragraph_text(paragraph: dict) -> tuple[str, int]:
    """The paragraph's text with Vision's line breaks resolved, and its line count."""
    parts: list[str] = []
    for word in paragraph.get("words") or []:
        for symbol in word.get("symbols") or []:
            char = symbol.get("text", "")
            kind = ((symbol.get("property") or {}).get("detectedBreak") or {}).get("type", "")
            if kind == "HYPHEN":
                # The line ended inside a word. Vision normally leaves the printed
                # hyphen out of the text; when it kept it, the case rule decides.
                parts.append("-\n" if char in HYPHENS else char + SOFT_WRAP)
                continue
            parts.append(char)
            if kind in ("SPACE", "SURE_SPACE"):
                parts.append(" ")
            elif kind in ("EOL_SURE_SPACE", "LINE_BREAK"):
                parts.append("\n")
    raw = "".join(parts).strip()
    if not raw:
        return "", 0
    if raw.endswith(SOFT_WRAP):
        # The word goes on in the next column or on the next page.
        raw = raw[:-1] + "-"
    lines = raw.count("\n") + raw.count(SOFT_WRAP) + 1
    text = LINE_END_HYPHEN_RE.sub(_rejoin, raw.replace(SOFT_WRAP, ""))
    return " ".join(text.split()), lines


def _is_tall(char: str) -> bool:
    return len(char) == 1 and (char.isdigit() or char.isupper() or char in TALL_LOWERCASE)


def _glyph_size(paragraph: dict, frame: _Frame) -> tuple[float, int]:
    """The paragraph's type size, measured as the median height of its tall glyphs.

    Whole words are a poor ruler (``was`` is shorter than ``Bold``), but capitals,
    digits and ascenders all reach the same height in a given font size. Scripts
    without them fall back to all letters.
    """
    tall: list[float] = []
    other: list[float] = []
    for word in paragraph.get("words") or []:
        for symbol in word.get("symbols") or []:
            char = symbol.get("text", "")
            box = frame.box(symbol.get("boundingBox"))
            if not box or box[3] <= box[1]:
                continue
            if _is_tall(char):
                tall.append(box[3] - box[1])
            elif char.isalnum():
                other.append(box[3] - box[1])
    sample = tall or other
    return (statistics.median(sample), len(sample)) if sample else (0.0, 0)


def parse_response(result: dict, index: int, image_size: tuple[int, int],
                   points_per_pixel: float) -> Page:
    """Turn one AnnotateImageResponse into the compact page model that is cached."""
    raw: list[dict] = []
    for vision_page in (result.get("fullTextAnnotation") or {}).get("pages") or []:
        for block in vision_page.get("blocks") or []:
            if block.get("blockType") in SKIPPED_BLOCKS:
                continue
            raw.extend(block.get("paragraphs") or [])
    frame = _Frame(_text_direction(raw), *image_size)
    scale = points_per_pixel
    paragraphs: list[Paragraph] = []
    for item in raw:
        text, lines = _paragraph_text(item)
        box = frame.box(item.get("boundingBox"))
        if not text or box is None:
            continue
        size, weight = _glyph_size(item, frame)
        if not size:
            size = (box[3] - box[1]) / lines
        paragraphs.append(Paragraph(
            text=text,
            left=round(box[0] * scale, 2),
            top=round(box[1] * scale, 2),
            right=round(box[2] * scale, 2),
            bottom=round(box[3] * scale, 2),
            size=round(size * scale, 3),
            weight=weight,
            lines=lines,
            # A missing confidence is "not reported", not zero.
            confidence=round(float(item.get("confidence", 1.0)), 3),
        ))
    return Page(index=index, width=round(frame.width * scale, 2),
                height=round(frame.height * scale, 2), paragraphs=paragraphs)


# --------------------------------------------------------------------------- #
# The page model -> markdown
# --------------------------------------------------------------------------- #


@dataclass
class _Block:
    kind: str  # heading | text | note | dropcap
    text: str
    page: int
    ratio: float  # type size relative to the body text
    top: float
    bottom: float
    level: int = 0


def _weighted_median(pairs: list[tuple[float, int]]) -> float:
    items = sorted((value, weight) for value, weight in pairs if weight > 0)
    total = sum(weight for _, weight in items)
    running = 0
    for value, weight in items:
        running += weight
        if running * 2 >= total:
            return value
    return 0.0


def _page_body(page: Page, body: float) -> float:
    """This page's body-text size, so a scan whose pages vary in scale still works.

    Pages with little text (chapter openers, title pages) use the book's value.
    """
    pairs = [(paragraph.size, paragraph.weight) for paragraph in page.paragraphs]
    if sum(weight for _, weight in pairs) < 150:
        return body
    local = _weighted_median(pairs)
    return local if 0.8 * body <= local <= 1.25 * body else body


def _repeat_key(text: str) -> str:
    """What stays the same in a running head from page to page (not its number)."""
    words = [
        word for word in re.sub(r"[\W\d_]+", " ", text.lower()).split()
        if not re.fullmatch(ROMAN, word.upper())
    ]
    return " ".join(words) if sum(len(word) for word in words) >= 3 else ""


def _margin_paragraphs(page: Page, body: float) -> list[Paragraph]:
    """Short lines on the very top or bottom edge of the page, with nothing beyond them."""
    found = []
    for paragraph in page.paragraphs:
        if paragraph.size >= body * HEADING_RATIO or len(paragraph.text) > 90 or paragraph.lines > 2:
            continue
        slack = paragraph.size / 2
        others = [other for other in page.paragraphs if other is not paragraph]
        at_top = paragraph.top <= page.height * MARGIN_BAND and all(
            other.bottom > paragraph.top + slack for other in others
        )
        at_bottom = paragraph.bottom >= page.height * (1 - MARGIN_BAND) and all(
            other.top < paragraph.bottom - slack for other in others
        )
        if at_top or at_bottom:
            found.append(paragraph)
    return found


def _running_heads(pages: list[Page], body: float) -> set[int]:
    """ids of the running heads, running feet and page numbers to drop."""
    candidates = [(page.index, paragraph) for page in pages
                  for paragraph in _margin_paragraphs(page, body)]
    seen_on: dict[str, set[int]] = {}
    for page_index, paragraph in candidates:
        key = _repeat_key(paragraph.text)
        if key:
            seen_on.setdefault(key, set()).add(page_index)
    # Running heads repeat the book's or the chapter's title, so a margin line that
    # reads like a heading is one even when a short chapter shows it only once.
    headings = {
        f" {_repeat_key(paragraph.text)} " for page in pages for paragraph in page.paragraphs
        if paragraph.size >= body * HEADING_RATIO
    }
    dropped = set()
    for _, paragraph in candidates:
        key = _repeat_key(paragraph.text)
        if PAGE_NUMBER_RE.fullmatch(paragraph.text) or (key and (
            len(seen_on[key]) >= 2
            or (len(key) >= 5 and any(f" {key} " in heading for heading in headings))
        )):
            dropped.add(id(paragraph))
    return dropped


def _assign_levels(blocks: list[_Block]) -> None:
    """One heading level per distinct type size, the largest first."""
    levels: dict[float, int] = {}
    level, largest = 0, 0.0
    for ratio in sorted({block.ratio for block in blocks if block.kind == "heading"}, reverse=True):
        if not level or ratio < largest * LEVEL_STEP:
            level, largest = min(level + 1, 6), ratio
        levels[ratio] = level
    for block in blocks:
        if block.kind == "heading":
            block.level = levels[block.ratio]


def _escape(text: str) -> str:
    return MARKDOWN_SPECIAL_RE.sub(r"\\\1", text)


def _escape_paragraph(text: str) -> str:
    """Escape what pandoc would read as markup, including list markers at the start."""
    text = ORDERED_LIST_START_RE.sub(r"\1\\\2", _escape(text))
    return OTHER_BLOCK_START_RE.sub(lambda match: "\\" + match.group(0), text)


def _join(first: str, second: str) -> str:
    if len(first) > 1 and first[-1] in HYPHENS and first[-2].isalpha():
        return first[:-1] + second  # "exam-" + "ple"
    return f"{first} {second}"


def pages_to_markdown(pages: list[Page]) -> tuple[str, dict]:
    """Rebuild the book's markdown from its pages. Returns (markdown, statistics)."""
    for page in pages:
        page.paragraphs = [
            paragraph for paragraph in page.paragraphs
            if paragraph.confidence >= NOISE_CONFIDENCE or len(paragraph.text) >= NOISE_MAX_CHARS
        ]
    every = [paragraph for page in pages for paragraph in page.paragraphs]
    stats: dict[str, Any] = {
        "paragraphs": len(every), "body_pt": 0.0, "headings": {}, "running_heads": 0,
        "joined": 0, "drop_caps": 0,
    }
    body = _weighted_median([(paragraph.size, paragraph.weight) for paragraph in every])
    if not every or not body:
        return "", stats
    stats["body_pt"] = body
    dropped = _running_heads(pages, body)
    stats["running_heads"] = len(dropped)

    blocks: list[_Block] = []
    for page in pages:
        page_body = _page_body(page, body)
        kept = [paragraph for paragraph in page.paragraphs if id(paragraph) not in dropped]

        def small(paragraph: Paragraph) -> bool:
            return paragraph.size <= page_body * NOTE_RATIO

        for paragraph in kept:
            ratio = paragraph.size / page_body
            if (small(paragraph) and paragraph.top >= page.height / 2
                    and all(small(other) for other in kept if other.top > paragraph.top)):
                kind = "note"  # footnotes: small print with only small print below it
            elif (ratio >= HEADING_RATIO and len(paragraph.text) <= MAX_HEADING_CHARS
                  and paragraph.lines <= MAX_HEADING_LINES
                  and any(char.isalnum() for char in paragraph.text)):
                kind = "heading"
            else:
                kind = "text"
            blocks.append(_Block(kind, paragraph.text, page.index, ratio,
                                 paragraph.top, paragraph.bottom))

    # Drop caps: a large single letter that starts the paragraph after it. Found
    # before the levels are assigned so they do not claim a heading level.
    for block, following in zip(blocks, blocks[1:]):
        if (block.kind == "heading" and len(block.text) == 1 and block.text.isalpha()
                and following.kind == "text" and following.page == block.page
                and not (NUMERIC_LABEL_RE.fullmatch(block.text) and following.text[:1].isupper())):
            block.kind = "dropcap"
    _assign_levels(blocks)

    merged: list[_Block] = []
    for block in blocks:
        previous = merged[-1] if merged else None
        same_page = previous is not None and previous.page == block.page
        if same_page and previous.kind == "dropcap":
            block.text = previous.text + block.text
            merged[-1] = block
            stats["drop_caps"] += 1
            continue
        if (same_page and block.kind == "heading" and previous.kind in ("heading", "text")
                and len(previous.text) <= 30 and CHAPTER_LABEL_RE.fullmatch(previous.text)):
            # "CHAPTER 3" (or a big "3") above the chapter's title: one heading.
            label = previous.text.rstrip(".")
            joiner = ". " if NUMERIC_LABEL_RE.fullmatch(previous.text) else ": "
            block.text = label + joiner + block.text
            if previous.kind == "heading":
                block.level = min(block.level, previous.level)
            merged[-1] = block
            continue
        if (same_page and block.kind == "heading" and previous.kind == "heading"
                and block.level == previous.level
                and block.top - previous.bottom < (previous.bottom - previous.top)):
            # A title Vision split into one paragraph per line.
            previous.text = f"{previous.text} {block.text}"
            previous.bottom = block.bottom
            continue
        merged.append(block)

    # Paragraphs a page break, a column break or a footnote cut in two: the first
    # part does not end a sentence and the second starts in lower case.
    joined: list[_Block] = []
    open_text: _Block | None = None
    for block in merged:
        if (block.kind == "text" and open_text is not None
                and not SENTENCE_END_RE.search(open_text.text) and block.text[:1].islower()):
            open_text.text = _join(open_text.text, block.text)
            stats["joined"] += 1
            continue
        joined.append(block)
        if block.kind == "text":
            open_text = block
        elif block.kind != "note":
            open_text = None  # a heading ends the paragraph; footnotes do not

    lines = []
    for block in joined:
        if block.kind == "heading":
            lines.append(f"{'#' * block.level} {_escape(block.text)}")
            stats["headings"][block.level] = stats["headings"].get(block.level, 0) + 1
        else:
            lines.append(_escape_paragraph(block.text))
    return "\n\n".join(lines) + "\n", stats


# --------------------------------------------------------------------------- #
# The whole run
# --------------------------------------------------------------------------- #


def _read_page(index: int, image: bytes, size: tuple[int, int], points_per_pixel: float,
               credentials: Credentials, folder: Path) -> Page:
    result = annotate(image, credentials)
    # Counted here, not when the page is collected: a request still under way
    # when a run is cancelled is billed as well.
    record_usage(1)
    page = parse_response(result, index, size, points_per_pixel)
    _save_cached(folder, page)
    return page


def _read_pages(document: Any, indices: list[int], folder: Path, credentials: Credentials,
                pages: dict[int, Page | None], log: LogFn, should_stop: StopFn | None) -> None:
    """OCR the pages a few at a time; each result is counted and cached on arrival."""
    queue = list(indices)
    in_flight: set[Future] = set()
    done = 0
    executor = ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS, thread_name_prefix="vision")
    finished_cleanly = False
    try:
        while queue or in_flight:
            # Rendering happens here, in one thread: pdfium is not thread-safe.
            while queue and len(in_flight) < PARALLEL_REQUESTS * 2:
                if should_stop and should_stop():
                    raise VisionOcrError("cancelled")
                index = queue.pop(0)
                image, size, points_per_pixel = render_page(document[index])
                in_flight.add(executor.submit(
                    _read_page, index, image, size, points_per_pixel, credentials, folder
                ))
            if should_stop and should_stop():
                raise VisionOcrError("cancelled")
            finished, in_flight = wait(in_flight, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in finished:
                page = future.result()
                pages[page.index] = page
                done += 1
                if done % 10 == 0 or done == len(indices):
                    log(f"   ... {done}/{len(indices)} pages read")
        finished_cleanly = True
    finally:
        # On an error or a cancel, requests already under way finish in the
        # background; their pages still land in the cache.
        executor.shutdown(wait=finished_cleanly, cancel_futures=True)


def ocr_pdf(
    pdf: Path,
    work: Path,
    log: LogFn,
    should_stop: StopFn | None = None,
    page_range: str | None = None,
    monthly_limit: int = FREE_PAGES_PER_MONTH,
) -> Path:
    """OCR a scanned PDF with Cloud Vision and return the markdown file it wrote.

    ``work`` holds one folder per PDF (named after its content) with the cached
    result of every page, so only pages that were never read before are sent.
    ``monthly_limit`` caps the pages sent per calendar month; 0 switches it off.
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf)
    try:
        indices = parse_page_range(page_range, len(document))
        folder = work / _fingerprint(pdf)
        folder.mkdir(parents=True, exist_ok=True)
        pages: dict[int, Page | None] = {index: _load_cached(folder, index) for index in indices}
        missing = [index for index in indices if pages[index] is None]
        log("   pipeline:        Google Cloud Vision OCR (no GPU)")
        log(f"   pages:           {len(indices)} to read, "
            f"{len(indices) - len(missing)} already cached from an earlier run")
        if missing:
            credentials = find_credentials()
            if credentials is None:
                raise VisionOcrError("no Google Cloud credentials found. " + SETUP_HINT)
            used = pages_used_this_month()
            log(f"   credentials:     {credentials.source}")
            log(f"   this month:      {used} page(s) sent from this computer so far"
                + (f" (limit {monthly_limit})" if monthly_limit > 0 else " (no limit set)"))
            if monthly_limit > 0 and used + len(missing) > monthly_limit:
                raise VisionOcrError(
                    f"this book needs {len(missing)} page(s) read by Google Cloud Vision, "
                    f"which would take this month's count from {used} past the limit of "
                    f"{monthly_limit} pages (counted on this computer). Nothing was sent.\n"
                    f"The first {FREE_PAGES_PER_MONTH} pages a month are free, after that "
                    "Google charges about US$1.50 per 1,000 pages. To go ahead anyway, raise "
                    "the limit (--vision-monthly-limit, or the field in the UI), or set it to "
                    "0 to switch the check off."
                )
            if used + len(missing) > FREE_PAGES_PER_MONTH:
                log(f"   note: this goes past the {FREE_PAGES_PER_MONTH} free pages a month; "
                    "Google bills the rest (about US$1.50 per 1,000 pages)")
            _read_pages(document, missing, folder, credentials, pages, log, should_stop)
    finally:
        document.close()

    markdown, stats = pages_to_markdown([page for page in (pages[i] for i in indices) if page])
    if not stats["paragraphs"]:
        raise VisionOcrError("Google Cloud Vision found no text on these pages")
    headings = ", ".join(f"h{level}: {count}" for level, count in sorted(stats["headings"].items()))
    log(f"   text:            {stats['paragraphs']} paragraphs, body text "
        f"{stats['body_pt']:.1f} pt (capital height)")
    log(f"   headings by size: {headings or 'none found'}")
    if stats["running_heads"]:
        log(f"   removed {stats['running_heads']} running head(s), feet and page number(s)")
    if stats["joined"]:
        log(f"   joined {stats['joined']} paragraph(s) split by a page or column break")
    if stats["drop_caps"]:
        log(f"   attached {stats['drop_caps']} drop cap(s) to their paragraph")
    target = folder / f"{pdf.stem}.md"
    target.write_text(markdown, encoding="utf-8")
    return target
