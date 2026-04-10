import re
import logging

from nltk.stem import SnowballStemmer

logger = logging.getLogger(__name__)

# ---- Synonym dictionary (A5) ----
# Common ecommerce synonyms: each tuple is a synonym group.
# The first term is canonical; all others are replaced with it.
SYNONYM_GROUPS: list[tuple[str, ...]] = [
    ("sofa", "couch"),
    ("nightstand", "bedside table", "night stand"),
    ("lamp", "light"),
    ("rug", "carpet"),
    ("dresser", "chest of drawers"),
    ("curtain", "drape"),
    ("wardrobe", "armoire"),
    ("bookshelf", "bookcase"),
    ("ottoman", "footstool", "pouf"),
    ("loveseat", "love seat"),
    ("bureau", "desk"),
    ("credenza", "sideboard", "buffet"),
    ("throw pillow", "accent pillow", "toss pillow"),
    ("comforter", "duvet"),
    ("faucet", "tap"),
    ("countertop", "counter top"),
]

# Build a replacement map: variant -> canonical
_SYNONYM_MAP: dict[str, str] = {}
for group in SYNONYM_GROUPS:
    canonical = group[0]
    for variant in group[1:]:
        _SYNONYM_MAP[variant] = canonical


def _build_synonym_pattern() -> re.Pattern | None:
    if not _SYNONYM_MAP:
        return None
    # Sort by length descending so longer phrases match first
    variants = sorted(_SYNONYM_MAP.keys(), key=len, reverse=True)
    pattern = "|".join(re.escape(v) for v in variants)
    return re.compile(pattern, re.IGNORECASE)


_SYNONYM_RE = _build_synonym_pattern()


# ---- Text preprocessor (A1) ----

_stemmer = SnowballStemmer("english")

# Fix spaced apostrophes: "it ' s" -> "it's", "don ' t" -> "don't"
_SPACED_APOS_RE = re.compile(r"\b(\w+) ' (s|t|re|ve|ll|d|m)\b", re.IGNORECASE)

# Normalize '' (double single-quotes for inches) -> "inch"
_INCHES_RE = re.compile(r"(\d+)\s*''")

# Strip non-alphanumeric except spaces and hyphens
_PUNCT_RE = re.compile(r"[^a-z0-9\s\-]")

# Multiple spaces
_MULTI_SPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str, *, stem: bool = True, synonyms: bool = True) -> str:
    """Clean and normalize text for search indexing/querying."""
    t = text.lower()

    # Fix spaced apostrophes
    t = _SPACED_APOS_RE.sub(r"\1'\2", t)

    # Normalize inches notation
    t = _INCHES_RE.sub(r"\1 inch", t)

    # Strip punctuation (keep hyphens and spaces)
    t = _PUNCT_RE.sub(" ", t)

    # Collapse whitespace
    t = _MULTI_SPACE_RE.sub(" ", t).strip()

    # Synonym expansion
    if synonyms and _SYNONYM_RE:
        t = _SYNONYM_RE.sub(lambda m: _SYNONYM_MAP[m.group(0).lower()], t)

    # Stemming
    if stem:
        t = " ".join(_stemmer.stem(w) for w in t.split())

    return t


def make_preprocessor(
    stem: bool = True, synonyms: bool = True
) -> "Callable[[str], str]":
    """Return a preprocessor function with the given options baked in."""

    def _preprocess(text: str) -> str:
        return preprocess_text(text, stem=stem, synonyms=synonyms)

    return _preprocess


# ---- Product features parser (A2) ----

_FEATURE_SEP_RE = re.compile(r"\|")
_FEATURE_KV_RE = re.compile(r"^([^:]+):(.+)$")


def parse_product_features(features: str) -> str:
    """Parse pipe-separated key:value product features into clean text.

    Input:  'overallwidth-sidetoside:64.7|dsprimaryproductstyle:modern|...'
    Output: 'overallwidth sidetoside 64.7 dsprimaryproductstyle modern ...'
    """
    if not isinstance(features, str) or not features.strip():
        return ""

    parts = []
    for pair in _FEATURE_SEP_RE.split(features):
        pair = pair.strip()
        match = _FEATURE_KV_RE.match(pair)
        if match:
            key = match.group(1).strip().replace("-", " ")
            val = match.group(2).strip()
            parts.append(f"{key} {val}")
        elif pair:
            parts.append(pair)

    return " ".join(parts)
