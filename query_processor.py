"""Query expansion and understanding module (Phase D).

Provides two expansion strategies using the OpenAI API (GPT-4o-mini):
- ``LLMQueryExpander`` — generates synonyms / reformulations.
- ``LLMIntentExpander`` — extracts structured intent (category, material, colour,
  style) and appends the extracted attributes to the query.

Expansions are cached to a local JSON file so repeated runs don't hit the API.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / ".cache"

# Load .env file if present (no extra dependency needed)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


# ====================================================================== #
# Abstract base                                                          #
# ====================================================================== #


class QueryExpander(ABC):
    """Base class for query expansion strategies."""

    @abstractmethod
    def expand(self, query: str) -> str:
        """Return an expanded version of *query*."""


def _get_openai_client():
    """Return an OpenAI client, importing lazily."""
    from openai import OpenAI

    return OpenAI()  # uses OPENAI_API_KEY env var


# ====================================================================== #
# Persistent JSON cache                                                  #
# ====================================================================== #


class _JsonCache:
    """Simple JSON-file backed dict cache."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, str] = {}
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info("Loaded %d cached entries from %s", len(self._data), self._path.name)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d entries to %s", len(self._data), self._path.name)


# ====================================================================== #
# D1 — LLM synonym expansion (GPT)                                      #
# ====================================================================== #


class LLMQueryExpander(QueryExpander):
    """Generate search synonyms and reformulations via GPT-4o-mini.

    Results are cached to ``.cache/synonym_cache.json`` so repeated runs
    don't call the API.  Set the ``OPENAI_API_KEY`` environment variable.

    Parameters
    ----------
    model : str
        OpenAI model name (default: ``gpt-4o-mini``).
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
        self._cache = _JsonCache(_CACHE_DIR / "synonym_cache.json")

    def _ensure_client(self):
        if self._client is None:
            self._client = _get_openai_client()

    def expand(self, query: str) -> str:
        key = query.strip()
        if key in self._cache:
            return self._cache[key]

        self._ensure_client()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful e-commerce search assistant. "
                        "Given a search query, output a short list of synonyms "
                        "and related product terms (no explanation, just terms "
                        "separated by commas). Keep it under 30 words."
                    ),
                },
                {"role": "user", "content": key},
            ],
            temperature=0,
            max_tokens=80,
        )
        output = response.choices[0].message.content.strip()
        expanded = f"{key} {output}" if output else key
        self._cache[key] = expanded
        logger.debug("LLMQueryExpander — '%s' → '%s'", key, expanded)
        return expanded

    def save_cache(self) -> None:
        """Persist the expansion cache to disk."""
        self._cache.save()

    def pre_expand(self, queries) -> None:
        """Expand all queries and save the cache."""
        for q in queries:
            self.expand(q)
        self.save_cache()


# ====================================================================== #
# D2 — LLM structured intent extraction (GPT)                           #
# ====================================================================== #

_ATTR_RE = re.compile(
    r"(?:category|material|color|colour|style|brand)\s*:\s*(.+)",
    re.IGNORECASE,
)


class LLMIntentExpander(QueryExpander):
    """Extract structured attributes and append them to the query.

    Uses GPT-4o-mini to identify category, material, colour, style, and
    brand from the raw query, then appends any extracted values as
    additional retrieval tokens.

    Parameters
    ----------
    model : str
        OpenAI model name (default: ``gpt-4o-mini``).
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
        self._cache = _JsonCache(_CACHE_DIR / "intent_cache.json")

    def _ensure_client(self):
        if self._client is None:
            self._client = _get_openai_client()

    def expand(self, query: str) -> str:
        key = query.strip()
        if key in self._cache:
            return self._cache[key]

        self._ensure_client()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an e-commerce product search expert. "
                        "Given a search query, extract product attributes. "
                        "Return ONLY a comma-separated list of attribute values "
                        "(category, material, color, style, brand) that are "
                        "clearly implied. No labels, no explanation. "
                        "If nothing is clearly implied, return the query unchanged."
                    ),
                },
                {"role": "user", "content": key},
            ],
            temperature=0,
            max_tokens=80,
        )
        output = response.choices[0].message.content.strip()
        expanded = f"{key} {output}" if output and output.lower() != key.lower() else key
        self._cache[key] = expanded
        logger.debug("LLMIntentExpander — '%s' → '%s'", key, expanded)
        return expanded

    def save_cache(self) -> None:
        """Persist the extraction cache to disk."""
        self._cache.save()

    def pre_expand(self, queries) -> None:
        """Expand all queries and save the cache."""
        for q in queries:
            self.expand(q)
        self.save_cache()
