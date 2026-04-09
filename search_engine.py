import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SearchEngine(ABC):
    """Abstract base class for all search engine implementations.

    Subclasses must implement ``fit`` to build an index from a product
    DataFrame and ``search_with_scores`` to retrieve the top-N product
    *indices* with their scores for a given query string.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this engine (used in logs and reports)."""

    @abstractmethod
    def fit(self, product_df: pd.DataFrame) -> None:
        """Build the search index from a product DataFrame."""

    @abstractmethod
    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        """Return the top-N (index, score) pairs for *query*."""

    def search(self, query: str, top_n: int = 10) -> list[int]:
        """Return the top-N product indices for *query*."""
        return [idx for idx, _ in self.search_with_scores(query, top_n)]

    # ---- helpers shared by all engines ----

    def _combine_columns(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.Series:
        """Concatenate *columns* into a single text series, filling NaN."""
        combined = df[columns[0]].fillna("")
        for col in columns[1:]:
            combined = combined + " " + df[col].fillna("")
        return combined.astype(str)


class TfidfSearchEngine(SearchEngine):
    """TF-IDF based search engine using cosine similarity.

    Parameters
    ----------
    columns : list[str]
        Product DataFrame columns to combine into the indexed text.
    vectorizer_kwargs : dict | None
        Extra keyword arguments forwarded to ``TfidfVectorizer``.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        vectorizer_kwargs: dict[str, Any] | None = None,
        preprocessor: Callable[[str], str] | None = None,
        name_override: str | None = None,
    ):
        self.columns = columns or ["product_name", "product_description"]
        self.vectorizer_kwargs = vectorizer_kwargs or {}
        self.preprocessor = preprocessor
        self._name_override = name_override
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None

    @property
    def name(self) -> str:
        return self._name_override or f"TF-IDF ({', '.join(self.columns)})"

    def fit(self, product_df: pd.DataFrame) -> None:
        combined = self._combine_columns(product_df, self.columns)
        if self.preprocessor:
            combined = combined.map(self.preprocessor)
        self._vectorizer = TfidfVectorizer(**self.vectorizer_kwargs)
        self._tfidf_matrix = self._vectorizer.fit_transform(combined)
        logger.info(
            "%s — indexed %d documents, vocabulary size %d",
            self.name,
            self._tfidf_matrix.shape[0],
            len(self._vectorizer.vocabulary_),
        )

    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        if self._vectorizer is None or self._tfidf_matrix is None:
            raise RuntimeError("Call fit() before search().")

        t0 = time.perf_counter()
        q = self.preprocessor(query) if self.preprocessor else query
        query_vec = self._vectorizer.transform([q])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        result = [(int(i), float(scores[i])) for i in top_indices]
        elapsed = time.perf_counter() - t0
        logger.debug("%s — query '%s' in %.4fs", self.name, query, elapsed)
        return result


class BM25SearchEngine(SearchEngine):
    """BM25 (Okapi) search engine.

    Parameters
    ----------
    columns : list[str]
        Product DataFrame columns to combine into the indexed text.
    tokenizer : callable | None
        Tokenizer function applied to each document and query.
        Defaults to ``str.lower().split()``.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        preprocessor: Callable[[str], str] | None = None,
        name_override: str | None = None,
    ):
        self.columns = columns or ["product_name", "product_description"]
        self.tokenizer = tokenizer or self._default_tokenizer
        self.preprocessor = preprocessor
        self._name_override = name_override
        self._bm25: BM25Okapi | None = None

    @property
    def name(self) -> str:
        return self._name_override or f"BM25 ({', '.join(self.columns)})"

    @staticmethod
    def _default_tokenizer(text: str) -> list[str]:
        return text.lower().split()

    def fit(self, product_df: pd.DataFrame) -> None:
        combined = self._combine_columns(product_df, self.columns)
        if self.preprocessor:
            combined = combined.map(self.preprocessor)
        corpus = [self.tokenizer(doc) for doc in combined]
        self._bm25 = BM25Okapi(corpus)
        logger.info(
            "%s — indexed %d documents",
            self.name,
            len(corpus),
        )

    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("Call fit() before search().")

        t0 = time.perf_counter()
        q = self.preprocessor(query) if self.preprocessor else query
        tokenized_query = self.tokenizer(q)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_n:][::-1]
        result = [(int(i), float(scores[i])) for i in top_indices]
        elapsed = time.perf_counter() - t0
        logger.debug("%s — query '%s' in %.4fs", self.name, query, elapsed)
        return result


class SentenceTransformerSearchEngine(SearchEngine):
    """Dense semantic search using sentence-transformers.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
    columns : list[str]
        Product DataFrame columns to combine for encoding.
    batch_size : int
        Batch size for encoding products.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        columns: list[str] | None = None,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.columns = columns or ["product_name", "product_description"]
        self.batch_size = batch_size
        self._model = None
        self._embeddings: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"SentenceTransformer ({self.model_name})"

    def fit(self, product_df: pd.DataFrame) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        combined = self._combine_columns(product_df, self.columns).tolist()
        logger.info(
            "%s — encoding %d documents (batch_size=%d) ...",
            self.name,
            len(combined),
            self.batch_size,
        )
        t0 = time.perf_counter()
        self._embeddings = self._model.encode(
            combined, batch_size=self.batch_size, show_progress_bar=True
        )
        elapsed = time.perf_counter() - t0
        logger.info("%s — encoding finished in %.1fs", self.name, elapsed)

    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        if self._model is None or self._embeddings is None:
            raise RuntimeError("Call fit() before search().")

        t0 = time.perf_counter()
        query_emb = self._model.encode([query])
        scores = cosine_similarity(query_emb, self._embeddings).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        result = [(int(i), float(scores[i])) for i in top_indices]
        elapsed = time.perf_counter() - t0
        logger.debug("%s — query '%s' in %.4fs", self.name, query, elapsed)
        return result


class WeightedTfidfSearchEngine(SearchEngine):
    """TF-IDF with per-field weighting.

    Fits a separate ``TfidfVectorizer`` for each column and combines
    the cosine-similarity scores using the supplied weights.

    Parameters
    ----------
    column_weights : dict[str, float]
        Mapping of column name -> weight (e.g. ``{"product_name": 3.0}``).
    vectorizer_kwargs : dict | None
        Shared kwargs forwarded to every ``TfidfVectorizer``.
    preprocessor : callable | None
        Text preprocessing function applied to each field.
    name_override : str | None
        Custom display name for this engine.
    """

    def __init__(
        self,
        column_weights: dict[str, float],
        vectorizer_kwargs: dict[str, Any] | None = None,
        preprocessor: Callable[[str], str] | None = None,
        name_override: str | None = None,
    ):
        self.column_weights = column_weights
        self.vectorizer_kwargs = vectorizer_kwargs or {}
        self.preprocessor = preprocessor
        self._name_override = name_override
        self._vectorizers: dict[str, TfidfVectorizer] = {}
        self._matrices: dict[str, Any] = {}

    @property
    def name(self) -> str:
        if self._name_override:
            return self._name_override
        parts = [f"{c}×{w}" for c, w in self.column_weights.items()]
        return f"Weighted TF-IDF ({', '.join(parts)})"

    def fit(self, product_df: pd.DataFrame) -> None:
        for col in self.column_weights:
            text = product_df[col].fillna("").astype(str)
            if self.preprocessor:
                text = text.map(self.preprocessor)
            vec = TfidfVectorizer(**self.vectorizer_kwargs)
            self._matrices[col] = vec.fit_transform(text)
            self._vectorizers[col] = vec
        total_docs = next(iter(self._matrices.values())).shape[0]
        logger.info("%s — indexed %d documents across %d fields",
                    self.name, total_docs, len(self.column_weights))

    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        if not self._vectorizers:
            raise RuntimeError("Call fit() before search().")

        t0 = time.perf_counter()
        q = self.preprocessor(query) if self.preprocessor else query
        n_docs = next(iter(self._matrices.values())).shape[0]
        combined_scores = np.zeros(n_docs)

        for col, weight in self.column_weights.items():
            qvec = self._vectorizers[col].transform([q])
            scores = cosine_similarity(qvec, self._matrices[col]).flatten()
            combined_scores += weight * scores

        top_indices = combined_scores.argsort()[-top_n:][::-1]
        result = [(int(i), float(combined_scores[i])) for i in top_indices]
        elapsed = time.perf_counter() - t0
        logger.debug("%s — query '%s' in %.4fs", self.name, query, elapsed)
        return result
