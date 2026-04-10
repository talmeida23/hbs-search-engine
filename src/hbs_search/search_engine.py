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
            combined, batch_size=self.batch_size, show_progress_bar=False
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


class HybridSearchEngine(SearchEngine):
    """Score-fusion hybrid that blends two engines.

    Supports two fusion strategies:

    * **weighted** (default): min-max normalises each engine's scores and
      computes ``alpha * scores_a + (1 - alpha) * scores_b``.
    * **rrf** (Reciprocal Rank Fusion): parameter-free fusion using
      ``1 / (k + rank)`` for each engine, then summed.

    Both engines must already be fitted before calling :meth:`fit` on
    the hybrid (which is a no-op).

    Parameters
    ----------
    engine_a, engine_b : SearchEngine
        Two fitted engines to fuse.
    alpha : float
        Blending weight for *engine_a* when ``strategy="weighted"``.
        Ignored when ``strategy="rrf"``.
    strategy : str
        ``"weighted"`` or ``"rrf"``.
    retrieve_n : int
        How many candidates to pull from each engine before fusion.
    name_override : str | None
        Custom display name.
    """

    def __init__(
        self,
        engine_a: SearchEngine,
        engine_b: SearchEngine,
        alpha: float = 0.5,
        strategy: str = "weighted",
        retrieve_n: int = 100,
        name_override: str | None = None,
    ):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.alpha = alpha
        self.strategy = strategy
        self.retrieve_n = retrieve_n
        self._name_override = name_override

    @property
    def name(self) -> str:
        if self._name_override:
            return self._name_override
        return (
            f"Hybrid({self.engine_a.name} + {self.engine_b.name}, "
            f"α={self.alpha}, {self.strategy})"
        )

    def fit(self, product_df: pd.DataFrame) -> None:
        # Both sub-engines should already be fitted.
        logger.info("%s — using pre-fitted sub-engines", self.name)

    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        t0 = time.perf_counter()

        results_a = self.engine_a.search_with_scores(query, self.retrieve_n)
        results_b = self.engine_b.search_with_scores(query, self.retrieve_n)

        if self.strategy == "rrf":
            fused = self._rrf(results_a, results_b)
        else:
            fused = self._weighted(results_a, results_b, self.alpha)

        # Sort descending by fused score, take top_n
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result = [(idx, score) for idx, score in ranked]
        elapsed = time.perf_counter() - t0
        logger.debug("%s — query '%s' in %.4fs", self.name, query, elapsed)
        return result

    # ------------------------------------------------------------------ #

    @staticmethod
    def _minmax(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return scores
        lo = min(scores.values())
        hi = max(scores.values())
        rng = hi - lo
        if rng == 0:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / rng for k, v in scores.items()}

    @classmethod
    def _weighted(
        cls,
        results_a: list[tuple[int, float]],
        results_b: list[tuple[int, float]],
        alpha: float,
    ) -> dict[int, float]:
        scores_a = cls._minmax(dict(results_a))
        scores_b = cls._minmax(dict(results_b))
        all_ids = set(scores_a) | set(scores_b)
        return {
            idx: alpha * scores_a.get(idx, 0.0)
            + (1 - alpha) * scores_b.get(idx, 0.0)
            for idx in all_ids
        }

    @staticmethod
    def _rrf(
        results_a: list[tuple[int, float]],
        results_b: list[tuple[int, float]],
        k: int = 60,
    ) -> dict[int, float]:
        fused: dict[int, float] = {}
        for rank, (idx, _) in enumerate(results_a, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(results_b, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
        return fused


# ====================================================================== #
# Phase C: Re-ranking engines                                            #
# ====================================================================== #


class ReRankingEngine(SearchEngine):
    """Two-stage retrieve-then-rerank engine.

    A first-stage *retriever* fetches ``retrieve_n`` candidates, then a
    *reranker* callable rescores them and the top ``top_n`` are returned.

    Parameters
    ----------
    retriever : SearchEngine
        Fitted first-stage engine.
    reranker : callable
        ``reranker(query, candidates, product_df) -> list[tuple[int, float]]``
        where *candidates* is ``list[tuple[int, float]]`` from the retriever.
    retrieve_n : int
        Number of candidates passed to the reranker.
    name_override : str | None
        Custom display name.
    """

    def __init__(
        self,
        retriever: SearchEngine,
        reranker: Callable,
        retrieve_n: int = 100,
        name_override: str | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_n = retrieve_n
        self._name_override = name_override
        self._product_df: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        if self._name_override:
            return self._name_override
        return f"ReRank({self.retriever.name})"

    def fit(self, product_df: pd.DataFrame) -> None:
        self._product_df = product_df
        logger.info("%s — ready (retriever already fitted)", self.name)

    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        t0 = time.perf_counter()
        candidates = self.retriever.search_with_scores(query, self.retrieve_n)
        reranked = self.reranker(query, candidates, self._product_df)
        result = reranked[:top_n]
        elapsed = time.perf_counter() - t0
        logger.debug("%s — query '%s' in %.4fs", self.name, query, elapsed)
        return result


class CrossEncoderReranker:
    """Reranker using a cross-encoder model from sentence-transformers.

    Scores each ``(query, document_text)`` pair and re-sorts by
    cross-encoder score.  Intended for use with :class:`ReRankingEngine`.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID, e.g. ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
    columns : list[str]
        Product columns to concatenate as the document text.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        columns: list[str] | None = None,
    ):
        self.model_name = model_name
        self.columns = columns or ["product_name", "product_description"]
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("CrossEncoderReranker — loaded %s", self.model_name)

    def release_model(self):
        """Free the cross-encoder model from memory (lazy-reloads on next call)."""
        self._model = None

    def __call__(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        product_df: pd.DataFrame,
    ) -> list[tuple[int, float]]:
        self._ensure_model()
        pairs = []
        idxs = []
        for idx, _ in candidates:
            doc = " ".join(
                str(product_df.at[idx, c]) for c in self.columns if c in product_df.columns
            )
            pairs.append((query, doc))
            idxs.append(idx)
        scores = self._model.predict(pairs, show_progress_bar=False)
        scored = list(zip(idxs, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ====================================================================== #
# Phase D: Query-expanded engine wrapper                                 #
# ====================================================================== #


class QueryExpandedSearchEngine(SearchEngine):
    """Wraps any SearchEngine, applying query expansion before retrieval.

    The *expander* is a callable with signature ``expand(query) -> str``
    (any :class:`query_processor.QueryExpander` instance works).

    Parameters
    ----------
    engine : SearchEngine
        A **fitted** inner search engine.
    expander : callable
        ``expander.expand(query) -> expanded_query``.
    name_override : str | None
        Custom display name.
    """

    def __init__(
        self,
        engine: "SearchEngine",
        expander: Any,
        name_override: str | None = None,
    ):
        self.engine = engine
        self.expander = expander
        self._name_override = name_override

    @property
    def name(self) -> str:
        if self._name_override:
            return self._name_override
        return f"Expanded({self.engine.name})"

    def fit(self, product_df: pd.DataFrame) -> None:
        # Inner engine is assumed already fitted.
        logger.info("%s — ready (inner engine already fitted)", self.name)

    def search_with_scores(
        self, query: str, top_n: int = 10
    ) -> list[tuple[int, float]]:
        expanded = self.expander.expand(query)
        return self.engine.search_with_scores(expanded, top_n)


class MetadataBoostReranker:
    """Reranker that boosts retrieval scores with product metadata.

    ``final = retrieval_score_norm + beta * rating_norm + gamma * popularity_norm``

    where ``popularity_norm`` is log-scaled review_count.

    Parameters
    ----------
    beta : float
        Weight for normalised average_rating.
    gamma : float
        Weight for normalised log(review_count + 1).
    rating_col : str
        Column name for average rating.
    review_col : str
        Column name for review count.
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma: float = 0.05,
        rating_col: str = "average_rating",
        review_col: str = "review_count",
    ):
        self.beta = beta
        self.gamma = gamma
        self.rating_col = rating_col
        self.review_col = review_col

    def __call__(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        product_df: pd.DataFrame,
    ) -> list[tuple[int, float]]:
        if not candidates:
            return candidates

        # Min-max normalise retrieval scores
        scores = {idx: s for idx, s in candidates}
        lo, hi = min(scores.values()), max(scores.values())
        rng = hi - lo if hi != lo else 1.0
        norm_scores = {idx: (s - lo) / rng for idx, s in scores.items()}

        # Pre-compute rating/popularity normalisation bounds from candidates
        ratings = []
        popularities = []
        for idx, _ in candidates:
            ratings.append(float(product_df.at[idx, self.rating_col] or 0))
            popularities.append(np.log1p(float(product_df.at[idx, self.review_col] or 0)))

        r_lo, r_hi = min(ratings), max(ratings)
        r_rng = r_hi - r_lo if r_hi != r_lo else 1.0
        p_lo, p_hi = min(popularities), max(popularities)
        p_rng = p_hi - p_lo if p_hi != p_lo else 1.0

        boosted = []
        for i, (idx, _) in enumerate(candidates):
            r_norm = (ratings[i] - r_lo) / r_rng
            p_norm = (popularities[i] - p_lo) / p_rng
            final = norm_scores[idx] + self.beta * r_norm + self.gamma * p_norm
            boosted.append((idx, final))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted
