import logging
from typing import Literal

import numpy as np
import pandas as pd

from search_engine import SearchEngine

logger = logging.getLogger(__name__)

# Graded relevance mapping for NDCG
RELEVANCE_GRADES: dict[str, int] = {
    "Exact": 2,
    "Partial": 1,
    "Irrelevant": 0,
}


class Evaluator:
    """Evaluates search engine performance against labelled ground truth.

    Supports MAP@K (binary, Exact-only) and NDCG@K (graded relevance
    using Exact=2, Partial=1, Irrelevant=0).
    """

    # ------------------------------------------------------------------
    # Metric computations (static, reusable outside the class)
    # ------------------------------------------------------------------

    @staticmethod
    def map_at_k(true_ids: np.ndarray, predicted_ids: list[int], k: int = 10) -> float:
        """Mean Average Precision @ K (binary relevance: Exact only).

        Identical to the original notebook implementation, preserved for
        backward-compatibility and baseline comparison.
        """
        if not len(true_ids) or not len(predicted_ids):
            return 0.0

        score = 0.0
        num_hits = 0.0

        for i, p_id in enumerate(predicted_ids[:k]):
            if p_id in true_ids and p_id not in predicted_ids[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(true_ids), k)

    @staticmethod
    def ndcg_at_k(
        predicted_ids: list[int],
        relevance_lookup: dict[int, int],
        k: int = 10,
    ) -> float:
        """Normalized Discounted Cumulative Gain @ K.

        Parameters
        ----------
        predicted_ids : list[int]
            Ranked list of product IDs returned by the search engine.
        relevance_lookup : dict[int, int]
            Mapping from product_id -> relevance grade (2/1/0) for the
            current query.  Products absent from the dict are assumed
            irrelevant (grade 0).
        k : int
            Cutoff rank.

        Returns
        -------
        float
            NDCG@K score in [0, 1].  Returns 0 if no relevant documents
            exist for the query.
        """
        gains = [relevance_lookup.get(pid, 0) for pid in predicted_ids[:k]]

        # DCG
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

        # Ideal gains: sort all known relevance grades descending
        ideal_gains = sorted(relevance_lookup.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    # ------------------------------------------------------------------
    # Full evaluation pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _build_relevance_lookup(
        label_group: pd.DataFrame,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Return (exact_ids, {product_id: grade}) for one query group."""
        exact_ids = label_group.loc[
            label_group["label"] == "Exact", "product_id"
        ].values

        relevance_lookup = {
            row["product_id"]: RELEVANCE_GRADES.get(row["label"], 0)
            for _, row in label_group.iterrows()
        }
        return exact_ids, relevance_lookup

    @classmethod
    def evaluate_queries(
        cls,
        engine: SearchEngine,
        product_df: pd.DataFrame,
        query_df: pd.DataFrame,
        label_df: pd.DataFrame,
        k: int = 10,
        label_filter: Literal["Exact", "Exact+Partial"] = "Exact",
    ) -> pd.DataFrame:
        """Run *engine* on every query and compute per-query metrics.

        Parameters
        ----------
        engine : SearchEngine
            A fitted search engine instance.
        product_df : pd.DataFrame
            Product catalogue (must match the one used to fit *engine*).
        query_df : pd.DataFrame
            Queries with ``query_id`` and ``query`` columns.
        label_df : pd.DataFrame
            Ground-truth labels with ``query_id``, ``product_id``, ``label``.
        k : int
            Cutoff for MAP@K and NDCG@K.
        label_filter : str
            Which labels count as "relevant" for MAP.
            ``"Exact"`` (default) matches the original notebook.
            ``"Exact+Partial"`` also treats partial matches as relevant.

        Returns
        -------
        pd.DataFrame
            Copy of *query_df* with added columns:
            ``top_product_ids``, ``relevant_ids``, ``map@k``, ``ndcg@k``.
        """
        grouped = label_df.groupby("query_id")
        result = query_df.copy()

        map_scores = []
        ndcg_scores = []
        top_ids_list = []
        relevant_ids_list = []

        for _, row in result.iterrows():
            qid = row["query_id"]
            query = row["query"]

            # Retrieve top-K product IDs
            indices = engine.search(query, top_n=k)
            top_product_ids = product_df.iloc[indices]["product_id"].tolist()
            top_ids_list.append(top_product_ids)

            # Ground truth for this query
            if qid in grouped.groups:
                label_group = grouped.get_group(qid)
                exact_ids, relevance_lookup = cls._build_relevance_lookup(label_group)
            else:
                logger.warning("No labels found for query_id=%s", qid)
                exact_ids = np.array([])
                relevance_lookup = {}

            # Decide which IDs are "relevant" for MAP
            if label_filter == "Exact+Partial":
                relevant = label_group.loc[
                    label_group["label"].isin(["Exact", "Partial"]), "product_id"
                ].values if qid in grouped.groups else np.array([])
            else:
                relevant = exact_ids

            relevant_ids_list.append(relevant)
            map_scores.append(cls.map_at_k(relevant, top_product_ids, k=k))
            ndcg_scores.append(
                cls.ndcg_at_k(top_product_ids, relevance_lookup, k=k)
            )

        result["top_product_ids"] = top_ids_list
        result["relevant_ids"] = relevant_ids_list
        result["map@k"] = map_scores
        result["ndcg@k"] = ndcg_scores

        logger.info(
            "%s — MAP@%d: %.4f | NDCG@%d: %.4f",
            engine.name,
            k,
            result["map@k"].mean(),
            k,
            result["ndcg@k"].mean(),
        )
        return result
