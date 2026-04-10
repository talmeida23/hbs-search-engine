"""Run a compact baseline vs improved lexical comparison."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hbs_search.data_loader import WANDSDataLoader
from hbs_search.evaluation import Evaluator
from hbs_search.preprocessing import make_preprocessor, parse_product_features
from hbs_search.search_engine import BM25SearchEngine, TfidfSearchEngine


def main() -> None:
    loader = WANDSDataLoader()
    loader.ensure_data()
    product_df = loader.load_products().copy()
    query_df = loader.load_queries()
    label_df = loader.load_labels()

    product_df["product_features_parsed"] = product_df["product_features"].map(
        parse_product_features
    )
    preprocessor = make_preprocessor(stem=True, synonyms=True)

    engines = [
        TfidfSearchEngine(
            columns=["product_name", "product_description"],
            name_override="TF-IDF baseline",
        ),
        BM25SearchEngine(
            columns=[
                "product_name",
                "product_description",
                "product_class",
                "category_hierarchy",
                "product_features_parsed",
            ],
            preprocessor=preprocessor,
            name_override="BM25 enriched + stem/syn",
        ),
    ]

    for engine in engines:
        engine.fit(product_df)
        result_df = Evaluator.evaluate_queries(engine, product_df, query_df, label_df, k=10)
        print(
            f"{engine.name} — MAP@10: {result_df['map@k'].mean():.4f} | "
            f"NDCG@10: {result_df['ndcg@k'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
