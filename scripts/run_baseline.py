"""Run baseline TF-IDF evaluation for quick sanity checks."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hbs_search.data_loader import WANDSDataLoader
from hbs_search.evaluation import Evaluator
from hbs_search.search_engine import TfidfSearchEngine


def main() -> None:
    loader = WANDSDataLoader()
    loader.ensure_data()
    product_df = loader.load_products()
    query_df = loader.load_queries()
    label_df = loader.load_labels()

    engine = TfidfSearchEngine(columns=["product_name", "product_description"])
    engine.fit(product_df)
    result_df = Evaluator.evaluate_queries(engine, product_df, query_df, label_df, k=10)

    print(f"{engine.name} — MAP@10: {result_df['map@k'].mean():.4f}")
    print(f"{engine.name} — NDCG@10: {result_df['ndcg@k'].mean():.4f}")


if __name__ == "__main__":
    main()
