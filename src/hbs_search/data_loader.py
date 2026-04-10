import logging
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class WANDSDataLoader:
    """Loads and manages the WANDS dataset (queries, products, labels).

    Handles cloning the repository if not already present and provides
    convenient access to the dataset files with proper error handling.
    """

    REPO_URL = "https://github.com/wayfair/WANDS.git"

    def __init__(self, base_path: str | None = None, repo_dir: str = "WANDS"):
        # Default to project root when used from the packaged src layout.
        default_base = Path(__file__).resolve().parents[2]
        self.base_path = Path(base_path) if base_path is not None else default_base
        self.repo_path = self.base_path / repo_dir
        self.dataset_path = self.repo_path / "dataset"

        self._products: pd.DataFrame | None = None
        self._queries: pd.DataFrame | None = None
        self._labels: pd.DataFrame | None = None

    def ensure_data(self) -> None:
        """Clone the WANDS repo if it doesn't already exist."""
        if self.repo_path.exists():
            logger.info("WANDS repo already exists at %s", self.repo_path)
            return

        logger.info("Cloning WANDS repo from %s ...", self.REPO_URL)
        subprocess.run(
            ["git", "clone", self.REPO_URL, str(self.repo_path)],
            check=True,
        )
        logger.info("Clone complete.")

    def _read_csv(self, filename: str) -> pd.DataFrame:
        """Read a tab-separated CSV from the dataset directory."""
        path = self.dataset_path / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. Run ensure_data() first."
            )

        df = pd.read_csv(path, sep="\t")
        if df.empty:
            raise ValueError(f"Dataset file is empty: {path}")

        logger.info("Loaded %s: %d rows, %d columns", filename, len(df), len(df.columns))
        return df

    def load_products(self) -> pd.DataFrame:
        """Load product.csv and cache the result."""
        if self._products is None:
            self._products = self._read_csv("product.csv")
        return self._products

    def load_queries(self) -> pd.DataFrame:
        """Load query.csv and cache the result."""
        if self._queries is None:
            self._queries = self._read_csv("query.csv")
        return self._queries

    def load_labels(self) -> pd.DataFrame:
        """Load label.csv and cache the result."""
        if self._labels is None:
            self._labels = self._read_csv("label.csv")
        return self._labels

    def get_grouped_labels(self) -> pd.core.groupby.DataFrameGroupBy:
        """Return labels grouped by query_id."""
        return self.load_labels().groupby("query_id")
