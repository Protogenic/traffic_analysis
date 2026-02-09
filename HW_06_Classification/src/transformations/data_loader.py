"""Data loading transformations."""

import logging
from pathlib import Path

import pandas as pd

from src.core import DataContainer, DataStage

logger = logging.getLogger(__name__)


class CSVLoaderStage(DataStage):
    """Stage for loading CSV data into the pipeline."""
    
    def __init__(self, csv_path: Path):
        """
        Initialize CSV loader.
        
        Args:
            csv_path: Path to the CSV file to load.
        """
        super().__init__("CSVLoader")
        self.csv_path = csv_path
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Load CSV file into container."""
        logger.info(f"Loading data from {self.csv_path}")
        container.raw_data = pd.read_csv(
            self.csv_path,
            sep=",",
            quotechar='"',
            engine="python",
            encoding="utf-8",
            index_col=0
        )
        container.processed_data = container.raw_data.copy()
        logger.info(f"Loaded {len(container.processed_data)} rows, {len(container.processed_data.columns)} columns")
        return container

