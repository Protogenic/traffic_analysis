import logging

import numpy as np
import pandas as pd

from src.core import Handler, PipelineContext

class LoadCSVHandler(Handler):
    """Handler for loading data from a CSV file."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Load the CSV file specified in the context path into a DataFrame.

        Args:
            ctx: Pipeline context containing the path to the CSV file.

        Returns:
            PipelineContext: Context updated with the loaded DataFrame.
        """
        logging.info(f"LoadCSVHandler: Starting to load {ctx.csv_path}")
        ctx.df = pd.read_csv(
            ctx.csv_path,
            sep=",",
            quotechar='"',
            engine="python",
            encoding="utf-8",
            index_col=0
        )
        logging.info(f"LoadCSVHandler: Loaded {ctx.csv_path} with {ctx.df.shape[0]} rows and {ctx.df.shape[1]} columns")
        return ctx

class SaveDataHandler(Handler):
    """Handler for saving the dataset into X.npy and y.npy files."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Save the processed features (X) and target (y) to NumPy files.

        Args:
            ctx: Pipeline context containing features and target arrays.

        Returns:
            PipelineContext: Unmodified context.
        """
        logging.info(f"SaveDataHandler: Saving data")
        np.save("X.npy", ctx.features)
        np.save("y.npy", ctx.target)
        logging.info(f"SaveDataHandler: Data was saved to X.npy and y.npy files")
        return ctx
