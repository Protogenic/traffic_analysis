import logging

import pandas as pd

from src.core import Handler, PipelineContext

class EncodeCategoricalFeaturesHandler(Handler):
    """Handler for encoding categorical features."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        One-hot encode categorical columns, excluding the target 'grade'.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context with encoded dataframe.
        """
        logging.info(f"EncodeCategoricalFeaturesHandler: Encoding categorical features")
        df = ctx.df.copy()

        # Exclude target 'grade' from encoding
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if 'grade' in cat_cols:
            cat_cols.remove('grade')
            
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        ctx.df = df
        logging.info(f"EncodeCategoricalFeaturesHandler: Done (features: {df.shape[1]})")
        return ctx

class SplitDataHandler(Handler):
    """Handler for splitting data for regression (salary target)."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Split dataframe into features and target (salary_rub).

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context with features and target populated.
        """
        logging.info("SplitDataHandler: Splitting X/y")
        ctx.features = ctx.df.drop(columns=["salary_rub"])
        ctx.target = ctx.df["salary_rub"]
        ctx.df = None
        return ctx

class SplitClassificationDataHandler(Handler):
    """Handler for splitting data for classification (grade target)."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Split dataframe into features and target (grade).

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context with features, target, and feature_names populated.
        """
        logging.info("SplitClassificationDataHandler: Splitting X/y")
        df = ctx.df.copy()
        
        if 'grade' not in df.columns:
            logging.error("Grade column not found!")
            return ctx
            
        ctx.target = df['grade'].values
        ctx.features = df.drop(columns=['grade']).values
        ctx.feature_names = df.drop(columns=['grade']).columns.tolist()
        ctx.df = None
        logging.info("SplitClassificationDataHandler: Done")
        return ctx
