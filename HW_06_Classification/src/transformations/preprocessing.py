"""Data preprocessing transformations."""

import logging

import pandas as pd

from src.core import DataContainer, DataStage

logger = logging.getLogger(__name__)


class CategoricalEncoderStage(DataStage):
    """Stage for encoding categorical features using one-hot encoding."""
    
    def __init__(self):
        """Initialize categorical encoder."""
        super().__init__("CategoricalEncoder")
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Encode categorical features, excluding target variable."""
        df = container.processed_data.copy()
        
        # Identify categorical columns (object type)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclude target variable if present
        if 'developer_level' in categorical_cols:
            categorical_cols.remove('developer_level')
        
        if categorical_cols:
            # Apply one-hot encoding
            df_encoded = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=True,
                prefix_sep='_'
            )
            
            # Preserve non-categorical columns
            numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
            if 'developer_level' in numeric_cols:
                numeric_cols.remove('developer_level')
            
            for col in numeric_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = df[col]
            
            if 'developer_level' in df.columns:
                df_encoded['developer_level'] = df['developer_level']
            
            df = df_encoded
        
        container.processed_data = df
        logger.info(
            f"Encoded categorical features. "
            f"Final feature count: {len(df.columns)}"
        )
        return container


class FeatureTargetSplitterStage(DataStage):
    """Stage for splitting features and target variable."""
    
    def __init__(self):
        """Initialize feature-target splitter."""
        super().__init__("FeatureTargetSplitter")
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Split data into feature matrix and target vector."""
        df = container.processed_data.copy()
        
        if 'developer_level' not in df.columns:
            logger.error("Target variable 'developer_level' not found!")
            return container
        
        # Extract target
        container.target_vector = df['developer_level'].values
        
        # Extract features
        feature_df = df.drop(columns=['developer_level'])
        container.feature_matrix = feature_df.values
        container.feature_labels = feature_df.columns.tolist()
        
        # Clear processed data to save memory
        container.processed_data = None
        
        logger.info(
            f"Split complete. Features shape: {container.feature_matrix.shape}, "
            f"Target shape: {container.target_vector.shape}"
        )
        
        return container

