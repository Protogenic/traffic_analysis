"""Target variable labeling transformations."""

import logging

from src.core import DataContainer, DataStage

logger = logging.getLogger(__name__)


class DeveloperLevelLabelerStage(DataStage):
    """Stage for labeling developer levels (Junior/Middle/Senior)."""
    
    JUNIOR_INDICATORS = [
        'junior', 'jun ', 'jr ', 'trainee', 'стажер', 'стажёр',
        'intern', 'младший', 'начинающий'
    ]
    
    SENIOR_INDICATORS = [
        'senior', 'sr ', 'lead', 'principal', 'staff', 'ведущий',
        'главный', 'руководитель', 'team lead', 'architect',
        'head of', 'expert'
    ]
    
    MIDDLE_INDICATORS = [
        'middle', 'mid ', 'мидл', 'мидлл'
    ]
    
    def __init__(self):
        """Initialize developer level labeler."""
        super().__init__("DeveloperLevelLabeler")
    
    def _determine_level(self, row) -> str:
        """
        Determine developer level based on job title and experience.
        
        Args:
            row: DataFrame row with job title and experience.
            
        Returns:
            Developer level: 'Junior', 'Middle', or 'Senior'.
        """
        job_title = str(row.get('_job_title', '')).lower()
        experience_months = row.get('experience_months', 0)
        
        has_junior = any(indicator in job_title for indicator in self.JUNIOR_INDICATORS)
        has_senior = any(indicator in job_title for indicator in self.SENIOR_INDICATORS)
        has_middle = any(indicator in job_title for indicator in self.MIDDLE_INDICATORS)
        
        # Handle conflicting indicators
        if has_senior and has_junior:
            return 'Senior' if experience_months > 36 else 'Junior'
        
        if has_senior:
            return 'Senior'
        
        if has_junior:
            return 'Middle' if experience_months > 60 else 'Junior'
        
        if has_middle:
            return 'Senior' if experience_months > 96 else 'Middle'
        
        # Default based on experience only
        if experience_months <= 18:
            return 'Junior'
        elif experience_months <= 60:
            return 'Middle'
        else:
            return 'Senior'
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Label developer levels and remove source columns."""
        df = container.processed_data.copy()
        
        # Extract job title for labeling
        job_title_col = 'Ищет работу на должность:'
        if job_title_col in df.columns:
            df['_job_title'] = df[job_title_col].fillna('').astype(str)
        else:
            df['_job_title'] = ''
        
        # Create target variable
        df['developer_level'] = df.apply(self._determine_level, axis=1)
        
        # Remove columns used for labeling (prevent data leakage)
        columns_to_remove = ['_job_title']
        if job_title_col in df.columns:
            columns_to_remove.append(job_title_col)
        if 'experience_months' in df.columns:
            columns_to_remove.append('experience_months')
        
        df = df.drop(columns=columns_to_remove)
        
        # Store target
        container.target_vector = df['developer_level'].values
        container.processed_data = df
        
        # Log distribution
        level_counts = df['developer_level'].value_counts()
        logger.info(f"Developer level distribution:\n{level_counts.to_string()}")
        logger.info(f"Removed columns to prevent leakage: {columns_to_remove}")
        
        return container

