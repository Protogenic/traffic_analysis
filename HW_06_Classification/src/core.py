"""Core pipeline infrastructure for data processing."""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


@dataclass
class DataContainer:
    """Container for processed data and metadata."""
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None
    feature_matrix: Optional[np.ndarray] = None
    target_vector: Optional[np.ndarray] = None
    feature_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata dictionary."""
        if not self.metadata:
            self.metadata = {}


class DataStage(ABC):
    """Abstract base class for data processing stages."""
    
    def __init__(self, name: str):
        """
        Initialize a data processing stage.
        
        Args:
            name: Human-readable name of the stage.
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def execute(self, container: DataContainer) -> DataContainer:
        """
        Execute the processing stage.
        
        Args:
            container: Data container with current state.
            
        Returns:
            Updated data container.
        """
        pass
    
    def __call__(self, container: DataContainer) -> DataContainer:
        """Allow stage to be called as a function."""
        self.logger.info(f"Executing stage: {self.name}")
        try:
            result = self.execute(container)
            self.logger.info(f"Stage {self.name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Stage {self.name} failed: {e}")
            raise


class ProcessingPipeline:
    """Pipeline for sequential data processing stages."""
    
    def __init__(self, stages: list[DataStage] | None = None):
        """
        Initialize processing pipeline.
        
        Args:
            stages: Optional list of stages to add during initialization.
        """
        self.stages: list[DataStage] = stages or []
        self.logger = logging.getLogger(f"{__name__}.ProcessingPipeline")
    
    def add_stage(self, stage: DataStage) -> ProcessingPipeline:
        """
        Add a processing stage to the pipeline.
        
        Args:
            stage: Stage to add.
            
        Returns:
            Self for method chaining.
        """
        self.stages.append(stage)
        return self
    
    def run(self, container: DataContainer) -> DataContainer:
        """
        Execute all stages in sequence.
        
        Args:
            container: Initial data container.
            
        Returns:
            Final processed data container.
        """
        self.logger.info(f"Starting pipeline with {len(self.stages)} stages")
        current = container
        
        for idx, stage in enumerate(self.stages, 1):
            self.logger.info(f"Processing stage {idx}/{len(self.stages)}: {stage.name}")
            current = stage(current)
        
        self.logger.info("Pipeline execution completed")
        return current
