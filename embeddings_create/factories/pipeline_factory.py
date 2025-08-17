"""Factory for creating pipeline components following SOLID principles."""

from ..interfaces.pipeline_interfaces import (
    IDatasetOrganizer,
    IEmbeddingGenerator,
    IFeatureExtractor,
    ILogger,
    IPipelineOrchestrator,
)
from ..pipeline.implementations import (
    ClickLogger,
    CompletePipelineOrchestrator,
    FourClassDatasetOrganizer,
    NeuralEmbeddingGenerator,
    SpatialFeatureExtractor,
)


class PipelineComponentFactory:
    """Factory for creating pipeline components with dependency injection."""

    @staticmethod
    def create_logger(verbose: bool = False) -> ILogger:
        """Create logger instance."""
        return ClickLogger(verbose=verbose)

    @staticmethod
    def create_feature_extractor(logger: ILogger) -> IFeatureExtractor:
        """Create feature extractor instance."""
        return SpatialFeatureExtractor(logger=logger)

    @staticmethod
    def create_dataset_organizer(logger: ILogger) -> IDatasetOrganizer:
        """Create dataset organizer instance."""
        return FourClassDatasetOrganizer(logger=logger)

    @staticmethod
    def create_embedding_generator(logger: ILogger) -> IEmbeddingGenerator:
        """Create embedding generator instance."""
        return NeuralEmbeddingGenerator(logger=logger)

    @staticmethod
    def create_complete_orchestrator(
        feature_extractor: IFeatureExtractor,
        dataset_organizer: IDatasetOrganizer,
        embedding_generator: IEmbeddingGenerator,
        logger: ILogger,
    ) -> IPipelineOrchestrator:
        """Create complete pipeline orchestrator with dependency injection."""
        return CompletePipelineOrchestrator(
            feature_extractor=feature_extractor,
            dataset_organizer=dataset_organizer,
            embedding_generator=embedding_generator,
            logger=logger,
        )

    @classmethod
    def create_complete_pipeline(cls, verbose: bool = False) -> IPipelineOrchestrator:
        """Factory method to create complete pipeline with all dependencies."""
        logger = cls.create_logger(verbose=verbose)
        feature_extractor = cls.create_feature_extractor(logger=logger)
        dataset_organizer = cls.create_dataset_organizer(logger=logger)
        embedding_generator = cls.create_embedding_generator(logger=logger)

        return cls.create_complete_orchestrator(
            feature_extractor=feature_extractor,
            dataset_organizer=dataset_organizer,
            embedding_generator=embedding_generator,
            logger=logger,
        )
