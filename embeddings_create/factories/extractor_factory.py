"""Factory for creating feature extractors."""

from typing import Any, ClassVar

from embeddings_create.extractors.diffuseness_extractor import DiffusenessExtractor
from embeddings_create.extractors.iacc_extractor import IACCExtractor
from embeddings_create.extractors.intensity_doa_extractor import IntensityDOAExtractor
from embeddings_create.extractors.salsa_extractor import SalsaExtractor, SalsaLiteExtractor
from embeddings_create.interfaces.feature_extractor import IFeatureExtractor
from embeddings_create.models.feature_config import FeatureType


class ExtractorFactory:
    """Factory class for creating feature extractors.

    This factory implements the Factory pattern, following the Open/Closed
    Principle by allowing new extractors to be added without modifying
    existing code. It also follows the Dependency Inversion Principle
    by returning interface types rather than concrete implementations.
    """

    _extractors: ClassVar[dict[FeatureType, type[IFeatureExtractor]]] = {
        FeatureType.SALSA: SalsaExtractor,
        FeatureType.SALSA_LITE: SalsaLiteExtractor,
        FeatureType.INTENSITY_DOA: IntensityDOAExtractor,
        FeatureType.DIFFUSENESS: DiffusenessExtractor,
        FeatureType.IACC: IACCExtractor,
    }

    @classmethod
    def create_extractor(cls, feature_type: FeatureType) -> IFeatureExtractor:
        """Create a feature extractor for the specified type.

        Args:
            feature_type: The type of feature extractor to create

        Returns:
            A feature extractor instance implementing IFeatureExtractor

        Raises:
            ValueError: If feature_type is not supported
        """
        if feature_type not in cls._extractors:
            available_types = list(cls._extractors.keys())
            raise ValueError(
                f"Unsupported feature type: {feature_type}. " f"Available types: {available_types}"
            )

        extractor_class = cls._extractors[feature_type]
        return extractor_class()

    @classmethod
    def register_extractor(
        cls, feature_type: FeatureType, extractor_class: type[IFeatureExtractor]
    ) -> None:
        """Register a new feature extractor type.

        This method allows extending the factory with new extractors
        without modifying the existing code, following the Open/Closed Principle.

        Args:
            feature_type: The feature type to register
            extractor_class: The extractor class to register

        Raises:
            TypeError: If extractor_class doesn't implement IFeatureExtractor
        """
        if not issubclass(extractor_class, IFeatureExtractor):
            raise TypeError(f"Extractor class {extractor_class} must implement IFeatureExtractor")

        cls._extractors[feature_type] = extractor_class

    @classmethod
    def get_supported_types(cls) -> list[FeatureType]:
        """Get list of supported feature types.

        Returns:
            List of supported FeatureType values
        """
        return list(cls._extractors.keys())

    @classmethod
    def is_supported(cls, feature_type: FeatureType) -> bool:
        """Check if a feature type is supported.

        Args:
            feature_type: The feature type to check

        Returns:
            True if feature type is supported
        """
        return feature_type in cls._extractors

    @classmethod
    def create_all_extractors(cls) -> dict[FeatureType, IFeatureExtractor]:
        """Create instances of all available extractors.

        Returns:
            Dictionary mapping feature types to extractor instances
        """
        return {
            feature_type: cls.create_extractor(feature_type) for feature_type in cls._extractors
        }

    @classmethod
    def get_extractor_info(cls, feature_type: FeatureType) -> dict[str, Any]:
        """Get information about a specific extractor.

        Args:
            feature_type: The feature type to get info for

        Returns:
            Dictionary containing extractor information

        Raises:
            ValueError: If feature_type is not supported
        """
        if not cls.is_supported(feature_type):
            raise ValueError(f"Unsupported feature type: {feature_type}")

        extractor = cls.create_extractor(feature_type)
        return {
            "name": extractor.name,
            "required_channels": extractor.required_channels,
            "default_config": extractor.get_default_config(),
            "class_name": extractor.__class__.__name__,
        }
