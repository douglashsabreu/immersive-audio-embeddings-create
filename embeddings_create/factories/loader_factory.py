"""Factory for creating audio loaders."""

from typing import Any, ClassVar

from embeddings_create.interfaces.audio_loader import IAudioLoader
from embeddings_create.utils.audio_loader import AudioLoader


class LoaderFactory:
    """Factory class for creating audio loaders.

    This factory implements the Factory pattern, following the Open/Closed
    Principle by allowing new loaders to be added without modifying
    existing code. Currently supports the standard AudioLoader implementation.
    """

    _loaders: ClassVar[dict[str, type[IAudioLoader]]] = {"standard": AudioLoader}

    @classmethod
    def create_loader(cls, loader_type: str = "standard", **kwargs: Any) -> IAudioLoader:
        """Create an audio loader of the specified type.

        Args:
            loader_type: The type of loader to create
            **kwargs: Additional arguments for loader initialization

        Returns:
            An audio loader instance implementing IAudioLoader

        Raises:
            ValueError: If loader_type is not supported
        """
        if loader_type not in cls._loaders:
            available_types = list(cls._loaders.keys())
            raise ValueError(
                f"Unsupported loader type: {loader_type}. " f"Available types: {available_types}"
            )

        loader_class = cls._loaders[loader_type]
        return loader_class(**kwargs)

    @classmethod
    def register_loader(cls, loader_type: str, loader_class: type[IAudioLoader]) -> None:
        """Register a new loader type.

        This method allows extending the factory with new loaders
        without modifying the existing code, following the Open/Closed Principle.

        Args:
            loader_type: The loader type identifier
            loader_class: The loader class to register

        Raises:
            TypeError: If loader_class doesn't implement IAudioLoader
        """
        if not issubclass(loader_class, IAudioLoader):
            raise TypeError(f"Loader class {loader_class} must implement IAudioLoader")

        cls._loaders[loader_type] = loader_class

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported loader types.

        Returns:
            List of supported loader type identifiers
        """
        return list(cls._loaders.keys())

    @classmethod
    def is_supported(cls, loader_type: str) -> bool:
        """Check if a loader type is supported.

        Args:
            loader_type: The loader type to check

        Returns:
            True if loader type is supported
        """
        return loader_type in cls._loaders
