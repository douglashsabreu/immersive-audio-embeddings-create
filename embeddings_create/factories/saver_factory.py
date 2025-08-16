"""Factory for creating result savers."""

from typing import Any, ClassVar

from embeddings_create.interfaces.result_saver import IResultSaver
from embeddings_create.utils.result_saver import ResultSaver


class SaverFactory:
    """Factory class for creating result savers.

    This factory implements the Factory pattern, following the Open/Closed
    Principle by allowing new savers to be added without modifying
    existing code. Currently supports the standard ResultSaver implementation.
    """

    _savers: ClassVar[dict[str, type[IResultSaver]]] = {"standard": ResultSaver}

    @classmethod
    def create_saver(cls, saver_type: str = "standard", **kwargs: Any) -> IResultSaver:
        """Create a result saver of the specified type.

        Args:
            saver_type: The type of saver to create
            **kwargs: Additional arguments for saver initialization

        Returns:
            A result saver instance implementing IResultSaver

        Raises:
            ValueError: If saver_type is not supported
        """
        if saver_type not in cls._savers:
            available_types = list(cls._savers.keys())
            raise ValueError(
                f"Unsupported saver type: {saver_type}. " f"Available types: {available_types}"
            )

        saver_class = cls._savers[saver_type]
        return saver_class(**kwargs)

    @classmethod
    def register_saver(cls, saver_type: str, saver_class: type[IResultSaver]) -> None:
        """Register a new saver type.

        This method allows extending the factory with new savers
        without modifying the existing code, following the Open/Closed Principle.

        Args:
            saver_type: The saver type identifier
            saver_class: The saver class to register

        Raises:
            TypeError: If saver_class doesn't implement IResultSaver
        """
        if not issubclass(saver_class, IResultSaver):
            raise TypeError(f"Saver class {saver_class} must implement IResultSaver")

        cls._savers[saver_type] = saver_class

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported saver types.

        Returns:
            List of supported saver type identifiers
        """
        return list(cls._savers.keys())

    @classmethod
    def is_supported(cls, saver_type: str) -> bool:
        """Check if a saver type is supported.

        Args:
            saver_type: The saver type to check

        Returns:
            True if saver type is supported
        """
        return saver_type in cls._savers
