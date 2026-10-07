"""Map YOLO classes into the required vehicle categories."""

from __future__ import annotations

from collections.abc import Mapping


class VehicleMapper:
    """Resolve model classes to normalized vehicle categories."""

    def __init__(self, class_mapping: Mapping[str, str]) -> None:
        self._mapping = {
            self._normalize(name): category
            for name, category in class_mapping.items()
        }
        self._categories = tuple(dict.fromkeys(self._mapping.values()))

    @property
    def categories(self) -> tuple[str, ...]:
        """Return categories in deterministic display order."""
        return self._categories

    def category_for(self, class_name: str) -> str | None:
        """Map a model class name to a custom category.

        Args:
            class_name: Class label emitted by YOLO.

        Returns:
            Custom category or None when the class is not configured.
        """
        return self._mapping.get(self._normalize(class_name))

    def model_class_ids(
        self,
        model_names: Mapping[int, str] | list[str],
    ) -> tuple[int, ...]:
        """Resolve configured vehicle names against model metadata.

        Args:
            model_names: YOLO class ID-to-name metadata.

        Returns:
            Sorted class IDs accepted by the vehicle pipeline.

        Raises:
            ValueError: If the model contains no configured vehicle class.
        """
        items = (
            model_names.items()
            if isinstance(model_names, Mapping)
            else enumerate(model_names)
        )
        class_ids = tuple(
            sorted(
                int(class_id)
                for class_id, name in items
                if self.category_for(str(name)) is not None
            )
        )
        if not class_ids:
            raise ValueError(
                "The model has no classes listed in VEHICLE_CLASS_MAPPING."
            )
        return class_ids

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().replace("_", " ").split())
