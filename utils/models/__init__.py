from typing import Dict, TypeVar

from .base import BaseModel
from .yolo import YoloModel
from .kmeans import KMeansModel


ModelType = TypeVar("ModelType", bound=BaseModel)


class Model:
    models: Dict[str, ModelType] = {
        "yolo": YoloModel,
        "kmeans": KMeansModel
    }

    @classmethod
    def get_model(cls, name: str, task: str, device: str) -> ModelType:
        model_class = cls.models.get(name.lower())
        if not model_class:
            raise ValueError(
                f"No Model class found for type {type}. Available types are: {list(cls.models.keys())}"
            )
        return model_class(task, device)
