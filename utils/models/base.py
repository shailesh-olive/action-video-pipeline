from abc import ABC, abstractmethod

import numpy as np
from supervision import Detections


class BaseModel(ABC):
    @abstractmethod
    def run(self, frame: np.ndarray, *args, **kwargs) -> Detections | dict:
        pass
