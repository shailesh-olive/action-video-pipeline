from abc import ABC, abstractmethod
from typing import List, Union, Iterator
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

import cv2


def resize_to_square(frame):
    h, w = frame.shape[:2]
    size = max(h, w)
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, (h, w)


@dataclass
class BaseTask(ABC):
    model: str
    video_info: sv.VideoInfo
    device: Union[str, int] = "cpu"
    colors: List[str] = field(
        default_factory=lambda: ["#59E659", "#474aff", "#DC143C", "#FF6347", "#030303"]
    )

    @abstractmethod
    def _process_frame(self, frame: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Process a single frame and return the annotated frame."""
        pass

    @abstractmethod
    def __post_init__(self, *args, **kwargs):
        """Perform any necessary setup for the detector."""
        pass

    def run(self, source_path: str) -> Iterator[np.ndarray]:
        frame_generator = sv.get_video_frames_generator(source_path=source_path)
        for idx, frame in enumerate(frame_generator):
            frame, _ = resize_to_square(frame)
            yield self._process_frame(frame, idx, source_path)
