import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import settings
from .base import BaseModel


class YoloModel(BaseModel):
    def __init__(self, task: str, device: str) -> None:
        self.task = task
        self.model = YOLO(settings.models.yolo.nfl.get(self.task)).to(device=device)

    def run(self, frame: np.ndarray, verbose: bool = False) -> sv.Detections:
        result = self.model(frame, imgsz=640, verbose=verbose)[0]

        if self.task == "pitch":
            keypoints = sv.KeyPoints.from_ultralytics(result)
            return self.__filter_keypoints(
                keypoints, settings.infer.keypoint.conf_thres
            )
        elif self.task == "pose":
            return sv.KeyPoints.from_ultralytics(result)
        elif self.task == "player":
            return sv.Detections.from_ultralytics(result)
        else:
            raise NotImplementedError(
                f"Task '{self.task}' not allowed! "
                f"Available tasks are 'pitch' and 'player'"
            )

    def __filter_keypoints(
        self, keypoints: sv.KeyPoints, conf_thres: float
    ) -> sv.KeyPoints:
        if not np.any(keypoints.xy) or not np.any(keypoints.confidence):
            return keypoints

        confidence_mask = keypoints.confidence <= conf_thres

        keypoints.xy[confidence_mask] = 0
        # keypoints.confidence[confidence_mask] = 0

        # if keypoints.class_id is not None:
        #     keypoints.class_id[confidence_mask] = 0

        # if keypoints.data is not None:
        #     keypoints.data[confidence_mask] = 0

        return keypoints

    def __filter_detections(
        self, detections: sv.Detections, conf_thres: float
    ) -> sv.Detections:
        if not np.any(detections.xyxy) or not np.any(detections.confidence):
            return detections

        confidence_mask = detections.confidence <= conf_thres

        detections.xyxy[confidence_mask] = 0

        return detections
