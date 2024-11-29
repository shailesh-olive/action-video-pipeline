from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import settings
from utils.model_utils import Model
from utils.tracker_utils.core import image_track, setup_cfg
from reid.utils.build import build_model

from base import BaseTask
from utils.tracker_utils.bot_sort import BoTSORT
from utils.tracker_utils.jersey_ocr import JerseyOCR


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

@dataclass
class PlayerFeatureExtractor(BaseTask):
    def __post_init__(self):
        self.pmodel = Model.get_model(self.model, "player", self.device)
        self.cfg = setup_cfg(
            settings.models.resnet.config,
            ["MODEL.WEIGHTS", settings.models.resnet.model_path],
        )
        self.jersey_ocr = JerseyOCR()
        self.model = build_model(self.cfg)
        self.model.eval()


        if self.device != "cpu":
            self.model = self.model.eval().to(device="cuda").half()
        else:
            self.model = self.model.eval()
        self.embeddings = []
        self.detections = []

    def _process_frame(
        self, frame: np.ndarray, idx: int, _, verbose: bool = False
    ) -> np.ndarray:
        result = self.pmodel.run(frame, verbose=verbose)
        self.__get_player_feature(idx, frame, result)
        return frame

    def __get_player_feature(self, idx: int, frame: np.ndarray, detections: np.ndarray):
        jersey_numbers = []
        patches = []
        batch_patches = []
        for detection in detections.xyxy:
            x1, y1, x2, y2 = detection.astype(int)
            img_crop = frame[y1:y2, x1:x2]
            # img_crop = Image.fromarray(img_crop)
            org_crop = img_crop.copy()

            patch = img_crop[:, :, ::-1]
            patch = cv2.resize(
                patch,
                tuple(self.cfg.INPUT.SIZE_TEST[::-1]),
                interpolation=cv2.INTER_LINEAR,
            )

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.to(device="cuda").half()

            patches.append(patch)

            img = np.array(org_crop)
            jersey_num, _ = self.jersey_ocr.get_jersey_number(img)
            jersey_numbers.append(jersey_num)

        patches = torch.stack(patches, dim=0)
        batch_patches.append(patches)

        features = np.zeros((0, 2048))

        for patches in batch_patches:
            # Run model
            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            features = np.vstack((features, feat))
            self.embeddings.extend(features)

        img_ids = np.array([idx] * len(detections))
        track_ids = np.array([1] * len(detections))
        cam_ids = np.array(["c001"] * len(detections))

        detection_info = np.hstack(
            (
                cam_ids[:, np.newaxis],
                img_ids[:, np.newaxis],
                track_ids[:, np.newaxis],
                detections.xyxy.astype(int),
                detections.confidence[:, np.newaxis],
                np.array(jersey_numbers)[:, np.newaxis],
                detections.class_id[:, np.newaxis],
            )
        )
        self.detections.extend(detection_info)


@dataclass
class PlayerTracker(BaseTask):
    def __post_init__(self):
        #self.max_age = 2
        self.tracker = BoTSORT(frame_rate=self.video_info.fps) #max_age=self.max_age)
        self.annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex(self.colors), thickness=2
        )
        self.lannotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(self.colors),
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=3,
            text_thickness=1,
        )
        # changes
        self.tannotator = sv.TraceAnnotator(
            color=sv.ColorPalette.from_hex(self.colors),
            trace_length = 5,

        )
        self.team_assign = Model.get_model("kmeans", "team", self.device)
        self.tracker_ids = None

    def _process_frame(
        self, frame: np.ndarray, idx: int, source_path: str, annotate: bool = True
    ) -> np.ndarray:
        if self.tracker_ids is None:
            self.tracker_ids = self.setup(source_path)

        # detections = self.__get_player_tracking(idx, frame)
        # #active_detections = [d for d in detections if d.age <= self.max_age]

        # if annotate:
        #     labels = [f"{tracker_id}" for tracker_id in detections.tracker_id]
        #     annotated_frame = frame.copy()
        #     annotated_frame = self.annotator.annotate(annotated_frame, detections)
        #     return self.lannotator.annotate(annotated_frame, detections, labels=labels)

        return self.tracker_ids

    def setup(self, source_path) -> np.ndarray:
        pbar = tqdm(total=self.video_info.total_frames)
        extractor = PlayerFeatureExtractor(
            model=self.model, video_info=self.video_info, device=self.device
        )
        frames_store = []
        for frame in extractor.run(source_path):
            frames_store.append(frame)
            pbar.update(1)
        pbar.close()

        # TODO: Currently 'cam' feature is redundant
        # Keeping it here for future reference
        cam = "c001"
        detections = np.array(extractor.detections)
        # cam_ids, img_ids, track_ids, xyxy, confidence, jersey_numbers, class_id
        mask = (detections[:, 0] == cam) & (detections[:, -1] == "1")
        player_detections = detections[mask][:, 1:].astype(float)
        other_detections = detections[~mask][:, 1:].astype(float)
        embeddings = np.array(extractor.embeddings)[mask]

        tracker_ids = image_track(
            player_detections[:, :-1], embeddings, self.video_info.fps
        )
        # return np.array(tracker_ids)
        tracker_ids = np.array(tracker_ids)
        
        return tracker_ids
        # import pickle
        # with open('pkl_files/match_4_video_38-deter.pkl', 'wb') as file:
        #     pickle.dump(tracker_ids, file)
        
        # from_team_classify = self.__cluster_as_team(
        #     frames_store, player_detections[:, :-1], tracker_ids, team_labels=(1, 4)
        # )

        # # Default value addition for unknown variables
        # other_detections[:, 1] = 0  # tracker ID for non-playing entity
        # other_detections[:, 7] = -1
        # other_detections = np.array(
        #     [np.insert(row, -1, [-1, -1]) for row in other_detections]
        # )

        # return np.concatenate((from_team_classify, other_detections), axis=0)

    # def __cluster_as_team(
    #     self,
    #     frames_store: list[np.ndarray],
    #     tracker_ids: np.ndarray,
    #     team_labels: Optional[Tuple[int, int]],
    # ) -> np.ndarray:
    #     # Step 1: Get team id for each tracked players
    #     p_teams = self.team_assign.run(frames_store, tracker_ids)

    #     # Step 2: Assign custom labels or use default 0,1
    #     if team_labels is None or len(team_labels) != 2:
    #         team_labels = [0, 1]

    #     player_teams = {
    #         tracker_id: team_labels[label] for tracker_id, label in p_teams.items()
    #     }

    #     # Step 3: Add team_id to tracker_ids array
    #     tracker_ids = np.pad(
    #         tracker_ids, ((0, 0), (0, 1)), mode="constant", constant_values=None
    #     )
    #     for i, row in enumerate(tracker_ids):
    #         tracker_id = row[1]
    #         if tracker_id in player_teams:
    #             tracker_ids[i, -1] = player_teams[tracker_id]

    #     return tracker_ids

    # def __get_player_tracking(self, idx: int, frame: np.ndarray) -> np.ndarray:
    #     mask = self.tracker_ids[:, 0] == idx

    #     xyxy = self.tracker_ids[mask][:, 2:6].astype(float)
    #     tracker = self.tracker_ids[mask][:, 1].astype(int)
    #     confidence = self.tracker_ids[mask][:, 6].astype(float)
    #     class_id = self.tracker_ids[mask][:, -1]
    #     class_id = np.where(class_id is None, 1, class_id).astype(int)
    #     # class_id = np.array([1] * len(xyxy))

    #     return Detection.to_supervision_detections(
    #         xyxy, confidence, class_id, tracker=tracker
    #     )
