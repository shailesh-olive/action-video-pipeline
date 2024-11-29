import numpy as np
import torch.nn.functional as F
from seperated_player_label.tracker_utils.bot_sort import BoTSORT
from seperated_player_label.reid.config import get_cfg


def nms_fast(boxes, probs=None, overlap_thresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )
    # return only the bounding boxes that were picked
    return boxes[pick].astype("float"), pick


def image_track(cam_detections, cam_embedding, fps):
    tracker = BoTSORT(frame_rate=fps)
    results = []
    # num_frames = int(str(cam_detections[-1][0])[-4:-2])

    jersey_track_map = {}
    new_frame_id = -1
    for frame_id in cam_detections[:, 0]:
        if frame_id != new_frame_id:
            new_frame_id = frame_id
            print("frame id: ", frame_id)

            inds = cam_detections[:, 0] == frame_id
            detections = cam_detections[inds][:, 2:-1].astype(
                float
            )  # x1,y1,x2,y2,score
            embedding = cam_embedding[inds]
            jersey_nums = cam_detections[inds][:, -1]

            # remove detections that overlaps another detection
            detections, pick = nms_fast(detections, None, overlap_thresh=0.9)
            embedding = embedding[pick]

            if detections is not None:
                online_targets = tracker.update(detections, embedding, jersey_nums)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    j_num = t.jersey_num
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    vertical = False

                    # add jersey number and track id in jersey_track_map
                    if str(j_num) != "0.0" and j_num not in jersey_track_map:
                        jersey_track_map[j_num] = tid
                    elif j_num in jersey_track_map:
                        # if jersey number is present in the mapping then
                        # get the previous track id of that jersey number
                        tid = jersey_track_map[j_num]

                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            [
                                frame_id,
                                tid,
                                tlwh[0],
                                tlwh[1],
                                tlwh[2] + tlwh[0],
                                tlwh[3] + tlwh[1],
                                t.score,
                                -1,
                                -1,
                                -1,
                            ]
                        )

            else:
                pass

    return results


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features
