import numpy as np

from scripts.track import matching
from basetrack import BaseTrack, TrackState
from kalman_filter import KalmanFilter

from copy import deepcopy
from collections import defaultdict


track_high_thresh = 0.2
track_low_thresh = 0.1
new_track_thresh = 0.2
track_buffer = 30
match_thresh = 0.8
aspect_ratio_thresh = 1.6
min_box_area = 10
proximity_thresh = 0.8
appearance_thresh = 0.25
img_width = 1920
img_height = 1080
border_thresh_right_x2 = img_width - 30
border_thresh_bottom_y2 = img_height - 150
border_thresh_left_x1 = 25
border_thresh_top_y1 = 25


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, jersey_num=0):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        # self.features = deque([], maxlen=feat_history)
        self.features = []
        self.times = []
        self.alpha = 0.9
        self.jersey_num = jersey_num

    def update_features(self, feat):
        # feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        self.smooth_feat = feat
        # if self.smooth_feat is None:
        #     self.smooth_feat = feat
        # else:
        #     self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        # self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xywh(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.jersey_num = new_track.jersey_num

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh)
        )

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
            self.times.append(frame_id)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.jersey_num = new_track.jersey_num

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{}-{})".format(
            self.track_id, self.start_frame, self.end_frame, self.jersey_num
        )


class BoTSORT(object):
    def __init__(self, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.lost_border_stracks = []
        self.lost_stracks_left = []
        self.lost_stracks_right = []
        self.lost_stracks_top = []
        self.lost_stracks_bottom = []
        self.found_track = False
        BaseTrack.clear_count()

        self.frame_id = 0
        # self.args = args

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        # Tracklet recorder
        self.tracklets = defaultdict()
        self.with_reid = True

    def update(self, output_results, embedding, jersey_nums):
        """
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        """

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4].astype(int)
                # classes = output_results[:, -1]
            else:
                raise ValueError(
                    "Wrong detection size {}".format(output_results.shape[1])
                )

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]

            # breakpoint()
            if self.with_reid:
                embedding = embedding[lowest_inds]
                features_keep = embedding[remain_inds]

        else:
            bboxes = []
            scores = []
            # classes = []
            dets = []
            scores_keep = []
            features_keep = []
            # classes_keep = []

        if len(dets) > 0:
            """Detections"""
            if self.with_reid:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, f, j_num)
                    for (tlbr, s, f, j_num) in zip(
                        dets, scores_keep, features_keep, jersey_nums
                    )
                ]
            else:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s)
                    for (tlbr, s) in zip(dets, scores_keep)
                ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        prev_frame_stracks = deepcopy(self.tracked_stracks)
        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        if self.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)
            dists = ious_dists
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            features_second = embedding[inds_second]
        else:
            dets_second = []
            scores_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, f, j_num)
                for (tlbr, s, f, j_num) in zip(
                    dets_second, scores_second, features_second, jersey_nums
                )
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        if self.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)
            dists = ious_dists

        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]

        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
        #     self.tracked_stracks, self.lost_stracks
        # )

        output_stracks = [track for track in self.tracked_stracks]

        for track in output_stracks:
            self.tracklets[track.track_id] = track

        # get all the tracks that left the frame
        for t in self.lost_stracks:
            x1, y1, x2, y2 = t.tlbr
            if (
                x1 > border_thresh_left_x1
                and x2 < border_thresh_right_x2
                and y1 > border_thresh_top_y1
                and y2 < border_thresh_bottom_y2
            ):
                pass
            elif t not in self.lost_border_stracks:
                self.lost_border_stracks.append(t)

        # get the tracks from left, right, top and bottom of the frame
        for t in self.lost_stracks:
            x1, y1, x2, y2 = t.tlbr
            if x1 < border_thresh_left_x1 and t not in self.lost_stracks_left:
                self.lost_stracks_left.append(t)
            elif x2 > border_thresh_right_x2 and t not in self.lost_stracks_right:
                self.lost_stracks_right.append(t)
            elif y1 < border_thresh_top_y1 and t not in self.lost_stracks_top:
                self.lost_stracks_top.append(t)
            if y2 > border_thresh_bottom_y2 and t not in self.lost_stracks_bottom:
                self.lost_stracks_bottom.append(t)

        lost_features = [t.curr_feat for t in self.lost_border_stracks]
        lost_track_ids = [t.track_id for t in self.lost_border_stracks]

        lost_features_left = [t.curr_feat for t in self.lost_stracks_left]
        lost_features_right = [t.curr_feat for t in self.lost_stracks_right]
        lost_features_top = [t.curr_feat for t in self.lost_stracks_top]
        lost_features_bottom = [t.curr_feat for t in self.lost_stracks_bottom]

        lost_track_ids_left = [t.track_id for t in self.lost_stracks_left]
        lost_track_ids_right = [t.track_id for t in self.lost_stracks_right]
        lost_track_ids_top = [t.track_id for t in self.lost_stracks_top]
        lost_track_ids_bottom = [t.track_id for t in self.lost_stracks_bottom]

        for t in activated_starcks:
            x1, y1, x2, y2 = t.tlbr
            if t.track_id > 30:
                # match all the players that appear from the left of the frame
                if x1 < border_thresh_left_x1 and len(self.lost_stracks_left) > 0:
                    feature = t.curr_feat
                    feature_norm = feature / np.linalg.norm(feature)
                    feature_list_norm = lost_features_left / np.linalg.norm(
                        lost_features_left, axis=1, keepdims=True
                    )

                    # Compute cosine similarity using matrix multiplication
                    cosine_similarities = np.dot(feature_list_norm, feature_norm)
                    best_match_index = np.argmax(cosine_similarities)
                    best_match_score = cosine_similarities[best_match_index]

                    if best_match_score > 0.4:
                        self.found_track = True
                        t.track_id = lost_track_ids_left[best_match_index]

                        self.tracked_stracks.append(t)
                        if best_match_index < len(self.lost_stracks_left):
                            self.lost_stracks_left.remove(
                                self.lost_stracks_left[best_match_index]
                            )
                        t.state == TrackState.Tracked

                        self.removed_stracks = [
                            remove_t
                            for remove_t in self.removed_stracks
                            if remove_t.track_id != t.track_id
                        ]
                    else:
                        t.track_id = -1
                # match all the players that appear from the right of the frame
                elif x2 > border_thresh_right_x2 and len(self.lost_stracks_right) > 0:
                    feature = t.curr_feat
                    feature_norm = feature / np.linalg.norm(feature)
                    feature_list_norm = lost_features_right / np.linalg.norm(
                        lost_features_right, axis=1, keepdims=True
                    )

                    # Compute cosine similarity using matrix multiplication
                    cosine_similarities = np.dot(feature_list_norm, feature_norm)
                    best_match_index = np.argmax(cosine_similarities)
                    best_match_score = cosine_similarities[best_match_index]

                    if best_match_score > 0.4:
                        self.found_track = True
                        t.track_id = lost_track_ids_right[best_match_index]

                        self.tracked_stracks.append(t)
                        if best_match_index < len(self.lost_stracks_right):
                            self.lost_stracks_right.remove(
                                self.lost_stracks_right[best_match_index]
                            )
                        t.state == TrackState.Tracked
                        self.removed_stracks = [
                            remove_t
                            for remove_t in self.removed_stracks
                            if remove_t.track_id != t.track_id
                        ]
                    else:
                        t.track_id = -1
                # match all the players that appear from the top of the frame
                elif y1 < border_thresh_top_y1 and len(self.lost_stracks_top) > 0:
                    feature = t.curr_feat
                    feature_norm = feature / np.linalg.norm(feature)
                    feature_list_norm = lost_features_top / np.linalg.norm(
                        lost_features_top, axis=1, keepdims=True
                    )

                    # Compute cosine similarity using matrix multiplication
                    cosine_similarities = np.dot(feature_list_norm, feature_norm)
                    best_match_index = np.argmax(cosine_similarities)
                    best_match_score = cosine_similarities[best_match_index]

                    if best_match_score > 0.4:
                        self.found_track = True
                        t.track_id = lost_track_ids_top[best_match_index]

                        self.tracked_stracks.append(t)
                        if best_match_index < len(self.lost_stracks_top):
                            self.lost_stracks_top.remove(
                                self.lost_stracks_top[best_match_index]
                            )
                        t.state == TrackState.Tracked
                        self.removed_stracks = [
                            remove_t
                            for remove_t in self.removed_stracks
                            if remove_t.track_id != t.track_id
                        ]
                    else:
                        t.track_id = -1
                # match all the players that appear from the bottom of the frame
                elif y2 > border_thresh_bottom_y2 and len(self.lost_stracks_bottom) > 0:
                    feature = t.curr_feat
                    feature_norm = feature / np.linalg.norm(feature)
                    feature_list_norm = lost_features_bottom / np.linalg.norm(
                        lost_features_bottom, axis=1, keepdims=True
                    )

                    # Compute cosine similarity using matrix multiplication
                    cosine_similarities = np.dot(feature_list_norm, feature_norm)
                    best_match_index = np.argmax(cosine_similarities)
                    best_match_score = cosine_similarities[best_match_index]

                    if best_match_score > 0.4:
                        self.found_track = True
                        t.track_id = lost_track_ids_bottom[best_match_index]

                        self.tracked_stracks.append(t)
                        if best_match_index < len(self.lost_stracks_bottom):
                            self.lost_stracks_bottom.remove(
                                self.lost_stracks_bottom[best_match_index]
                            )
                        t.state == TrackState.Tracked
                        self.removed_stracks = [
                            remove_t
                            for remove_t in self.removed_stracks
                            if remove_t.track_id != t.track_id
                        ]
                    else:
                        t.track_id = -1

        # match the remaining players that are not matched in above method
        for t in activated_starcks:
            x1, y1, x2, y2 = t.tlbr
            if t.track_id == -1:
                feature = t.curr_feat
                feature_norm = feature / np.linalg.norm(feature)
                feature_list_norm = lost_features / np.linalg.norm(
                    lost_features, axis=1, keepdims=True
                )

                # Compute cosine similarity using matrix multiplication
                cosine_similarities = np.dot(feature_list_norm, feature_norm)
                best_match_index = np.argmax(cosine_similarities)
                best_match_score = cosine_similarities[best_match_index]

                if best_match_score > 0.7:
                    t.track_id = lost_track_ids[best_match_index]
                    self.tracked_stracks.append(t)
        # if the same player appears again in the frame then remove the player from removed_stracks list
        if self.found_track:
            self.removed_stracks = [
                t for t in self.removed_stracks if t not in self.tracked_stracks
            ]

        prev_frame_bbox = [track.tlbr for track in prev_frame_stracks]
        prev_frame_features = [t.curr_feat for t in prev_frame_stracks]
        prev_frame_trackids = [t.track_id for t in prev_frame_stracks]
        cur_trackids = [track.track_id for track in output_stracks]

        # if the detection disappears in the crowd and appears again after certain frame then
        # find the nearest detection from previous frame and get the matched track id
        for t in output_stracks:
            if int(t.track_id) > 30:
                cur_bbox = t.tlbr

                distances = [
                    (i, bbox_distance(cur_bbox, bbox))
                    for i, bbox in enumerate(prev_frame_bbox)
                ]
                top_3_nearest_indices = sorted(distances, key=lambda x: x[1])[:3]
                neareset_feat = [
                    prev_frame_features[i] for i, _ in top_3_nearest_indices
                ]
                nearest_track_ids = [
                    prev_frame_trackids[i] for i, _ in top_3_nearest_indices
                ]

                feature = t.curr_feat
                feature_norm = feature / np.linalg.norm(feature)
                feature_list_norm = neareset_feat / np.linalg.norm(
                    neareset_feat, axis=1, keepdims=True
                )

                # Compute cosine similarity using matrix multiplication
                cosine_similarities = np.dot(feature_list_norm, feature_norm)
                best_match_index = np.argmax(cosine_similarities)
                best_match_score = cosine_similarities[best_match_index]

                matched_trackid = nearest_track_ids[best_match_index]
                if best_match_score > 0.7 and matched_trackid not in cur_trackids:
                    t.track_id = matched_trackid
                    self.tracked_stracks.append(t)

                cur_trackids = [track.track_id for track in output_stracks]
        return output_stracks


def bbox_distance(bbox1, bbox2):
    # Calculate the center of each bounding box
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

    # Calculate Euclidean distance between centers
    return np.linalg.norm(np.array(center1) - np.array(center2))


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
