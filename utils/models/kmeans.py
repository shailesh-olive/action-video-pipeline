import cv2
import numpy as np
from skimage import color
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


def voting_algorithm(tracker_ids, team_list):
    # Initialize a dictionary to store teams for each tracker_id
    tracker_dict = defaultdict(list)

    # Step 1: Populate the dictionary with tracker_id as key and teams as values
    for tracker_id, team in zip(tracker_ids, team_list):
        tracker_dict[tracker_id].append(team)

    # Step 2: For each tracker_id, find the most common team
    result = {}
    for tracker_id, teams in tracker_dict.items():
        # Use Counter to find the most common team
        most_common_team = Counter(teams).most_common(1)[0][0]
        result[tracker_id] = most_common_team

    return result


def count_players_from_teams(result, team1=0, team2=1):
    # Use Counter to count occurrences of each team
    team_counts = Counter(result.values())
    
    # Get counts for Team 1 and Team 2, with default 0 if the team is not present
    team1_count = team_counts.get(team1, 0)
    team2_count = team_counts.get(team2, 0)

    return team1_count, team2_count


class KMeansModel(BaseModel):
    def __init__(self, task: str, device: str) -> None:
        self.task = task
        self.model = KMeans(n_clusters=2, random_state=789, max_iter=500, tol=0.001)
        self.scaler = StandardScaler()
        self.player_team_dict = {}

    def run(self, frames: np.ndarray, tracker_ids: np.ndarray):
        track_id = tracker_ids[:, 1]
        all_feature_list = []
        new_frame_id = -1

        for idx in tracker_ids[:, 0]:
            if idx != new_frame_id:
                new_frame_id = idx

                inds = tracker_ids[:, 0] == idx
                bboxes = tracker_ids[inds][:, 2:6]  # x1,y1,x2,y2

                all_feature_list.append(
                    self.__get_players_feature(bboxes, frames[int(idx)])
                )

        # Flatten and stack each players feature to ensure it's 2D
        teams = self.__runs_kmeans(np.vstack(all_feature_list))

        # Assigning team id to each tracked players
        team_with_id = voting_algorithm(track_id, teams)

        # NOTE: Line below gives the Player Count for each team
        # from tactix.utils.team import count_players_from_teams
        # team_0_players, team_1_players = count_players_from_teams(team_with_id)

        return team_with_id

    def __get_players_feature(self, boxes: np.ndarray, image: np.ndarray):
        dominant_features = []
        grass_color = self._get_bg_color(image)  # Now returns HSV
        grass_hsv = np.array(grass_color)

        for bbox in boxes:
            # Extract the bounding box from the image
            # Replace negative values with zero
            bbox[bbox < 0] = 0
            x1, y1, x2, y2 = map(int, bbox)
            player_crop = image[y1:y2, x1:x2]
            if len(player_crop) == 0:
                dominant_features.append(np.zeros(3))  # Placeholder for color features
            else:
                dominant_features.append(
                    self.__extract_color_features(player_crop, grass_hsv)
                )

        # Convert BGR to RGB before passing to LAB conversion
        return [
            color.rgb2lab(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_BGR2RGB)).flatten()
            for rgb in dominant_features
        ]

    def _get_bg_color(self, frame):
        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define a large range of green color in HSV
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Create a mask for the green regions in the image
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Return the average HSV color of the masked green regions
        return cv2.mean(hsv_img, mask=mask)[:3]

    def __extract_color_features(self, player_img, bg_color_hsv):
        # Convert player image to HSV color space
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

        # Define the range of green color in HSV
        lower_green = np.array([bg_color_hsv[0] - 10, 40, 40])
        upper_green = np.array([bg_color_hsv[0] + 10, 255, 255])

        # Threshold the HSV image to isolate green (grass)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)  # Invert the mask to keep non-green areas

        # Only keep the upper half of the player image
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0 : player_img.shape[0] // 2, :] = 255
        mask = cv2.bitwise_and(mask, upper_mask)

        # Apply the mask to the player image
        masked_img = cv2.bitwise_and(player_img, player_img, mask=mask)

        # Calculate the player's kit color by averaging the remaining pixels
        player_kit_color = np.array(cv2.mean(masked_img, mask=mask)[:3])

        return player_kit_color

    def __runs_kmeans(self, data):
        scaled_data = self.scaler.fit_transform(data)  # scaling the data
        kmeans = self.model.fit(scaled_data)
        return kmeans.labels_
