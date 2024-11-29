import torch
import supervision as sv
from preprocessor import run_preprocess

from track import PlayerTracker


model = "yolo"
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(source_path: str, target_path: str) -> None:
    video_info = sv.VideoInfo.from_video_path(source_path)

    task_handler = PlayerTracker(model=model, video_info=video_info, device=device)

    tracker_generator = task_handler.run(source_path)
    player_id_array = next(tracker_generator)
    
    run_preprocess(source_path=source_path, tracker_ids=player_id_array, target_path = target_path)
    
    return player_id_array


if __name__ == "__main__":

    input_video_path = "/home/shailesh/Downloads/vid32.mp4"
    output_dump_path = "artifacts/final_output_video"
    

    final = main(input_video_path, output_dump_path)
