import cv2
import subprocess
from tqdm import tqdm


def resize_to_square(frame):
    h, w = frame.shape[:2]
    size = max(h, w)
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, (h, w)

def add_audio_to_video(video_path, output_path, audio_path="utils/short-audio-file.mp3"):
    # ffmpeg command to add audio to video
    command = [
        'ffmpeg',
        '-i', video_path,        # Input video file
        '-i', audio_path,        # Input audio file
        '-c:v', 'libx264',              # Use H.264 video codec
        '-profile:v', 'high',           # Set profile to High
        '-preset', 'fast',              # Set encoding speed/quality balance        
        '-c:a', 'aac',           # Encode audio to AAC format
        '-shortest',             # Stop encoding when the shortest stream ends
        '-loglevel', 'error',
        '-movflags', 'faststart',       # Allow for quicker playback on the web
        output_path,
        '-y'
    ]

    # Run the command
    subprocess.run(command, check=True)

def run_preprocess(source_path, tracker_ids, target_path):
    # tqdm_iter = tqdm(video_path)
    # for each_name in tqdm_iter:
    each_name = source_path.split("/")[-1].replace(".mp4","")
    myvar = tracker_ids

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .avi files; use 'mp4v' for .mp4 files
    final_shape = (150,250)
    video = cv2.VideoCapture(source_path)

    # Pre-compute constants
    frame_store = {key: [] for key in set(myvar[:, 1])}
    frame_rate = 5  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    p_id = 1

    # Loop over frames and track IDs to create cropped images
    last_frame = -1
    for f_no, track_id, x_min, y_min, x_max, y_max in myvar[:, 0:6]:
        # Only read a new frame when the frame number changes
        if last_frame != f_no:
            last_frame = f_no
            _, frame = video.read()
            resized_image, orgi_size = resize_to_square(frame)
            height, width, _ = resized_image.shape  # Get dimensions only once

        # Clamp bounding box coordinates within image dimensions
        x_min, y_min = int(max(0, x_min)), int(max(0, y_min))
        x_max, y_max = int(min(width, x_max)), int(min(height, y_max))
        # cropped_image = resized_image[y_min:y_max, x_min:x_max]
        frame_store[track_id].append(resized_image[y_min:y_max, x_min:x_max])

    # Create video files for each tracker
    tracker_id_iter = tqdm(frame_store.items())
    for frame, p_images in tracker_id_iter:
        
        tracker_id_iter.set_description_str(f"Working with tracker {p_id}")
        # Skip frames with fewer than 5 images
        if len(p_images) <= 5:
            continue

        # Define output paths
        output_video_path = f'artifacts/temp/output_{each_name}_{frame}_{p_id}.mp4'
        output_path = f'{target_path}/output_{each_name}_{frame}_{p_id}.mp4'

        # Initialize video writer with the shape of resized images
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, final_shape)

        # Write each cropped image to the video file
        for img in p_images:
            resized_img = cv2.resize(img, final_shape)
            video_writer.write(resized_img)

        # Release resources after writing the video
        video_writer.release()
        add_audio_to_video(output_video_path, output_path)

        p_id += 1

    print(f"Completed video {each_name} !")
    return 0

