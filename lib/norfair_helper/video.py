import cv2
import numpy as np
from norfair import Tracker, draw_boxes

from lib.norfair_helper.utils import compute_embeddings, yolo_to_norfair_detection
from lib.sequence import Sequence


def generate_tracking_video(
    sequence: Sequence,
    tracker: Tracker,
    frame_size: tuple,
    output_path: str,
    add_embedding: bool = False,
) -> str:
    """
    Generate a video with the tracking results.

    Args:
        sequence: The sequence of frames and detections.
        tracker: The tracker to use.
        frame_size: The size of the frames.
        output_path: The path to save the video to.
        add_embedding: Whether to add the embedding to the video.

    Returns:
        The path to the video.

    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Changed codec to 'mp4v' for compatibility with Mac
    out = cv2.VideoWriter(output_path, fourcc, 20.0, frame_size)  # Changed file extension to .mp4

    for frame, detection in sequence:
        frame = np.array(frame)
        detections_list = yolo_to_norfair_detection(detection, frame_size)
        if add_embedding:
            detections_list = compute_embeddings(detections_list, frame)
        tracked_objects = tracker.update(detections=detections_list)
        frame_detected = draw_boxes(frame, tracked_objects, draw_ids=True, color="by_label")
        frame_detected = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB)
        out.write(frame_detected)
    out.release()
    return output_path
