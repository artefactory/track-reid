import cv2
import numpy as np


def draw_all_bbox_on_image(image, tracking_objects: np.ndarray):
    """
    A list of of detections with track id, class id and confidence.
            [
                [x, y, x, y, track_id, class_id, conf],
                [x, y, x, y, track_id, class_id, conf],
                ...
            ]

    Plot this on the image with the track id, class id and confidence.
    """
    for detection in tracking_objects:
        x1, y1, x2, y2, track_id, class_id, conf = detection
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"ID: {track_id}, Class: {class_id}, Conf: {conf}",
            (x1, y1 - 10),
            0,
            1,
            (0, 255, 0),
            2,
        )
    return image


def yolo_results_with_track_id(detections, tracking_output: np.ndarray):
    """Transforms YOLO detections and tracking info into easy to plot format.

    Args:
        detections: A list of YOLO detections.
        tracking_results: The output of tracking with format [x1, y1, x2, y2, track_id, class_id, conf].


    Returns:
        A list of of detections with track id and class.
            [
                [x, y, x, y, track_id, class, conf],
                [x, y, x, y, track_id, class, conf],
                ...
            ]
    """
    # Check if all arrays are the same length
    if len(detections) != len(tracking_output):
        raise ValueError("Length of detections and tracking_output must be the same.")

    if len(tracking_output) == 0:
        return np.array([])
    boxes = detections.numpy().boxes.xyxy
    scores = detections.numpy().boxes.conf
    classes = detections.numpy().boxes.cls
    return np.stack(
        [
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2],
            boxes[:, 3],
            tracking_output[:, 4],
            classes,
            scores,
        ],
        axis=1,
    )


def yolo_results_to_bytetrack_format(detections):
    """Transforms YOLO detections into the bytetrack format.

    Args:
        detections: A list of YOLO detections.

    Returns:
        A list of bytetrack detections.
    """
    boxes = detections.numpy().boxes.xyxyn
    scores = detections.numpy().boxes.conf
    classes = detections.numpy().boxes.cls
    return np.stack(
        [
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
            scores,
            classes,
        ],
        axis=1,
    )
