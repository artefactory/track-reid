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
        x1, y1, x2, y2, track_id, _, conf = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{int(track_id)} ({conf:.2f})",
            (x1, y1 - 10),
            0,
            1,
            (0, 255, 0),
            2,
        )
    return image


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
            boxes[:, 2],
            boxes[:, 3],
            scores,
            classes,
        ],
        axis=1,
    )


def scale_bbox_as_xyxy(bbox: np.ndarray, target_img_size: tuple):
    """Scales a bounding box to a target image size.

    Args:
        bbox: A bounding box in the format [x, y, x, y].
        target_img_size: The target image size as a tuple (h, W).

    Returns:
        The scaled bounding box.
    """
    x1, y1, x2, y2 = bbox
    h, w = target_img_size
    scaled_bbox = np.array([x1 * w, y1 * h, x2 * w, y2 * h])
    return scaled_bbox
