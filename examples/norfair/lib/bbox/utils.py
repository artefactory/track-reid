import numpy as np


def xy_center_to_xyxy(bbox: np.array) -> np.array:
    """Convert bounding box from xy_center to xyxy format"""
    return np.array(
        [
            bbox[0] - bbox[2] / 2,
            bbox[1] - bbox[3] / 2,
            bbox[0] + bbox[2] / 2,
            bbox[1] + bbox[3] / 2,
        ]
    )


def rescale_bbox(bbox: np.array, original_size: tuple) -> np.array:
    """Rescale bounding box from current_size to original_size"""
    return np.array(
        [
            bbox[0] * original_size[0],
            bbox[1] * original_size[1],
            bbox[2] * original_size[0],
            bbox[3] * original_size[1],
        ]
    )
