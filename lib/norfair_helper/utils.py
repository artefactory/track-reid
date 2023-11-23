from typing import List

import cv2
import numpy as np
from norfair import Detection, get_cutout

from lib.bbox.utils import rescale_bbox, xy_center_to_xyxy


def yolo_to_norfair_detection(
    yolo_detections: np.array,
    original_img_size: tuple,
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []
    for detection_output in yolo_detections:
        bbox = np.array(
            [
                [detection_output[1].item(), detection_output[2].item()],
                [detection_output[3].item(), detection_output[4].item()],
            ]
        )
        bbox = xy_center_to_xyxy(bbox.flatten()).reshape(2, 2)
        bbox = rescale_bbox(bbox.flatten(), original_img_size).reshape(2, 2)
        scores = np.array([detection_output[5].item(), detection_output[5].item()])
        norfair_detections.append(Detection(points=bbox, scores=scores, label=detection_output[0]))
    return norfair_detections


def compute_embeddings(norfair_detections: List[Detection], image: np.array):
    """
    Add embedding attribute to all Detection objects in norfair_detections.
    """
    for detection in norfair_detections:
        object = get_cutout(detection.points, image)
        if object.shape[0] > 0 and object.shape[1] > 0:
            detection.embedding = get_hist(object)
    return norfair_detections


def get_hist(image: np.array):
    """Compute an embedding with histograms"""
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()
