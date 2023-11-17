import numpy as np

from trackreid.tracked_object import TrackedObject


def bounding_box_distance(candidate: TrackedObject, switcher: TrackedObject):
    # Get the bounding boxes from the Metadata of each TrackedObject
    bbox1 = candidate.metadata.bbox
    bbox2 = switcher.metadata.bbox

    # Calculate the Euclidean distance between the centers of the bounding boxes
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    return distance
