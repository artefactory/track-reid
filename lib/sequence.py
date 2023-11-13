from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class Sequence:
    """This class represents a sequence of frames and detections and make the data iterable."""

    frame_paths: List[str]

    def __post_init__(self):
        self.detection_paths: List[str] = [
            f.replace("frames", "detections").replace(".jpg", ".txt") for f in self.frame_paths
        ]

    def __repr__(self):
        return (
            f"Sequence(n_frames={len(self.frame_paths)}, n_detections={len(self.detection_paths)})"
        )

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.frame_paths):
            raise StopIteration

        try:
            frame = Image.open(self.frame_paths[self.index])
        except OSError:  # file doesn't exist not detection return empty file
            frame = np.array([])

        try:
            detection = np.loadtxt(self.detection_paths[self.index], dtype="float")
        except OSError:  # file doesn't exist not detection return empty file
            detection = np.array([])

        self.index += 1
        return frame, detection
