from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class Sequence:
    """This class represents a sequence of frames and detections and make the data iterable."""

    frame_paths: List[str]
    detection_path: str

    def __repr__(self):
        return (
            f"Sequence(n_frames={len(self.frame_paths)}, n_detections={len(self.detection_path)})"
        )

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.frame_paths):
            raise StopIteration

        frame = Image.open(self.frame_paths[self.index])
        detection_file = self.detection_path[self.index]
        detection = np.loadtxt(detection_file, dtype="float")

        self.index += 1
        return frame, detection
