import json
from pathlib import Path

from track_reid.args.reid_args import POSSIBLE_CLASSES


class TrackedObjectMetaData:
    def __init__(self, data_line):
        self.first_frame_id = int(data_line[0])
        self.class_counts = {class_name: 0 for class_name in POSSIBLE_CLASSES}
        self.observations = 0
        self.confidence_sum = 0
        self.confidence = 0
        self.update(data_line)

    def update(self, data_line):
        self.last_frame_id = int(data_line[0])
        class_name = data_line[2]
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        self.bbox = list(map(int, data_line[3:7]))
        confidence = float(data_line[7])
        self.confidence = confidence
        self.confidence_sum += confidence
        self.observations += 1

    def merge(self, other_object):
        if not isinstance(other_object, TrackedObjectMetaData):
            raise TypeError("Can only merge with another TrackedObjectMetaData.")

        self.observations += other_object.observations
        self.confidence_sum += other_object.confidence_sum
        self.confidence = other_object.confidence
        self.bbox = other_object.bbox
        self.last_frame_id = other_object.last_frame_id
        for class_name in POSSIBLE_CLASSES:
            self.class_counts[class_name] = self.class_counts.get(
                class_name, 0
            ) + other_object.class_counts.get(class_name, 0)

    def copy(self):
        # Create a new instance of TrackedObjectMetaData with the same data
        copy_obj = TrackedObjectMetaData(
            [self.first_frame_id, 0, list(self.class_counts.keys())[0], *self.bbox, self.confidence]
        )
        # Update the copied instance with the actual class counts and observations
        copy_obj.class_counts = self.class_counts.copy()
        copy_obj.observations = self.observations
        copy_obj.confidence_sum = self.confidence_sum
        copy_obj.confidence = self.confidence

        return copy_obj

    def save_to_json(self, filename):
        data = {
            "first_frame_id": self.first_frame_id,
            "class_counts": self.class_counts,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "confidence_sum": self.confidence_sum,
            "observations": self.observations,
        }

        with Path.open(filename, "w") as file:
            json.dump(data, file)

    @classmethod
    def load_from_json(cls, filename):
        with Path.open(filename, "r") as file:
            data = json.load(file)
            obj = cls.__new__(cls)
            obj.first_frame_id = data["first_frame_id"]
            obj.class_counts = data["class_counts"]
            obj.bbox = data["bbox"]
            obj.confidence = data["confidence"]
            obj.confidence_sum = data["confidence_sum"]
            obj.observations = data["observations"]
            return obj

    def class_proportions(self):
        if self.observations > 0:
            proportions = {
                class_name: count / self.observations
                for class_name, count in self.class_counts.items()
            }
        else:
            proportions = {class_name: 0.0 for class_name in POSSIBLE_CLASSES}
        return proportions

    def percentage_of_time_seen(self, frame_id):
        if self.observations > 0:
            percentage = (self.observations / (frame_id - self.first_frame_id + 1)) * 100
        else:
            percentage = 0.0
        return percentage

    def mean_confidence(self):
        if self.observations > 0:
            return self.confidence_sum / self.observations
        else:
            return 0.0

    def __repr__(self) -> str:
        return f"TrackedObjectMetaData(bbox={self.bbox})"

    def __str__(self):
        return (
            f"First frame seen: {self.first_frame_id}, nb observations: {self.observations}, "
            + "class Proportions: {self.class_proportions()}, Bounding Box: {self.bbox}, "
            + "Mean Confidence: {self.mean_confidence()}"
        )
