import json

from trackreid.configs.input_data_positions import input_data_postitions


class TrackedObjectMetaData:
    def __init__(self, data_line, frame_id):
        self.first_frame_id = frame_id
        self.class_counts = {}
        self.observations = 0
        self.confidence_sum = 0
        self.confidence = 0
        self.update(data_line, frame_id)

    def update(self, data_line, frame_id):
        self.last_frame_id = frame_id

        class_name = int(data_line[input_data_postitions.category])
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        self.bbox = list(map(int, data_line[input_data_postitions.bbox]))
        confidence = float(data_line[input_data_postitions.confidence])
        self.confidence = confidence
        self.confidence_sum += confidence
        self.observations += 1

    def merge(self, other_object):
        if not isinstance(other_object, type(self)):
            raise TypeError("Can only merge with another TrackedObjectMetaData.")

        self.observations += other_object.observations
        self.confidence_sum += other_object.confidence_sum
        self.confidence = other_object.confidence
        self.bbox = other_object.bbox
        self.last_frame_id = other_object.last_frame_id
        for class_name in other_object.class_counts.keys():
            self.class_counts[class_name] = self.class_counts.get(
                class_name, 0
            ) + other_object.class_counts.get(class_name, 0)

    def copy(self):
        copy_obj = TrackedObjectMetaData.__new__(TrackedObjectMetaData)
        # Update the copied instance with the actual class counts and observations
        copy_obj.bbox = self.bbox.copy()
        copy_obj.class_counts = self.class_counts.copy()
        copy_obj.observations = self.observations
        copy_obj.confidence_sum = self.confidence_sum
        copy_obj.confidence = self.confidence
        copy_obj.first_frame_id = self.first_frame_id
        copy_obj.last_frame_id = self.last_frame_id

        return copy_obj

    def to_dict(self):
        class_counts_str = {
            str(class_name): count for class_name, count in self.class_counts.items()
        }
        data = {
            "first_frame_id": int(self.first_frame_id),
            "last_frame_id": int(self.last_frame_id),
            "class_counts": class_counts_str,
            "bbox": [int(i) for i in self.bbox],
            "confidence": float(self.confidence),
            "confidence_sum": float(self.confidence_sum),
            "observations": int(self.observations),
        }
        return data

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, data: dict):
        class_counts_str = data["class_counts"]
        class_counts = {int(class_name): count for class_name, count in class_counts_str.items()}
        obj = cls.__new__(cls)
        obj.first_frame_id = data["first_frame_id"]
        obj.last_frame_id = data["last_frame_id"]
        obj.class_counts = class_counts
        obj.bbox = data["bbox"]
        obj.confidence = data["confidence"]
        obj.confidence_sum = data["confidence_sum"]
        obj.observations = data["observations"]
        return obj

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    def class_proportions(self):
        if self.observations > 0:
            proportions = {
                class_name: count / self.observations
                for class_name, count in self.class_counts.items()
            }
        else:
            proportions = None
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
            + f"class proportions: {self.class_proportions()}, bbox: {self.bbox}, "
            + f"mean confidence: {self.mean_confidence()}"
        )
