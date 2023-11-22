import json

import numpy as np

from trackreid.configs.input_data_positions import input_data_positions


class TrackedObjectMetaData:
    """
    The TrackedObjectMetaData class is used to store and manage metadata for tracked objects in a video frame.
    This metadata includes information such as the frame ID where the object was first seen, the class counts
    (how many times each class was detected), the bounding box coordinates, and the confidence level of the detection.

    This metadata is then use in selection and cost functions to compute likelihood of a match between two objects.

    Usage:
    An instance of TrackedObjectMetaData is created by passing a data_line (which contains the detection data
    for a single frame) and a frame_id (which identifies the frame where the object was detected).
    """

    def __init__(self, data_line: np.ndarray, frame_id: int):
        self.first_frame_id = frame_id
        self.class_counts = {}
        self.observations = 0
        self.confidence_sum = 0
        self.confidence = 0
        self.update(data_line, frame_id)

    def update(self, data_line: np.ndarray, frame_id: int):
        """
        Updates the metadata of a tracked object based on new detection data.

        This method is used to update the metadata of a tracked object whenever new detection data is available.
        It updates the last frame id, class counts, bounding box, confidence, confidence sum, and observations.

        Args:
            data_line (np.ndarra): The detection data for a single frame. It contains information such as the
            class name, bounding box coordinates, and confidence level of the detection.

            frame_id (int): The frame id where the object was detected. This is used to update the last frame id of
            the tracked object.

        Updates:
            last_frame_id: The last frame id is updated to the frame id where the object was detected.

            class_counts: The class counts are updated by incrementing the count of the detected class by 1.

            bbox: The bounding box is updated to the bounding box coordinates from the detection data.

            confidence: The confidence is updated to the confidence level from the detection data.

            confidence_sum: The confidence sum is updated by adding the confidence level from the detection data.

            observations: The total number of observations is incremented by 1.
        """
        self.last_frame_id = frame_id

        class_name = int(data_line[input_data_positions.category])
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        self.bbox = list(map(int, data_line[input_data_positions.bbox]))
        confidence = float(data_line[input_data_positions.confidence])
        self.confidence = confidence
        self.confidence_sum += confidence
        self.observations += 1

    def merge(self, other_object):
        """
        Merges the metadata of another TrackedObjectMetaData instance into the current one.

        Args:
            other_object (TrackedObjectMetaData): The other TrackedObjectMetaData instance whose metadata
            is to be merged with the current instance.

        Raises:
            TypeError: If the other_object is not an instance of TrackedObjectMetaData.

        Updates:
            observations: The total number of observations is updated by adding the observations of the other object.
            confidence_sum: The total confidence sum is updated by adding the confidence sum of the other object.
            confidence: The confidence is updated to the confidence of the other object.
            bbox: The bounding box is updated to the bounding box of the other object.
            last_frame_id: The last frame id is updated to the last frame id of the other object.
            class_counts: The class counts are updated by adding the class counts of the other object for each class.
        """
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
        """
        Creates a copy of the current TrackedObjectMetaData instance.

        Returns:
            TrackedObjectMetaData: A new instance of TrackedObjectMetaData with the same
            properties as the current instance.
        """
        copy_obj = TrackedObjectMetaData.__new__(TrackedObjectMetaData)
        copy_obj.bbox = self.bbox.copy()
        copy_obj.class_counts = self.class_counts.copy()
        copy_obj.observations = self.observations
        copy_obj.confidence_sum = self.confidence_sum
        copy_obj.confidence = self.confidence
        copy_obj.first_frame_id = self.first_frame_id
        copy_obj.last_frame_id = self.last_frame_id

        return copy_obj

    def to_dict(self):
        """
        Converts the TrackedObjectMetaData instance to a dictionary.

        The class_counts dictionary is converted to a string-keyed dictionary.
        The bounding box list is converted to a list of integers.
        The first_frame_id, last_frame_id, confidence, confidence_sum, and observations are converted to their
        respective types.

        Returns:
            dict: A dictionary representation of the TrackedObjectMetaData instance.
        """
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
        """
        Converts the TrackedObjectMetaData instance to a JSON string.

        Returns:
            str: A JSON string representation of the TrackedObjectMetaData instance.
        """
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a new instance of the class from a dictionary.

        The dictionary should contain the following keys: "first_frame_id", "last_frame_id", "class_counts",
        "bbox", "confidence", "confidence_sum", and "observations". The "class_counts" key should map to a
        dictionary where the keys are class names (as integers) and the values are counts.

        Args:
            data (dict): A dictionary containing the data to populate the new instance.

        Returns:
            TrackedObjectMetaData: A new instance of TrackedObjectMetaData populated with the data from the dictionary.
        """
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
        """
        Creates a new instance of the class from a JSON string.

        Args:
            json_str (str): A JSON string representation of the TrackedObjectMetaData instance.

        Returns:
            TrackedObjectMetaData: A new instance of TrackedObjectMetaData populated with the data from the JSON string.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def class_proportions(self):
        """
        Calculates the proportions of each class in the tracked object.

        Returns:
            dict: A dictionary where the keys are class names and the values are the proportions of each class.
        """
        if self.observations > 0:
            proportions = {
                class_name: count / self.observations
                for class_name, count in self.class_counts.items()
            }
        else:
            proportions = None
        return proportions

    def percentage_of_time_seen(self, frame_id: int):
        """
        Calculates the percentage of time the tracked object has been seen.

        Args:
            frame_id (int): The current frame id.

        Returns:
            float: The percentage of time the tracked object has been seen.
        """
        if self.observations > 0:
            percentage = (self.observations / (frame_id - self.first_frame_id + 1)) * 100
        else:
            percentage = 0.0
        return percentage

    def mean_confidence(self):
        """
        Calculates the mean confidence of the tracked object.

        Returns:
            float: The mean confidence of the tracked object.
        """
        if self.observations > 0:
            return self.confidence_sum / self.observations
        else:
            return 0.0

    def __repr__(self) -> str:
        """
        Returns a string representation of the TrackedObjectMetaData instance.

        Returns:
            str: A string representation of the TrackedObjectMetaData instance.
        """
        return f"TrackedObjectMetaData(bbox={self.bbox})"

    def __str__(self):
        """
        Returns a string representation of the TrackedObjectMetaData instance.

        Returns:
            str: A string representation of the TrackedObjectMetaData instance.
        """
        return (
            f"First frame seen: {self.first_frame_id}, nb observations: {self.observations}, "
            + f"class proportions: {self.class_proportions()}, bbox: {self.bbox}, "
            + f"mean confidence: {self.mean_confidence()}"
        )
