from __future__ import annotations

import json
from typing import Optional, Union

import numpy as np
from llist import sllist

from trackreid.configs.reid_constants import reid_constants
from trackreid.tracked_object_metadata import TrackedObjectMetaData
from trackreid.utils import split_list_around_value


class TrackedObject:
    """
    The TrackedObject class represents an object that is being tracked in a video frame.
    It contains information about the object's state, its unique identifiers, and metadata.

    The object's state is an integer that represents the current state of the object in the
    reid process. The states can take the following values:

    - LOST_FOREVER (-3): "Switcher never rematched"
    - TRACKER_OUTPUT (-2): "Tracker output not in reid process"
    - FILTERED_OUTPUT (-1): "Tracker output entering reid process"
    - STABLE (0): "Stable object"
    - SWITCHER (1): "Lost object to be re-matched"
    - CANDIDATE (2): "New object to be matched"

    The object's unique identifiers are stored in a singly linked list (sllist) called re_id_chain. The re_id_chain
    is a crucial component in the codebase. It stores the history of the object's unique identifiers, allowing for
    tracking of the object across different frames. The first value in the re_id_chain
    is the original object ID, while the last value is the most recent tracker ID assigned to the object.

    The metadata is an instance of the TrackedObjectMetaData class, which contains additional information
    about the object.

    The TrackedObject class provides several methods for manipulating and accessing the data it contains.
    These include methods for merging two TrackedObject instances, updating the metadata, and converting the
    TrackedObject instance to a dictionary or JSON string.

    The TrackedObject class also provides several properties for accessing specific pieces of data, such as the object's
    unique identifier, its state, and its metadata.

    Args:
        object_ids (Union[Union[float, int], sllist]): The unique identifiers for the object.
        state (int): The current state of the object.
        metadata (Union[np.ndarray, TrackedObjectMetaData]): The metadata for the object. It can be either a
        TrackedObjectMetaData object, or a data line, i.e. output of detection model. If metadata is initialized
        with a TrackedObjectMetaData object, a frame_id must be given.
        frame_id (Optional[int], optional): The frame ID where the object was first seen. Defaults to None.

    Raises:
        NameError: If the type of object_ids or metadata is unrecognized.
    """

    def __init__(
        self,
        object_ids: Union[Union[float, int], sllist],
        state: int,
        metadata: Union[np.ndarray, TrackedObjectMetaData],
        frame_id: Optional[int] = None,
    ):
        self.state = state

        if isinstance(object_ids, Union[float, int]):
            self.re_id_chain = sllist([object_ids])
        elif isinstance(object_ids, sllist):
            self.re_id_chain = sllist(object_ids)
        else:
            raise NameError("unrocognized type for object_ids.")
        if isinstance(metadata, np.ndarray):
            assert (
                frame_id is not None
            ), "Please provide a frame_id for TrackedObject initialization"
            self.metadata = TrackedObjectMetaData(metadata, frame_id)
        elif isinstance(metadata, TrackedObjectMetaData):
            self.metadata = metadata.copy()
        else:
            raise NameError("unrocognized type for metadata.")

    def copy(self):
        return TrackedObject(object_ids=self.re_id_chain, state=self.state, metadata=self.metadata)

    def merge(self, other_object):
        if not isinstance(other_object, TrackedObject):
            raise TypeError("Can only merge with another TrackedObject.")

        # Merge the re_id_chains
        self.re_id_chain.extend(other_object.re_id_chain)
        self.metadata.merge(other_object.metadata)
        self.state = other_object.state

        # Return the merged object
        return self

    @property
    def object_id(self):
        """
        Returns the first value in the re_id_chain which represents the object id.
        """
        return self.re_id_chain.first.value

    @property
    def tracker_id(self):
        """
        Returns the last value in the re_id_chain which represents the last tracker id.
        """
        return self.re_id_chain.last.value

    @property
    def category(self):
        """
        Returns the category with the maximum count in the class_counts dictionary of the metadata.
        """
        return max(self.metadata.class_counts, key=self.metadata.class_counts.get)

    @property
    def confidence(self):
        """
        Returns the confidence value from the metadata.
        """
        return self.metadata.confidence

    @property
    def mean_confidence(self):
        """
        Returns the mean confidence value from the metadata.
        """
        return self.metadata.mean_confidence()

    @property
    def bbox(self):
        """
        Returns the bounding box coordinates from the metadata.
        """
        return self.metadata.bbox

    @property
    def nb_ids(self):
        """
        Returns the number of ids in the re_id_chain.
        """
        return len(self.re_id_chain)

    @property
    def nb_corrections(self):
        """
        Returns the number of corrections which is the number of ids in the re_id_chain minus one.
        """
        return self.nb_ids - 1

    def get_age(self, frame_id):
        """
        Calculates and returns the age of the tracked object based on the given frame id.
        Age is defined as the difference between the current frame id and the first frame id where
        the object was detected.
        """
        return frame_id - self.metadata.first_frame_id

    def get_nb_frames_since_last_appearance(self, frame_id):
        """
        Calculates and returns the number of frames since the last appearance of the tracked object.
        This is computed as the difference between the current frame id and the last frame id where
        the object was detected.
        """
        return frame_id - self.metadata.last_frame_id

    def get_state(self):
        """
        Returns the current state of the tracked object.
        """
        return self.state

    def __hash__(self):
        return hash(self.object_id)

    def __repr__(self):
        return (
            f"TrackedObject(current_id={self.object_id}, re_id_chain={list(self.re_id_chain)}"
            + f", state={self.state}: {reid_constants.STATES.DESCRIPTION[self.state]})"
        )

    def __str__(self):
        return f"{self.__repr__()}, metadata : {self.metadata}"

    def update_metadata(self, data_line: np.ndarray, frame_id: int):
        """
        Updates the metadata of the tracked object based on new detection data.

        This method is used to update the metadata of a tracked object whenever new detection data is available.
        It updates the metadata by calling the update method of the TrackedObjectMetaData instance associated with
        the tracked object.

        Args:
            data_line (np.ndarray): The detection data for a single frame. It contains information such as the
            class name, bounding box coordinates, and confidence level of the detection.

            frame_id (int): The frame id where the object was detected. This is used to update the last frame id of
            the tracked object.
        """
        self.metadata.update(data_line=data_line, frame_id=frame_id)

    def __eq__(self, other):
        if isinstance(other, Union[float, int]):
            return other in self.re_id_chain
        elif isinstance(other, TrackedObject):
            return self.re_id_chain == other.re_id_chain
        return False

    def cut(self, object_id: int):
        """
        Splits the re_id_chain of the tracked object at the specified object_id and creates a new TrackedObject
        instance with the remaining part of the re_id_chain. The original TrackedObject instance retains the part
        of the re_id_chain before the specified object_id.

        Args:
            object_id (int): The object_id at which to split the re_id_chain.

        Raises:
            NameError: If the specified object_id is not found in the re_id_chain of the tracked object.

        Returns:
            tuple: A tuple containing the new TrackedObject instance and the original TrackedObject instance.
        """
        if object_id not in self.re_id_chain:
            raise NameError(
                f"Trying to cut object {self} with {object_id} that is not in the re-id chain."
            )

        before, after = split_list_around_value(self.re_id_chain, object_id)
        self.re_id_chain = before

        new_object = TrackedObject(
            state=reid_constants.STATES.STABLE, object_ids=after, metadata=self.metadata
        )
        # set potential age 0 for new object
        new_object.metadata.first_frame_id = new_object.metadata.last_frame_id
        return new_object, self

    def to_dict(self):
        """
        Converts the TrackedObject instance to a dictionary.

        Returns:
            dict: A dictionary representation of the TrackedObject instance.
        """
        data = {
            "object_id": float(self.object_id),
            "state": int(self.state),
            "re_id_chain": list(self.re_id_chain),
            "metadata": self.metadata.to_dict(),
        }
        return data

    def to_json(self):
        """
        Converts the TrackedObject instance to a JSON string.

        Returns:
            str: A JSON string representation of the TrackedObject instance.
        """
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a new TrackedObject instance from a dictionary.

        Args:
            data (dict): A dictionary containing the data for the TrackedObject instance.

        Returns:
            TrackedObject: A new TrackedObject instance created from the dictionary.
        """
        obj = cls.__new__(cls)
        obj.state = data["state"]
        obj.re_id_chain = sllist(data["re_id_chain"])
        obj.metadata = TrackedObjectMetaData.from_dict(data["metadata"])
        return obj

    @classmethod
    def from_json(cls, json_str: str):
        """
        Creates a new TrackedObject instance from a JSON string.

        Args:
            json_str (str): A JSON string containing the data for the TrackedObject instance.

        Returns:
            TrackedObject: A new TrackedObject instance created from the JSON string.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
