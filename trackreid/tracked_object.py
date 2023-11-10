from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from llist import sllist

from trackreid.constants.reid_constants import reid_constants
from trackreid.tracked_object_metadata import TrackedObjectMetaData
from trackreid.utils import split_list_around_value


class TrackedObject:
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
            self.re_id_chain = object_ids
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
        return self.re_id_chain.first.value

    @property
    def category(self):
        return max(self.metadata.class_counts, key=self.metadata.class_counts.get)

    @property
    def confidence(self):
        return self.metadata.confidence

    @property
    def mean_confidence(self):
        return self.metadata.mean_confidence()

    @property
    def bbox(self):
        return self.metadata.bbox

    @property
    def nb_corrections(self):
        return len(self.re_id_chain)

    def get_age(self, frame_id):
        return frame_id - self.metadata.first_frame_id

    def get_nb_frames_since_last_appearance(self, frame_id):
        return frame_id - self.metadata.last_frame_id

    def get_state(self):
        return self.state

    def __hash__(self):
        return hash(self.object_id)

    def __repr__(self):
        return (
            f"TrackedObject(current_id={self.object_id}, re_id_chain={list(self.re_id_chain)}"
            + f", state={self.state}: {reid_constants.DESCRIPTION[self.state]})"
        )

    def __str__(self):
        return f"{self.__repr__()}, metadata : {self.metadata}"

    def update_metadata(self, data_line: np.ndarray, frame_id: int):
        self.metadata.update(data_line=data_line, frame_id=frame_id)

    def __eq__(self, other):
        if isinstance(other, Union[float, int]):
            return other in self.re_id_chain
        elif isinstance(other, TrackedObject):
            return self.re_id_chain == other.re_id_chain
        return False

    def cut(self, object_id: int):
        if object_id not in self.re_id_chain:
            raise NameError(
                f"Trying to cut object {self} with {object_id} that is not in the re-id chain."
            )

        before, after = split_list_around_value(self.re_id_chain, object_id)
        self.re_id_chain = before

        new_object = TrackedObject(
            state=reid_constants.STABLE, object_ids=after, metadata=self.metadata
        )
        return new_object, self

    def format_data(self):
        return [
            self.object_id,
            self.category,
            self.bbox[0],
            self.bbox[1],
            self.bbox[2],
            self.bbox[3],
            self.confidence,
        ]

    def to_dict(self):
        data = {
            "state": self.state,
            "re_id_chain": list(self.re_id_chain),
            "metadata": self.metadata.to_dict(),
        }
        return data

    def save_to_json(self, filename):
        with Path.open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    @classmethod
    def load_from_json(cls, filename):
        with Path.open(filename, "r") as file:
            data = json.load(file)
            obj = cls.__new__(cls)
            obj.state = data["state"]
            obj.re_id_chain = sllist(data["re_id_chain"])
            obj.metadata = TrackedObjectMetaData.load_from_json(data["metadata"])
        return obj
