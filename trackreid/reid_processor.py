from __future__ import annotations

from typing import Dict, List, Set, Union

import numpy as np

from trackreid.args.reid_args import INPUT_POSITIONS, OUTPUT_POSITIONS
from trackreid.constants.reid_constants import reid_constants
from trackreid.matcher import Matcher
from trackreid.tracked_object import TrackedObject
from trackreid.tracked_object_filter import TrackedObjectFilter
from trackreid.utils import (
    filter_objects_by_state,
    get_nb_output_cols,
    get_top_list_correction,
    reshape_tracker_result,
)


class ReidProcessor:
    def __init__(
        self,
        filter_confidence_threshold,
        filter_time_threshold,
        cost_function,
        selection_function,
        max_frames_to_rematch: int = 100,
        max_attempt_to_rematch: int = 1,
    ) -> None:
        self.matcher = Matcher(cost_function=cost_function, selection_function=selection_function)

        self.tracked_filter = TrackedObjectFilter(
            confidence_threshold=filter_confidence_threshold,
            frames_seen_threshold=filter_time_threshold,
        )

        self.all_tracked_objects: List[TrackedObject] = []
        self.last_frame_tracked_objects: Set[TrackedObject] = set()

        self.switchers: List[TrackedObject] = []
        self.candidates: List[TrackedObject] = []

        self.max_frames_to_rematch = max_frames_to_rematch
        self.max_attempt_to_rematch = max_attempt_to_rematch

        self.frame_id = 0
        self.nb_output_cols = get_nb_output_cols(output_positions=OUTPUT_POSITIONS)

    def process(self, tracker_output: np.ndarray, frame_id: int):
        if tracker_output.size:  # empty tracking
            reshaped_tracker_output = reshape_tracker_result(tracker_output=tracker_output)
            self.all_tracked_objects = self._preprocess(
                tracker_output=reshaped_tracker_output, frame_id=frame_id
            )
            self._perform_reid_process(tracker_output=reshaped_tracker_output)
            reid_output = self._postprocess(tracker_output=tracker_output)
            return reid_output
        else:
            return tracker_output

    def _preprocess(self, tracker_output: np.ndarray, frame_id: int) -> List["TrackedObject"]:
        self.all_tracked_objects = self._update_tracked_objects(
            tracker_output=tracker_output, frame_id=frame_id
        )
        self.all_tracked_objects = self._apply_filtering()
        return self.all_tracked_objects

    def _update_tracked_objects(self, tracker_output: np.ndarray, frame_id: int):
        self.frame_id = frame_id
        for object_id, data_line in zip(
            tracker_output[:, INPUT_POSITIONS["object_id"]], tracker_output
        ):
            if object_id not in self.all_tracked_objects:
                new_tracked_object = TrackedObject(
                    object_ids=object_id,
                    state=reid_constants.TRACKER_OUTPUT,
                    frame_id=frame_id,
                    metadata=data_line,
                )
                self.all_tracked_objects.append(new_tracked_object)
            else:
                self.all_tracked_objects[self.all_tracked_objects.index(object_id)].update_metadata(
                    data_line, frame_id=frame_id
                )

        return self.all_tracked_objects

    def _get_current_tracked_objects(self, current_tracker_ids: Set[Union[int, float]]):
        tracked_objects = filter_objects_by_state(
            self.all_tracked_objects, states=reid_constants.TRACKER_OUTPUT, exclusion=True
        )

        current_tracked_objects = set(
            [tracked_id for tracked_id in tracked_objects if tracked_id in current_tracker_ids]
        )

        return tracked_objects, current_tracked_objects

    def _apply_filtering(self):
        for tracked_object in self.all_tracked_objects:
            self.tracked_filter.update(tracked_object)

        return self.all_tracked_objects

    def _perform_reid_process(self, tracker_output: np.ndarray):
        current_tracker_ids: List[Union[int, float]] = list(
            tracker_output[:, INPUT_POSITIONS["object_id"]]
        )

        # TODO: we can get rid of self.switchers and self.candidates by
        # applying:
        # candidates = filter_objects_by_state(
        #     self.all_tracked_objects, states=reid_constants.CANDIDATE, exclusion=False
        # )
        # switchers = filter_objects_by_state(
        #     self.all_tracked_objects, states=reid_constants.SWITCHER, exclusion=False
        # )

        self.all_tracked_objects, self.switchers = self.correct_reid_chains(
            all_tracked_objects=self.all_tracked_objects,
            current_tracker_ids=current_tracker_ids,
            switchers=self.switchers,
        )

        tracked_objects, current_tracked_objects = self._get_current_tracked_objects(
            current_tracker_ids=current_tracker_ids
        )

        self.switchers = self.drop_switchers(
            switchers=self.switchers,
            current_tracked_objects=current_tracked_objects,
            max_frames_to_rematch=self.max_frames_to_rematch,
            frame_id=self.frame_id,
        )

        self.candidates = self.drop_candidates(
            self.candidates, self.max_attempt_to_rematch, self.frame_id
        )

        self.candidates.extend(self.identify_candidates(tracked_objects=tracked_objects))

        self.switchers.extend(
            self.identify_switchers(
                current_tracked_objects=current_tracked_objects,
                last_frame_tracked_objects=self.last_frame_tracked_objects,
                all_tracked_objects=self.all_tracked_objects,
            )
        )

        matches = self.matcher.match(self.candidates, self.switchers)

        self.all_tracked_objects, self.switchers, self.candidates = self.process_matches(
            all_tracked_objects=self.all_tracked_objects,
            matches=matches,
            candidates=self.candidates,
            switchers=self.switchers,
        )

        _, current_tracked_objects = self._get_current_tracked_objects(
            current_tracker_ids=current_tracker_ids
        )

        self.last_frame_tracked_objects = current_tracked_objects.copy()

    @staticmethod
    def identify_switchers(
        all_tracked_objects: List["TrackedObject"],
        current_tracked_objects: Set["TrackedObject"],
        last_frame_tracked_objects: Set["TrackedObject"],
    ):
        switchers = []
        lost_ids = last_frame_tracked_objects - current_tracked_objects

        for tracked_id in all_tracked_objects:
            if tracked_id in lost_ids:
                switchers.append(tracked_id)
                tracked_id.state = reid_constants.SWITCHER

        return switchers

    @staticmethod
    def identify_candidates(tracked_objects: List["TrackedObject"]):
        candidates = []
        for current_object in tracked_objects:
            if current_object.state == reid_constants.FILTERED_OUTPUT:
                current_object.state = reid_constants.CANDIDATE
                candidates.append(current_object)
        return candidates

    @staticmethod
    def correct_reid_chains(
        all_tracked_objects: List["TrackedObject"],
        current_tracker_ids: List[Union[int, float]],
        switchers: List["TrackedObject"],
    ):
        top_list_correction = get_top_list_correction(all_tracked_objects)

        for current_object in current_tracker_ids:
            tracked_id = all_tracked_objects[all_tracked_objects.index(current_object)]
            object_state = tracked_id.state
            if current_object not in top_list_correction:
                all_tracked_objects.remove(tracked_id)
                if object_state == reid_constants.SWITCHER:
                    switchers.remove(tracked_id)

                new_object, tracked_id = tracked_id.cut(current_object)

                tracked_id.state = reid_constants.STABLE
                all_tracked_objects.append(tracked_id)

                # 2 cases to take :
                if new_object in current_tracker_ids:
                    new_object.state = reid_constants.STABLE
                    all_tracked_objects.append(new_object)

                elif new_object.nb_corrections > 1:
                    new_object.state = reid_constants.SWITCHER
                    switchers.append(new_object)
                    all_tracked_objects.append(new_object)

        return all_tracked_objects, switchers

    @staticmethod
    def process_matches(
        all_tracked_objects: List["TrackedObject"],
        matches: Dict["TrackedObject", "TrackedObject"],
        switchers: List["TrackedObject"],
        candidates: List["TrackedObject"],
    ):
        for match in matches:
            candidate_match, switcher_match = match.popitem()

            switcher_match.merge(candidate_match)
            switcher_match.state = reid_constants.STABLE
            all_tracked_objects.remove(candidate_match)
            switchers.remove(switcher_match)
            candidates.remove(candidate_match)

        return all_tracked_objects, switchers, candidates

    @staticmethod
    def drop_switchers(
        switchers: List["TrackedObject"],
        current_tracked_objects: Set["TrackedObject"],
        max_frames_to_rematch: int,
        frame_id: int,
    ):
        switchers_to_drop = set(switchers).intersection(current_tracked_objects)
        filtered_switchers = switchers.copy()

        for switcher in switchers:
            if switcher in switchers_to_drop:
                switcher.state = reid_constants.STABLE
                filtered_switchers.remove(switcher)
            elif switcher.get_nb_frames_since_last_appearance(frame_id) > max_frames_to_rematch:
                switcher.state = reid_constants.LOST_FOREVER
                filtered_switchers.remove(switcher)

        return filtered_switchers

    @staticmethod
    def drop_candidates(
        candidates: List["TrackedObject"], max_attempt_to_rematch: int, frame_id: int
    ):
        filtered_candidates = candidates.copy()
        # for now drop candidates if there was no match
        for candidate in filtered_candidates:
            if candidate.get_age(frame_id) >= max_attempt_to_rematch:
                candidate.state = reid_constants.STABLE
                candidates.remove(candidate)
        return candidates

    def _postprocess(self, tracker_output: np.ndarray):
        filtered_objects = list(
            filter(
                lambda obj: obj.get_state() == reid_constants.STABLE
                and obj in tracker_output[:, INPUT_POSITIONS["object_id"]],
                self.all_tracked_objects,
            )
        )

        reid_output = np.zeros((len(filtered_objects), self.nb_output_cols))

        for idx, object in enumerate(filtered_objects):
            for required_variable in OUTPUT_POSITIONS:
                if required_variable == "frame_id":
                    output = self.frame_id
                else:
                    try:
                        output = getattr(object, required_variable)
                    except:  # noqa: E722
                        raise NameError(
                            f"Attribute {required_variable} not in TrackedObject.Check your required output names."
                        )

                reid_output[idx, OUTPUT_POSITIONS[required_variable]] = output

        return reid_output

    def to_dict(self):
        data = dict()
        for tracked_object in self.all_tracked_objects:
            data[tracked_object.object_id] = tracked_object.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls.__new__(cls)
        obj.all_tracked_objects = [
            TrackedObject.from_dict(data_object) for data_object in data.values()
        ]
        return obj