from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
from track_reid.constants.reid_constants import reid_constants
from track_reid.matcher import Matcher
from track_reid.tracked_object import TrackedObject
from track_reid.tracked_object_filter import TrackedObjectFilter
from track_reid.utils import filter_objects_by_state, get_top_list_correction


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
        self.switchers: List[TrackedObject] = []
        self.candidates: List[TrackedObject] = []

        self.last_tracker_ids: Set[int] = set()

        self.max_frames_to_rematch = max_frames_to_rematch
        self.max_attempt_to_rematch = max_attempt_to_rematch

        self.frame_id = 0

    def update(self, tracker_output: np.ndarray):
        reshaped_tracker_output = self._reshape_input(tracker_output)
        self._preprocess(tracker_output=reshaped_tracker_output)
        self._perform_reid_process(tracker_output=reshaped_tracker_output)
        reid_output = self._postprocess(tracker_output=tracker_output)
        return reid_output

    def _preprocess(self, tracker_output: np.ndarray):
        self.all_tracked_objects = self._update_tracked_objects(tracker_output=tracker_output)
        self.all_tracked_objects = self._apply_filtering()

    def _update_tracked_objects(self, tracker_output: np.ndarray):
        self.frame_id = tracker_output[0, 0]
        for object_id, data_line in zip(tracker_output[:, 1], tracker_output):
            if object_id not in self.all_tracked_objects:
                new_tracked_object = TrackedObject(
                    object_ids=object_id, state=reid_constants.BYETRACK_OUTPUT, metadata=data_line
                )
                self.all_tracked_objects.append(new_tracked_object)
            else:
                self.all_tracked_objects[self.all_tracked_objects.index(object_id)].update_metadata(
                    data_line
                )

        return self.all_tracked_objects

    @staticmethod
    def _reshape_input(bytetrack_output: np.ndarray):
        if bytetrack_output.ndim == 1:
            bytetrack_output = np.expand_dims(bytetrack_output, 0)
        return bytetrack_output

    def _apply_filtering(self):
        for tracked_object in self.all_tracked_objects:
            self.tracked_filter.update(tracked_object)

        return self.all_tracked_objects

    def _perform_reid_process(self, tracker_output: np.ndarray):
        tracked_ids = filter_objects_by_state(
            self.all_tracked_objects, states=reid_constants.BYETRACK_OUTPUT, exclusion=True
        )

        current_tracker_ids = set(tracker_output[:, 1]).intersection(set(tracked_ids))

        self.compute_stable_objects(
            current_tracker_ids=current_tracker_ids, tracked_ids=self.all_tracked_objects
        )

        self.switchers = self.drop_switchers(
            self.switchers,
            current_tracker_ids,
            max_frames_to_rematch=self.max_frames_to_rematch,
            frame_id=self.frame_id,
        )

        self.candidates.extend(self.identify_candidates(tracked_ids=tracked_ids))

        self.switchers.extend(
            self.identify_switchers(
                current_tracker_ids=current_tracker_ids,
                last_bytetrack_ids=self.last_tracker_ids,
                tracked_ids=tracked_ids,
            )
        )

        matches = self.matcher.match(self.candidates, self.switchers)

        self.process_matches(
            all_tracked_objects=self.all_tracked_objects,
            matches=matches,
            candidates=self.candidates,
            switchers=self.switchers,
            current_tracker_ids=current_tracker_ids,
        )

        self.candidates = self.drop_candidates(
            self.candidates,
        )

        self.last_tracker_ids = current_tracker_ids.copy()

    @staticmethod
    def identify_switchers(
        tracked_ids: List["TrackedObject"],
        current_tracker_ids: Set[int],
        last_bytetrack_ids: Set[int],
    ):
        switchers = []
        lost_ids = last_bytetrack_ids - current_tracker_ids

        for tracked_id in tracked_ids:
            if tracked_id in lost_ids:
                switchers.append(tracked_id)
                tracked_id.state = reid_constants.SWITCHER

        return switchers

    @staticmethod
    def identify_candidates(tracked_ids: List["TrackedObject"]):
        candidates = []
        for current_object in tracked_ids:
            if current_object.state == reid_constants.FILTERED_OUTPUT:
                current_object.state = reid_constants.CANDIDATE
                candidates.append(current_object)
        return candidates

    @staticmethod
    def compute_stable_objects(tracked_ids: list, current_tracker_ids: Set[int]):
        top_list_correction = get_top_list_correction(tracked_ids)

        for current_object in current_tracker_ids:
            tracked_id = tracked_ids[tracked_ids.index(current_object)]
            if current_object not in top_list_correction:
                tracked_ids.remove(tracked_id)
                new_object, tracked_id = tracked_id.cut(current_object)

                new_object.state = reid_constants.STABLE
                tracked_id.state = reid_constants.STABLE

                tracked_ids.append(new_object)
                tracked_ids.append(tracked_id)

    @staticmethod
    def process_matches(
        all_tracked_objects: List["TrackedObject"],
        matches: Dict["TrackedObject", "TrackedObject"],
        switchers: List["TrackedObject"],
        candidates: List["TrackedObject"],
        current_tracker_ids: Set[int],
    ):
        for match in matches:
            candidate_match, switcher_match = match.popitem()

            switcher_match.merge(candidate_match)
            all_tracked_objects.remove(candidate_match)
            switchers.remove(switcher_match)
            candidates.remove(candidate_match)

            current_tracker_ids.discard(candidate_match.id)
            current_tracker_ids.add(switcher_match.id)

    @staticmethod
    def drop_switchers(
        switchers: List["TrackedObject"],
        current_tracker_ids: Set[int],
        max_frames_to_rematch: int,
        frame_id: int,
    ):
        switchers_to_drop = set(switchers).intersection(current_tracker_ids)
        filtered_switchers = switchers.copy()

        for switcher in switchers:
            if switcher in switchers_to_drop:
                switcher.state = reid_constants.STABLE
                filtered_switchers.remove(switcher)
            elif switcher.get_nb_frames_since_last_appearance(frame_id) > max_frames_to_rematch:
                filtered_switchers.remove(switcher)

        return filtered_switchers

    @staticmethod
    def drop_candidates(candidates: List["TrackedObject"]):
        # for now drop candidates if there was no match
        for candidate in candidates:
            candidate.state = reid_constants.STABLE
        return []

    def _postprocess(self, tracker_output: np.ndarray):
        filtered_objects = list(
            filter(
                lambda obj: obj.get_state() == reid_constants.STABLE
                and obj in tracker_output[:, 1],
                self.all_tracked_objects,
            )
        )
        reid_output = []
        for object in filtered_objects:
            reid_output.append(
                [
                    self.frame_id,
                    object.id,
                    object.category,
                    object.bbox[0],
                    object.bbox[1],
                    object.bbox[2],
                    object.bbox[3],
                    object.confidence,
                ]
            )

        return reid_output
