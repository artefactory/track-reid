from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np

from trackreid.args.reid_args import INPUT_POSITIONS, MAPPING_CLASSES, OUTPUT_POSITIONS
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
        filter_confidence_threshold: float,
        filter_time_threshold: int,
        cost_function: Callable,
        selection_function: Callable,
        max_frames_to_rematch: int,
        max_attempt_to_match: int,
        cost_function_threshold: Optional[Union[int, float]] = None,
    ) -> None:
        """
        Initializes the ReidProcessor class.

        Args:
            filter_confidence_threshold: Confidence threshold for the filter.
            filter_time_threshold: Time threshold for the filter.
            cost_function: Cost function to be used.
            selection_function: Selection function to be used.
            max_frames_to_rematch (int): Maximum number of frames to rematch.
            max_attempt_to_match (int): Maximum attempts to match.
            cost_function_threshold (int, float): Maximum cost to rematch 2 objects.
        """

        self.matcher = Matcher(
            cost_function=cost_function,
            selection_function=selection_function,
            cost_function_threshold=cost_function_threshold,
        )

        self.tracked_filter = TrackedObjectFilter(
            confidence_threshold=filter_confidence_threshold,
            frames_seen_threshold=filter_time_threshold,
        )

        self.all_tracked_objects: List[TrackedObject] = []
        self.last_frame_tracked_objects: Set[TrackedObject] = set()

        self.max_frames_to_rematch = max_frames_to_rematch
        self.max_attempt_to_match = max_attempt_to_match

        self.frame_id = 0
        self.nb_output_cols = get_nb_output_cols(output_positions=OUTPUT_POSITIONS)

    @property
    def nb_corrections(self) -> int:
        nb_corrections = 0
        for obj in self.all_tracked_objects:
            nb_corrections += obj.nb_corrections
        return nb_corrections

    @property
    def nb_tracker_ids(self) -> int:
        tracker_ids = 0
        for obj in self.all_tracked_objects:
            tracker_ids += obj.nb_ids
        return tracker_ids

    @property
    def corrected_objects(self) -> List["TrackedObject"]:
        return [obj for obj in self.all_tracked_objects if obj.nb_corrections]

    @property
    def seen_objects(self) -> List["TrackedObject"]:
        return filter_objects_by_state(
            tracked_objects=self.all_tracked_objects,
            states=[reid_constants.STATES.TRACKER_OUTPUT, reid_constants.STATES.FILTERED_OUTPUT],
            exclusion=True,
        )

    @property
    def mean_nb_corrections(self) -> float:
        return self.nb_corrections / len(self.all_tracked_objects)

    def update(
        self, tracker_output: np.ndarray, frame_id: int
    ) -> Union[np.ndarray, List[TrackedObject]]:
        """
        Processes the tracker output.

        Args:
            tracker_output (np.ndarray): The tracker output.
            frame_id (int): The frame id.

        Returns:
            Union[np.ndarray, List[TrackedObject]]: The processed output.
        """
        if tracker_output.size:  # empty tracking
            self.all_tracked_objects, current_tracker_ids = self._preprocess(
                tracker_output=tracker_output, frame_id=frame_id
            )
            self._perform_reid_process(current_tracker_ids=current_tracker_ids)
            reid_output = self._postprocess(current_tracker_ids=current_tracker_ids)
            return reid_output
        else:
            return tracker_output

    def _preprocess(self, tracker_output: np.ndarray, frame_id: int) -> List["TrackedObject"]:
        """
        Preprocesses the tracker output.

        Args:
            tracker_output (np.ndarray): The tracker output.
            frame_id (int): The frame id.

        Returns:
            List["TrackedObject"]: The preprocessed output.
        """
        reshaped_tracker_output = reshape_tracker_result(tracker_output=tracker_output)
        current_tracker_ids = list(reshaped_tracker_output[:, INPUT_POSITIONS["object_id"]])

        self.all_tracked_objects = self._update_tracked_objects(
            tracker_output=reshaped_tracker_output, frame_id=frame_id
        )
        self.all_tracked_objects = self._apply_filtering()
        return self.all_tracked_objects, current_tracker_ids

    def _update_tracked_objects(
        self, tracker_output: np.ndarray, frame_id: int
    ) -> List[TrackedObject]:
        """
        Updates the tracked objects.

        Args:
            tracker_output (np.ndarray): The tracker output.
            frame_id (int): The frame id.

        Returns:
            List[TrackedObject]: The updated tracked objects.
        """
        self.frame_id = frame_id
        for object_id, data_line in zip(
            tracker_output[:, INPUT_POSITIONS["object_id"]], tracker_output
        ):
            if object_id not in self.all_tracked_objects:
                new_tracked_object = TrackedObject(
                    object_ids=object_id,
                    state=reid_constants.STATES.TRACKER_OUTPUT,
                    frame_id=frame_id,
                    metadata=data_line,
                )
                self.all_tracked_objects.append(new_tracked_object)
            else:
                self.all_tracked_objects[self.all_tracked_objects.index(object_id)].update_metadata(
                    data_line, frame_id=frame_id
                )

        return self.all_tracked_objects

    def _get_current_frame_tracked_objects(
        self, current_tracker_ids: Set[Union[int, float]]
    ) -> Set[Union[int, float]]:
        """
        Retrieves the tracked objects for the current frame.

        Args:
            current_tracker_ids (Set[Union[int, float]]): The set of current tracker IDs.

        Returns:
            Set[Union[int, float]]: The set of tracked objects for the current frame.
        """
        tracked_objects = filter_objects_by_state(
            self.all_tracked_objects, states=reid_constants.STATES.TRACKER_OUTPUT, exclusion=True
        )

        current_frame_tracked_objects = set(
            [tracked_id for tracked_id in tracked_objects if tracked_id in current_tracker_ids]
        )

        return current_frame_tracked_objects

    def _apply_filtering(self) -> List[TrackedObject]:
        """
        Applies filtering to the tracked objects.

        Returns:
            List[TrackedObject]: The filtered tracked objects.
        """
        for tracked_object in self.all_tracked_objects:
            self.tracked_filter.update(tracked_object)

        return self.all_tracked_objects

    def _perform_reid_process(self, current_tracker_ids: List[Union[int, float]]) -> None:
        """
        Performs the reid process.

        Args:
            current_tracker_ids (List[Union[int, float]]): The current tracker IDs.
        """
        self.all_tracked_objects = self.correct_reid_chains(
            all_tracked_objects=self.all_tracked_objects, current_tracker_ids=current_tracker_ids
        )

        current_frame_tracked_objects = self._get_current_frame_tracked_objects(
            current_tracker_ids=current_tracker_ids
        )

        self.all_tracked_objects = self.update_switchers_states(
            all_tracked_objects=self.all_tracked_objects,
            current_frame_tracked_objects=current_frame_tracked_objects,
            max_frames_to_rematch=self.max_frames_to_rematch,
            frame_id=self.frame_id,
        )

        self.all_tracked_objects = self.update_candidates_states(
            all_tracked_objects=self.all_tracked_objects,
            max_attempt_to_match=self.max_attempt_to_match,
            frame_id=self.frame_id,
        )

        self.all_tracked_objects = self.identify_switchers(
            current_frame_tracked_objects=current_frame_tracked_objects,
            last_frame_tracked_objects=self.last_frame_tracked_objects,
            all_tracked_objects=self.all_tracked_objects,
        )

        self.all_tracked_objects = self.identify_candidates(
            all_tracked_objects=self.all_tracked_objects
        )

        candidates = filter_objects_by_state(
            self.all_tracked_objects, states=reid_constants.STATES.CANDIDATE, exclusion=False
        )
        switchers = filter_objects_by_state(
            self.all_tracked_objects, states=reid_constants.STATES.SWITCHER, exclusion=False
        )

        matches = self.matcher.match(candidates, switchers)

        self.all_tracked_objects = self.process_matches(
            all_tracked_objects=self.all_tracked_objects,
            matches=matches,
        )

        current_frame_tracked_objects = self._get_current_frame_tracked_objects(
            current_tracker_ids=current_tracker_ids
        )

        self.last_frame_tracked_objects = current_frame_tracked_objects.copy()

    @staticmethod
    def identify_switchers(
        all_tracked_objects: List["TrackedObject"],
        current_frame_tracked_objects: Set["TrackedObject"],
        last_frame_tracked_objects: Set["TrackedObject"],
    ) -> List["TrackedObject"]:
        """
        Identifies switchers in the list of all tracked objects, and
        update their states. A switcher is an object that is lost, and probably
        needs to be rematched.

        Args:
            all_tracked_objects (List["TrackedObject"]): List of all objects being tracked.
            current_frame_tracked_objects (Set["TrackedObject"]): Set of currently tracked objects.
            last_frame_tracked_objects Set["TrackedObject"]: Set of last timestep tracked objects.

        Returns:
            List["TrackedObject"]: Updated list of all tracked objects after state changes.
        """
        lost_objects = last_frame_tracked_objects - current_frame_tracked_objects

        for tracked_object in all_tracked_objects:
            if tracked_object in lost_objects:
                tracked_object.state = reid_constants.STATES.SWITCHER

        return all_tracked_objects

    @staticmethod
    def identify_candidates(all_tracked_objects: List["TrackedObject"]) -> List["TrackedObject"]:
        """
        Identifies candidates in the list of all tracked objects, and
        update their states. A candidate is an object that was never seen before and
        that probably needs to be rematched.

        Args:
            all_tracked_objects (List["TrackedObject"]): List of all objects being tracked.

        Returns:
            List["TrackedObject"]: Updated list of all tracked objects after state changes.
        """
        tracked_objects = filter_objects_by_state(
            all_tracked_objects, states=reid_constants.STATES.TRACKER_OUTPUT, exclusion=True
        )
        for current_object in tracked_objects:
            if current_object.state == reid_constants.STATES.FILTERED_OUTPUT:
                current_object.state = reid_constants.STATES.CANDIDATE
        return all_tracked_objects

    @staticmethod
    def correct_reid_chains(
        all_tracked_objects: List["TrackedObject"],
        current_tracker_ids: List[Union[int, float]],
    ) -> List["TrackedObject"]:
        """
        Corrects the reid chains to prevent duplicates when an object reappears with a corrected id.
        For instance, if an object has a reid chain [1, 3, 6, 7], only the id 7 should be in the tracker's output.
        If another id from the chain (e.g., 3) is in the tracker's output, the reid chain is split into two:
        [1, 3] and [6, 7]. The first object's state is set to stable as 3 is in the current tracker output,
        and a new object with reid chain [6, 7] is created.
        The new object's state can be:
            - stable, if the tracker output is in the new reid chain
            - switcher, if not
            - nothing, if this is a singleton object, in which case the reid process is performed automatically.

        Args:
            all_tracked_objects (List["TrackedObject"]): List of all objects being tracked.
            current_tracker_ids (List[Union[int, float]]): The current tracker IDs.

        Returns:
            List["TrackedObject"]: The corrected tracked objects.
        """
        top_list_correction = get_top_list_correction(all_tracked_objects)
        to_correct = set(current_tracker_ids) - set(top_list_correction)

        for current_object in to_correct:
            tracked_id = all_tracked_objects[all_tracked_objects.index(current_object)]
            all_tracked_objects.remove(tracked_id)
            new_object, tracked_id = tracked_id.cut(current_object)

            tracked_id.state = reid_constants.STATES.STABLE
            all_tracked_objects.append(tracked_id)

            if new_object in current_tracker_ids:
                new_object.state = reid_constants.STATES.STABLE
                all_tracked_objects.append(new_object)

            elif new_object.nb_corrections > 1:
                new_object.state = reid_constants.STATES.SWITCHER
                all_tracked_objects.append(new_object)

        return all_tracked_objects

    @staticmethod
    def process_matches(
        all_tracked_objects: List["TrackedObject"],
        matches: Dict["TrackedObject", "TrackedObject"],
    ) -> List["TrackedObject"]:
        """
        Processes the matches.

        Args:
            all_tracked_objects (List["TrackedObject"]): List of all objects being tracked.
            matches (Dict["TrackedObject", "TrackedObject"]): The matches.

        Returns:
            List["TrackedObject"]: The processed tracked objects.
        """
        for match in matches:
            candidate_match, switcher_match = match.popitem()
            switcher_match.merge(candidate_match)
            switcher_match.state = reid_constants.STATES.STABLE
            all_tracked_objects.remove(candidate_match)

        return all_tracked_objects

    @staticmethod
    def update_switchers_states(
        all_tracked_objects: List["TrackedObject"],
        current_frame_tracked_objects: Set["TrackedObject"],
        max_frames_to_rematch: int,
        frame_id: int,
    ) -> List["TrackedObject"]:
        """
        Updates the state of switchers in the list of all tracked objects:
            - If a switcher is lost for too long, it will be flaged as lost forever
            - If a switcher reapears in the tracking output, it will be flaged as
            a stable object.

        Args:
            all_tracked_objects (List["TrackedObject"]): List of all objects being tracked.
            current_frame_tracked_objects (Set["TrackedObject"]): Set of currently tracked objects.
            max_frames_to_rematch (int): Maximum number of frames to rematch.
            frame_id (int): Current frame id.

        Returns:
            List["TrackedObject"]: Updated list of all tracked objects after state changes.
        """
        switchers = filter_objects_by_state(
            all_tracked_objects, reid_constants.STATES.SWITCHER, exclusion=False
        )
        switchers_to_drop = set(switchers).intersection(current_frame_tracked_objects)

        for switcher in switchers:
            if switcher in switchers_to_drop:
                switcher.state = reid_constants.STATES.STABLE
            elif switcher.get_nb_frames_since_last_appearance(frame_id) > max_frames_to_rematch:
                switcher.state = reid_constants.STATES.LOST_FOREVER

        return all_tracked_objects

    @staticmethod
    def update_candidates_states(
        all_tracked_objects: List["TrackedObject"], max_attempt_to_match: int, frame_id: int
    ) -> List["TrackedObject"]:
        """
        Updates the state of candidates in the list of all tracked objects.
        If a candidate has not been rematched despite max_attempt_to_match attempts,
        if will be flaged as a stable object.

        Args:
            all_tracked_objects (List["TrackedObject"]): List of all objects being tracked.
            max_attempt_to_match (int): Maximum attempt to match a candidate.
            frame_id (int): Current frame id.

        Returns:
            List["TrackedObject"]: Updated list of all tracked objects after state changes.
        """
        candidates = filter_objects_by_state(
            tracked_objects=all_tracked_objects,
            states=reid_constants.STATES.CANDIDATE,
            exclusion=False,
        )

        for candidate in candidates:
            if candidate.get_age(frame_id) >= max_attempt_to_match:
                candidate.state = reid_constants.STATES.STABLE
        return all_tracked_objects

    def _postprocess(self, current_tracker_ids: List[Union[int, float]]) -> np.ndarray:
        """
        Postprocesses the current tracker IDs.

        Args:
            current_tracker_ids (List[Union[int, float]]): The current tracker IDs.

        Returns:
            np.ndarray: The postprocessed output.
        """
        stable_objects = [
            obj
            for obj in self.all_tracked_objects
            if obj.get_state() == reid_constants.STATES.STABLE and obj in current_tracker_ids
        ]

        reid_output = np.zeros((len(stable_objects), self.nb_output_cols))

        for idx, stable_object in enumerate(stable_objects):
            for required_variable in OUTPUT_POSITIONS:
                output = (
                    self.frame_id
                    if required_variable == "frame_id"
                    else getattr(stable_object, required_variable, None)
                )
                if output is None:
                    raise NameError(
                        f"Attribute {required_variable} not in TrackedObject. Check your required output names."
                    )
                if required_variable == "category":
                    inverted_dict = {v: k for k, v in MAPPING_CLASSES.items()}
                    output = inverted_dict[output]
                reid_output[idx, OUTPUT_POSITIONS[required_variable]] = output

        return reid_output

    def to_dict(self) -> Dict:
        """
        Converts the tracked objects to a dictionary.

        Returns:
            Dict: The dictionary representation of the tracked objects.
        """
        data = dict()
        for tracked_object in self.all_tracked_objects:
            data[tracked_object.object_id] = tracked_object.to_dict()
        return data
