from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np

from trackreid.configs.input_data_positions import input_data_positions
from trackreid.configs.output_data_positions import output_data_positions
from trackreid.configs.reid_constants import reid_constants
from trackreid.cost_functions import bounding_box_distance
from trackreid.matcher import Matcher
from trackreid.selection_functions import select_by_category
from trackreid.tracked_object import TrackedObject
from trackreid.tracked_object_filter import TrackedObjectFilter
from trackreid.utils import (
    filter_objects_by_state,
    get_nb_output_cols,
    get_top_list_correction,
    reshape_tracker_result,
)


class ReidProcessor:
    """
    The ReidProcessor class is designed to correct the results of tracking algorithms by reconciling and reassigning
    lost or misidentified IDs. This ensures a consistent and accurate tracking of objects over time.

    All input data should be of numeric type, either integers or floats.
    Here's an example of how the input data should look like based on the schema:

    | bbox (0-3)      | object_id (4) | category (5) | confidence (6) |
    |-----------------|---------------|--------------|----------------|
    | 50, 60, 120, 80 |       1       |       1      |       0.91     |
    | 50, 60, 120, 80 |       2       |       0      |       0.54     |

    Each row represents a detected object. The first four columns represent the bounding box coordinates
    (x, y, width, height), the fifth column represents the object ID assigned by the tracker,
    the sixth column represents the category of the detected object, and the seventh column represents
    the confidence score of the detection.

    You can use ReidProcessor.print_input_data_requirements() for more insight.

    Here's an example of how the output data looks like based on the schema:

    | frame_id (0) | object_id (1) | category (2) | bbox (3-6)      | confidence (7) | mean_confidence (8) | tracker_id (9) |
    |--------------|---------------|--------------|-----------------|----------------|---------------------|----------------|
    | 1            | 1             | 1            | 50, 60, 120, 80 | 0.91           | 0.85                | 1              |
    | 2            | 2             | 0            | 50, 60, 120, 80 | 0.54           | 0.60                | 2              |

    You can use ReidProcessor.print_output_data_format_information() for more insight.


    Args:
        filter_confidence_threshold (float): Confidence threshold for the filter. The filter will only consider tracked objects that have a mean confidence score during the all transaction above this threshold.

        filter_time_threshold (int): Time threshold for the filter. The filter will only consider tracked objects that have been seen for a number of frames above this threshold.

        max_frames_to_rematch (int): Maximum number of frames to rematch. If a switcher is lost for a number of frames greater than this value, it will be flagged as lost forever.

        max_attempt_to_match (int): Maximum number of attempts to match a candidate. If a candidate has not been rematched despite a number of attempts equal to this value, it will be flagged as a stable object.

        selection_function (Callable): A function that determines whether two objects should be considered for matching. The selection function should take two TrackedObject instances as input and return a binary value (0 or 1). A return value of 1 indicates that the pair should be considered for matching, while a return value of 0 indicates that the pair should not be considered.

        cost_function (Callable): A function that calculates the cost of matching two objects. The cost function should take two TrackedObject instances as input and return a numerical value representing the cost of matching these two objects. A lower cost indicates a higher likelihood of a match.

        cost_function_threshold (Optional[Union[int, float]]): An maximal threshold value for the cost function. If provided, any pair of objects with a matching cost greater than this threshold will not be considered for matching. If not provided, all selected pairs will be considered regardless of their matching cost.\n

        save_to_txt (bool): A flag indicating whether to save the results to a text file. If set to True, the results will be saved to a text file specified by the file_path parameter.

        file_path (str): The path to the text file where the results will be saved if save_to_txt is set to True.
    """  # noqa: E501

    def __init__(
        self,
        filter_confidence_threshold: float,
        filter_time_threshold: int,
        max_frames_to_rematch: int,
        max_attempt_to_match: int,
        selection_function: Callable = select_by_category,
        cost_function: Callable = bounding_box_distance,
        cost_function_threshold: Optional[Union[int, float]] = None,
        save_to_txt: bool = False,
        file_path: str = "tracks.txt",
    ) -> None:
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
        self.nb_output_cols = get_nb_output_cols(output_positions=output_data_positions)

        self.save_to_txt = save_to_txt
        self.file_path = file_path

    def reset(self) -> None:
        """
        Resets the ReID processor state for a new processing sequence.

        This method resets the frame counter to zero and clears all tracked objects
        and the last frame's tracked objects from memory, preparing the processor
        for a new sequence of frames.
        """
        self.frame_id = 0
        self.all_tracked_objects: List[TrackedObject] = []
        self.last_frame_tracked_objects: Set[TrackedObject] = set()

    def set_file_path(self, new_file_path: str) -> None:
        """
        Sets a new file path for saving txt data.

        Args:
            new_file_path (str): The new file path.
        """
        self.file_path = new_file_path

    @property
    def nb_corrections(self) -> int:
        """
        Calculates and returns the total number of corrections made across all tracked objects.

        Returns:
            int: Total number of corrections.
        """
        nb_corrections = 0
        for obj in self.all_tracked_objects:
            nb_corrections += obj.nb_corrections
        return nb_corrections

    @property
    def nb_tracker_ids(self) -> int:
        """
        Calculates and returns the total number of tracker IDs across all tracked objects.

        Returns:
            int: Total number of tracker IDs.
        """
        tracker_ids = 0
        for obj in self.all_tracked_objects:
            tracker_ids += obj.nb_ids
        return tracker_ids

    @property
    def corrected_objects(self) -> List["TrackedObject"]:
        """
        Returns a list of tracked objects that have been corrected.

        Returns:
            List[TrackedObject]: List of corrected tracked objects.
        """
        return [obj for obj in self.all_tracked_objects if obj.nb_corrections]

    @property
    def seen_objects(self) -> List["TrackedObject"]:
        """
        Returns a list of tracked objects that have been seen, excluding those in the
        states TRACKER_OUTPUT and FILTERED_OUTPUT.

        Returns:
            List[TrackedObject]: List of seen tracked objects.
        """
        return filter_objects_by_state(
            tracked_objects=self.all_tracked_objects,
            states=[reid_constants.STATES.TRACKER_OUTPUT, reid_constants.STATES.FILTERED_OUTPUT],
            exclusion=True,
        )

    @property
    def mean_nb_corrections(self) -> float:
        """
        Calculates and returns the mean number of corrections across all tracked objects.

        Returns:
            float: Mean number of corrections.
        """
        if len(self.all_tracked_objects):
            return self.nb_corrections / len(self.all_tracked_objects)
        else:
            return 0

    def update(self, tracker_output: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Processes the tracker output and updates internal states.

        All input data should be of numeric type, either integers or floats.
        Here's an example of how the input data should look like based on the schema:

        | bbox (0-3)      | object_id (4) | category (5) | confidence (6) |
        |-----------------|---------------|--------------|----------------|
        | 50, 60, 120, 80 |       1       |       1      |       0.91     |
        | 50, 60, 120, 80 |       2       |       0      |       0.54     |

        Each row represents a detected object. The first four columns represent the bounding box coordinates
        (x, y, width, height), the fifth column represents the object ID assigned by the tracker,
        the sixth column represents the category of the detected object, and the seventh column represents
        the confidence score of the detection.

        You can use ReidProcessor.print_input_data_requirements() for more insight.

        Here's an example of how the output data looks like based on the schema:

        | frame_id (0) | object_id (1) | category (2) | bbox (3-6)      | confidence (7) | mean_confidence (8) | tracker_id (9) |
        |--------------|---------------|--------------|-----------------|----------------|---------------------|----------------|
        | 1            | 1             | 1            | 50, 60, 120, 80 | 0.91           | 0.85                | 1              |
        | 2            | 2             | 0            | 50, 60, 120, 80 | 0.54           | 0.60                | 2              |

        You can use ReidProcessor.print_output_data_format_information() for more insight.

        Args:
            tracker_output (np.ndarray): The tracker output.
            frame_id (int): The frame id.

        Returns:
            np.ndarray: The processed output.
        """  # noqa: E501
        if tracker_output.size:  # empty tracking
            self.all_tracked_objects, current_tracker_ids = self._preprocess(
                tracker_output=tracker_output, frame_id=frame_id
            )
            self._perform_reid_process(current_tracker_ids=current_tracker_ids)
            reid_output = self._postprocess(current_tracker_ids=current_tracker_ids)

        else:
            reid_output = tracker_output

        if self.save_to_txt:
            self._save_results_to_txt(file_path=self.file_path, reid_output=reid_output)

        return reid_output

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
        current_tracker_ids = list(reshaped_tracker_output[:, input_data_positions.object_id])

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
            tracker_output[:, input_data_positions.object_id], tracker_output
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
        Performs the re-identification process on tracked objects.

        This method is responsible for managing the state of tracked objects and identifying potential
        candidates for re-identification. It follows these steps:

        1.  _correct_reid_chains: Corrects the re-identification chains of all tracked objects
        based on the current tracker IDs. This avoids potential duplicates.
        2.  _update_switchers_states: Updates the states of switchers (objects that have switched IDs)
        based on the current frame's tracked objects, the maximum number of frames to rematch, and the current frame ID.
        3.  _update_candidates_states: Updates the states of candidate objects (potential matches for re-identification)
        based on the maximum number of attempts to match and the current frame ID.
        4.  _identify_switchers: Identifies switchers based on the current and last frame's tracked objects and
        updates the state of all tracked objects accordingly.
        5.  _identify_candidates: Identifies candidates for re-identification and updates the state of all
        tracked objects accordingly.
        6.  match: Matches candidates with switchers using Jonker-Volgenant algorithm.
        7.  _process_matches: Processes the matches and updates the state of all tracked objects accordingly.

        Args:
            current_tracker_ids (List[Union[int, float]]): The current tracker IDs.
        """

        self.all_tracked_objects = self._correct_reid_chains(
            all_tracked_objects=self.all_tracked_objects, current_tracker_ids=current_tracker_ids
        )

        current_frame_tracked_objects = self._get_current_frame_tracked_objects(
            current_tracker_ids=current_tracker_ids
        )

        self.all_tracked_objects = self._update_switchers_states(
            all_tracked_objects=self.all_tracked_objects,
            current_frame_tracked_objects=current_frame_tracked_objects,
            max_frames_to_rematch=self.max_frames_to_rematch,
            frame_id=self.frame_id,
        )

        self.all_tracked_objects = self._update_candidates_states(
            all_tracked_objects=self.all_tracked_objects,
            max_attempt_to_match=self.max_attempt_to_match,
            frame_id=self.frame_id,
        )

        self.all_tracked_objects = self._identify_switchers(
            current_frame_tracked_objects=current_frame_tracked_objects,
            last_frame_tracked_objects=self.last_frame_tracked_objects,
            all_tracked_objects=self.all_tracked_objects,
        )

        self.all_tracked_objects = self._identify_candidates(
            all_tracked_objects=self.all_tracked_objects
        )

        candidates = filter_objects_by_state(
            self.all_tracked_objects, states=reid_constants.STATES.CANDIDATE, exclusion=False
        )
        switchers = filter_objects_by_state(
            self.all_tracked_objects, states=reid_constants.STATES.SWITCHER, exclusion=False
        )

        matches = self.matcher.match(candidates, switchers)

        self.all_tracked_objects = self._process_matches(
            all_tracked_objects=self.all_tracked_objects,
            matches=matches,
        )

        current_frame_tracked_objects = self._get_current_frame_tracked_objects(
            current_tracker_ids=current_tracker_ids
        )

        self.last_frame_tracked_objects = current_frame_tracked_objects.copy()

    @staticmethod
    def _identify_switchers(
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
    def _identify_candidates(all_tracked_objects: List["TrackedObject"]) -> List["TrackedObject"]:
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
    def _correct_reid_chains(
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
                new_object.state = reid_constants.STATES.CANDIDATE
                all_tracked_objects.append(new_object)

            elif new_object.nb_corrections > 1:
                new_object.state = reid_constants.STATES.SWITCHER
                all_tracked_objects.append(new_object)

        return all_tracked_objects

    @staticmethod
    def _process_matches(
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
    def _update_switchers_states(
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
    def _update_candidates_states(
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

    def _postprocess(
        self,
        current_tracker_ids: List[Union[int, float]],
    ) -> np.ndarray:
        """
        Postprocesses the current tracker IDs.
        It selects the stable TrackedObjects, and formats their datas in the output
        to match requirements.

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
            for required_variable in output_data_positions.model_json_schema()["properties"].keys():
                output = (
                    self.frame_id
                    if required_variable == "frame_id"
                    else getattr(stable_object, required_variable, None)
                )
                if output is None:
                    raise NameError(
                        f"Attribute {required_variable} not in TrackedObject. Check your required output names."
                    )
                reid_output[idx, getattr(output_data_positions, required_variable)] = output

        return reid_output

    def _save_results_to_txt(self, file_path: str, reid_output: np.ndarray) -> None:
        """
        Saves the reid_output to a txt file.

        Args:
            file_path (str): The path to the txt file.
            reid_output (np.ndarray): The output of _post_process.
        """
        with open(file_path, "a") as f:  # noqa: PTH123
            for row in reid_output:
                line = " ".join(
                    str(int(val)) if val.is_integer() else "{:.6f}".format(val) for val in row
                )
                f.write(line + "\n")

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

    @staticmethod
    def print_input_data_format_requirements():
        """

        Prints the input data format requirements.

        All input data should be of numeric type, either integers or floats.
        Here's an example of how the input data should look like based on the schema:

        |    bbox (0-3)   | object_id (4) | category (5) | confidence (6) |
        |-----------------|---------------|--------------|----------------|
        | 50, 60, 120, 80 |       1       |       1      |       0.91     |
        | 50, 60, 120, 80 |       2       |       0      |       0.54     |

        Each row represents a detected object. The first four columns represent the bounding box coordinates
        (x, y, width, height), the fifth column represents the object ID assigned by the tracker,
        the sixth column represents the category of the detected object, and the seventh column represents
        the confidence score of the detection.
        """
        input_schema = input_data_positions.model_json_schema()

        print("Input Data Format Requirements:")
        for name, properties in input_schema["properties"].items():
            print("-" * 50)
            print(f"{name}: {properties['description']}")
            print(
                f"{name} (position of {name} in the input array must be): {properties['default']}"
            )

    @staticmethod
    def print_output_data_format_information():
        """
        Prints the output data format information.

        Here's an example of how the output data looks like based on the schema:

        | frame_id (0) | object_id (1) | category (2) | bbox (3-6) | confidence (7) | mean_confidence (8) | tracker_id (9) |
        |--------------|---------------|--------------|------------|----------------|-------------------|------------------|
        | 1            | 1             | 1            | 50,60,120,80 | 0.91         | 0.85              | 1                |
        | 2            | 2             | 0            | 50,60,120,80 | 0.54         | 0.60              | 2                |

        """  # noqa: E501
        output_schema = output_data_positions.model_json_schema()

        print("\nOutput Data Format:")
        for name, properties in output_schema["properties"].items():
            print("-" * 50)
            print(f"{name}: {properties['description']}")
            print(
                f"{name} (position of {name} in the output array will be): {properties['default']}"
            )
