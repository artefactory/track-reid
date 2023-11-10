from typing import List, Union

import numpy as np
from llist import sllist


def get_top_list_correction(tracked_ids: list):
    top_list_correction = [tracked_id.re_id_chain.last.value for tracked_id in tracked_ids]

    return top_list_correction


def split_list_around_value(my_list: sllist, value_to_split: float):
    if value_to_split == my_list.last.value:
        raise NameError("split on the last")
    if value_to_split not in my_list:
        raise NameError(f"{value_to_split} is not in the list")

    before = sllist()
    after = sllist()

    current = my_list.first

    while current:
        before.append(current.value)
        if current.value == value_to_split:
            break
        current = current.next

    current = current.next
    while current:
        after.append(current.value)
        current = current.next

    return before, after


def filter_objects_by_state(tracked_objects: List, states: Union[int, list], exclusion=False):
    if isinstance(states, int):
        states = [states]
    if exclusion:
        filtered_objects = [obj for obj in tracked_objects if obj.state not in states]
    else:
        filtered_objects = [obj for obj in tracked_objects if obj.state in states]
    return filtered_objects


def reshape_tracker_result(tracker_output: np.ndarray):
    if tracker_output.ndim == 1:
        tracker_output = np.expand_dims(tracker_output, 0)
    return tracker_output
