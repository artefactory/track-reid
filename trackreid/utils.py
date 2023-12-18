from typing import List, Union

import numpy as np
from llist import sllist

from trackreid.configs.output_data_positions import OutputDataPositions


def get_top_list_correction(tracked_ids: List):
    """
    Function to get the last value of each re_id_chain in tracked_ids.

    Args:
        tracked_ids (list): List of tracked ids.

    Returns:
        list: List of last values of each re_id_chain in tracked_ids.
    """
    top_list_correction = [tracked_id.re_id_chain.last.value for tracked_id in tracked_ids]

    return top_list_correction


def split_list_around_value(my_list: sllist, value_to_split: float):
    """
    Function to split a list around a given value.

    Args:
        my_list (sllist): The list to split.
        value_to_split (float): The value to split the list around.

    Returns:
        tuple: Two lists, before and after the split value.
    """
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


def filter_objects_by_state(tracked_objects: List, states: Union[int, List[int]], exclusion=False):
    """
    Function to filter tracked objects by their state.

    Args:
        tracked_objects (List): List of tracked objects.
        states (Union[int, list]): State or list of states to filter by.
        exclusion (bool, optional): If True, exclude objects with the given states. Defaults to False.

    Returns:
        list: List of filtered tracked objects.
    """
    if isinstance(states, int):
        states = [states]
    if exclusion:
        filtered_objects = [obj for obj in tracked_objects if obj.state not in states]
    else:
        filtered_objects = [obj for obj in tracked_objects if obj.state in states]
    return filtered_objects


def filter_objects_by_category(
    tracked_objects: List,
    category: Union[Union[float, int], List[Union[float, int]]],
    exclusion=False,
):
    """
    Function to filter tracked objects by their category.

    Args:
        tracked_objects (List): List of tracked objects.
        category (Union[Union[float, int], list]): Category or list of categories to filter by.
        exclusion (bool, optional): If True, exclude objects with the given categories. Defaults to False.

    Returns:
        list: List of filtered tracked objects.
    """
    if isinstance(category, (float, int)):
        category = [category]
    if exclusion:
        filtered_objects = [obj for obj in tracked_objects if obj.category not in category]
    else:
        filtered_objects = [obj for obj in tracked_objects if obj.category in category]
    return filtered_objects


def reshape_tracker_result(tracker_output: np.ndarray):
    """
    Function to reshape the tracker output if it has only one dimension.

    Args:
        tracker_output (np.ndarray): The tracker output to reshape.

    Returns:
        np.ndarray: The reshaped tracker output.
    """
    if tracker_output.ndim == 1:
        tracker_output = np.expand_dims(tracker_output, 0)
    return tracker_output


def get_nb_output_cols(output_positions: OutputDataPositions):
    """
    Function to get the number of output columns based on the model json schema.

    Args:
        output_positions (OutputDataPositions): The output data positions.

    Returns:
        int: The number of output columns.
    """
    schema = output_positions.model_json_schema()
    nb_cols = 0
    for feature in schema["properties"]:
        if schema["properties"][feature]["type"] == "integer":
            nb_cols += 1
        elif schema["properties"][feature]["type"] == "array":
            nb_cols += len(schema["properties"][feature]["default"])
        else:
            raise TypeError("Unknown type in required output positions.")

    return nb_cols
