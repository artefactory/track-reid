import json
from pathlib import Path

import numpy as np
from llist import sllist

from trackreid import utils
from trackreid.tracked_object import TrackedObject

# Load tracked object data
INPUT_FOLDER = Path("tests/data/unit_tests/tracked_objects")
LIST_TRACKED_OBJECTS = ["object_1.json", "object_4.json", "object_24.json"]

ALL_TRACKED_OBJECTS = []
for tracked_object in LIST_TRACKED_OBJECTS:
    with Path.open(INPUT_FOLDER / tracked_object) as file:
        ALL_TRACKED_OBJECTS.append(TrackedObject.from_dict(json.load(file)))


# Define tests
def test_get_top_list_correction():
    top_list_correction = utils.get_top_list_correction(ALL_TRACKED_OBJECTS)
    assert top_list_correction == [21.0, 13.0, 24.0]


def test_split_list_around_value_1():
    my_list = sllist([1, 2, 3, 4, 5])
    value_to_split = 3
    before, after = utils.split_list_around_value(my_list, value_to_split)
    assert list(before) == [1, 2, 3]
    assert list(after) == [4, 5]


def test_split_list_around_value_2():
    my_list = sllist([1, 2, 3, 4, 5])
    value_to_split = 1
    before, after = utils.split_list_around_value(my_list, value_to_split)
    assert list(before) == [1]
    assert list(after) == [2, 3, 4, 5]


def test_split_list_around_value_3():
    my_list = sllist([1, 2, 3, 4, 5])
    value_to_split = 4
    before, after = utils.split_list_around_value(my_list, value_to_split)
    assert list(before) == [1, 2, 3, 4]
    assert list(after) == [5]


def test_filter_objects_by_state():
    states = 0
    assert utils.filter_objects_by_state(ALL_TRACKED_OBJECTS, states, exclusion=False) == [
        ALL_TRACKED_OBJECTS[0],
        ALL_TRACKED_OBJECTS[1],
    ]


def test_filter_objects_by_state_2():
    states = -2
    assert utils.filter_objects_by_state(ALL_TRACKED_OBJECTS, states, exclusion=True) == [
        ALL_TRACKED_OBJECTS[0],
        ALL_TRACKED_OBJECTS[1],
    ]


def test_filter_objects_by_category():
    category = 0
    assert utils.filter_objects_by_category(ALL_TRACKED_OBJECTS, category, exclusion=False) == [
        ALL_TRACKED_OBJECTS[0],
        ALL_TRACKED_OBJECTS[2],
    ]


def test_filter_objects_by_category_2():
    category = 1
    assert utils.filter_objects_by_category(ALL_TRACKED_OBJECTS, category, exclusion=True) == [
        ALL_TRACKED_OBJECTS[0],
        ALL_TRACKED_OBJECTS[2],
    ]


def test_reshape_tracker_result():
    tracker_output = np.array([1, 1, 3, 4, 5, 6, 7])
    assert np.array_equal(
        utils.reshape_tracker_result(tracker_output), np.array([[1, 1, 3, 4, 5, 6, 7]])
    )


def test_get_nb_output_cols():
    output_positions = {"feature1": 1, "feature2": [1, 2, 3]}
    assert utils.get_nb_output_cols(output_positions) == 4
