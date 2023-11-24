import json
from pathlib import Path

from trackreid.tracked_object import TrackedObject

INPUT_FOLDER = Path("tests/assets/unit_tests/data/tracked_objects")
LIST_TRACKED_OBJECTS = ["tracked_object_1.json", "tracked_object_4.json", "tracked_object_24.json"]

ALL_TRACKED_OBJECTS = []
for tracked_object in LIST_TRACKED_OBJECTS:
    with Path.open(INPUT_FOLDER / tracked_object) as file:
        ALL_TRACKED_OBJECTS.append(TrackedObject.from_dict(json.load(file)))


def test_tracked_object_copy():
    tracked_object = ALL_TRACKED_OBJECTS[0].copy()
    copied_object = tracked_object.copy()
    assert copied_object.object_id == tracked_object.object_id
    assert copied_object.state == tracked_object.state
    assert copied_object.category == tracked_object.category
    assert round(copied_object.confidence, 2) == round(tracked_object.confidence, 2)
    assert round(copied_object.mean_confidence, 2) == round(tracked_object.mean_confidence, 2)
    assert copied_object.bbox == tracked_object.bbox
    assert copied_object.nb_ids == tracked_object.nb_ids
    assert copied_object.nb_corrections == tracked_object.nb_corrections

    tracked_object_2 = ALL_TRACKED_OBJECTS[1].copy()
    tracked_object.merge(tracked_object_2)

    assert round(copied_object.confidence, 2) != round(tracked_object.confidence, 2)
    assert round(copied_object.mean_confidence, 2) != round(tracked_object.mean_confidence, 2)
    assert copied_object.bbox != tracked_object.bbox
    assert copied_object.nb_ids != tracked_object.nb_ids
    assert copied_object.nb_corrections != tracked_object.nb_corrections


def test_tracked_object_properties():
    tracked_object = ALL_TRACKED_OBJECTS[0].copy()
    assert tracked_object.object_id == 1.0
    assert tracked_object.state == 0
    assert tracked_object.category == 0
    assert round(tracked_object.confidence, 2) == 0.61
    assert round(tracked_object.mean_confidence, 2) == 0.64
    assert tracked_object.bbox == [598, 208, 814, 447]
    assert tracked_object.nb_ids == 5
    assert tracked_object.nb_corrections == 4


def test_tracked_object_merge():
    tracked_object_1 = ALL_TRACKED_OBJECTS[0].copy()
    tracked_object_2 = ALL_TRACKED_OBJECTS[1].copy()
    tracked_object_1.merge(tracked_object_2)
    assert tracked_object_1.object_id == 1.0
    assert tracked_object_1.state == 0
    assert tracked_object_1.category == 1
    assert round(tracked_object_1.confidence, 2) == 0.70
    assert round(tracked_object_1.mean_confidence, 2) == 0.67
    assert tracked_object_1.bbox == [548, 455, 846, 645]
    assert tracked_object_1.nb_ids == 7
    assert tracked_object_1.nb_corrections == 6


def test_tracked_object_cut():
    tracked_object = ALL_TRACKED_OBJECTS[0].copy()
    new_object, cut_object = tracked_object.cut(2.0)
    assert new_object.object_id == 14.0
    assert new_object.state == 0
    assert new_object.category == 0
    assert round(new_object.confidence, 2) == 0.61
    assert round(new_object.mean_confidence, 2) == 0.64
    assert new_object.bbox == [598, 208, 814, 447]
    assert new_object.nb_ids == 3
    assert new_object.nb_corrections == 2
    assert cut_object.object_id == 1.0
    assert cut_object.state == 0
    assert cut_object.category == 0
    assert round(cut_object.confidence, 2) == 0.61
    assert round(cut_object.mean_confidence, 2) == 0.64
    assert cut_object.bbox == [598, 208, 814, 447]
    assert cut_object.nb_ids == 2
    assert cut_object.nb_corrections == 1


def test_get_age():
    tracked_object = ALL_TRACKED_OBJECTS[0].copy()
    assert tracked_object.get_age(100) == 85


def test_get_nb_frames_since_last_appearance():
    tracked_object = ALL_TRACKED_OBJECTS[0].copy()
    assert tracked_object.get_nb_frames_since_last_appearance(300) == 49


def test_get_state():
    tracked_object = ALL_TRACKED_OBJECTS[0].copy()
    assert tracked_object.get_state() == 0
