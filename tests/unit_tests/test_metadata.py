import json
from pathlib import Path

from trackreid.tracked_object_metadata import TrackedObjectMetaData

INPUT_FOLDER = Path("tests/data/unit_tests/tracked_objects")
LIST_TRACKED_OBJECTS = ["object_1.json", "object_4.json", "object_24.json"]

ALL_TRACKED_METADATA = []
for tracked_object in LIST_TRACKED_OBJECTS:
    with Path.open(INPUT_FOLDER / tracked_object) as file:
        ALL_TRACKED_METADATA.append(TrackedObjectMetaData.from_dict(json.load(file)["metadata"]))


def test_tracked_metadata_copy():
    tracked_metadata = ALL_TRACKED_METADATA[0].copy()
    copied_metadata = tracked_metadata.copy()
    assert copied_metadata.first_frame_id == 15
    assert copied_metadata.last_frame_id == 251
    assert copied_metadata.class_counts == {"shop_item": 175, "personal_item": 0}
    assert copied_metadata.bbox == [598, 208, 814, 447]
    assert copied_metadata.confidence == 0.610211
    assert copied_metadata.confidence_sum == 111.30582399999996
    assert copied_metadata.observations == 175

    assert round(copied_metadata.percentage_of_time_seen(251), 2) == 73.84
    class_proportions = copied_metadata.class_proportions()
    assert round(class_proportions["shop_item"], 2) == 1.0
    assert round(class_proportions["personal_item"], 2) == 0.0

    tracked_metadata_2 = ALL_TRACKED_METADATA[1].copy()
    tracked_metadata.merge(tracked_metadata_2)
    # test impact of merge inplace in a copy, should be none

    assert copied_metadata.class_counts == {"shop_item": 175, "personal_item": 0}
    assert copied_metadata.bbox == [598, 208, 814, 447]
    assert copied_metadata.confidence == 0.610211
    assert copied_metadata.confidence_sum == 111.30582399999996
    assert copied_metadata.observations == 175


def test_tracked_metadata_merge():
    tracked_metadata_1 = ALL_TRACKED_METADATA[0].copy()
    tracked_metadata_2 = ALL_TRACKED_METADATA[1].copy()
    tracked_metadata_1.merge(tracked_metadata_2)
    assert tracked_metadata_1.last_frame_id == 251
    assert tracked_metadata_1.class_counts["shop_item"] == 175
    assert tracked_metadata_1.class_counts["personal_item"] == 216
    assert tracked_metadata_1.bbox == [548, 455, 846, 645]
    assert tracked_metadata_1.confidence == 0.700626
    assert tracked_metadata_1.confidence_sum == 260.988185
    assert tracked_metadata_1.observations == 391
