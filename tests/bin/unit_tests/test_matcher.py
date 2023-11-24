import json
from pathlib import Path

from trackreid.matcher import Matcher
from trackreid.tracked_object import TrackedObject

INPUT_FOLDER = Path("tests/assets/unit_tests/data/tracked_objects")
LIST_TRACKED_OBJECTS = ["object_1.json", "object_4.json", "object_24.json"]

ALL_TRACKED_OBJECTS = []
for tracked_object in LIST_TRACKED_OBJECTS:
    with Path.open(INPUT_FOLDER / tracked_object) as file:
        ALL_TRACKED_OBJECTS.append(TrackedObject.from_dict(json.load(file)))


def test_matcher_no_match():
    def dummy_cost_function(candidate, switcher):
        return abs(candidate.object_id - switcher.object_id)

    def dummy_selection_function(candidate, switcher):  # noqa: ARG001
        return 0

    matcher = Matcher(dummy_cost_function, dummy_selection_function)

    candidates = []
    switchers = []
    for obj in ALL_TRACKED_OBJECTS:
        candidates.append(obj)
        switchers.append(obj)

    matches = matcher.match(candidates, switchers)

    assert len(matches) == 0


def test_matcher_all_match():
    def dummy_cost_function(candidate, switcher):
        return abs(candidate.object_id - switcher.object_id)

    def dummy_selection_function(candidate, switcher):  # noqa: ARG001
        return 1

    matcher = Matcher(dummy_cost_function, dummy_selection_function)

    candidates = []
    switchers = []
    for obj in ALL_TRACKED_OBJECTS:
        candidates.append(obj)
        switchers.append(obj)

    matches = matcher.match(candidates, switchers)

    assert len(matches) == 3
    for i in range(3):
        assert matches[i][candidates[i]] == switchers[i]


def test_matcher_middle_case():
    def dummy_cost_function(candidate, switcher):
        return abs(candidate.object_id - switcher.object_id)

    def dummy_selection_function(candidate, switcher):
        return (candidate.object_id % 2 == switcher.object_id % 2) and (
            candidate.object_id != switcher.object_id
        )

    matcher = Matcher(dummy_cost_function, dummy_selection_function)

    candidates = []
    switchers = []
    for obj in ALL_TRACKED_OBJECTS:
        candidates.append(obj)
        switchers.append(obj)

    matches = matcher.match(candidates, switchers)

    assert len(matches) == 2
    for match in matches:
        for candidate, switcher in match.items():
            assert candidate.object_id % 2 == switcher.object_id % 2
