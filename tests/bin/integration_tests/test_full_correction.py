import json
from pathlib import Path

import numpy as np

from tests.utils.file_utils import compare_files, reset_output_folder
from trackreid.reid_processor import ReidProcessor
from trackreid.tracked_object import TrackedObject

INPUT_FOLDER = Path("tests/assets/integration_tests/data/")
INPUT_FILE = "tracker_output.txt"

OUTPUT_FOLDER = Path("tests/results/integration_tests/")
OUTPUT_FILE = "corrected_tracks.txt"

EXPECTED_OUTPUT_FOLDER = Path("tests/assets/integration_tests/expected_outputs")
EXPECTED_OUTPUT_FILES = [
    OUTPUT_FILE,
    "tracked_object_0.json",
    "tracked_object_1.json",
    "tracked_object_2.json",
    "tracked_object_3.json",
    "tracked_object_4.json",
]


def save_tracked_objects(reid_processor: ReidProcessor, file_path_folder: Path):
    for tracked_object_id, tracked_object in enumerate(reid_processor.all_tracked_objects):
        file_path = file_path_folder / f"tracked_object_{tracked_object_id}.json"
        with file_path.open("w") as file:
            json.dump(tracked_object.to_dict(), file)


def test_full_correction():
    # 1. Define dummy cost and selection functions
    def dummy_cost_function(candidate: TrackedObject, switcher: TrackedObject):  # noqa: ARG001
        return 0

    def dummy_selection_function(candidate: TrackedObject, switcher: TrackedObject):  # noqa: ARG001
        return 1

    # 2. Initialize ReidProcessor
    reid_processor = ReidProcessor(
        filter_confidence_threshold=0.1,
        filter_time_threshold=1,
        max_frames_to_rematch=100,
        max_attempt_to_match=5,
        cost_function=dummy_cost_function,
        selection_function=dummy_selection_function,
        save_to_txt=True,
        file_path=OUTPUT_FOLDER / OUTPUT_FILE,
    )

    # 3. Reset output folder, load data and group by frame_id

    reset_output_folder(output_folder=OUTPUT_FOLDER, create_folder=True)

    tracker_output = np.loadtxt(INPUT_FOLDER / INPUT_FILE)
    frame_ids, indexes = np.unique(tracker_output[:, 0], return_index=True)
    frame_tracker_outputs = np.split(tracker_output[:, 1:], indexes)[1:]

    # 4. perform reid process and save corrected objects

    for frame_id, frame_tracker_output in zip(frame_ids, frame_tracker_outputs):
        reid_processor.update(frame_id=frame_id, tracker_output=frame_tracker_output)

    save_tracked_objects(reid_processor=reid_processor, file_path_folder=OUTPUT_FOLDER)

    # 5. Perform evaluation and delete files if test succeed

    for expected_output_file in EXPECTED_OUTPUT_FILES:
        compare_files(
            EXPECTED_OUTPUT_FOLDER / expected_output_file, OUTPUT_FOLDER / expected_output_file
        )

    reset_output_folder(output_folder=OUTPUT_FOLDER, create_folder=False)
