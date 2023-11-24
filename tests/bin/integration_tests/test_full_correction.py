from pathlib import Path

import numpy as np

from tests.utils.file_utils import compare_files, reset_output_folder
from trackreid.reid_processor import ReidProcessor
from trackreid.tracked_object import TrackedObject

INPUT_FOLDER = Path("tests/assets/integration_tests/data/")
INPUT_FILE = "tracker_output.txt"

OUTPUT_FOLDER = Path("tests/results/integration_tests/")
OUTPUT_FILE = "tracks.txt"

EXPECTED_OUTPUT_FOLDER = Path("tests/assets/integration_tests/expected_outputs")
EXPECTED_OUTPUT_FILE = "corrected_tracks.txt"


def test_full_correction():
    def dummy_cost_function(candidate: TrackedObject, switcher: TrackedObject):  # noqa: ARG001
        return 0

    def dummy_selection_function(candidate: TrackedObject, switcher: TrackedObject):  # noqa: ARG001
        return 1

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

    reset_output_folder(output_folder=OUTPUT_FOLDER, create_folder=True)

    tracker_output = np.loadtxt(INPUT_FOLDER / INPUT_FILE)

    frame_ids, indexes = np.unique(tracker_output[:, 0], return_index=True)
    frame_tracker_outputs = np.split(tracker_output[:, 1:], indexes)[1:]

    for frame_id, frame_tracker_output in zip(frame_ids, frame_tracker_outputs):
        reid_processor.update(frame_id=frame_id, tracker_output=frame_tracker_output)

    print(type(EXPECTED_OUTPUT_FOLDER / EXPECTED_OUTPUT_FILE))

    compare_files(EXPECTED_OUTPUT_FOLDER / EXPECTED_OUTPUT_FILE, OUTPUT_FOLDER / OUTPUT_FILE)
    reset_output_folder(output_folder=OUTPUT_FOLDER, create_folder=False)
