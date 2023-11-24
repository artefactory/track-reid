import filecmp
import json
import shutil
from pathlib import Path


def reset_output_folder(output_folder: Path, create_folder=True):
    """
    Each test should start and end by deleting the output folder
    """
    if output_folder.exists():
        shutil.rmtree(output_folder, ignore_errors=True)

    if create_folder:
        output_folder.mkdir(parents=True, exist_ok=True)


def _compare_txt_files(file1: Path, file2: Path):
    """Compare two txt files line by line"""
    with file1.open("r") as f1:
        with file2.open("r") as f2:
            for line1, line2 in zip(f1, f2):
                assert line1.strip() == line2.strip(), (line1, line2)


def _compare_json_files(file1: Path, file2: Path):
    with file1.open("r") as f1:
        json1 = json.load(f1)

    with file2.open("r") as f2:
        json2 = json.load(f2)

    # converting to string with same key order
    json1, json2 = json.dumps(json1, sort_keys=True), json.dumps(json2, sort_keys=True)
    assert json1 == json2, (json1, json2)


def compare_files(file1: Path, file2: Path):
    """
    Compare two files, regardless of their extension
    """
    assert file1.exists(), f"file1 {file1} does not exist"
    assert file2.exists(), f"File {file2} does not exist"

    extension1 = file1.suffix
    extension2 = file2.suffix

    if extension1 != extension2:
        raise ValueError(
            f"Cannot compare {file1} and {file2} because they have different extensions"
        )

    if extension1 == ".json":
        return _compare_json_files(file1, file2)

    elif extension1 == ".txt":  # we don't use filecmp to compare line by line
        return _compare_txt_files(file1, file2)

    else:
        assert filecmp.cmp(file1, file2)
