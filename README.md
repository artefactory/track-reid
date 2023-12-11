<div align="center">

# track-reid

[![CI status](https://github.com/artefactory-fr/track-reid/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory-fr/track-reid/actions/workflows/ci.yaml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory-fr/track-reid/blob/main/.pre-commit-config.yaml)
</div>

This Git repository is dedicated to the development of a Python library aimed at correcting the results of tracking algorithms. The primary goal of this library is to reconcile and reassign lost or misidentified IDs, ensuring a consistent and accurate tracking of objects over time.

[See the detailed documentation of this project](https://artefactory-fr.github.io/track-reid/)

## trackreid + bytetrack VS bytetrack
<p align="center">
  <img src="https://storage.googleapis.com/track-reid/assets/demo_with_reid_small.gif" width="400"/><br>
  <b>Bytetrack x yolov8l x trackreid, 4 objects</b>
</p>
<p align="center">
  <img src="https://storage.googleapis.com/track-reid/assets/demo_no_reid_small.gif" width="400"/><br>
  <b>Bytetrack x yolov8l, 42 objects</b>
</p>
<p align="center">
    <a href="https://artefactory-fr.github.io/track-reid/">Demo with better quality can be found here</a>
</p>

<div align="center">

 | Model | Time difference | Time per iteration |
 | --- | --- | --- |
 | yolo + bytetrack | -- |  -- | -- |
 | yolo + bytetrack + trackreid | +0.95% | +2e-6 s/it |

</div>
<div align="center">

## Installation

To install the library, run the following command:
```bash
pip install git+https://github.com/artefactory-fr/track-reid.git@main
```

To install a specific version, run the following command:
```bash
pip install git+https://github.com/artefactory-fr/track-reid.git@x.y.z
```

</div>


## Usage

Suppose you have a list of frames, a model and a tracker. You can call the `ReidProcessor` update method on each outputs of your tracker as follow:

```python
for frame_id, image_filename in enumerate(available_frames):
    img = cv2.imread(image_filename)
    detections = model.predict(img)
    tracked_objects = tracker.update(detections, frame_id)
    corrected_tracked_objects = reid_processor.update(tracked_objects, frame_id)

```

At the end of the for loop, information about the correction can be retrieved using the `ReidProcessor` properties. For instance, the list of tracked object can be accessed using:

```python
reid_processor.seen_objects()
```

For a complete example you can refer to [examples/trackreid/starter_kit_reid.ipynb](/examples/trackreid/starter_kit_reid.ipynb)
