<div align="center">

# track-reid

[![CI status](https://github.com/artefactory-fr/track-reid/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory-fr/track-reid/actions/workflows/ci.yaml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)]()

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory-fr/track-reid/blob/main/.pre-commit-config.yaml)
</div>

This Git repository is dedicated to the development of a Python library aimed at correcting the results of tracking algorithms. The primary goal of this library is to reconcile and reassign lost or misidentified IDs, ensuring a consistent and accurate tracking of objects over time.

## Table of Contents

- [track-reid](#track-reid)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Repository Structure](#repository-structure)

## Installation

First, install poetry:

```bash
make download-poetry
```

To install the required packages in a virtual environment, run the following command:

```bash
make install
```

You can then activate the env with the following command:

```bash
poetry shell
```

A complete list of available commands can be found using the following command:

```bash
make help
```

## Usage

For a quickstart, please refer to the documentation [here](https://artefactory-fr.github.io/track-reid/quickstart_user/). You also have at disposal a demo notebook in `notebooks/starer_kit_reid.ipynb`.

Lets say you have a `dataset` iterable object, composed for each iteartion of a frame id and its associated tracking results. You can call the `ReidProcessor` update class using the following:

```python
for frame_id, tracker_output in dataset:
    corrected_results = reid_processor.update(frame_id = frame_id,                  tracker_output=tracker_output)
```

At the end of the for loop, information about the correction can be retrieved using the `ReidProcessor` properties. For instance, the list of tracked object can be accessed using:

```python
reid_processor.seen_objects()
```

## Documentation

A detailed documentation of this project is available [here](https://artefactory-fr.github.io/track-reid/)

To serve the documentation locally, run the following command:

```bash
mkdocs serve
```

To build it and deploy it to GitHub pages, run the following command:

```bash
make deploy_docs
```

## Repository Structure

```
.
├── .github    <- GitHub Actions workflows and PR template
├── bin        <- Bash files
├── config     <- Configuration files
├── docs       <- Documentation files (mkdocs)
├── lib        <- Python modules
├── notebooks  <- Jupyter notebooks
├── secrets    <- Secret files (ignored by git)
└── tests      <- Unit tests
```
