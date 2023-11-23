# Quickstart developers

## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/artefactory-fr/track-reid.git
```

Then, navigate to the project directory:

```bash
cd track-reid
```

To install the necessary dependencies, we use Poetry. If you don't have Poetry installed, you can download it using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Now, you can install the dependencies:

```bash
make install
```

This will create a virtual environment and install the necessary dependencies.
To activate the virtual environment in your terminal, you can use the following command:

```bash
poetry shell
```

You can also update the requirements using the following command:

```bash
make update-requirements
```

Then, you are ready to go !
For more detailed information, please refer to the `Makefile`.

## Tests

In this project, we have designed both integration tests and unit tests. These tests are located in the `tests` directory of the project.

Integration tests are designed to test the interaction between different parts of the system, ensuring that they work together as expected. Those tests can be found in the `tests/integration_tests` directory of the project.

Unit tests, on the other hand, are designed to test individual components of the system in isolation. We provided a bench of unit tests to test key functions of the project, those can be found in `tests/unit_tests`.

To run all tests, you can use the following command:

```bash
make run_tests
```
