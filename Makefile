PYTHON_VERSION = 3.11
# help: help					- Display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: download-poetry			- Download poetry
download-poetry:
	curl -sSL https://install.python-poetry.org | python3 -

# help: install					- Install python dependencies using poetry
.PHONY: install
install:
	@poetry config virtualenvs.create true
	@poetry env use $(PYTHON_VERSION)
	@poetry lock -n
	@poetry install -n
	@poetry run pre-commit install -t pre-commit -t pre-push

.PHONY: install-requirements
# help: requirements					- Install Python Dependencies
install-requirements:
	@poetry install -n

.PHONY: install-dev-requirements
## Install Python Dependencies for development
install-dev-requirements:
	@poetry install -n --with dev

.PHONY: update-requirements
#help: update-requirements				- Update Python Dependencies (requirements.txt and requirements-dev.txt)
update-requirements:
	@poetry lock -n

.PHONY: format-code
#help: format-code						- Format/lint all-files using pre-commit hooks (black, flake8, isort, ...)
format-code:
	@poetry run pre-commit run -a

.PHONY: run-tests
#help: run-tests						- Run all tests with pytest
run-tests:
	@export PYTHONPATH=.
	@poetry run pytest

# help: deploy_docs				- Deploy documentation to GitHub Pages
.PHONY: deploy_docs
deploy_docs:
	@mkdocs build
	@mkdocs gh-deploy
