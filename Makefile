PIP_TOOLS_VERSION = 6.13.0
SETUPTOOLS_VERSION = 67.7.2
NUMPY_VERSION = $(shell cat requirements.in | sed 's/ /\n/g' | grep "numpy==" | sed 's/numpy==//g')

USE_CONDA ?= 1
INSTALL_SCRIPT = install_with_conda.sh
ifeq (false,$(USE_CONDA))
	INSTALL_SCRIPT = install_with_venv.sh
endif

# help: help					- Display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: install					- Create a virtual environment and install dependencies
.PHONY: install
install:
	@bash bin/$(INSTALL_SCRIPT)

# help: install_pip_tools		- Install piptools and setuptools
.PHONY: install_pip_tools
install_pip_tools:
	@echo "Installing pip-tools"
	@pip install pip-tools==$(PIP_TOOLS_VERSION) setuptools==$(SETUPTOOLS_VERSION)

# help: install_project_requirements		- Install prohect requirements
.PHONY: install_project_requirements
install_project_requirements: install_pip_tools
	@pip install numpy==${NUMPY_VERSION}
	@pip-compile requirements.in
	@pip install -r requirements.txt

# help: install_precommit			- Install pre-commit hooks
.PHONY: install_precommit
install_precommit:
	@pre-commit install -t pre-commit
	@pre-commit install -t pre-push

# help: deploy_docs				- Deploy documentation to GitHub Pages
.PHONY: deploy_docs
deploy_docs:
	@mkdocs build
	@mkdocs gh-deploy
