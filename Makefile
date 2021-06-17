all: clean requirements dotenv
.PHONY: all

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = multiling-twitter
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: create_environment
ifeq (True,$(HAS_CONDA))
	conda develop -n $(PROJECT_NAME) .
else
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
endif

## Build with name of project to have it as tab name.
jpt-build:
	conda activate $(PROJECT_NAME)
	jupyter lab build --name='$(PROJECT_NAME)'

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

dotenv:
	echo "PROJ_DIR=\"$(PROJECT_DIR)\"" >> $(PROJECT_DIR)/.env

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create -f environment.yml
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m venv $(PROJECT_DIR)/.venv
endif
