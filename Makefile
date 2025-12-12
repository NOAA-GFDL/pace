SHELL := /bin/bash
include docker/Makefile.image_names
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))



define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
CMD ?=bash
PULL ?=True
DEV ?=y
CHECK_CHANGED_SCRIPT=$(CWD)/changed_from_main.py
CONTAINER_CMD?=docker
SAVEPOINT_SETUP=pip3 list

PORT ?=8888
APP_NAME ?=Pace_dev

VOLUMES ?=
BUILD_FLAGS ?=

### Testing variables

NUM_RANKS ?=6
MPIRUN_ARGS ?=--oversubscribe --mca btl_vader_single_copy_mechanism none
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS) $(MPIRUN_ARGS)
TEST_ARGS ?=-v
FV3CORE_THRESH_ARGS=--threshold_overrides_file=$(PACE_PATH)/pyFV3/tests/savepoint/translate/overrides/standard.yaml
PHYSICS_THRESH_ARGS=--threshold_overrides_file=$(PACE_PATH)/pySHiELD/tests/savepoint/translate/overrides/standard.yaml

TEST_DATA_LOC ?=test_data/
TEST_DATA_VERSION ?=8.1.3
TEST_DATA_HOST ?= https://portal.nccs.nasa.gov/datashare/astg/smt/pace-regression-data/
TEST_RESOLUTION ?= c12
TEST_CONFIG ?= $(TEST_DATA_LOC)$(TEST_RESOLUTION)_$(NUM_RANKS)ranks

RUN_FLAGS ?=--rm
ifeq ("$(CONTAINER_CMD)","")
	PACE_PATH?=$(ROOT_DIR)
else
ifeq ("$(CONTAINER_CMD)","srun")
	PACE_PATH?=$(ROOT_DIR)
else
	PACE_PATH?=/pace
endif
endif
ifeq ("$(CONTAINER_CMD)","")
	EXPERIMENT_DATA_RUN=$(TEST_CONFIG)
else
ifeq ("$(CONTAINER_CMD)","srun")
	EXPERIMENT_DATA_RUN=$(TEST_CONFIG)
else
	EXPERIMENT_DATA_RUN=$(TEST_CONFIG)
endif
endif
ifeq ($(DEV),y)
	VOLUMES += -v $(ROOT_DIR):/pace
else
	VOLUMES += -v $(EXPERIMENT_DATA_RUN)
endif
ifeq ($(CONTAINER_CMD),docker)
	CONTAINER_FLAGS=run $(RUN_FLAGS) $(VOLUMES) --env GT_CACHE_ROOT=/pace/.gt_cache $(PACE_IMAGE)
else
	CONTAINER_FLAGS=
endif

###


build:
ifneq ($(findstring docker,$(CONTAINER_CMD)),)  # only build if using docker
ifeq ($(DEV),n)  # rebuild container if not running in dev mode
	$(MAKE) _force_build
else  # build even if running in dev mode if there is no environment image
ifeq ($(shell docker images -q us.gcr.io/vcm-ml/pace 2> /dev/null),)
	$(MAKE) _force_build
endif
endif
endif

_force_build:
	DOCKER_BUILDKIT=1 docker build \
		$(BUILD_FLAGS) \
		-f $(CWD)/Dockerfile \
		-t $(PACE_IMAGE) \
		.

enter:
	docker run --rm -it \
		$(VOLUMES) \
		-p=$(PORT):$(PORT) \
		--name="$(APP_NAME)" \
	$(PACE_IMAGE) $(CMD)

dev:
	DEV=y $(MAKE) enter

notebook:
	CMD="jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --notebook-dir=/pace/examples/notebooks" \
	DEV=y \
	$(MAKE) enter

get_standard_test_data:
	if [ ! -d $(TEST_DATA_LOC) ]; then \
	    mkdir -p $(TEST_DATA_LOC); \
	fi ; \
	if [ ! -f $(TEST_CONFIG)_standard/dycore/input.nml ] ; then \
		wget $(TEST_DATA_HOST)/8.1.3_c12_6ranks_standard.tar.gz; \
		tar -xzvf $(ROOT_DIR)/8.1.3_c12_6ranks_standard.tar.gz; \
		mv $(ROOT_DIR)/8.1.3/* $(TEST_DATA_LOC); \
		rm -rf 8.1.3*; \
	fi

get_physics_test_data:
	if [ ! -d $(TEST_DATA_LOC) ]; then \
	    mkdir -p $(TEST_DATA_LOC); \
	fi ; \
	if [ ! -f $(TEST_CONFIG)_baroclinic/physics/input.nml ] ; then \
		wget $(TEST_DATA_HOST)/8.1.3_c12_6ranks_baroclinic.physics.tar.gz ; \
		tar -xzvf $(ROOT_DIR)/8.1.3_c12_6ranks_baroclinic.physics.tar.gz; \
		mv $(ROOT_DIR)/8.1.3/* $(TEST_DATA_LOC); \
		rm -rf 8.1.3*; \
	fi

get_test_data:
	$(MAKE) get_standard_test_data; \
	$(MAKE) get_physics_test_data

test_util:
	if [ $(shell $(CHECK_CHANGED_SCRIPT) util) != false ]; then \
		$(MAKE) -C util test; \
	fi

savepoint_tests: build  ## dycore-only savepoint tests
	TARGET=dycore $(MAKE) get_test_data
	$(CONTAINER_CMD) $(CONTAINER_FLAGS) bash -c "$(SAVEPOINT_SETUP) && cd $(PACE_PATH) && pytest --data_path=$(TEST_DATA_LOC)/standard/dycore/ $(TEST_ARGS) $(FV3CORE_THRESH_ARGS) $(PACE_PATH)/pyFV3/tests/savepoint"

savepoint_tests_mpi: build
	TARGET=dycore $(MAKE) get_test_data
	$(CONTAINER_CMD) $(CONTAINER_FLAGS) bash -c "$(SAVEPOINT_SETUP) && cd $(PACE_PATH) && $(MPIRUN_CALL) python3 -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_LOC)/dycore/ $(TEST_ARGS) $(FV3CORE_THRESH_ARGS) -m parallel $(PACE_PATH)/pyFV3/tests/savepoint"

dependencies.svg: dependencies.dot
	dot -Tsvg $< -o $@

physics_savepoint_tests: build
	TARGET=physics $(MAKE) get_test_data
	$(CONTAINER_CMD) $(CONTAINER_FLAGS) bash -c "$(SAVEPOINT_SETUP) && cd $(PACE_PATH) && pytest --data_path=$(EXPERIMENT_DATA_RUN)/physics/ $(TEST_ARGS) $(PHYSICS_THRESH_ARGS) $(PACE_PATH)/pySHiELD/tests/savepoint"

physics_savepoint_tests_mpi: build
	TARGET=physics $(MAKE) get_test_data
	$(CONTAINER_CMD) $(CONTAINER_FLAGS) bash -c "$(SAVEPOINT_SETUP) && cd $(PACE_PATH) && $(MPIRUN_CALL) python -m mpi4py -m pytest --maxfail=1 --data_path=$(EXPERIMENT_DATA_RUN)/physics/ $(TEST_ARGS) $(PHYSICS_THRESH_ARGS) -m parallel $(PACE_PATH)/pySHiELD/tests/savepoint"

test_main: build
	$(CONTAINER_CMD) $(CONTAINER_FLAGS) bash -c "$(SAVEPOINT_SETUP) && cd $(PACE_PATH) && pytest $(TEST_ARGS) $(PACE_PATH)/tests/main"

test_savepoint:  ## top level savepoint tests
	TARGET=dycore $(MAKE) get_test_data
	$(CONTAINER_CMD) $(CONTAINER_FLAGS) bash -c "$(SAVEPOINT_SETUP) && cd $(PACE_PATH) && $(MPIRUN_CALL) python -m pytest --data_path=$(EXPERIMENT_DATA_RUN)/dycore/ $(TEST_ARGS) $(PACE_PATH)/tests/savepoint"

test_notebooks:  ## tests for jupyter notebooks, must be run in correct Python environment
	pytest --nbmake "examples/notebooks"

test_mpi_54rank:
	mpirun -n 54 $(MPIRUN_ARGS) python3 -m mpi4py -m pytest tests/mpi_54rank

driver_savepoint_tests_mpi: build
	TARGET=pace $(MAKE) get_test_data
	$(CONTAINER_CMD) $(CONTAINER_FLAGS) bash -c "$(SAVEPOINT_SETUP) && cd $(PACE_PATH) && $(MPIRUN_CALL) python -m mpi4py -m pytest --maxfail=1 --data_path=$(EXPERIMENT_DATA_RUN)/physics/ $(TEST_ARGS) $(PHYSICS_THRESH_ARGS) -m parallel $(PACE_PATH)/pySHiELD/tests/savepoint"

docs: ## generate Sphinx HTML documentation
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

doctest: ## run Sphinx doctest
	$(MAKE) -C docs doctest

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

lint:
	pre-commit run --all-files

.PHONY: docs doctest servedocs build
