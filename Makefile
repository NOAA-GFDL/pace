SHELL := /bin/bash
include docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
CHECK_CHANGED_SCRIPT=$(CWD)/changed_from_main.py
VOLUMES = -v $(CWD):/port_dev
ifeq ($(DEV),y)
	VOLUMES = -v $(CWD)/fv3core:/fv3core -v $(CWD)/fv3gfs-physics:/fv3gfs-physics -v $(CWD)/stencils:/stencils -v $(CWD)/dsl:/dsl -v $(CWD)/pace-util:/pace-util -v $(CWD)/test_data:/test_data -v $(CWD)/examples:/port_dev/examples
endif

build:
	PULL=$(PULL) $(MAKE) -C docker fv3gfs_image

dev:
	docker run --rm -it \
		--network host \
		$(VOLUMES) \
		$(FV3GFS_IMAGE) bash

test_util:
	if [ $(shell $(CHECK_CHANGED_SCRIPT) pace-util) != false ]; then \
		$(MAKE) -C pace-util test; \
	fi

savepoint_tests:
	$(MAKE) -C fv3core $@

savepoint_tests_mpi:
	$(MAKE) -C fv3core $@

dependencies.svg: dependencies.dot
	dot -Tsvg $< -o $@

constraints.txt: fv3core/requirements.txt fv3core/requirements/requirements_wrapper.txt fv3core/requirements/requirements_lint.txt pace-util/requirements.txt fv3gfs-physics/requirements.txt
	pip-compile $^ --output-file constraints.txt

physics_savepoint_tests:
	$(MAKE) -C fv3gfs-physics $@

physics_savepoint_tests_mpi:
	$(MAKE) -C fv3gfs-physics $@

update_submodules_venv:
	if [ ! -f $(CWD)/external/daint_venv/install.sh  ]; then \
                git submodule update --init external/daint_venv; \
        fi
