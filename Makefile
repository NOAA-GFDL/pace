include docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
CHECK_CHANGED_SCRIPT=$(CWD)/changed_from_main.py


build:
	PULL=$(PULL) $(MAKE) -C docker fv3gfs_image

dev:
	docker run --rm -it \
		--network host \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) bash

test_util:
	if [ $(shell $(CHECK_CHANGED_SCRIPT) fv3gfs-util) != false ]; then \
		$(MAKE) -C fv3gfs-util test; \
	fi

savepoint_tests:
	$(MAKE) -C fv3core $@

savepoint_tests_mpi:
	$(MAKE) -C fv3core $@

dependencies.svg: dependencies.dot
	dot -Tsvg $< -o $@

constraints.txt: fv3core/requirements.txt fv3core/requirements/requirements_wrapper.txt fv3core/requirements/requirements_lint.txt fv3gfs-util/requirements.txt fv3gfs-physics/requirements.txt
	pip-compile $^ --output-file constraints.txt
