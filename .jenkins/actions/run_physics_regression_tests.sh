#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
XML_REPORT="sequential_test_results.xml"
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "


if [ ${python_env} == "virtualenv" ]; then
    export TEST_ARGS="${TEST_ARGS} --junitxml=${jenkins_dir}/${XML_REPORT}"
    CONTAINER_CMD="srun" make physics_savepoint_tests
else
    export TEST_ARGS="${TEST_ARGS} --junitxml=/${XML_REPORT}"
    make physics_savepoint_tests
fi
