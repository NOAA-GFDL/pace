#!/bin/bash
# This is the master script used to trigger Jenkins actions.
# The idea of this script is to keep the amount of code in the "Execute shell" field small
#
# Example syntax:
# .jenkins/jenkins.sh run_regression_tests
#
# Other actions such as test/build/deploy can be defined.

### Some environment variables available from Jenkins
### Note: for a complete list see https://jenkins.ginko.ch/env-vars.html
# slave              The name of the build host (daint, kesch, ...).
# BUILD_NUMBER       The current build number, such as "153".
# BUILD_ID           The current build id, such as "2005-08-22_23-59-59" (YYYY-MM-DD_hh-mm-ss).
# BUILD_DISPLAY_NAME The display name of the current build, something like "#153" by default.
# NODE_NAME          Name of the host.
# NODE_LABELS        Whitespace-separated list of labels that the node is assigned.
# JENKINS_HOME       The absolute path of the data storage directory assigned on the master node.
# JENKINS_URL        Full URL of Jenkins, like http://server:port/jenkins/
# BUILD_URL          Full URL of this build, like http://server:port/jenkins/job/foo/15/
# JOB_URL            Full URL of this job, like http://server:port/jenkins/job/foo/

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

set -x

# echo basic setup
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

# start timer
T="$(date +%s)"

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass an argument"
test -n "${slave}" || exitError 1005 ${LINENO} "slave is not defined"

# GTC backend name fix: passed as gtc_gt_* but their real name are gtc:gt:*
#                       OR gtc_* but their real name is gtc:*
input_backend="$2"
if [[ $input_backend = gtc_gt_* ]] ; then
    # sed explained: replace _ with :, two times
    input_backend=`echo $input_backend | sed 's/_/:/;s/_/:/'`
fi
if [[ $input_backend = gtc_* ]] ; then
    # sed explained: replace _ with :
    input_backend=`echo $input_backend | sed 's/_/:/'`
fi

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILDENV_DIR=$JENKINS_DIR/../buildenv

# Read arguments
action="$1"
backend="$input_backend"
experiment="$3"

# check presence of env directory
pushd `dirname $0` > /dev/null
popd > /dev/null
shopt -s expand_aliases

# setup module environment and default queue
test -f ${BUILDENV_DIR}/machineEnvironment.sh || exitError 1201 ${LINENO} "cannot find machineEnvironment.sh script"
. ${BUILDENV_DIR}/machineEnvironment.sh
export python_env=${python_env}
echo "PYTHON env ${python_env}"


if [ -z "${GT4PY_VERSION}" ]; then
    export GT4PY_VERSION=`cat GT4PY_VERSION.txt`
fi
# If the backend is a GTC backend we fetch the caches
if [[ $backend != *numpy* ]];then
    echo "Fetching for exisintg gt_caches"
    . ${JENKINS_DIR}/actions/fetch_caches.sh $backend $experiment
fi

# load machine dependent environment
if [ ! -f ${BUILDENV_DIR}/env.${host}.sh ] ; then
    exitError 1202 ${LINENO} "could not find ${BUILDENV_DIR}/env.${host}.sh"
fi
. ${BUILDENV_DIR}/env.${host}.sh

# check if action script exists
script="${JENKINS_DIR}/actions/${action}.sh"
test -f "${script}" || exitError 1301 ${LINENO} "cannot find script ${script}"

# load scheduler tools
. ${BUILDENV_DIR}/schedulerTools.sh
scheduler_script="${BUILDENV_DIR}/submit.${host}.${scheduler}"

# if there is a scheduler script, make a copy for this job
if [ -f ${scheduler_script} ] ; then
    if [ "${action}" == "setup" ]; then
	scheduler="none"
    else
	cp  ${scheduler_script} job_${action}.sh
	scheduler_script=job_${action}.sh
    fi
fi


# if the environment variable is set to long_job we skip timing restrictions:
if [ -v LONG_EXECUTION ]; then
    sed -i 's|00:45:00|03:30:00|g' ${scheduler_script}
fi

# if this is a parallel job and the number of ranks is specified in the experiment argument, set NUM_RANKS
# and update the scheduler script if there is one
if grep -q "parallel" <<< "${script}"; then
    if grep -q "ranks" <<< "${experiment}"; then
	export NUM_RANKS=`echo ${experiment} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
	echo "Setting NUM_RANKS=${NUM_RANKS}"
	if [ -f ${scheduler_script} ] ; then
	    sed -i 's|<NTASKS>|<NTASKS>\n#SBATCH \-\-hint=multithread\n#SBATCH --ntasks-per-core=2|g' ${scheduler_script}
	    sed -i 's|45|30|g' ${scheduler_script}
	    if [ "$NUM_RANKS" -gt "6" ] && [ ! -v LONG_EXECUTION ]; then
            sed -i 's|cscsci|debug|g' ${scheduler_script}
        elif [ "$NUM_RANKS" -gt "6" ]; then
            sed -i 's|cscsci|normal|g' ${scheduler_script}
        fi
	    sed -i 's|<NTASKS>|"'${NUM_RANKS}'"|g' ${scheduler_script}
	    sed -i 's|<NTASKSPERNODE>|"24"|g' ${scheduler_script}
	fi
    fi
fi

# get the test data version from the Makefile
export DATA_VERSION=`grep "FORTRAN_SERIALIZED_DATA_VERSION=" Makefile  | cut -d '=' -f 2`

# Set the SCRATCH directory to the working directory if not set (e.g. for running on gce)
if [ -z ${SCRATCH} ] ; then
    export SCRATCH=`pwd`
fi

# Set the host data head directory location
export TEST_DATA_DIR="/project/s1053/fv3core_serialized_test_data/${DATA_VERSION}"
export TEST_DATA_DIR="${SCRATCH}/jenkins/scratch/fv3core_fortran_data/${DATA_VERSION}"
export FV3_STENCIL_REBUILD_FLAG=False
# Set the host data location
export TEST_DATA_HOST="${TEST_DATA_DIR}/${experiment}/"
export EXPERIMENT=${experiment}
if [ -z ${JENKINS_TAG} ]; then
    export JENKINS_TAG=${JOB_NAME}${BUILD_NUMBER}
    if [ -z ${JENKINS_TAG} ]; then
	export JENKINS_TAG=test
    fi
fi
export JENKINS_TAG=${JENKINS_TAG//[,=\/]/-}
if [ ${#JENKINS_TAG} -gt 85 ]; then
	NAME=`echo ${JENKINS_TAG} | md5sum | cut -f1 -d" "`
	export JENKINS_TAG=${NAME//[,=\/]/-}-${BUILD_NUMBER}
fi
echo "JENKINS TAG ${JENKINS_TAG}"

if [ -z ${VIRTUALENV} ]; then
    echo "setting VIRTUALENV"
    export VIRTUALENV=${JENKINS_DIR}/../venv_${JENKINS_TAG}
fi

if [ ${python_env} == "virtualenv" ]; then
    if [ -d ${VIRTUALENV} ]; then
	echo "Using existing virtualenv ${VIRTUALENV}"
    else
	echo "virtualenv ${VIRTUALENV} is not setup yet, installing now"
	export PACE_INSTALL_FLAGS="-e"
	${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
    fi
    source ${VIRTUALENV}/bin/activate
    if grep -q "parallel" <<< "${script}"; then
	export MPIRUN_CALL="srun"
    fi
    export PACE_PATH="${JENKINS_DIR}/../"
    export TEST_DATA_RUN_LOC=${TEST_DATA_HOST}
fi

export DOCKER_BUILDKIT=1

run_command "${script} ${backend} ${experiment} " Job${action} ${scheduler_script}

if [ $? -ne 0 ] ; then
  exitError 1510 ${LINENO} "problem while executing script ${script}"
fi
echo "### ACTION ${action} SUCCESSFUL"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0

# so long, Earthling!
