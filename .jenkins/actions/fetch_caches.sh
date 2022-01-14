#!/bin/bash
BACKEND=$1
EXPNAME=$2
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'
CACHE_DIR="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/${EXPNAME}/${SANITIZED_BACKEND}"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$SCRIPT_DIR/../../

if [ -z "${GT4PY_VERSION}" ]; then
    export GT4PY_VERSION=`git submodule status ${PACE_DIR}/external/gt4py | awk '{print $1;}'`
fi

if [ ! -d $(pwd)/fv3gfs-physics/.gt_cache ]; then
    if [ -d ${CACHE_DIR} ]; then
        cache_filename=${CACHE_DIR}/${GT4PY_VERSION}.tar.gz
        if [ -f "${cache_filename}" ]; then
            tar -xf ${cache_filename} -C fv3gfs-physics/.
            find fv3gfs-physics/. -name m_\*.py -exec sed -i "s|PLACEHOLDER/fv3gfs-physics|$(pwd)/fv3gfs-physics|g" {} +
            echo ".gt_cache successfully fetched from ${cache_filename}"
        else
            echo ".gt_cache not fetched, cache not found at ${cache_filename}"
        fi
    fi
fi
