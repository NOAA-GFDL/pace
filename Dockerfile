FROM python:3.11-slim-bookworm@sha256:7cd3fa11d619688317226bc93dc59bc8966e9aec6bc2a6abb847e8ab7d656706

RUN apt-get update && apt-get install -y make \
    software-properties-common \
    libgeos-dev \
    libopenmpi3 \
    libopenmpi-dev \
    libboost-all-dev \
    libhdf5-serial-dev \
    netcdf-bin \
    libnetcdf-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    python3 \
    python3-pip \
    git

RUN pip3 install --upgrade pip setuptools wheel

COPY . /pace

RUN cd /pace && \
    pip3 install -r /pace/requirements_dev.txt -c /pace/constraints.txt

RUN cd / && \
    git clone https://github.com/ai2cm/fv3net

ENV CFLAGS="-I/usr/include -DACCEPT_USE_OF_DEPRECATED_PROJ_API_H=1"

RUN python3 -m pip install \
    matplotlib==3.5.2 \
    ipyparallel==8.4.1 \
    jupyterlab==3.4.4 \
    shapely==1.8.5 \
    cartopy==0.23.0 \
    jupyterlab_code_formatter==1.5.2 \
    isort==5.10.1 \
    black==22.3.0 \
    /fv3net/external/vcm

ENV PYTHONPATH=/fv3net/external/fv3viz:/pace/external/gt4py/src

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
