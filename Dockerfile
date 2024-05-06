FROM python:3.8.13-bullseye@sha256:2a01d88a1684e6d7f08030cf5ae73b536926c64076cab197e9e3d9f699255283

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
    python3-pip

RUN pip3 install --upgrade setuptools wheel

COPY constraints.txt /pace/constraints.txt

RUN pip3 install -r /pace/constraints.txt

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
    cartopy==0.18.0 \
    jupyterlab_code_formatter==1.5.2 \
    isort==5.10.1 \
    black==22.3.0 \
    /fv3net/external/vcm

ENV PYTHONPATH=/fv3net/external/fv3viz:/pace/external/gt4py/src

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
