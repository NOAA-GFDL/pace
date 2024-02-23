#***********************************************************************
#*                   GNU Lesser General Public License
#*
#* This file is part of the GFDL Flexible Modeling System (FMS).
#*
#* FMS is free software: you can redistribute it and/or modify it under
#* the terms of the GNU Lesser General Public License as published by
#* the Free Software Foundation, either version 3 of the License, or (at
#* your option) any later version.
#*
#* FMS is distributed in the hope that it will be useful, but WITHOUT
#* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#* for more details.
#*
#* You should have received a copy of the GNU Lesser General Public
#* License along with FMS.  If not, see <http://www.gnu.org/licenses/>.
#***********************************************************************
# Build stage with Spack pre-installed and ready to be used
# From rem1776
FROM spack/rockylinux9:latest as builder


# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir /opt/spack-environment \
    &&  (echo 'spack:' \
    &&   echo '  specs:' \
    &&   echo '  - gcc@13.2.0' \
    &&   echo '  - mpich@4.1.2' \
    &&   echo '  - netcdf-c@4.9.2 ^hdf5@1.14.2' \
    &&   echo '  - netcdf-fortran@4.6.1' \
    &&   echo '  - git@2.42.0' \
    &&   echo '  - miniconda3@22.11.1' \
    &&   echo '  concretizer:' \
    &&   echo '    unify: true' \
    &&   echo '    reuse: true' \
    &&   echo '  packages:' \
    &&   echo '    all:' \
    &&   echo '      compiler: [gcc@13.2.0]' \
    &&   echo '  # template file for any extra steps' \
    &&   echo '  config:' \
    &&   echo '    template_dirs:' \
    &&   echo '    - ./' \
    &&   echo '    install_tree: /opt/software' \
    &&   echo '  view: /opt/view') > /opt/spack-environment/spack.yaml

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment \
    && spack env activate -d . \
    && spack install --fail-fast

RUN dnf install -y autoconf make automake m4 libtool pkg-config
RUN cd /home \
    && git clone https://github.com/NOAA-GFDL/FRE-NCTOOLS.git \
    && cd FRE-NCTOOLS \
    && export FC=/opt/view/bin/mpif90 \
    && export CC=/opt/view/bin/mpicc \
    && export FCFLAGS="-I/opt/view/include -g -O2" \
    && export CFLAGS="-I/opt/view/include -g -O2" \
    && export LDFLAGS="-L/opt/view/lib" \
    && export PATH=$PATH:/opt/view/bin \
    && autoreconf -iv \
    && mkdir build && cd build \
    && ../configure --with-mpi --prefix=/opt/view \
    && make install \
    && cd .. && rm -rf FRE-NCTOOLS.git


# Bare OS image to run the installed executables
FROM docker.io/rockylinux:9

COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/view /opt/view
COPY test_scripts /home/scripts

WORKDIR /home

RUN dnf install -y libtool pkg-config

ENV LD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64:/usr/local/lib:/usr/local/lib64"
ENV PATH="/opt/view/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/view/fre-nctools/bin"

RUN conda init bash && source /root/.bashrc

ENTRYPOINT [ "/bin/bash" ]

LABEL version=3.8
