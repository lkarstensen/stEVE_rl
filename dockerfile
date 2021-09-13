FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ARG USER
ARG PW

# Install APT tools and repos
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    apt-utils \
    software-properties-common
RUN add-apt-repository ppa:git-core/ppa
RUN apt-get update && apt-get upgrade -y

# Install system tools
RUN apt-get install -y \
    git \
    net-tools \
    wget \
    curl \
    zip \
    unzip \
    patchelf

# Install compilers
RUN apt-get install -y \
    build-essential \
    gcc-7=7.5.\*

# Install core deps
RUN apt-get install -y \
    libglew-dev \
    freeglut3-dev \
    zlib1g-dev \
    libeigen3-dev \
    libboost-all-dev

# Install plugins deps
RUN apt-get install -y \
    python3-dev python3-pip \
    libpng-dev libjpeg-dev libtiff-dev \
    libblas-dev \
    liblapack-dev \
    libsuitesparse-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libassimp-dev \
    libbullet-dev \
    liboce-ocaf-dev \
    libzmq3-dev liboscpack-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy scipy pygame

# Install GUI deps
#RUN apt-get install -y qt5-default

# Install cmake
RUN apt-get install -y cmake

ENV PYBIND11_DIR /tmp/pybind11

RUN git clone -b v2.4 --depth 1 https://github.com/pybind/pybind11.git $PYBIND11_DIR/src
RUN mkdir $PYBIND11_DIR/build 
RUN cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYBIND11_TEST=OFF -S ${PYBIND11_DIR}/src -B ${PYBIND11_DIR}/build
RUN cmake --build ${PYBIND11_DIR}/build
RUN cmake --install ${PYBIND11_DIR}/build 

ENV SOFA_DIR /opt/sofa

RUN mkdir $SOFA_DIR
RUN git clone -b v21.06 https://github.com/sofa-framework/sofa.git ${SOFA_DIR}/src
RUN mkdir ${SOFA_DIR}/build

RUN cmake -G"CodeBlocks - Unix Makefiles" \
    -DSOFAGUI_QT=OFF -DSOFA_BUILD_TESTS=OFF \
    -DSOFA_FETCH_SOFAPYTHON3=ON \
    -DSP3_LINK_TO_USER_SITE=ON \
    -DSP3_PYTHON_PACKAGES_LINK_DIRECTORY=/usr/lib/python3.8/dist-packages \
    -S ${SOFA_DIR}/src \
    -B ${SOFA_DIR}/build
RUN cmake --build ${SOFA_DIR}/build -j

RUN cmake --install ${SOFA_DIR}/build

ENV SOFA_ROOT ${SOFA_DIR}/build/install/

RUN git clone -b pamb_v2106_testing https://${USER}:${PW}@gitlab.cc-asp.fraunhofer.de/stacie/sofa_beamadapter.git $SOFA_DIR/src/applications/plugins/BeamAdapter
RUN sed -i '$asofa_add_plugin(BeamAdapter BeamAdapter)' $SOFA_DIR/src/applications/plugins/CMakeLists.txt

RUN cmake -DPLUGIN_SOFADISTANCEGRID=ON -DPLUGIN_SOFAIMPLICITFIELD=ON -DPLUGIN_BEAMADAPTER=ON -DBEAMADAPTER_BUILD_TESTS=OFF -S ${SOFA_DIR}/src -B ${SOFA_DIR}/build
RUN cmake --build ${SOFA_DIR}/build -j

RUN git clone https://${USER}:${PW}@gitlab.cc-asp.fraunhofer.de/stacie/eve.git /opt/eve
RUN python3 -m pip install /opt/eve[all]

RUN git clone https://${USER}:${PW}@gitlab.cc-asp.fraunhofer.de/stacie/toy-problems/tiltmaze.git /opt/tiltmaze 
RUN python3 -m pip install /opt/tiltmaze

RUN python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://${USER}:${PW}@gitlab.cc-asp.fraunhofer.de/stacie/stacierl.git /opt/stacierl
RUN python3 -m pip install /opt/stacierl

RUN python3 -m pip install optuna