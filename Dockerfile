FROM ninai/pipeline:base

#FROM ubuntu:xenial

RUN apt-get update && apt-get install -q -y \
    build-essential \
    python \
    python-numpy \
    git \
    g++ \
    libeigen3-dev \
    qt5-qmake \
    qtscript5-dev \
    libqt5xmlpatterns5-dev \
    libqt5opengl5-dev \
    assimp-utils \
    nano \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/hamzamerzic/vcglib -b devel
RUN git clone https://github.com/hamzamerzic/meshlab -b devel

ARG QMAKE_FLAGS="-spec linux-g++ CONFIG+=release CONFIG+=qml_release CONFIG+=c++11 QMAKE_CXXFLAGS+=-fPIC QMAKE_CXXFLAGS+=-std=c++11 QMAKE_CXXFLAGS+=-fpermissive INCLUDEPATH+=/usr/include/eigen3 LIBS+=-L/meshlab/src/external/lib/linux-g++"
ARG MAKE_FLAGS="-j"

WORKDIR /meshlab/src/external
RUN qmake -qt=5 external.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src/common
RUN qmake -qt=5 common.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src
RUN qmake -qt=5 meshlab_mini.pro $QMAKE_FLAGS && make $MAKE_FLAGS
RUN qmake -qt=5 meshlab_full.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /notebooks

#add the cgal scripts
EXPOSE 8895

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts


#FROM hamzamerzic/meshlab
#FROM hamzamerzic/meshlab
ENTRYPOINT ["/scripts/run_jupyter.sh"]
