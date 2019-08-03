FROM ubuntu:18.04
LABEL maintainer="Brendan Papadopoulos"
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y update && \
    apt-get -y install graphviz libxml2-dev python3-cairosvg parallel


RUN apt-get -y install python3.7
RUN apt -y install python3-pip
RUN pip3 install datajoint --upgrade
RUN pip3 install python-igraph xlrd


WORKDIR /src

RUN pip3 install ipyvolume jupyterlab statsmodels pycircstat nose
RUN pip3 install seaborn --upgrade
RUN pip3 install jgraph



RUN apt-get -y install vim
RUN . /etc/profile

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



WORKDIR /

#add the cgal scripts
EXPOSE 8888

#add the cgal scripts
#RUN git clone https://github.com/sdorkenw/MeshParty.git
#WORKDIR /MeshParty
#RUN pip3 install . --upgrade

RUN pip3 install pykdtree trimesh

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]
