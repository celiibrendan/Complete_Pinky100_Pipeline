FROM hamzamerzic/meshlab
FROM ninai/pipeline:base

LABEL maintainer="Christos Papadopoulos"

#ADD /meshlab /meshlab
#COPY --from=hamzamerzic/meshlab ./README.md /meshlab/README.md
RUN apt-get -y update
#RUN apt-get -y install software-properties-common
#RUN add-apt-repository ppa:zarquon42/meshlab
#RUN apt-get -y update
RUN apt-get -y install meshlab
RUN apt-get -y install xvfb

RUN apt-get -y update && \
    apt-get -y install graphviz libxml2-dev python3-cairosvg parallel

# CGAL Dependencies ########################################################
RUN apt-get -y install libboost-all-dev libgmp-dev libmpfr-dev libcgal-dev libboost-wave-dev libeigen3-dev
############################################################################

RUN pip3 install datajoint --upgrade
RUN pip3 install python-igraph xlrd


WORKDIR /src

RUN pip3 install ipyvolume jupyterlab statsmodels pycircstat nose
RUN pip3 install seaborn --upgrade
RUN pip3 install jgraph

#RUN add-apt-repository ppa:boost-latest/ppa
#RUN apt-get update

RUN apt-get -y install vim
RUN . /etc/profile
ADD ./CGAL /src/CGAL
RUN pip3 install -e /src/CGAL


#-------The part of the script that add meshlabserver to path ----#
#RUN bash
#RUN /bin/bash export PATH=${PATH}:/meshlab/src/distrib
#RUN /bin/bash function meshlabserver() { xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@; }
#RUN /bin/bash export -f meshlabserver


WORKDIR /notebooks

#add the cgal scripts
EXPOSE 8895

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./setup_meshlabserver.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
RUN /usr/bin/env bash -c "source /scripts/setup_meshlabserver.sh"


#FROM hamzamerzic/meshlab
#FROM hamzamerzic/meshlab
ENTRYPOINT ["/scripts/run_jupyter.sh"]
