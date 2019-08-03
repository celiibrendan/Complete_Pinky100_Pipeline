FROM ninai/pipeline:base
LABEL maintainer="Brendan Papadopoulos"

RUN apt-get -y update && \
    apt-get -y install graphviz libxml2-dev python3-cairosvg parallel


#install meshlab and dependencies
RUN apt-get -y install meshlab xvfb

# CGAL Dependencies ########################################################
RUN apt-get -y install libboost-all-dev libgmp-dev libmpfr-dev libcgal-dev libboost-wave-dev
############################################################################

RUN pip3 install datajoint --upgrade
RUN pip3 install python-igraph xlrd


WORKDIR /src

RUN pip3 install ipyvolume jupyterlab statsmodels pycircstat nose
RUN pip3 install seaborn --upgrade
RUN pip3 install jgraph



RUN apt-get -y install vim
RUN . /etc/profile

#add the segmentation python library
ADD ./CGAL/cgal_segmentation /src/CGAL/cgal_segmentation
RUN pip3 install -e /src/CGAL/cgal_segmentation

#add the skeletonization python library
ADD ./CGAL/cgal_skeleton /src/CGAL/cgal_skeleton
RUN pip3 install -e /src/CGAL/cgal_skeleton




WORKDIR /notebooks

#add the cgal scripts
EXPOSE 8895

#add the cgal scripts

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]
