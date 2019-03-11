FROM hamzamerzic/meshlab
LABEL maintainer="Brendan Papadopoulos"
RUN apt-get -y update

###############################################################################
# Install some optimization libraries (used by many libraries below)
RUN apt-get update && \
    apt-get install -y libopenblas-dev libatlas-base-dev libeigen3-dev && \
    export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

###############################################################################
# Install Python 3
RUN apt-get update && \
    apt-get install -y python3-dev python3-pip python3-numpy python3-scipy \
        python3-matplotlib

###############################################################################
# Install OpenCV 3.4.3 with CUDA support (take a seat, this takes hours)
# With no CUDA support: sudo apt-get install python3-opencv
RUN pip3 install --upgrade pip
RUN pip3 install nose2 jupyterlab

#add the cgal scripts
EXPOSE 8896

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts


#FROM hamzamerzic/meshlab
#FROM hamzamerzic/meshlab
ENTRYPOINT ["/scripts/run_jupyter.sh"]
