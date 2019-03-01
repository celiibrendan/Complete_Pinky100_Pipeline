FROM hamzamerzic/meshlab

RUN apt-get -y update
WORKDIR /src

RUN apt-get -y install vim


WORKDIR /notebooks

#add the cgal scripts
EXPOSE 8896

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]
