#!/usr/bin/env bash

#export PATH=${PATH}:/meshlab/src/distrib

#function meshlabserver() { xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@; }
#export -f meshlabserver

#source meshlab_source.lib

jupyter lab "$@" --allow-root
