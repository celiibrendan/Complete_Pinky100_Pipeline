#!/bin/bash
[[ ":$PATH:" != *":/meshlab/src/distrib"* ]] && PATH="/meshlab/src/distrib:${PATH}"
#export PATH=${PATH}:/meshlab/src/distrib
function meshlabserver() { xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@; }
export -f meshlabserver
#meshlabserver -i neuron_648518346341366885_new.off -o neuron_648518346341366885_new_2.off -s remeshing_script_4.mlx