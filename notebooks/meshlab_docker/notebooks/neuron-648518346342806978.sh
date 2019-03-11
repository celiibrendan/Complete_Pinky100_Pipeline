#!/bin/bash
echo "hello world"
function meshlabserver() { xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@; }
export -f meshlabserver
meshlabserver -i ./temp/neuron_648518346342806978.off -o ./temp/neuron_648518346342806978_new.off -s remeshing_script_5.mlx