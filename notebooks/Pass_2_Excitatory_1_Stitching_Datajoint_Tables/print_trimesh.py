import os
import contextlib

def print_trimesh(current_mesh,file_name):
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        current_mesh.export(file_name)