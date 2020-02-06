import os, contextlib
import pathlib
import subprocess

def run_meshlab_script(mlx_script,input_mesh_file,output_mesh_file):
    script_command = (" -i " + str(input_mesh_file) + " -o " + 
                                    str(output_mesh_file) + " -s " + str(mlx_script))
    #return script_command
    command_to_run = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + script_command
    #command_to_run = 'meshlabserver ' + script_command
    
    print(command_to_run)
    subprocess_result = subprocess.run(command_to_run,shell=True)
    
    return subprocess_result

def meshlab_fix_manifold_path_specific_mls(input_path_and_filename,
                                           output_path_and_filename="",
                                           segment_id=-1,meshlab_script=""):
    #fix the path if it comes with the extension
    if input_path_and_filename[-4:] == ".off":
        path_and_filename = input_path_and_filename[:-4]
        input_mesh = input_path_and_filename
    else:
        raise Exception("Not passed off file")
    
    
    if output_path_and_filename == "":
        output_mesh = path_and_filename+"_mls.off"
    else:
        output_mesh = output_path_and_filename
    
    if meshlab_script == "":
        meshlab_script = str(pathlib.Path.cwd()) + "/" + "remeshing_remove_non_man_edges.mls"
    
    #print("meshlab_script = " + str(meshlab_script))
    #print("starting meshlabserver fixing non-manifolds")
    subprocess_result_1 = run_meshlab_script(meshlab_script,
                      input_mesh,
                      output_mesh)
    #print("Poisson subprocess_result= "+ str(subprocess_result_1))
    
    if str(subprocess_result_1)[-13:] != "returncode=0)":
        raise Exception('neuron' + str(segment_id) + 
                         ' did not fix the manifold edges')
    
    return output_mesh

# create a temp folder if doesn't already exist
import os
folder_name = "decimation_temp"
directory = "./" + str(folder_name)
if not os.path.exists(directory):
    os.makedirs(directory)

import trimesh

def decimate_mesh(vertices,faces,segment_id,current_folder):
    #write the file to the temp folder
#     input_file_base = write_Whole_Neuron_Off_file(vertices, faces, segment_id, folder=current_folder)
    input_file_base = os.path.join(current_folder, f'neuron_{segment_id}')
    trimesh.Trimesh(vertices=vertices, faces=faces).export(input_file_base+".off", )
    output_file = input_file_base + "_decimated"
    
    script_name = "decimation_meshlab.mls"
    meshlab_script_path_and_name = str(pathlib.Path.cwd()) + "/" + script_name
    
    meshlab_fix_manifold_path_specific_mls(
        input_path_and_filename=input_file_base + ".off",
        output_path_and_filename=output_file + ".off",
        meshlab_script=meshlab_script_path_and_name
    )
    
    #read in the output mesh and return the vertices and faces
    current_mesh = trimesh.load_mesh(output_file + '.off')
    
    #check if file exists and then delete the temporary decimated mesh filess
    if os.path.exists(input_file_base + ".off"):
        os.remove(input_file_base + ".off")
    if os.path.exists(output_file + ".off"):
        os.remove(output_file + ".off")
 
    return current_mesh.vertices, current_mesh.faces

from minfig import * # Required for the adapters to be used with locally defined tables

# Virtual module accessors
minnie = configure_minnie(return_virtual_module=True) # virtual module with the adapted attribute for mesh access from .h5 files

@minnie.schema
class Decimation(dj.Computed):
    definition = minnie.Decimation.definition
    
    # Creates hf file at the proper location, returns the filepath of the newly created file
    @classmethod
    def make_file(cls, segment_id, version, vertices, faces):
        """Creates hf file at the proper location, returns the filepath of the newly created file"""
        
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3

        filename = f'{segment_id}_{version}.h5'
        filepath = os.path.join(external_decimated_mesh_path, filename)
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('segment_id', data=segment_id)
            hf.create_dataset('version', data=version)
            hf.create_dataset('vertices', data=vertices)
            hf.create_dataset('faces', data=faces)

        return filepath

    @classmethod
    def make_entry(cls, segment_id, version, decimation_ratio, vertices, faces):
        key = dict(
            segment_id=segment_id,
            version=version,
            decimation_ratio=decimation_ratio,
            n_vertices=len(vertices),
            n_faces=len(faces)
        )

        filepath = cls.make_file(segment_id, version, vertices, faces)

        cls.insert1(dict(key, mesh=filepath), allow_direct_insert=True)

    key_source = minnie.Mesh - minnie.DecimationError & 'n_vertices < 50000000'

    def make(self, key):
        mesh = (minnie.Mesh & key).fetch1('mesh')
        
        version = 0
        decimation_ratio = 0.25 # Only the value in the .mls can change the ratio though.
        
        try:
            new_vertices, new_faces = decimate_mesh(mesh.vertices, mesh.faces, key['segment_id'], folder_name)
            self.make_entry(
                segment_id=key['segment_id'],
                version=version,
                decimation_ratio=decimation_ratio,
                vertices=new_vertices,
                faces=new_faces
            )
        except Exception as e:
            key['version'] = version
            key['decimation_ratio'] = decimation_ratio
            minnie.DecimationError.insert1(dict(key, log=str(e)))
            print(e)
#             raise e

if __name__ == '__main__':
    import random
    import time

    # Random sleep delay to avoid concurrent key_source queries from hangin
    time.sleep(random.randint(0, 360))
    print('Populate Started')
    Decimation.populate(reserve_jobs=True, suppress_errors=False, order='random')