#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Purpose: To use the soma extraction algorithm to extract soma centers, meshes and bounding
boxes for the pinky data set (to be used for theoretical neuroscience project)

Thing have to do to make sure meshlab is there: export PATH=$PATH:/meshlab/src/distrib
"""


# In[2]:


import cgal_Segmentation_Module as csm
from whole_neuron_classifier_datajoint_adapted import extract_branches_whole_neuron
import time
import trimesh
import numpy as np
import datajoint as dj

dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
pinky = dj.create_virtual_module('pinky', 'microns_pinky')
schema = dj.schema("microns_pinky")

dj.config["display.limit"] = 40
dj.config["enable_python_native_blobs"] = True


# In[3]:


def run_meshlab_script(mlx_script,input_mesh_file,output_mesh_file):
    script_command = (" -i " + str(input_mesh_file) + " -o " + 
                                    str(output_mesh_file) + " -s " + str(mlx_script))
    #return script_command
    command_to_run = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + script_command
    #command_to_run = 'meshlabserver ' + script_command
    
    print(command_to_run)
    subprocess_result = subprocess.run(command_to_run,shell=True)
    
    return subprocess_result

import os, contextlib
import pathlib
import subprocess
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


# In[4]:


# soma_mesh = trimesh.load_mesh("../Platinum_Blender/soma_test.off")
# soma_mesh.show()
# soma_mesh.bounding_box.vertices
# soma_bounding_box_corners = np.stack([np.min(soma_mesh.bounding_box.vertices,axis=0),
# np.max(soma_mesh.bounding_box.vertices,axis=0)])
# soma_bounding_box_corners[1,0]


# In[5]:


pinky.PymeshfixDecimatedExcitatoryStitchedMesh()


# In[6]:


@schema
class SomaCenters(dj.Computed):
    
    definition="""
    -> pinky.PymeshfixDecimatedExcitatoryStitchedMesh
    ---
    soma_center             : longblob                 # the xyz coordinates of the soma (with the [4,4,40] adjustment already applied)
    vertices            : longblob                     # vertices for soma mesh
    faces                : longblob                    # faces array for soma mesh
    bounding_box         : longblob                    # upper and lower corners for soma bounding box
    bbox_x_min           : int unsigned                # minimum x value for the soma bounding box
    bbox_x_max           : int unsigned                # maximum x value for the soma bounding box
    bbox_y_min           : int unsigned                # minimum y value for the soma bounding box
    bbox_y_max           : int unsigned                # maximum y value for the soma bounding box
    bbox_z_min           : int unsigned                # minimum z value for the soma bounding box
    bbox_z_max           : int unsigned                # maximum z value for the soma bounding box
    
    """
    
    def make(self,key):
        
        segment_id = key["segment_id"]
        segmentation = key["segmentation"]
        print("\n\n******Working on neuron: " + str(segment_id) + "********")
        
        
        verts,faces = (pinky.PymeshfixDecimatedExcitatoryStitchedMesh & key).fetch("vertices","triangles")
        new_mesh_vertices = verts[0]
        new_mesh_faces = faces[0] 
        """
        .[************ retrieve the vertices and faces array of the mesh .[************
        new_mesh_vertices, new_mesh_faces =
        """
        
        """
        start MLS remeshing
        
        """
        
        # make sure temp folder exists, if not then create one
        import os
        directory = "./temp"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        original_main = trimesh.Trimesh(new_mesh_vertices,new_mesh_faces)
        output_mesh_name = "temp/" + str(segment_id) + "_original.off"
        original_main.export("./" + output_mesh_name)
        
        import pathlib
        # run the meshlab server script
        script_name = "poisson_working_meshlab.mls"
        meshlab_script_path_and_name = str(pathlib.Path.cwd()) + "/" + script_name
        input_path =str(pathlib.Path.cwd()) + "/" +  output_mesh_name

        indices = [i for i, a in enumerate(input_path) if a == "_"]
        stripped_ending = input_path[:-(len(input_path)-indices[-1])]

        output_path = stripped_ending + "_mls.off"
        print(meshlab_script_path_and_name)
        print(input_path)
        print(output_path)
        print("Running the mls function")
        meshlab_fix_manifold_path_specific_mls(input_path_and_filename=input_path,
                                                   output_path_and_filename=output_path,
                                                   segment_id=segment_id,
                                                   meshlab_script=meshlab_script_path_and_name)
        
        """
        start the CGAL segmentation:
        """
        new_mesh = trimesh.load_mesh(output_path)
        
        mesh_splits = new_mesh.split(only_watertight=True)

        len("Total mesh splits = " + str(mesh_splits))
        #get the largest mesh
        mesh_lengths = np.array([len(split.faces) for split in mesh_splits])

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # sns.set()
        # sns.distplot(mesh_lengths)

        largest_index = np.where(mesh_lengths == np.max(mesh_lengths))
        largest_mesh = mesh_splits[largest_index][0]


        indices = [i for i, a in enumerate(output_path) if a == "_"]
        stripped_ending = output_path[:-(len(output_path)-indices[-1])]
        largest_mesh_path = stripped_ending + "_largest_piece.off"

        largest_mesh.export(largest_mesh_path)
        print("done exporting")
        
        
        faces = np.array(largest_mesh.faces)
        verts = np.array(largest_mesh.vertices)
        #run the whole algorithm on the neuron to test
        verts_labels, faces_labels = extract_branches_whole_neuron(import_Off_Flag=False,segment_id=segment_id,vertices=verts,
                             triangles=faces,pymeshfix_Flag=False,
                             import_CGAL_Flag=False,
                             return_Only_Labels=True,
                             clusters=3,
                             smoothness=0.2)
        
        soma_faces = np.where(faces_labels == 5.0)[0]
        soma_mesh = largest_mesh.submesh([soma_faces],append=True)
        
        soma_center = soma_mesh.vertices.mean(axis=0).astype("float")
        soma_center = soma_center/np.array([4,4,40])
        print("Poor man's center from just averagin vertices = " + str(soma_center))
        
        soma_bounding_box_corners = np.stack([np.min(soma_mesh.bounding_box.vertices,axis=0),
            np.max(soma_mesh.bounding_box.vertices,axis=0)])

        
        insert_key = dict(key)
        insert_key["soma_center"] = soma_center
        insert_key["vertices"] = soma_mesh.vertices
        insert_key["faces"] = soma_mesh.faces
        insert_key["bounding_box"] = soma_bounding_box_corners
        insert_key["bbox_x_min"] = int(soma_bounding_box_corners[0,0])
        insert_key["bbox_x_max"] = int(soma_bounding_box_corners[1,0])
        insert_key["bbox_y_min"] = int(soma_bounding_box_corners[0,1])
        insert_key["bbox_y_max"] = int(soma_bounding_box_corners[1,1])
        insert_key["bbox_z_min"] = int(soma_bounding_box_corners[0,2])
        insert_key["bbox_z_max"] = int(soma_bounding_box_corners[1,2])
        
        #4) Insert the key into the table
        self.insert1(insert_key,skip_duplicates=True)


# In[10]:


#(schema.jobs & "table_name='__soma_centers'").delete()


# In[8]:


#(schema.jobs & "table_name='__whole_auto_annotations_label_clusters3'")#.delete()
import time
start_time = time.time()
SomaCenters.populate(reserve_jobs=True)
print(f"Total time for SomaCenters populate = {time.time() - start_time}")


# In[ ]:




