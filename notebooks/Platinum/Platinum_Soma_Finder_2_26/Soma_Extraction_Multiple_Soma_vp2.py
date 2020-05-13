#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Testing out the full soma extraction

Pseudocode for Algorithm: 
Load in mesh
Split mesh into largest pieces: 
    Iterate through all mesh pieces of a certain threshold
    Do the Poisson surface reconstruction:
    Find all the mesh pieces of a certain threshold:
        (Optional step) Run the screened poisson surface reconstruction
        Run the segmentation algorithm
        Identify all somas
        Save of the soma meshes


"""


# In[1]:


import cgal_Segmentation_Module as csm
from whole_neuron_classifier_datajoint_adapted import extract_branches_whole_neuron
import whole_neuron_classifier_datajoint_adapted as wcda 
import time
import trimesh
import numpy as np
import datajoint as dj
import os


# # Helper Functions

# In[2]:


def run_meshlab_script(mlx_script,input_mesh_file,output_mesh_file):
    script_command = (" -i " + str(input_mesh_file) + " -o " + 
                                    str(output_mesh_file) + " -s " + str(mlx_script))
    #return script_command
    command_to_run = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + script_command
    #command_to_run = 'meshlabserver ' + script_command
    
    print(command_to_run)
    subprocess_result = subprocess.run(command_to_run,shell=True)
    
    return subprocess_result


# In[3]:


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


# # Step 1) Import mesh and find all the significant pieces

# In[4]:


"""
Setting up the mesh file and the output files
"""

total_test_meshes = [
'110778132960975016_stitched.off']

output_file = total_test_meshes[0]
folder_name = "soma_extraction_tests_vp1/" 

output_mesh_name = folder_name + output_file
print(f"Working on {output_file}")

indices = [i for i, a in enumerate(output_file) if a == "_"]
indices
seg_id_stripped = output_file[:indices[0]]
n = dict(segment_id=int(seg_id_stripped))
segment_id = int(seg_id_stripped)


# # Helper Functions

# In[5]:


import pathlib
def run_poisson_surface_reconstruction(pre_largest_mesh_path,
                                       segment_id = "None",
                                      script_name = "poisson_working_meshlab.mls"):

    """
    Will run the poisson surface reconstruction
    
    """
    # run the meshlab server script

    meshlab_script_path_and_name = str(pathlib.Path.cwd()) + "/" + script_name
    input_path =str(pathlib.Path.cwd()) + "/" +  pre_largest_mesh_path

    indices = [i for i, a in enumerate(input_path) if a == "_"]
    stripped_ending = input_path[:-4]

    output_path = stripped_ending + "_mls.off"
    # print(meshlab_script_path_and_name)
    # print(input_path)
    # print(output_path)
    print("Running the mls function")
    meshlab_fix_manifold_path_specific_mls(input_path_and_filename=input_path,
                                               output_path_and_filename=output_path,
                                               segment_id=segment_id,
                                               meshlab_script=meshlab_script_path_and_name)
    return output_path


# In[ ]:


"""
Loop that will compute the soma meshes and locations

"""

# ------------parameters------------------
large_mesh_threshold = 600000
large_mesh_threshold_inner = 40000
soma_width_threshold = 0.35
soma_size_threshold = 10000

# ------------------------------

new_mesh = trimesh.load_mesh(output_mesh_name)
mesh_splits = new_mesh.split(only_watertight=False)

#len("Total mesh splits = " + str(mesh_splits))
#get the largest mesh
mesh_lengths = np.array([len(split.faces) for split in mesh_splits])

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# sns.distplot(mesh_lengths)

largest_index = np.where(mesh_lengths == np.max(mesh_lengths))
largest_mesh = mesh_splits[largest_index][0]

""" -- temporarily changing to the second largest mesh"""
total_mesh_split_lengths = [len(k.faces) for k in mesh_splits]
ordered_mesh_splits = mesh_splits[np.flip(np.argsort(total_mesh_split_lengths))]
list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold]

print(f"Total found significant pieces before Poisson = {list_of_largest_mesh}")

# total_soma_mesh = trimesh.Trimesh(vertices=np.array([]),
#                                  triangles = np.array([]))

total_soma_list = []
total_classifier_list = []
total_poisson_list = []

#start iterating through 
no_somas_found_in_big_loop = 0
for i,largest_mesh in enumerate(list_of_largest_mesh):
    print(f"----- working on large mesh #{i}: {largest_mesh}")
    
    somas_found_in_big_loop = False

    stripped_ending = output_mesh_name[:-4]
    pre_largest_mesh_path = stripped_ending + "_" + str(i) + "_largest_piece.off"

    largest_mesh.export(pre_largest_mesh_path)
    print("done exporting")
    
    output_path = run_poisson_surface_reconstruction(pre_largest_mesh_path)
    
    #---------------- Will carry out the cgal segmentation -------- #
    #import the mesh
    new_mesh_inner = trimesh.load_mesh(output_path)
    
    mesh_splits_inner = new_mesh_inner.split(only_watertight=False)
    total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
    ordered_mesh_splits_inner = mesh_splits_inner[np.flip(np.argsort(total_mesh_split_lengths_inner))]
    list_of_largest_mesh_inner = [k for k in ordered_mesh_splits_inner if len(k.faces) > large_mesh_threshold_inner]
    print(f"Total found significant pieces AFTER Poisson = {list_of_largest_mesh}")
    
    stripped_ending = output_path[:-4]
    print(f"stripped_ending 2 = {stripped_ending}")
    n_failed_inner_soma_loops = 0
    for j, largest_mesh_inner in enumerate(list_of_largest_mesh_inner):

        print(f"----- working on mesh after poisson #{j}: {largest_mesh_inner}")
        
        largest_mesh_path_inner = stripped_ending +"_" + str(j) + "_largest_inner.off"

        #DON'T NEED THIS WRITE NOW BECAUSE IT ALREADY OUTPUTS THE MESH
        largest_mesh_inner.export(largest_mesh_path_inner)
        print(f"done exporting {largest_mesh_path_inner}")
        
        # Starts the actual cgal segmentation:
        
        faces = np.array(largest_mesh_inner.faces)
        verts = np.array(largest_mesh_inner.vertices)
        #run the whole algorithm on the neuron to test
        verts_labels, faces_labels, soma_value,classifier = wcda.extract_branches_whole_neuron(
                            import_Off_Flag=False,
                            segment_id=segment_id,
                            vertices=verts,
                             triangles=faces,
                            pymeshfix_Flag=False,
                             import_CGAL_Flag=False,
                             return_Only_Labels=True,
                             clusters=3,
                             smoothness=0.2,
                            soma_only=True,
                            return_classifier = True
                            )
        
        total_classifier_list.append(classifier)
        total_poisson_list.append(largest_mesh_inner)

        # Save all of the portions that resemble a soma
        median_values = np.array([v["median"] for k,v in classifier.sdf_final_dict.items()])
        segmentation = np.array([k for k,v in classifier.sdf_final_dict.items()])

        #order the compartments by greatest to smallest
        sorted_medians = np.flip(np.argsort(median_values))
        print(f"segmentation[sorted_medians],median_values[sorted_medians] = {(segmentation[sorted_medians],median_values[sorted_medians])}")
        valid_soma_segments_width = [g for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                            and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold))]
        
        valid_soma_segments_width
        if len(valid_soma_segments_width) > 0:
            print(f"      ------ Found {len(valid_soma_segments_width)} viable somas: {valid_soma_segments_width}")
            somas_found_in_big_loop = True
            #get the meshes only if signfiicant length
            labels_list = classifier.labels_list
            for v in valid_soma_segments_width:
                interest_labels = [k for k in labels_list if k == v]
                soma_mesh = largest_mesh.submesh([interest_labels],append=True)
                total_soma_list.append(v)

            n_failed_inner_soma_loops = 0
            
        else:
            n_failed_inner_soma_loops += 1
            
        
        # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
        if n_failed_inner_soma_loops >= 2:
            print("breaking inner loop because 2 soma fails in a row")
            break
        
    
    # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
    if somas_found_in_big_loop == False:
        no_somas_found_in_big_loop += 1
        if no_somas_found_in_big_loop >= 2:
            print("breaking because 2 fails in a row in big loop")
            break
        
    else:
        no_somas_found_in_big_loop = 0
    
    
    
    


# In[ ]:


import numpy as np
np.savez("saved_4_neuron_mesh.npz",total_soma_list = total_soma_list,total_classifier_list = total_classifier_list, total_poisson_list = total_poisson_list)


# In[ ]:




