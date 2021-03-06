
# coding: utf-8

# In[1]:


"""
Purpose: To create the skeletons for the neurons
-- adapted from the Neurite Skeleton that was run successfully on the Neurites

Labels will have to pull from: 
UndecimatedNeuronLabels


"""


# In[2]:


"""
Will generate the skeletons for all of the
Exhitatory Neurons and Orphan Neurons

Process: 
1) Check which table the neuron is in
2) Filter away any error labels 
3) Run pymeshfix on neuron
4) Run skeletonization
5) Write to datajoint as array

"""


# In[3]:


import numpy as np
import datajoint as dj
import time
import pymeshfix
import os
import datetime
import calcification_Module as cm
from meshparty import trimesh_io
import trimesh

#for supressing the output
import os, contextlib
import pathlib
import subprocess

#for error counting
from collections import Counter

#for reading in the new raw_skeleton files
import csv

from Skeleton_Stitcher import stitch_skeleton_with_degree_check, find_skeleton_distance


# In[4]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 100

schema = dj.schema('microns_pinky')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')


# In[5]:


#pinky.NeuritePostsynAxon.drop()


# In[6]:


#output for the skeleton edges to be stored by datajoint
""" OLD WAY THAT DATAJOINT WAS GETTING MAD AT 
def read_skeleton(file_path):
    with open(file_path) as f:
        bones = list()
        for line in f.readlines():
            bones.append(np.array(line.split()[1:], float).reshape(-1, 3))
    return np.array(bones)
"""

""" NEW FLAT LIST WAY"""
#practice reading in dummy skeleton file
def read_skeleton_flat(file_path):
    with open(file_path) as f:
        bones = list()
        for line in f.readlines():
            for r in (np.array(line.split()[1:], float).reshape(-1, 3)):
                bones.append(r)
            bones.append([np.nan,np.nan,np.nan])
    return np.array(bones).astype(float)


""" New read function: for adjusted 2 vert skeleton output"""
def read_raw_skeleton(file_path):
    edges = list()
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        for i,row in enumerate(reader):
            v1 = (float(row[1]),float(row[2]),float(row[3]))
            v2 = (float(row[4]),float(row[5]),float(row[6]))
            edges.append((v1,v2))
    return np.array(edges).astype(float)

def read_skeleton_revised(file_path):
    with open(file_path) as f:
        bones = np.array([])
        for line in f.readlines():
            #print(line)
            line = (np.array(line.split()[1:], float).reshape(-1, 3))
            #print(line[:-1])
            #print(line[1:])

            #print(bones.size)
            if bones.size <= 0:
                bones = np.stack((line[:-1],line[1:]),axis=1)
            else:
                bones = np.vstack((bones,(np.stack((line[:-1],line[1:]),axis=1))))
            #print(bones)


    return np.array(bones).astype(float)


# In[7]:


#make sure there is a temp file in the directory, if not then make one
#if temp folder doesn't exist then create it
if (os.path.isdir(os.getcwd() + "/pymesh_NEURONS")) == False:
    os.mkdir("pymesh_NEURONS")


# In[8]:


#create the output file
##write the OFF file for the neuron
import pathlib
def write_Whole_Neuron_Off_file(neuron_ID,
                                vertices=[], 
                                triangles=[],
                                folder="pymesh_NEURONS"):
    #primary_key = dict(segmentation=1, segment_id=segment_id, decimation_ratio=0.35)
    #vertices, triangles = (mesh_Table_35 & primary_key).fetch1('vertices', 'triangles')
    
    num_vertices = (len(vertices))
    num_faces = len(triangles)
    
    #get the current file location
    file_loc = pathlib.Path.cwd() / folder
    filename = "neuron_" + str(neuron_ID)
    path_and_filename = file_loc / filename
    
    #print(file_loc)
    #print(path_and_filename)
    
    #open the file and start writing to it    
    f = open(str(path_and_filename) + ".off", "w")
    f.write("OFF\n")
    f.write(str(num_vertices) + " " + str(num_faces) + " 0\n" )
    
    
    #iterate through and write all of the vertices in the file
    for verts in vertices:
        f.write(str(verts[0]) + " " + str(verts[1]) + " " + str(verts[2])+"\n")
    
    #print("Done writing verts")
        
    for faces in triangles:
        f.write("3 " + str(faces[0]) + " " + str(faces[1]) + " " + str(faces[2])+"\n")
    
    print("Done writing OFF file")
    #f.write("end")
    
    return str(path_and_filename),str(filename),str(file_loc)


# In[9]:


def meshlab_fix_manifold(key,folder="pymesh_NEURONS"):
    
    file_loc = pathlib.Path.cwd() / folder
    filename = "neuron_" + str(key["segment_id"])
    path_and_filename = str(file_loc / filename)
    
    
    input_mesh = path_and_filename + ".off"
    output_mesh = path_and_filename+"_mls.off"
    
    
    meshlab_script = str(pathlib.Path.cwd()) + "/" + "remeshing_remove_non_man_edges.mls"
    
    print("starting meshlabserver fixing non-manifolds")
    subprocess_result_1 = run_meshlab_script(meshlab_script,
                      input_mesh,
                      output_mesh)
    #print("Poisson subprocess_result= "+ str(subprocess_result_1))
    
    if str(subprocess_result_1)[-13:] != "returncode=0)":
        raise Exception('neuron' + str(key["segment_id"]) + 
                         ' did not fix the manifold edges')
    
    return output_mesh

def meshlab_fix_manifold_path(path_and_filename,segment_id=-1):
    #fix the path if it comes with the extension
    if path_and_filename[-4:] == ".off":
        path_and_filename = path_and_filename[-4:]
    
    input_mesh = path_and_filename + ".off"
    output_mesh = path_and_filename+"_mls.off"
    
    #print("input_mesh = " + str(input_mesh))
    #print("output_mesh = " + str(output_mesh))
    
    meshlab_script = str(pathlib.Path.cwd()) + "/" + "remeshing_remove_non_man_edges.mls"
    
    print("starting meshlabserver fixing non-manifolds")
    subprocess_result_1 = run_meshlab_script(meshlab_script,
                      input_mesh,
                      output_mesh)
    #print("Poisson subprocess_result= "+ str(subprocess_result_1))
    
    if str(subprocess_result_1)[-13:] != "returncode=0)":
        raise Exception('neuron' + str(segment_id) + 
                         ' did not fix the manifold edges')
    
    return output_mesh

def meshlab_fix_manifold_path_specific_mls(path_and_filename,segment_id=-1,meshlab_script=""):
    #fix the path if it comes with the extension
    if path_and_filename[-4:] == ".off":
        path_and_filename = path_and_filename[:-4]
    
    input_mesh = path_and_filename + ".off"
    output_mesh = path_and_filename+"_mls.off"
    
    #print("input_mesh = " + str(input_mesh))
    #print("output_mesh = " + str(output_mesh))
    if meshlab_script == "":
        meshlab_script = str(pathlib.Path.cwd()) + "/" + "remeshing_remove_non_man_edges.mls"
    
    print("meshlab_script = " + str(meshlab_script))
    #print("starting meshlabserver fixing non-manifolds")
    subprocess_result_1 = run_meshlab_script(meshlab_script,
                      input_mesh,
                      output_mesh)
    #print("Poisson subprocess_result= "+ str(subprocess_result_1))
    
    if str(subprocess_result_1)[-13:] != "returncode=0)":
        raise Exception('neuron' + str(segment_id) + 
                         ' did not fix the manifold edges')
    
    return output_mesh


# In[10]:


def run_meshlab_script(mlx_script,input_mesh_file,output_mesh_file):
    script_command = (" -i " + str(input_mesh_file) + " -o " + 
                                    str(output_mesh_file) + " -s " + str(mlx_script))
    #return script_command
    print('xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + script_command)
    subprocess_result = subprocess.run('xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + 
                   script_command,shell=True)
    
    return subprocess_result


# In[11]:


def get_mesh_minus_errors_and_spines(key):
    """
    Will retrieve the mesh from the database
    Retrieve the labels and then subtract the spines and error segments
    from the mesh to get a suitable submesh
    """
    
    #get the mesh
    verts,faces = (pinky.Mesh & key).fetch1("vertices","triangles")
    
    coarse_triangles_labels,overlay_triangles_labels = (pinky.UndecimatedNeuronLabels &
                                                           key).fetch1("coarse_triangles_labels","overlay_triangles_labels")
    
    #get the mask that will filter out errors and 
    non_errors = coarse_triangles_labels != 10
    non_spines = overlay_triangles_labels<12
    
    total_mesh_mask = np.logical_and(non_errors,non_spines)
    
    total_indices = np.linspace(0,len(coarse_triangles_labels)-1,len(coarse_triangles_labels)).astype("int")
    face_indexes = total_indices[total_mesh_mask]
    
    #check to see if the face_indexes is empty
    if not face_indexes.any():
        #create an empty mesh
        current_mesh = trimesh.Trimesh(np.array([]),np.array([]),process=False)
        return current_mesh,np.array([])
    
    
    
    #get a trimesh object of this mesh and return the submesh
    current_mesh = trimesh.Trimesh(verts,faces,process=False)
    new_mesh = current_mesh.submesh([face_indexes],only_watertight=False,append=True)
    return new_mesh,face_indexes
    


# In[12]:


import warnings
from pymeshfix import _meshfix
import pymeshfix

def clean_from_arrays_celii(v, f, verbose=False, joincomp=False,
                      remove_smallest_components=True,clean=True):
    """
    Performs default cleaning procedure on vertex and face arrays
    Returns cleaned vertex and face arrays
    
    Parameters
    ----------
    v : numpy.ndarray
        Numpy n x 3 array of vertices
    f : numpy.ndarray
        Numpy n x 3 array of faces.
    verbose : bool, optional
        Prints progress to stdout.  Default True.
    joincomp : bool, optional
        Attempts to join nearby open components.  Default False
    remove_smallest_components : bool, optional
        Remove all but the largest isolated component from the mesh
        before beginning the repair process.  Default True.
    Examples
    --------
    >>>
    >>> CleanFromFile('inmesh.ply', 'outmesh.ply')
    """
    # Create mesh object and load from file
    tin = _meshfix.PyTMesh(verbose)
    tin.load_array(v, f)

    # repari and return vertex and face arrays
    repair_celii(tin, verbose, joincomp, remove_smallest_components,clean)
    return tin.return_arrays()

#what the repair function does: 
def repair_celii(tin, verbose=False, joincomp=True, remove_smallest_components=True,clean=True):
    """
    Performs mesh repair using default cleaning procedure using a tin object.
    Internal function.  Use CleanFromFile or CleanFromVF.
    """

    # Keep only the largest component (i.e. with most triangles)
    if remove_smallest_components:
        sc = tin.remove_smallest_components()
        if sc and verbose:
            print('Removed %d small components' % sc)

    # join closest components
    if joincomp:
        tin.join_closest_components()
    
    if tin.boundaries():
        if verbose:
            print('Patching holes...')
        holespatched = tin.fill_small_boundaries()
        if verbose:
            print('Patched %d holes' % holespatched)
    
    
        
    tin.boundaries()
    if clean == True:
        # Perform mesh cleaning
        if verbose:
            print('Fixing degeneracies and intersections')
        result = tin.clean()
    else:
        print("Skipping the degenerative cleaning")
        result = False

    # Check boundaries again
    if tin.boundaries():
        if verbose:
            print('Patching holes...')
        holespatched = tin.fill_small_boundaries()
        if verbose:
            print('Patched %d holes' % holespatched)
    
        if verbose:
            print('Performing final check...')
        if clean == True:
            if verbose:
                print('Fixing degeneracies and intersections')
            result = tin.clean()
        else:
            print("Skipping the degenerative cleaning")
            result = False

    if result:
        warnings.warn('MeshFix could not fix everything')


# In[13]:


def get_axon_mesh(key):
    """
    Will return a trimesh object representing the mesh for that soma
    """
    
    #get the mesh
    verts,faces = (pinky.Mesh & key).fetch1("vertices","triangles")
    current_mesh = trimesh.Trimesh(verts,faces,process=False)
    
    #get the labels for the mesh
    face_labels = (pinky.UndecimatedNeuronLabels & key).fetch1("coarse_triangles_labels")
    
    y = np.where(face_labels==6)
    z = np.where(face_labels==7)
    axon_indexes = np.hstack((y,z))
    
    print("axon_indexes = " + str(axon_indexes))
    
    if not axon_indexes.any():
        return trimesh.Trimesh(np.array([]),np.array([])),axon_indexes
    
    #get a submesh
    axon_mesh = current_mesh.submesh(axon_indexes,append=True)
    return axon_mesh,axon_indexes


# In[14]:


@schema
class NeuritePostsynAxon(dj.Computed):
    definition="""
    -> pinky.Mesh
    ---
    n_edges   :int unsigned #number of edges stored
    edges     :longblob #array storing edges on each row
    n_bodies    :int unsigned #the amount of segments the neurite was originally split into
    n_bodies_stitched  :int unsigned #the amount of segments whose skeletons were stitched back together (aka above the significance threshold)
    largest_mesh_perc : float #number of faces of largest submesh / number of faces of entire skeletal mesh
    largest_mesh_distance_perc : float #skeleton length of largest submesh / skeleton length of entire stitched mesh
    n_faces_in_filtered : int unsigned #number of faces in the mesh after spines and errors filtered away
    faces_in_filtered : longblob #array storing face indices used for submesh once errors and spines removed
    """

    
    key_source = (dj.U("segment_id","segmentation") & (pinky.CompartmentFinal.ComponentFinal() 
                                     & [dict(compartment_type=x) for x in ["Axon-Dendr","Axon-Soma"]]))  + (dj.U("segment_id","segmentation") & (pinky.CompartmentOrphan.ComponentOrphan() 
                                     & [dict(compartment_type=x) for x in ["Axon-Dendr","Axon-Soma"]])) 

    
    def make(self, key):
        print("----------Working on " + str(key['segment_id']) +  ":----------")
        
        split_significance_threshold = 200

        global_time = time.time()
        
        #get the mesh with the error segments filtered away
        start_time = time.time()
        
        
        #gets the axons
        mesh,face_indexes = get_axon_mesh(key)
        
        new_key = dict(segmentation=key["segmentation"],
                           segment_id=key["segment_id"])
        
        """
        Handle the poosibility that there is no mesh
        """
        if len(mesh.faces) < 5:
            print(f"There were only {len(mesh.faces)} faces in mesh so not doing skeletonization")
            new_key["n_edges"] = 0
            new_key["edges"] = np.array([]).astype(float)
            new_key["n_bodies"] = 0
            new_key["n_bodies_stitched"] = 0
            new_key["largest_mesh_perc"] = 0
            new_key["largest_mesh_distance_perc"] = 0
            new_key["n_faces_in_filtered"] = len(face_indexes)
            new_key["faces_in_filtered"] = face_indexes
            
            self.insert1(new_key,skip_duplicates=True)
        else:


            print(f"Step 1: Retrieving Mesh and removing error segments: {time.time() - start_time}")
            


            # Don't need these attributes      
            #vertices=key["vertices"],
            #                       triangles=new_key["triangles"],n_vertices=key["n_vertices"],
            #                       n_triangles=key["n_triangles"])


            start_time = time.time()
            
            """ OLDER WAY OF JUST GETTING THE LARGEST MESH PIECE
            count, labels = trimesh_io.trimesh.graph.csgraph.connected_components(
                                                                mesh.edges_sparse,
                                                                directed=False,
                                                                return_labels=True)


            label_counter = Counter(labels)


            new_key["n_bodies"] = count
            values = np.array(labels)


            list_counter = Counter(labels)
            max_counter = max(list_counter.values())

            max_label = -1
            for label_key,label_number in list_counter.items():
                if label_number==max_counter:
                    max_label = label_key
            print("max label = " + str(max_label))

            searchval = max_label

            ii = np.where(values == searchval)[0]
            new_key["largest_mesh_perc"] = len(ii)/len(labels)

            print("n_bodies = " + str(new_key["n_bodies"]))
            print("largest mesh perc = " + str(new_key["largest_mesh_perc"]))
            """



            total_splits = mesh.split(only_watertight=False)
            print(f"There were {len(total_splits)} after split and significance threshold")
            mesh_pieces = [k for k in total_splits if len(k.faces) > split_significance_threshold]
            print(f"There were {len(mesh_pieces)} after split and significance threshold")
            for g,mh in enumerate(mesh_pieces):
                print(f"Mesh piece {g} with number of faces {len(mh.faces)}")

            print(f"Step 2a: Getting the number of splits: {time.time() - start_time}")

            #get the largest mesh piece
            largest_mesh_index = -1
            largest_mesh_size = 0

            for t,msh in enumerate(mesh_pieces):
                if len(msh.faces) > largest_mesh_size:
                    largest_mesh_index = t
                    largest_mesh_size = len(msh.faces) 

            #largest mesh piece
            largest_mesh_perc = largest_mesh_size/len(mesh.faces)
            new_key["largest_mesh_perc"] = largest_mesh_perc
            print("largest mesh perc = " + str(largest_mesh_perc))

            largest_mesh_skeleton_distance = -1

            paths_used = []
            total_edges = np.array([])
            for h,m in enumerate(mesh_pieces): 
                print(f"Working on split {h} with face total = {len(m.faces)}")


    #             start_time = time.time()
    #             #pass the vertices and faces to pymeshfix to become watertight
    #             meshfix = pymeshfix.MeshFix(m.vertices,m.faces)
    #             meshfix.repair(verbose=False,joincomp=True,remove_smallest_components=False)
    #             print(f"Step 2b: Pymesh shrinkwrapping: {time.time() - start_time}")

    #             #print("Step 2: Writing Off File")
    #             start_time = time.time()
    #             #write the new mesh to off file
    #             path_and_filename,filename,file_loc = write_Whole_Neuron_Off_file(str(new_key["segment_id"]) + "_piece_" + str(h),meshfix.v,meshfix.f)
    #             print(f"Step 3: Writing shrinkwrap off file: {time.time() - start_time}")
                #add the path to be deleted later
        
                #send the data through the hole patching function from pymesh:
                cleaned_verts, cleaned_triangles= clean_from_arrays_celii(m.vertices,m.faces,verbose=True,joincomp=True,remove_smallest_components=False,clean=False)
                #cleaned_mesh = make_trimesh_object(m.vertices,m.faces)
            
            
        
                path_and_filename,filename,file_loc = write_Whole_Neuron_Off_file(str(new_key["segment_id"]) + "_piece_" + str(h),cleaned_verts, cleaned_triangles)

                paths_used.append(path_and_filename)


                #Run the meshlabserver scripts
                start_time = time.time()

                #output_mesh = meshlab_fix_manifold_path(path_and_filename,key["segment_id"])
                meshlab_script = str(pathlib.Path.cwd()) + "/" + "pymesh_fix_substitute_2.mls"
                output_mesh = meshlab_fix_manifold_path_specific_mls(path_and_filename,key["segment_id"],meshlab_script)
                
                
                meshlab_script_2 = str(pathlib.Path.cwd()) + "/" + "remove_intersecting_faces.mls"
                output_mesh = meshlab_fix_manifold_path_specific_mls(output_mesh,key["segment_id"],meshlab_script_2)
                
                print("About to run pymeshfix")
                start_time_py = time.time()
                loaded_mesh = trimesh.load_mesh(output_mesh,process=False)
                meshfix = pymeshfix.MeshFix(loaded_mesh.vertices,loaded_mesh.faces)
                meshfix.repair(verbose=False,joincomp=True,remove_smallest_components=False)
                print(f"Total time for pymeshfix = {time.time() - start_time_py}")
                
                path_and_filename,filename,file_loc = write_Whole_Neuron_Off_file(str(new_key["segment_id"]) + "_piece_" + str(h),meshfix.v, meshfix.f)
                
                
                print(f"Step 4: Meshlab fixing non-manifolds: {time.time() - start_time}")

                print(path_and_filename)

                #send to be skeletonized
                start_time = time.time()

                mls_mesh = trimesh.load_mesh(path_and_filename + ".off")

                if len(mls_mesh.faces) < 20:
                    print("Number of faces are less than 20 so not generating skeleton")
                    continue

                return_value = cm.calcification(path_and_filename)
                if return_value > 0:
                    raise Exception('skeletonization for neuron ' + str(new_key["segment_id"]) + 
                                    ' did not finish... exited with error code: ' + str(return_value))

                    print("Trying skeletonization with pymesh")

                    #try to run the same skeletonization but now with skeletonization

                    #             start_time = time.time()
                    #pass the vertices and faces to pymeshfix to become watertight


                    meshfix = pymeshfix.MeshFix(mls_mesh.vertices,mls_mesh.faces)
                    meshfix.repair(verbose=False,joincomp=True,remove_smallest_components=False)
                    print(f"Step 2b: Pymesh shrinkwrapping: {time.time() - start_time}")

                    if len(meshfix.f) < 20:
                        print("Number of faces are less than 20 so not generating skeleton")
                        continue

                    #print("Step 2: Writing Off File")
                    start_time = time.time()
                    #write the new mesh to off file
                    path_and_filename,filename,file_loc = write_Whole_Neuron_Off_file(str(new_key["segment_id"]) + "_piece_" + str(h),meshfix.v,meshfix.f)
                    print(f"Step 3: Writing shrinkwrap off file: {time.time() - start_time}")
                    #add the path to be deleted later
                    paths_used.append(path_and_filename)

                    #Run the meshlabserver scripts
                    start_time = time.time()

                    #output_mesh = meshlab_fix_manifold_path(path_and_filename,key["segment_id"])
                    meshlab_script = str(pathlib.Path.cwd()) + "/" + "pymesh_fix_substitute_2.mls"
                    output_mesh = meshlab_fix_manifold_path_specific_mls(path_and_filename,key["segment_id"],meshlab_script)
                    

                    print(f"Step 4: Meshlab fixing non-manifolds: {time.time() - start_time}")

                    #print(output_mesh[:-4])

                    #send to be skeletonized
                    start_time = time.time()

                    mls_mesh = trimesh.load_mesh(output_mesh)

                    if len(mls_mesh.faces) < 20:
                        print("Number of faces are less than 20 so not generating skeleton")
                        continue

                    return_value = cm.calcification(output_mesh[:-4])

                    if return_value > 0:
                        raise Exception('skeletonization for neuron ' + str(new_key["segment_id"]) + 
                                    ' did not finish EVEN AFTER TRYING PYMESH... exited with error code: ' + str(return_value))



                print(f"Step 5: Generating Skeleton: {time.time() - start_time}")



                #read in the skeleton files into an array
                bone_array = read_skeleton_revised(path_and_filename+"_skeleton.cgal")

                #print(bone_array)
                if len(bone_array) <= 0:
                    raise Exception('No skeleton generated for ' + str(new_key["segment_id"]))
                print(f"Step 5: Generating and reading Skeleton: {time.time() - start_time}")

                #get the largest mesh skeleton distance
                if h == largest_mesh_index:
                    largest_mesh_skeleton_distance = find_skeleton_distance(bone_array)

                #add the skeleton edges to the total edges
                if not total_edges.any():
                    total_edges = bone_array
                else:
                    total_edges = np.vstack([total_edges,bone_array])


            total_edges_stitched = stitch_skeleton_with_degree_check(total_edges)

            #get the total skeleton distance for the stitched skeleton
            total_skeleton_distance = find_skeleton_distance(total_edges_stitched)

            largest_mesh_distance_perc = largest_mesh_skeleton_distance/total_skeleton_distance


            start_time = time.time()
            new_key["n_edges"] = len(total_edges_stitched)
            new_key["edges"] = total_edges_stitched
            new_key["n_bodies"] = len(total_splits)
            new_key["n_bodies_stitched"] = len(mesh_pieces)
            new_key["largest_mesh_perc"] = largest_mesh_perc
            new_key["largest_mesh_distance_perc"] = largest_mesh_distance_perc
            new_key["n_faces_in_filtered"] = len(face_indexes)
            new_key["faces_in_filtered"] = face_indexes
            
            #print("new_key=" + str(new_key))

            self.insert1(new_key,skip_duplicates=True)
            print(f"Step 6: Inserting dictionary: {time.time() - start_time}")
            #raise Exception("done with one neuron")
            for path_and_filename in paths_used:
                os.system("rm "+str(path_and_filename)+"*")

            print(f"Total time: {time.time() - global_time}")
            print("\n\n")

        

                         
                                    


# In[15]:


#(schema.jobs & "table_name='__neurite_postsyn_axon'" & "status='error'").delete()


# In[16]:


#NeuronSkeletonStitched & "segment_id=neuron648518346349487734"


# In[ ]:


start = time.time()
NeuritePostsynAxon.populate(reserve_jobs=True)
print(time.time() - start)

