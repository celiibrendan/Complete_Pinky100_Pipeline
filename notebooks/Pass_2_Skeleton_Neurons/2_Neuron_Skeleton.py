
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
import datajoint as dj
import time
import pymeshfix
import os
import datetime
import calcification_Module as cm

#for supressing the output
import os, contextlib
import pathlib
import subprocess

#for error counting
from collections import Counter

#for reading in the new raw_skeleton files
import csv


# In[3]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

schema = dj.schema('microns_pinky')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')


# In[4]:


#ta3p100.CoarseLabelFinal() #ta3p100.CoarseLabelOrphan()


# In[5]:


#function that will filter out error triangles
def generate_neighborhood(triangles, num_vertices):
    neighborhood = dict()
    for i in range(num_vertices):
        neighborhood[i] = set()
    for node1, node2, node3 in triangles:
        neighborhood[node1].update([node2, node3])
        neighborhood[node2].update([node1, node3])
        neighborhood[node3].update([node1, node2])
    return neighborhood

def set_search_first(starting_node, neighborhood):
    """
    Modified Depth-First-Search utilizing sets to reduce duplicate checks:

    Neighborhood must be a dict with the keys being the vertex indices!
    """    
    visited_nodes = set()
    temp_stack = set()
    temp_stack.add(starting_node)
    while len(temp_stack) > 0:
        starting_node = temp_stack.pop()
        if starting_node not in visited_nodes:
            visited_nodes.add(starting_node)
            temp_stack.update(neighborhood[starting_node])
    return list(visited_nodes)
def get_connected_portions(neighborhood):
    neighborhood_copy = neighborhood.copy()
    portions = []
    while len(neighborhood_copy) > 0:
        starting_node = next(iter(neighborhood_copy))
        portion = set_search_first(starting_node, neighborhood_copy)
        for node in portion:
            neighborhood_copy.pop(node)
        portions.append(portion)
    return portions

def get_largest_portion_index(portions):
    portion_lengths = [len(portion) for portion in portions]
    return portion_lengths.index(max(portion_lengths))

def get_largest_portion(portions):
    return portions[get_largest_portion_index(portions)]

def remove_floating_artifacts(mesh,key,mesh_labels):    
    mesh_copy = mesh.copy()
    
#     #get the labels for the mesh
#     #find out if in Orphan Table or Regular Neuron Table
#     if len(ta3p100.CoarseLabelFinal() & key) > 0:
#         mesh_labels = (ta3p100.CoarseLabelFinal & key).fetch1()
#     elif len(ta3p100.CoarseLabelOrphan() & key) > 0:
#         mesh_labels = (ta3p100.CoarseLabelOrphan & key).fetch1()
#     else:
#         raise Exception('neuron' + str(key["segment_id"]) + 
#                         'not present in any labels!')

    
    #look for errors
    not_errors = [i for i,k in enumerate(mesh_labels["triangles"]) if k != 10]
    original_triangles = mesh["triangles"]
    """
    print(type(not_errors))
    print(len(not_errors))
    print("not_errors = "+ str(not_errors[:100]))
    print(type(original_triangles))
    print(len(original_triangles))
    #print(original_triangles)
    print("not_errors = " + str(original_triangles[not_errors]))
    """
    
    mesh_copy['triangles'] = np.array(original_triangles[not_errors])
    
    return mesh_copy


def remove_isolated_vertices(mesh):
    mesh_copy = mesh.copy()

    neighborhood = generate_neighborhood(mesh_copy['triangles'], len(mesh_copy['vertices']))
    isolated_nodes = [portion.pop() for portion in get_connected_portions(neighborhood) if len(portion) == 1]

    vertices = mesh_copy['vertices']
    triangles = mesh_copy['triangles']
    vertex_list = list(vertices)

    if len(isolated_nodes) > 0:
        num_isolated_nodes_passed = 0
        isolated_nodes_set = set(isolated_nodes)
        count_to_decrement = np.zeros(len(vertices))
        for i in range(len(vertices)):
            if i in isolated_nodes_set:
                num_isolated_nodes_passed += 1
            else:
                count_to_decrement[i] = num_isolated_nodes_passed

        for i, triangle in enumerate(triangles):
            start = time.time()
            node1, node2, node3 = triangle
            triangles[i][0] -= count_to_decrement[node1]
            triangles[i][1] -= count_to_decrement[node2]
            triangles[i][2] -= count_to_decrement[node3]
        for i, isolated_node in enumerate(isolated_nodes):
            vertex_list.pop(isolated_node - i)

    mesh_copy['vertices'] = np.array(vertex_list)

    return mesh_copy


def remove_error_segments(key):

    full_start = time.time()

    print(str(key['segment_id']) +  ":")
    start = time.time()

    #find out if in Orphan Table or Regular Neuron Table
    if len(pinky.CoarseLabelFinal() & key) > 0:
        mesh = (pinky.PymeshfixDecimatedExcitatoryStitchedMesh & key).fetch1()
        mesh_labels = (pinky.CoarseLabelFinal & key).fetch1()
    elif len(pinky.CoarseLabelOrphan() & key) > 0:
        mesh = (pinky.Decimation35OrphanStitched & key).fetch1()
        mesh_labels = (pinky.CoarseLabelOrphan & key).fetch1()
    else:
        raise Exception('neuron' + str(key["segment_id"]) + 
                        'not present in any labels!')
    
    print(key['segment_id'], "mesh fetched.", time.time() - start)
    start = time.time()
    
    #print(mesh['triangles'])
    myCounter = Counter(mesh_labels['triangles'])#.tolist())
    print(myCounter)
    
    keys = list(myCounter.keys())
    #print(len(keys))
    
    if len(keys) < 2 and keys[0] == 10:
        print("only error segments")
        key['n_vertices'] = 0
        key['n_triangles'] = 0
        key['vertices'] = np.ndarray([])
        key['triangles'] = np.ndarray([])
        
        print("This took ", time.time() - full_start, "seconds.")
        print()
        return key
    
    neighborhood = generate_neighborhood(mesh['triangles'], len(mesh['vertices']))
    print(key['segment_id'] , "neighborhood generated.", time.time() - start)
    start = time.time()
    
    mesh = remove_floating_artifacts(mesh,key,mesh_labels)
    print(key['segment_id'], "floating artifacts removed.", time.time() - start)
    start = time.time()

    mesh = remove_isolated_vertices(mesh)
    print(key['segment_id'], "isolated nodes removed.", time.time() - start)
    start = time.time()

    key['n_vertices'] = len(mesh['vertices'])
    key['n_triangles'] = len(mesh['triangles'])
    key['vertices'] = mesh['vertices']
    key['triangles'] = mesh['triangles']

    #self.insert1(key, skip_duplicates=True)
    print(key['segment_id'], "key successfully filtered.", time.time() - start)
    start = time.time()

    print("This took ", time.time() - full_start, "seconds.")
    print()
    return key


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

""" NEW FLAT LIST WAY, this is outdated for one below"""
#
def read_skeleton_flat(file_path):
    with open(file_path) as f:
        bones = list()
        for line in f.readlines():
            for r in (np.array(line.split()[1:], float).reshape(-1, 3)):
                bones.append(r)
            bones.append([np.nan,np.nan,np.nan])
    return np.array(bones).astype(float)


""" New read function: for adjusted 2 vert skeleton output"""
# def read_raw_skeleton(file_path):
#     edges = list()
#     with open(file_path) as f:
#         reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
#         for i,row in enumerate(reader):
#             v1 = (float(row[1]),float(row[2]),float(row[3]))
#             v2 = (float(row[4]),float(row[5]),float(row[6]))
#             edges.append((v1,v2))
#     return np.array(edges).astype(float)


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
if (os.path.isdir(os.getcwd() + "/pymesh_neurons")) == False:
    os.mkdir("pymesh_neurons")


# In[8]:


#keysource for neuron table
pinky.CoarseLabelFinal()
print(len(pinky.CoarseLabelFinal()))


# In[9]:


pinky.CoarseLabelOrphan()
print(len(pinky.CoarseLabelOrphan()))


# In[10]:


ns_table = ((dj.U("segmentation","segment_id") & pinky.CoarseLabelFinal.proj()) 
     + (dj.U("segmentation","segment_id") & pinky.CoarseLabelOrphan.proj()))
print(len(ns_table))
ns_table


# In[11]:


#create the output file
##write the OFF file for the neuron
import pathlib
def write_Whole_Neuron_Off_file(neuron_ID,
                                vertices=[], 
                                triangles=[],
                                folder="pymesh_neurons"):
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


# In[12]:


def meshlab_fix_manifold(key,folder="pymesh_neurons"):
    
    file_loc = pathlib.Path.cwd() / folder
    filename = "neuron_" + str(key["segment_id"])
    path_and_filename = str(file_loc / filename)
    
    
    input_mesh = path_and_filename + ".off"
    output_mesh = path_and_filename+"_mls.off"
    
    
    meshlab_script = str(pathlib.Path.cwd()) + "/" + "remeshing_remove_non_man_edges.mls"
    
    print("starting remeshing_remove_non_man_edges")
    subprocess_result_1 = run_meshlab_script(meshlab_script,
                      input_mesh,
                      output_mesh)
    #print("Poisson subprocess_result= "+ str(subprocess_result_1))
    
    if str(subprocess_result_1)[-13:] != "returncode=0)":
        raise Exception('neuron' + str(key["segment_id"]) + 
                         ' did not fix the manifold edges')
    
    return output_mesh


# In[13]:


def run_meshlab_script(mlx_script,input_mesh_file,output_mesh_file):
    script_command = (" -i " + str(input_mesh_file) + " -o " + 
                                    str(output_mesh_file) + " -s " + str(mlx_script))
    #return script_command
    subprocess_result = subprocess.run('xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + 
                   script_command,shell=True)
    
    return subprocess_result


# In[14]:


@schema
class NeuronSkeleton(dj.Computed):
    definition="""
    -> pinky.Mesh
    ---
    n_edges   :int unsigned #number of edges stored
    edges     :longblob #array storing edges on each row
    n_vertices   :int unsigned #number of vertices in mesh filtered of error segments
    n_triangles  :int unsigned #number of faces in mesh filtered of error segments
    vertices     :longblob #mesh data for vertices in mesh filtered of error segments
    triangles    :longblob #mesh data for faces in mesh filtered of error segments
     
    """
    
    key_source = ((dj.U("segmentation","segment_id") & pinky.CoarseLabelFinal.proj()) 
     + (dj.U("segmentation","segment_id") & pinky.CoarseLabelOrphan.proj()))
    
    #how you get the date and time  datetime.datetime.now()
    
    def make(self, key):
        global_time = time.time()
        #get the mesh with the error segments filtered away
        start_time = time.time()
        new_key = remove_error_segments(key)
        print(f"Step 1: Retrieving Mesh and removing error segments: {time.time() - start_time}")
        
        #where i deal with the error segments
        if new_key["vertices"].size<2:
            start_time = time.time()
            print("All faces were error segments, inserting dummy entry")
            #create the key with None
            new_key["n_vertices"] = 0
            new_key["n_triangles"] = 0
            new_key["vertices"] = np.array([]).astype(float)
            new_key["triangles"] = np.array([]).astype(float)
            new_key["n_edges"] = 0
            new_key["edges"] = np.array([]).astype(float)
            self.insert1(new_key,skip_duplicates=True)
            
            #insert dummy dictionary into correspondence table
#             new_correspondence_dict = dict(segmentation=key["segmentation"],
#                                            segment_id=key["segment_id"],
#                                            time_updated=str(datetime.datetime.now()),
#                                            n_correspondence = 0,
#                                            correspondence=np.array([]).astype(float))
            
#             #if all goes well then write to correspondence database
#             ta3p100.NeuronRawSkeletonCorrespondence.insert1(new_correspondence_dict,skip_duplicates=True)
            
            
            print(f"Step 2: Inserting dummy dictionary: {time.time() - start_time}")
            print(f"Total time: {time.time() - global_time}")
            print("\n\n")
        
        else:
        
            #print("Step 2: Remove all error semgents")
            start_time = time.time()
            #pass the vertices and faces to pymeshfix to become watertight
            meshfix = pymeshfix.MeshFix(new_key["vertices"],new_key["triangles"])
            meshfix.repair(verbose=False,joincomp=True,remove_smallest_components=False)
            print(f"Step 2: Pymesh shrinkwrapping: {time.time() - start_time}")

            #print("Step 2: Writing Off File")
            start_time = time.time()
            #write the new mesh to off file
            path_and_filename,filename,file_loc = write_Whole_Neuron_Off_file(str(new_key["segment_id"]),meshfix.v,meshfix.f)
            print(f"Step 3: Writing shrinkwrap off file: {time.time() - start_time}")

            #Run the meshlabserver scripts
            start_time = time.time()
            output_mesh = meshlab_fix_manifold(key)
            print(f"Step 4: Meshlab fixing non-manifolds: {time.time() - start_time}")

            print(output_mesh[:-4])

            #send to be skeletonized
            start_time = time.time()
            return_value = cm.calcification(output_mesh[:-4])
            if return_value > 0:
                raise Exception('skeletonization for neuron ' + str(new_key["segment_id"]) + 
                                ' did not finish... exited with error code: ' + str(return_value))
            #print(f"Step 5: Generating Skeleton: {time.time() - start_time}")



            #read in the skeleton files into an array
            #start_time = time.time()
            
            ##****** this needs to be changed for reading them in******
            bone_array = read_skeleton_revised(output_mesh[:-4]+"_skeleton.cgal")
            #correspondence_array = read_skeleton_revised(output_mesh[:-4]+"_correspondance.cgal")
            #print(bone_array)
            if len(bone_array) <= 0:
                raise Exception('No skeleton generated for ' + str(new_key["segment_id"]))
            
#             if len(correspondence_array) <= 0:
#                 raise Exception('No CORRESPONDENCE generated for ' + str(new_key["segment_id"]))
                
            print(f"Step 5: Generating and reading Skeleton: {time.time() - start_time}")


            start_time = time.time()
            
            new_key["n_edges"] = bone_array.shape[0]
            new_key["edges"] = bone_array
            #new_key["branches"] = []

            #print(key)
            #if all goes well then write to database
            self.insert1(new_key,skip_duplicates=True)
            
            #create a new dictionary key to be inserted into correspondence table
            """
            time_updated      :timestamp    # the time at which the component labels were updated
            ---
            n_correspondence   :int unsigned #number of mappings from skeleton vert to original surface vert 
            correspondence     :longblob #array storing mapping of every skeleton vert to original surface vert  
            """
#             #insert dummy dictionary into correspondence table
#             new_correspondence_dict = dict(segmentation=key["segmentation"],
#                                            segment_id=key["segment_id"],
#                                            time_updated=str(datetime.datetime.now()),
#                                            n_correspondence = correspondence_array.shape[0],
#                                            correspondence=correspondence_array)
            
#             #if all goes well then write to correspondence database
#             ta3p100.NeuronRawSkeletonCorrespondence.insert1(new_correspondence_dict,skip_duplicates=True)
            
            
            os.system("rm "+str(path_and_filename)+"*")
            print(f"Step 6: Inserting both dictionaries: {time.time() - start_time}")
            print(f"Total time: {time.time() - global_time}")
            print("\n\n")
          
                         
                                    


# In[21]:


#pinky.NeuronSkeleton()#.delete()
#(schema.jobs & "table_name='__neuron_skeleton'").delete()


# In[ ]:


start = time.time()
NeuronSkeleton.populate(reserve_jobs=True)
print(time.time() - start)

