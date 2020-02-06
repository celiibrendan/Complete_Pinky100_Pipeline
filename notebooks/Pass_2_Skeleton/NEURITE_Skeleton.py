
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
from meshparty import trimesh_io

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


# In[5]:


#make sure there is a temp file in the directory, if not then make one
#if temp folder doesn't exist then create it
if (os.path.isdir(os.getcwd() + "/pymesh_NEURITES")) == False:
    os.mkdir("pymesh_NEURITES")


# In[6]:


#create the output file
##write the OFF file for the neuron
import pathlib
def write_Whole_Neuron_Off_file(neuron_ID,
                                vertices=[], 
                                triangles=[],
                                folder="pymesh_NEURITES"):
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


# In[7]:


def meshlab_fix_manifold(key,folder="pymesh_NEURITES"):
    
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


# In[8]:


def run_meshlab_script(mlx_script,input_mesh_file,output_mesh_file):
    script_command = (" -i " + str(input_mesh_file) + " -o " + 
                                    str(output_mesh_file) + " -s " + str(mlx_script))
    #return script_command
    subprocess_result = subprocess.run('xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@ ' + 
                   script_command,shell=True)
    
    return subprocess_result


# In[9]:


pinky.Mesh() & pinky.Neurite() & pinky.CurrentSegmentation


# In[10]:


@schema
class NeuriteSkeleton(dj.Computed):
    definition="""
    -> pinky.Mesh
    ---
    n_edges   :int unsigned #number of edges stored
    edges     :longblob #array storing edges on each row
    n_bodies    :tinyint unsigned #the amount of segments the neurite was originally split into
    largest_mesh_perc : float #the percentage of the entire mesh that the largest submesh makes up
    
    """
    
    
    key_source = pinky.Mesh() & pinky.Neurite() & pinky.CurrentSegmentation
    #how you get the date and time  datetime.datetime.now()
    
    def make(self, key):
        global_time = time.time()
        #get the mesh with the error segments filtered away
        start_time = time.time()
        print(str(key['segment_id']) +  ":")
        my_dict = (pinky.Mesh & pinky.Neurite.proj() & pinky.CurrentSegmentation
                           & key).fetch1()
        print(f"Step 1: Retrieving Mesh and removing error segments: {time.time() - start_time}")
        new_key = dict(segmentation=key["segmentation"],
                       segment_id=key["segment_id"])
        
        
# Don't need these attributes      
#vertices=key["vertices"],
#                       triangles=new_key["triangles"],n_vertices=key["n_vertices"],
#                       n_triangles=key["n_triangles"])
        
        
        start_time = time.time()
        #pass the vertices and faces to pymeshfix to become watertight
        
        mesh = trimesh_io.Mesh(vertices=my_dict["vertices"], faces=my_dict["triangles"])
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
        print(f"Step 2a: Getting the number of splits: {time.time() - start_time}")
        
        
        
        start_time = time.time()
        #pass the vertices and faces to pymeshfix to become watertight
        meshfix = pymeshfix.MeshFix(my_dict["vertices"],my_dict["triangles"])
        meshfix.repair(verbose=False,joincomp=True,remove_smallest_components=False)
        print(f"Step 2b: Pymesh shrinkwrapping: {time.time() - start_time}")
        
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
        bone_array = read_skeleton_revised(output_mesh[:-4]+"_skeleton.cgal")
            
        #print(bone_array)
        if len(bone_array) <= 0:
            raise Exception('No skeleton generated for ' + str(new_key["segment_id"]))
        print(f"Step 5: Generating and reading Skeleton: {time.time() - start_time}")
        
              
        start_time = time.time()
        new_key["n_edges"] = len(bone_array)
        new_key["edges"] = bone_array
        #new_key["branches"] = []
        
        
        #print(key)
        #if all goes well then write to database
        self.insert1(new_key,skip_duplicates=True)
        #raise Exception("done with one neuron")
        os.system("rm "+str(path_and_filename)+"*")
        print(f"Step 6: Inserting dictionary: {time.time() - start_time}")
        print(f"Total time: {time.time() - global_time}")
        print("\n\n")
        
        

                         
                                    


# In[11]:


start = time.time()
NeuriteSkeleton.populate(reserve_jobs=True)
print(time.time() - start)


# In[12]:


#(schema.jobs & "table_name='__neurite_skeleton'").delete()

