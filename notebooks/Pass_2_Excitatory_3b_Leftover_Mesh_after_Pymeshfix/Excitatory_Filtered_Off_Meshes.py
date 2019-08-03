
# coding: utf-8

# In[1]:


from boolean import neuron_boolean_difference
import datajoint as dj
import numpy as np
import trimesh
import time


# In[2]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

schema = dj.schema('microns_pinky')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')


# In[ ]:


"""
Could have put these in the table but decided not to:
significan_threshold : int              # number of faces needed for seperated submesh to be considered significant and then recovered
distance_threshold   : int              # distance away for vertices to be considered part of filtered mesh (used in KDTree)
    

"""


# In[3]:


@schema
class ExcitatoryLeftoverMeshes(dj.Computed):
    definition="""
    -> pinky.PymeshfixDecimatedExcitatoryStitchedMesh
    ---
    n_vertices           : bigint           # number of vertices in this mesh pieces that were filtered away
    n_triangles          : bigint           # number of triangles in this mesh pieces that were filtered away
    vertices             : longblob         # x,y,z coordinates of vertices
    triangles            : longblob         # triangles (triplets of vertices)
    
    recovered_perc       : decimal(6,5)     # number of faces of this recovered mesh  / number of faces in filtered pymeshfix mesh
    """
    

    def make(self, key):
        print("\n\n------Working on segment " + str(key["segment_id"]) + "-------")
        global_time = time.time()
        #get the pymeshfixed mesh
        start_time = time.time()
        verts_fixed, faces_fixed = (pinky.PymeshfixDecimatedExcitatoryStitchedMesh & key).fetch1("vertices","triangles")

        #Step 2: get the fully decimated mesh
        verts_original, faces_original = (pinky.Decimation35 & key).fetch1("vertices","triangles")
        print(f"Retrieving Meshes: {time.time() - start_time}")
        
        #do the boolean mesh difference to get the leftover mesh
        start_time = time.time()
        verts_diff, faces_diff = neuron_boolean_difference(verts_fixed,
                                                   faces_fixed,
                                                   verts_original,
                                                   faces_original,
                                                   distance_threshold = 5,
                                                   significance_threshold=90,
                                                   n_sample_points=3)
        print(f"Boolean Mesh Difference: {time.time() - start_time}")
        new_key = dict(key,
                      n_vertices = len(verts_diff),
                      n_triangles = len(faces_diff),
                      vertices = verts_diff,
                      triangles = faces_diff,
                      recovered_perc = len(faces_diff) / len(faces_fixed))
        
        #insert the key
        self.insert1(new_key,skip_duplicates=True)
        print(f"Total time for processing leftover mesh: {time.time() - global_time}")
        
        

     


# In[8]:


#schema = dj.schema("microns_pinky")
#(schema.jobs & "table_name='__excitatory_leftover_meshes'").delete()


# In[5]:


start = time.time()
ExcitatoryLeftoverMeshes.populate(reserve_jobs=True)
print(time.time() - start)

