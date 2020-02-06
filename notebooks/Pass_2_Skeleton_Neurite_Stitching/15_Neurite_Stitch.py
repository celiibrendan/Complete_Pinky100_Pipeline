
# coding: utf-8

# In[1]:


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

from Stitcher_vp2_without_edges_check import stitch_neuron
import datajoint as dj
import numpy as np
import datajoint as dj
import trimesh
import time
import os


# In[2]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

schema = dj.schema('microns_pinky')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')


# In[3]:


@schema
class NeuriteStitched(dj.Computed):
    definition=""" 
    -> pinky.Mesh
    ---
    n_vertices           : bigint           # number of vertices in this mesh
    n_triangles          : bigint           # number of triangles in this mesh
    vertices             : longblob         # x,y,z coordinates of vertices
    triangles            : longblob         # triangles (triplets of vertices)
    n_pieces             : int              # number of unconnected mesh pieces outside the largest mesh piece
    largest_piece_perc   : decimal(6,5)     # number of faces percentage of largest mesh piece in respect to total mesh
    outside_perc         : decimal(6,5)     # number of faces percentage of mesh outside the biggest mesh piece
    n_stitched           : int              # number of mesh pieces stitched back to main mesh
    stitched_addon_perc  : decimal(6,5)     # number of faces percentage of pieces that were stitched back in respect to largest mesh piece
    n_unstitched         : int              # number of mesh pieces remaining unstitched back to main mesh        
    unstitched_perc     : decimal(6,5)     # number of faces percentage of pieces that were not in respect to largest mesh piece
    
    
    """
    
    
    key_source = pinky.Mesh() & pinky.Neurite() & pinky.CurrentSegmentation

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
        
        segment_id = my_dict["segment_id"]
        verts = my_dict["vertices"]
        faces = my_dict["triangles"]

        
        print("NOT Using a load file for meshes")

        [n_vertices,
         n_triangles,
         vertices,
         triangles,
         n_pieces,
         largest_piece_perc,
         outside_perc,
         n_stitched,
         stitched_addon_perc,
         n_unstitched,
         unstitched_perc] = stitch_neuron(segment_id=segment_id,
                                                      vertices=verts,
                                                      faces=faces,
                                                    import_from_off=False,
                                                     load_meshes_flag = False,
                                                 save_meshes_flag = False,
                                                    pymeshfix_flag = False
                                                   )
        
        
        #insert dummy dictionary into correspondence table
        insert_key = dict(key,
                          n_vertices=n_vertices,
                         n_triangles=n_triangles,
                         vertices=vertices,
                         triangles=triangles,
                         n_pieces=n_pieces,
                         largest_piece_perc=largest_piece_perc,
                         outside_perc=outside_perc,
                         n_stitched=n_stitched,
                         stitched_addon_perc=stitched_addon_perc,
                         n_unstitched=n_unstitched,
                         unstitched_perc=unstitched_perc)
                        
        
        self.insert1(insert_key,skip_duplicates=True)
        
    
                         
                                    


# In[4]:


start = time.time()
NeuriteStitched.populate(reserve_jobs=True)
print(time.time() - start)


# In[7]:


(schema.jobs & "table_name='__neurite_stitched'").delete()
#schema.jobs

