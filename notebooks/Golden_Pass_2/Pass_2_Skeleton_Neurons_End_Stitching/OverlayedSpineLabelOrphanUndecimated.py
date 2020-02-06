
# coding: utf-8

# In[1]:


import datajoint as dj
import numpy as np
from scipy.spatial import KDTree
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


# In[3]:


pinky.OverlayedSpineLabelOrphan()


# In[4]:


@schema
class OverlayedSpineLabelOrphanUndecimated(dj.Computed):
    definition = """
    # Segment labels with Spine labels overlayed for undecimated excitatory meshes
    -> pinky.OverlayedSpineLabelOrphan
    ---
    vertices             : longblob                     
    triangles            : longblob  
    """
    
    def make(self, key):
        print("Working on Neuron " + str(key["segment_id"]))
        """
        Pseudocode:
        1) Get the undecimated labels
        2) Get the decimated mesh
        3) Get the undecimated mesh
        4) Do a KD tree to map the decimated vertices to the undecimated and give it the labels
        """

        search_key = key
        new_key = dict(segment_id = search_key["segment_id"],segmentation=search_key["segmentation"])
        
        start_time = time.time()
        dec_vert_labels = (pinky.OverlayedSpineLabelOrphan & search_key).fetch1("vertices")
        print(f"Retrieved decimated vertex labels = {time.time() - start_time}")
        
        #get the decimated mesh
        start_time = time.time()
        dec_mesh_table = pinky.Decimation35OrphanStitched & search_key
        dec_vertices, dec_triangles = dec_mesh_table.fetch1("vertices","triangles")
        
        #get the undecimated mesh
        undec_mesh_table = pinky.Mesh & new_key
        undec_vertices, undec_triangles = undec_mesh_table.fetch1("vertices","triangles")

        print(f"Retrieved decimated and undecimated meshes = {time.time() - start_time}")
        
        start_time = time.time()
        dec_KDTree = KDTree(dec_vertices)
        print(f"KDTree creation: {time.time() - start_time}")
        
        start_time = time.time()
        distances, nearest_nodes = dec_KDTree.query(undec_vertices)
        print(f"KDTree mapping: {time.time() - start_time}")
    
        start_time = time.time()
        #get the labels for the undecimated mesh
        undecimated_vert_labels = dec_vert_labels[nearest_nodes]
        
        #get the first vertex of every triangle
        traingles_first_verts = undecimated_vert_labels[undec_triangles[:,0]]
        
        traingle_labels = undecimated_vert_labels[traingles_first_verts]
        print(f"Generated Undecimated vert and triangle labels: {time.time() - start_time}")
        
        #make sure the number of vertices matches the number
        #of vertices in the undecimated mesh
        
        if len(traingle_labels) != len(undec_triangles):
            print(f"len(traingle_labels) = " + str(len(traingle_labels)))
            print(f"len(undec_triangles) = " + str(len(undec_triangles)))
            raise Exception("Number of traingle labels doesn't match the number of undecimated traingles")
        if len(undecimated_vert_labels) != len(undec_vertices):
            print(f"len(undecimated_vert_labels) = " + str(len(undecimated_vert_labels)))
            print(f"len(undec_vertices) = " + str(len(undec_vertices)))
            raise Exception("Number of vertex labels doesn't match the number of undecimated vertices") 
        
        key["vertices"] = undecimated_vert_labels
        key["triangles"] = traingle_labels
        self.insert1(key,skip_duplicates=True)


# In[8]:


#(schema.jobs & "table_name='__overlayed_spine_label_orphan_undecimated'").delete()


# In[ ]:


start = time.time()
OverlayedSpineLabelOrphanUndecimated.populate(reserve_jobs=True)
print(time.time() - start)

