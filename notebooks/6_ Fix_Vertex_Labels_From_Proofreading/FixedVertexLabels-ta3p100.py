#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datajoint as dj
import numpy as np
import time
from tqdm import tqdm


# In[2]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')
schema = dj.schema('microns_ta3p100')


# In[4]:


# ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')
# vertices = ta3p100.CoarseLabel()


# ta3p100.ComponentLabel() #has the spine labels for vertices
# ta3p100.Compartment.Component() #has the indexes for each compartment


# neuron_comp_labels = (ta3p100.ComponentLabel & "segment_id=" + str(current_id)).fetch(as_dict=True)
        
# for comp_lab in neuron_comp_labels:
#     #get the corresponding indices
#     custom_key = dict(segment_id=comp_lab["segment_id"],
#                         compartment_type=comp_lab["compartment_type"],
#                         component_index=comp_lab["component_index"],
#                         segmentation=comp_lab["segmentation"],
#                         decimation_ratio=comp_lab["decimation_ratio"])
#     triangle_indices = (ta3p100.Compartment.Component & custom_key).fetch1("triangle_indices").tolist()
#     #go through the neuron and label those that are applying to spine 
#     triangle_labels = comp_lab["labeled_triangles"].tolist()
    
#     spine_head_label = 13
#     spine_label = 14
#     spin_neck_label = 15
#     spine_related_labels = [13,14,15]
            
            
            
#     for index,label in zip(triangle_indices,triangle_labels):
#         if int(label) in spine_related_labels:
#             faces_raw[index].material_index = int(label)


# In[ ]:


@schema
class CoarseLabel(dj.Computed):
    definition = """
    # Vertex labels for ta3p100.ProofreadLabel did not correctly match the triangle labels, so these are regenerated from the correct triangle labels.
    -> ta3p100.ProofreadLabel
    ---
    vertices  : longblob # Corrected vertex labels
    triangles : longblob # Same triangle labels as ta3p100.ProofreadLabel
    """
    
    key_source = ta3p100.ProofreadLabel & 'status="complete"'
    
    def make(self, key):
        start = time.time()
        
        print(key["segment_id"])
        labels = (ta3p100.ProofreadLabel & key).fetch1()
        corrected_vertex_labels = np.zeros(labels['vertices'].shape, np.uint8)
        
        mesh = (ta3p100.CleansedMesh & 'decimation_ratio=0.35' & dict(segment_id=key['segment_id'])).fetch1()
        mesh_triangles = mesh['triangles']
        
        vertex_label_dict = dict()
        for i, label in enumerate(labels['triangles']):
            triangle = mesh_triangles[i]
            for node in triangle:
                if node in vertex_label_dict:
                    if vertex_label_dict[node] < label:
                        vertex_label_dict[node] = label
                else:
                    vertex_label_dict[node] = label
                
        for node, label in vertex_label_dict.items():
            corrected_vertex_labels[node] = label
            
        self.insert1(dict(key,
                          vertices=corrected_vertex_labels,
                          triangles=labels['triangles']))
        
        print("Segment {} vertex labels regenerated in: {} seconds.".format(key['segment_id'], time.time() - start))


# In[ ]:


start = time.time()
CoarseLabel.populate(reserve_jobs=True)
print(time.time() - start)

