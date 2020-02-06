#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from OPTIMIZED_whole_neuron_classifier_datajoint_adapted import extract_branches_whole_neuron
import datajoint as dj
import numpy as np
import datajoint as dj
import trimesh
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels


# In[ ]:


dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config["display.limit"] = 20
    
#schema = dj.schema('microns_ta3p100')
#ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')
schema = dj.schema("microns_pinky")
pinky = dj.create_virtual_module("pinky","microns_pinky")


# In[ ]:


#pinky.WholeAutoAnnotationsClusters3Optimized.drop()


# In[ ]:


pinky.PymeshfixDecimatedExcitatoryStitchedMesh()


# In[ ]:


@schema
class WholeAutoAnnotationsClusters3Optimized(dj.Computed):
    definition = """
    # creates the labels for the mesh table
    -> pinky.PymeshfixDecimatedExcitatoryStitchedMesh
    ---
    vertices   : longblob     # label data for the vertices
    triangles  : longblob     # label data for the faces
    apical_perc : decimal(3,2) #proportion of true apicals correctly classified
    basal_perc : decimal(3,2) #proportion of true basal correctly classified
    oblique_perc : decimal(3,2) #proportion of true oblique correctly classified
    soma_perc : decimal(3,2) #proportion of true soma correctly classified
    axon_soma_perc : decimal(3,2) #proportion of true axon somas correctly classified
    axon_dendrite_perc : decimal(3,2) #proportion of true axon dendrite correctly classified
    dendrite_perc : decimal(3,2) #proportion of true dendrite correctly classified
    error_perc : decimal(3,2) #proportion of true error correctly classified
    unlabelable_perc : decimal(3,2) #proportion of true unlabelable correctly classified
    cilia_perc : decimal(3,2) #proportion of true cilia correctly classified
    unknown_perc: decimal(3,2) #proportion of true unknown correctly classified
    confusion_matrix : longblob #the confusion matrix for the classifier
    """
    #key_source = pinky.PymeshfixDecimatedExcitatoryStitchedMesh & [dict(segment_id=k) for k in [648518346349478700,648518346349485870,648518346349473813,648518346349475510,648518346349473597]]
    
    def make(self, key):
        start_time = time.time()
        
        print("\n\n*****Starting Auto Labeling for " + str(key["segment_id"]) + "******")
        
        segment_id = key["segment_id"]

        #get the vertices and faces from datajoint
        # get the newly stitched mesh
        # get the original mesh

        
        verts,faces = (pinky.PymeshfixDecimatedExcitatoryStitchedMesh() & key).fetch1("vertices","triangles")

#         cgal_location = "/notebooks/Pass_2_Excitatory_4_Auto_Classifier_Whole_Neuron_Run_2/automatic_classifier_revised_efficiency/temp/"
#         cgal_file_sdf = str(key["segment_id"]) + "_fixed-cgal_3_0.20_sdf.csv"
#         cgal_file_seg = str(key["segment_id"]) + "_fixed-cgal_3_0.20.csv"


        #run the whole algorithm on the neuron to test
        verts_labels, faces_labels = extract_branches_whole_neuron(import_Off_Flag=False,segment_id=segment_id,vertices=verts,
                             triangles=faces,pymeshfix_Flag=False,
                             import_CGAL_Flag=False,
                             #import_CGAL_paths = [cgal_location + cgal_file_seg,
                             #                    cgal_location + cgal_file_sdf],
                             return_Only_Labels=True,
                             clusters=3,
                             smoothness=0.2)
        
        
        #calculate the confusion matrix of this classification ---------------------------
        
        actual_labels = (pinky.ProofreadLabel & key).fetch1("triangles")
        
        Label_key_data = pinky.LabelKey.fetch(as_dict=True)
        kept_data = Label_key_data[2:9] + Label_key_data[10:13]

        classes = [k["description"] for k in kept_data]
        cm_labels = [k["numeric"] for k in kept_data]
        classes.append("unknown")
        cm_labels.append(13)
        
        #compute normalized confusion matrix
        cm = confusion_matrix(actual_labels, faces_labels,labels=cm_labels)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        
        # --------------------------------------------------------------------------------
        
        #create a dictionary that has all of the classification accuracies and the confusion matrix
        apical_perc = cm[0,0]
        basal_perc = cm[1,1]
        oblique_perc = cm[2,2]
        soma_perc = cm[3,3]
        axon_soma_perc = cm[4,4]
        axon_dendrite_perc = cm[5,5]
        dendrite_perc = cm[6,6]
        error_perc = cm[7,7]
        unlabelable_perc = cm[8,8]
        cilia_perc = cm[9,9]
        unknown_perc = cm[10,10]
        
        
        
        new_key = dict(
                  apical_perc=apical_perc,
                  basal_perc=basal_perc,
                  oblique_perc=oblique_perc,
                  soma_perc=soma_perc,
                  axon_soma_perc=axon_soma_perc,
                  axon_dendrite_perc=axon_dendrite_perc,
                  dendrite_perc=dendrite_perc,
                  error_perc=error_perc,
                  unlabelable_perc=unlabelable_perc,
                  cilia_perc=cilia_perc,
                  unknown_perc=unknown_perc)
        
        for k,i in new_key.items():
            if np.isnan(i):
                new_key[k] = -1

        new_key.update(key)
        
        new_key["confusion_matrix"] = cm
        new_key["vertices"]=verts_labels,
        new_key["triangles"]=faces_labels
        
        #insert the key
        self.insert1(new_key,skip_duplicates=True)


        
        print(f"Finished Auto Labeling: {time.time() - start_time}")
        


# In[ ]:


#(schema.jobs & "table_name='__whole_auto_annotations_clusters3_optimized'").delete()


# In[ ]:


start = time.time()
WholeAutoAnnotationsClusters3Optimized.populate(reserve_jobs=True)
print(time.time() - start)


# In[ ]:


#(schema.jobs & "table_name='__whole_auto_annotations_clusters3'").delete()


# In[ ]:


#schema.jobs


# In[ ]:


WholeAutoAnnotationsClusters3Optimized()


# In[ ]:




