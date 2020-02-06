#!/usr/bin/env python
# coding: utf-8

# In[6]:


import datajoint as dj
import numpy as np
import datetime
import math
import os

#from cloudvolume import CloudVolume
#from collections import Counter
#from funconnect import ta3


# In[7]:


import cgal_Segmentation_Module as csm


# In[8]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 10


# user: celiib
# pass: newceliipass
# host: at-database.ad.bcm.edu
# schemas: microns_% and celiib_%


# In[10]:


schema = dj.schema('microns_ta3p100')
ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')


# In[11]:


#if temp folder doesn't exist then create it
if (os.path.isdir(os.getcwd() + "/temp")) == False:
    os.mkdir("temp")


# In[12]:


import os
import pathlib

##write the OFF file for the neuron
def write_Whole_Neuron_Off_file(neuron_ID,vertices=[], triangles=[]):
    #primary_key = dict(segmentation=1, segment_id=segment_id, decimation_ratio=0.35)
    #vertices, triangles = (mesh_Table_35 & primary_key).fetch1('vertices', 'triangles')
    
    num_vertices = (len(vertices))
    num_faces = len(triangles)
    
    #get the current file location
    file_loc = pathlib.Path.cwd() / "temp"
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
    
    return str(path_and_filename),str(filename)


# In[17]:


#(schema.jobs & 'table_name="__component_auto_segment_whole"').delete()


# In[18]:


################# USE THIS FOR THE AUTOMATED PARAMETER TESTING OF THE SEGMENT
import cgal_Segmentation_Module as csm
import csv
import decimal
import time
import os

@schema
class ComponentAutoSegmentWhole(dj.Computed):
    definition = """
    # creates the labels for the mesh table
    -> ta3p100.CleansedMesh35
    clusters     : tinyint unsigned  #what the clustering parameter was set to
    smoothness   : decimal(3,2)             #what the smoothness parameter was set to, number betwee 0 and 1
    ---
    n_triangles  : int unsigned # number of faces
    seg_group    : longblob     # group segmentation ID's for faces from automatic CGAL segmentation
    sdf          : longblob     #  width values for faces from from automatic CGAL segmentation
    median_sdf   : decimal(6,5) # the median width value for the sdf values
    mean_sdf     : decimal(6,5) #the mean width value for the sdf values
    third_q      : decimal(6,5) #the upper quartile for the mean width values
    ninety_perc  : decimal(6,5) #the 90th percentile for the mean width values
    time_updated : timestamp    # the time at which the segmentation was performed
   
    
   """
    
    key_source = ta3p100.CleansedMesh35 & "decimation_ratio=0.35" #& [dict(segment_id=comp) for comp in [50467565,58045989]]#,481423,579228,694582]]
    
    whole_neuron_dicts = dict()
    
    
    def make(self, key):
        
        from cgal_Segmentation_Module import cgal_segmentation
        #key passed to function is just dictionary with the following attributes
        """segmentation
        segment_id
        decimation_ratio
        """
        
        
        #clusters_default = 18
        #smoothness_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        #cluster_list = [2,3,4,5,6]
        
        smoothness_list = [0.2]
        cluster_list = [3]
        
        entire_neuron = (ta3p100.CleansedMesh35 & key).fetch1()
        neuron_ID = key["segment_id"]
        component_size = int(entire_neuron["n_triangles"])
        
        print("inside make function with " + str(neuron_ID))
        
        total_dict = list()
        
        for smoothness in smoothness_list:
            
            for clusters in cluster_list:
                start_time = time.time()
                
                #print(str(entire_neuron["segment_id"]) + " cluster:" + str(clusters) 
                #      + " smoothness:" + str(smoothness))

                #generate the off file for each component
                #what need to send them:
                """----From cleansed Mesh---
                vertices
                triangles
                ----From component table--
                n_vertex_indices
                n_triangle_indices
                vertex_indices
                triangle_indices"""

                if key['segment_id'] not in self.whole_neuron_dicts:
                    self.whole_neuron_dicts[key['segment_id']] = (ta3p100.CleansedMesh35 & 'decimation_ratio=0.35' & dict(segment_id=key['segment_id'])).fetch1()
                
                
                path_and_filename, off_file_name = write_Whole_Neuron_Off_file(neuron_ID,
                                            self.whole_neuron_dicts[key['segment_id']]["vertices"],
                                            self.whole_neuron_dicts[key['segment_id']]["triangles"])
                
                #print("About to start segmentation")
                
                #will have generated the component file by now so now need to run the segmentation
                csm.cgal_segmentation(path_and_filename,clusters,smoothness)

                #generate the name of the files
                smoothness_str = str(smoothness)
                if(len(smoothness_str)<4):
                    smoothness_str = smoothness_str + "0"
                
                
                
                cgal_file_name = path_and_filename + "-cgal_" + str(clusters) + "_"+str(smoothness_str)
                group_csv_cgal_file = cgal_file_name + ".csv"
                sdf_csv_file_name = cgal_file_name+"_sdf.csv"


                #check if file actually exists
                import os
                exists = os.path.isfile(group_csv_cgal_file)
                
                if( not exists):
                    print("Segmentation not created for " + str(off_file_name))
                    print("################## " + str(neuron_ID) + " ##################")
                    
                    #delete the off file if it exists:
                    #off_exists = os.path.isfile(path_and_filename)
                    print(path_and_filename + ".off")
                    if os.path.isfile(path_and_filename + ".off"):
                        os.remove(path_and_filename + ".off")
                else:


                    with open(group_csv_cgal_file) as f:
                        reader = csv.reader(f)
                        your_list = list(reader)
                    group_list = []
                    for item in your_list:
                        group_list.append(int(item[0]))

                    with open(sdf_csv_file_name) as f:
                        reader = csv.reader(f)
                        your_list = list(reader)
                    sdf_list = []
                    for item in your_list:
                        sdf_list.append(float(item[0]))

                    #print(group_list)
                    #print(sdf_list)

                    #now write them to the datajoint table  
                    #table columns for ComponentAutoSegmentation: segmentation, segment_id, decimation_ratio, compartment_type, component_index, seg_group, sdf
                    comp_dict = dict(key,
                                        clusters=clusters,
                                        smoothness=smoothness,
                                        n_triangles=component_size,
                                        seg_group=group_list,
                                        sdf=sdf_list,
                                        median_sdf=np.median(sdf_list),
                                        mean_sdf=np.mean(sdf_list),
                                        third_q=np.percentile(sdf_list, 75),
                                        ninety_perc=np.percentile(sdf_list, 90),
                                        time_updated=str(datetime.datetime.now())[0:19])
                    
                    total_dict.append(comp_dict)
                    self.insert1(comp_dict,skip_duplicates=True)  #--> only inserting one at a time

                    #then go and erase all of the files used: the sdf files, 
                    real_off_file_name = path_and_filename + ".off"

                    files_to_delete = [group_csv_cgal_file,sdf_csv_file_name,real_off_file_name]
                    for fl in files_to_delete:
                        if os.path.exists(fl):
                            os.remove(fl)
                        else:
                            print(fl + " file does not exist")

                    print("finished")
                    print("--- %s seconds ---" % (time.time() - start_time))
            
            #self.insert(total_dict,skip_duplicates=True)
            #print("inserted all the dictionaries")
                


# In[19]:


ComponentAutoSegmentWhole.populate(reserve_jobs=True)


# In[ ]:




