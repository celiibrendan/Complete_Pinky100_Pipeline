"""
Notebook takes Final_spine_extraction and adds the classification functionality of finding the spine heads/necks and generic spines

Other things to add:
1) Size multiplier
2) Add in an error classifier for merge errors with spines

Change the way that things are labels: 
1) Backbones currently labeled as -1,
Head -2
Neck -3
Spine -4
Error -5

"""

import numpy as np
import math
from collections import Counter
import sys
#import matplotlib.pyplot as plt
import networkx as nx
import time
import csv
from pathlib import Path
import os
import trimesh

#for cgal segmentation
import cgal_Segmentation_Module as csm

class ClassifyMesh(object):
    
    #generates the mapping of vertices to the faces that are touching it
    def generate_verts_to_face_dictionary(self):
        verts_to_Face = {}

        #initialize the lookup dictionary as empty lists
        faces_raw = self.mesh.faces
        verts_raw = self.mesh.vertices
        
        for i,pre_vertex in enumerate(verts_raw):
            verts_to_Face[i] = []
        

        for i,verts in enumerate(faces_raw):
            #add the index to the list for each of the vertices
            for vertex in verts:
                verts_to_Face[vertex].append(i)

        return verts_to_Face
    
    def __init__(self,mesh_file_location,file_name):
    #import the mesh

        full_path = str(Path(mesh_file_location) / Path(file_name))
        self.mesh = trimesh.load_mesh(full_path)
        self.verts_to_Face = self.generate_verts_to_face_dictionary()
        #get the vertices to faces lookup table

    def find_neighbors(self,current_label):
        """will return the number of neighbors that border the segment"""

        #iterate over each face with that label
        #   get the vertices of that face
        #   get all the faces that have that vertice associated with that
        #   get the labels of all of the neighbor faces, for each of these labels, add it to the neighbors 
        #list if it is not already there and doesn't match the label you are currently checking
        #   return the list 

        labels_list = self.labels_list
        verts_to_Face = self.verts_to_Face
        faces_raw = self.mesh.faces
        
        
        #get the indexes of all of the faces with that label that you want to find the neighbors for
        index_list = []
        for i,x in enumerate(labels_list):
            if x == current_label:
                index_list.append(i)

        verts_checked = []
        faces_checked = []
        neighbors_list = []
        neighbors_shared_vert = {}
        for index in index_list:
            
            #get the vertices associates with face
            vertices = faces_raw[index]

            #get the faces associated with the vertices of that specific face
            for vert in vertices:
                #will only check each vertex once
                if vert not in verts_checked:
                    verts_checked.append(vert)
                    faces_associated_vert = verts_to_Face[vert]
                    for fac in faces_associated_vert:
                        #make sure it is not a fellow face with the label who we are looking for the neighbors of
                        if (fac not in index_list):
                            #check to see if checked the the face already
                            if (fac not in faces_checked):
                                if(labels_list[fac] not in neighbors_list):
                                    #add the vertex to the count of shared vertices
                                    neighbors_shared_vert[labels_list[fac]] = 0 
                                    #only store the faces that are different
                                    neighbors_list.append(labels_list[fac])
                                    #faces_to_check.append(fac)
                                    #faces_to_check.insert(0, fac)
                                #increment the number of times we have seen that label face
                                neighbors_shared_vert[labels_list[fac]] = neighbors_shared_vert[labels_list[fac]] + 1
                                #now add the face to the checked list
                                faces_checked.append(fac)

        #have all of the faces to check


        number_of_faces = len(index_list)

       

        return neighbors_list,neighbors_shared_vert,number_of_faces


    
    def find_neighbors_optomized(self,current_label):
        
        
        col1_member = self.adjacency_labels_col1  == current_label
        col2_member = self.adjacency_labels_col2  == current_label
        
        logical_xor = np.logical_xor(col1_member,col2_member)

        total_array = np.concatenate([self.adjacency_labels_col1[logical_xor],
              self.adjacency_labels_col2[logical_xor]])
        
        neighbors_shared_vert = dict(Counter(total_array))
        del neighbors_shared_vert[current_label]
        
        neighbors_list = list(neighbors_shared_vert.keys())
        number_of_faces = self.labels_list_counter[current_label]

        
        return neighbors_list,neighbors_shared_vert,number_of_faces
    
    def smooth_backbone_vp4_optomized(self,backbone_width_threshold = 0.35,
                                      max_backbone_threshold = 400,
                                      backbone_threshold=300,
                                      shared_vert_threshold=25,
                                      shared_vert_threshold_new = 5,
                                      backbone_neighbor_min=10):
        #print("at beginning of smooth backbone vp4")
        
        faces_raw = self.mesh.faces
        verts_raw = self.mesh.vertices

        #generate the easy lookup table
        verts_to_Face = self.verts_to_Face
        
        #new optomized way of getting initial backbone list
        total_items = np.array(sorted(self.labels_list_counter.items()))
        keys = total_items[:,0]
        values = total_items[:,1]
        big_threshold = values >= max_backbone_threshold

        small_threshold = values > backbone_threshold 
        sdf_threshold = np.array(list(self.sdf_final_dict.values())) >= backbone_width_threshold
        total_list = np.logical_or(big_threshold,np.logical_and(small_threshold,sdf_threshold))
        backbone_labels = keys[total_list]
 
        list_flag = False
    
        if list_flag == True:
            to_remove = []
        else:
            to_remove = set()

        backbone_neighbors_dict = {}

        
        
        #finds all of the neighbors and how many shared vertices they have
        for bkbone in backbone_labels:
            #find_neighbors Description of Return List:
            #1) neighbors_list = labels of all bordering neighbors
            #2) neighbors_shared_vert = number of faces for each bordering neighbor
            #3) number_of_faces = total number of faces for current label
            
            #neighbors_list,neighbors_shared_vert,number_of_faces = self.find_neighbors_optomized(bkbone)
            neighbors_list,neighbors_shared_vert,number_of_faces = self.find_neighbors_optomized(bkbone)
            #neighbors_list,neighbors_shared_vert,number_of_faces = self.find_neighbors(self.labels_list,bkbone)
            #add the neighbor stats and count to the dictionary corresponding to that label
            backbone_neighbors_dict[bkbone] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
                number_of_faces=number_of_faces)
            
        
         #beginning smoothing round that removes ones from backbone list
        for i in range(0,5):
            print("smoothing round " + str(i+1))
            counter = 0
            #iterates through all the groups that were designated as backbones
            for bkbone in backbone_labels:
                if bkbone not in to_remove: #if not already designated to be removed

                    #just retrieve the neighbor stats and count of faces that are already stored in dict
                    neighbors_list = backbone_neighbors_dict[bkbone]["neighbors_list"]
                    neighbors_shared_vert = backbone_neighbors_dict[bkbone]["neighbors_shared_vert"]
                    number_of_faces = backbone_neighbors_dict[bkbone]["number_of_faces"]

                    #counts up the number of shared vertices with backbone neighbors

                    #FUTURE OPTOMIZATION
                    backbone_count_flag = False
                    neighbor_counter = 0 #TOTAL NUMBER OF BACKBONE NEIGHBORS
                    #spine_neighbor_counter = 0
                    total_backbone_shared_verts = 0 #TOTAL NUMBER OF FACES SHARED WITH BACKBONE
                    for n in neighbors_list:         
                        if (n in backbone_labels) and (n not in to_remove):
                            neighbor_counter += 1
                            total_backbone_shared_verts = total_backbone_shared_verts + neighbors_shared_vert[n] 
                    

                    #FUTURE OPTOMIZATION
                    #if meets requirement of shared verts then activates flag     
                    if (total_backbone_shared_verts > shared_vert_threshold):
                        backbone_count_flag = True

                    #if there are no neighbor's that are backbones or does not share enough backbone vertices --> remove from backbone list
                    if neighbor_counter <= 0 or backbone_count_flag == False:
                        if list_flag == True:
                            to_remove.append(bkbone)
                        else:
                            to_remove.add(bkbone)
                        counter += 1


            #if 1 or less non-backbones were converted to remove list then go ahead to the next step
            if counter <= 1:
                #print("counter caused the break")
                break

        #print("just broke out of the loop")
        """
        Status: 
        1) Started with a tentative list of backbones
        2) Removed some potential backbone lists
        """


        #now go through and make sure no unconnected backbone segments

        """Pseudo-code for filtering algorithm
        1) iterate through all of the backbone labels
        2) Go get the neighbors of the backbone
        3) Add all of the neighbors who are too part of the backbone to the backbones to check list
        4) While backbone neighbor counter is less than the threshold or until list to check is empty
        5) Pop the next neighbor off the list and add it to the neighbors check list
        6) Get the neighbors of this guy
        7) for each of neighbors that is also on the backbone BUT HASN'T BEEN CHECKED YET append them to the list to be check and update counter
        8) continue at beginning of loop
        -- once loop breaks
        9) if the counter is below the threshold:
            Add all of values in the neighbros already checked list to the new_to_remove
        10) Use the new_backbone_labels and new_to_remove to rewrite the labels_list

        """

        #gets the new backbones list without the ones removed
        #new_backbone_labels = [bkbone for bkbone in backbone_labels if bkbone not in to_remove] #OPTOMIZE
        new_backbone_labels = list(set(backbone_labels).difference(to_remove))
        
        list_flag = True
        if list_flag == True:
            new_to_remove = []
            skip_labels = []
        else:
            new_to_remove = set({})
            skip_labels = set({})
        

        for bkbonz in new_backbone_labels:
            if bkbonz not in skip_labels:
                #print("working on backbone = " + str(bkbonz))
                if list_flag == True:
                    checked_backbone_neighbors = []
                    backbone_neighbors_to_check = []
                else:
                    checked_backbone_neighbors = set()
                    backbone_neighbors_to_check = set()
                new_backbone_neighbor_counter = 0


#                 if bkbonz not in backbone_neighbors_dict.keys(): #should never enter this loop..... #OPTOMIZE
#                     neighbors_list,neighbors_shared_vert,number_of_faces = self.find_neighbors(labels_list,bkbonz)
#                     backbone_neighbors_dict[bkbonz] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
#                         number_of_faces=number_of_faces)
                #gets the stats of the neighbors and count of current label
                neighbors_list = backbone_neighbors_dict[bkbonz]["neighbors_list"]
                neighbors_shared_vert = backbone_neighbors_dict[bkbonz]["neighbors_shared_vert"]
                number_of_faces = backbone_neighbors_dict[bkbonz]["number_of_faces"]

                for bb in neighbors_list:
                    #counts as viable backbone neighbor if meets following conditions:
                    #1) In the new backbone list
                    #2) hasn't been checked yet
                    #3) not in the new ones to remove
                    #4) The number of neighbors shared by that label is greater than raw threshold shared_vert_threshold_new

                    #OPTOMIZE: don't need checked_backbone_neighbors
                    if (bb in new_backbone_labels) and (bb not in checked_backbone_neighbors) and (bb not in new_to_remove) and neighbors_shared_vert[bb] > shared_vert_threshold_new:
                        if list_flag == True:
                            backbone_neighbors_to_check.append(bb)
                        else:
                            backbone_neighbors_to_check.add(bb)
                        new_backbone_neighbor_counter += 1

                #at this point have :
                #1) total number of backbone neighbors: new_backbone_neighbor_counter
                #2) backbone neighbors in list: backbone_neighbors_to_check

                if list_flag == True:
                    checked_backbone_neighbors = [nb for nb in backbone_neighbors_to_check]
                else:
                    checked_backbone_neighbors = set([nb for nb in backbone_neighbors_to_check])


                #4) While backbone neighbor counter is less than the threshold or until list to check is empty

                #Iterates through all possible backbone neighbors unitl:
                # A) new_backbone_neighbor_counter is greater than set threshold of backbone_neighbor_min OR
                # B) no more backbone neighbors to check

                #Goal: counts the backbone chain with that label, so in hopes if not high enough then not backbone piece
                while new_backbone_neighbor_counter < backbone_neighbor_min and len(backbone_neighbors_to_check)>0:
                    #5) Pop the next neighbor off the list and add it to the neighbors check list
                    if list_flag == True:
                        current_backbone = backbone_neighbors_to_check.pop(0)
                    else:
                        current_backbone = backbone_neighbors_to_check.pop()
                        
                    if current_backbone not in checked_backbone_neighbors:
                        if list_flag == True:
                            checked_backbone_neighbors.append(current_backbone) #mark it as checked
                        else:
                            checked_backbone_neighbors.add(current_backbone)
                    
                    #gets the current neighbors and counts of one of the possible neighbor backbones
                    neighbors_list = backbone_neighbors_dict[current_backbone]["neighbors_list"]
                    neighbors_shared_vert = backbone_neighbors_dict[current_backbone]["neighbors_shared_vert"]
                    number_of_faces = backbone_neighbors_dict[current_backbone]["number_of_faces"]

                    #7) for each of neighbors that is also on the backbone BUT HASN'T BEEN CHECKED YET append them to the list to be check and update counter
                    for bb in neighbors_list:
                        if (bb in new_backbone_labels) and (bb not in checked_backbone_neighbors) and (bb not in new_to_remove) and neighbors_shared_vert[bb] > shared_vert_threshold_new:
                            if list_flag == True:
                                backbone_neighbors_to_check.append(bb)
                            else:
                                backbone_neighbors_to_check.add(bb)
                            new_backbone_neighbor_counter += 1

                #9) if the counter is below the threshold --> Add all of values in the neighbros already checked list to the new_to_remove
                if new_backbone_neighbor_counter < backbone_neighbor_min:
                    for bz in checked_backbone_neighbors:
                        if bz not in new_to_remove:
                            if list_flag == True:
                                new_to_remove.append(bz)
                            else:
                                new_to_remove.add(bz)
                            #print("removed " + str(checked_backbone_neighbors))
                else:
                    
                    if list_flag == True:
                        skip_labels = skip_labels + checked_backbone_neighbors
                    else:
                        skip_labels.update(checked_backbone_neighbors)
                    
     
        #go through and switch the label of hte 
        #may not want to relabel until the end in order to preserve the labels in case label a big one wrong

        for i in range(0,len(self.labels_list)):
            if self.labels_list[i] in new_backbone_labels and self.labels_list[i] not in new_to_remove:
                self.labels_list[i] = -1


        #print("Done backbone extraction")
        return
    
    
    #used for when not pulling from datajoint
    def get_cgal_data_and_label_local_optomized(self,ob_name,labels_file,sdf_file):
        
        #reads int the cgal labels for all of the faces
        triangles_labels = np.zeros(len(self.mesh.faces)).astype("int64")
        with open(labels_file) as csvfile:
            #print("inside labels file")

            for i,row in enumerate(csv.reader(csvfile)):
                triangles_labels[i] = int(row[0])

        """ OLD WAY OF GETTING BLENDER MESH OBJECT
        ob = bpy.context.object
        me = ob.data
        verts_raw = ob.data.vertices
        faces_raw = ob.data.polygons
        """
        
        #converts the cgal labels into a list that
        # starts at 0
        # progresses in order for all unique labels (so no numbers are skipped and don't have corresponding face)
        verts_raw = self.mesh.vertices
        faces_raw = self.mesh.faces
        #gets a list of the unique labels
        unique_segments = list(Counter(triangles_labels).keys())
        segmentation_length = len(unique_segments) 
        unique_index_dict = {unique_segments[x]:x for x in range(0,segmentation_length )}
        
        labels_list = np.zeros(len(triangles_labels)).astype("int64")
        for i,tri in enumerate(triangles_labels):

            #assembles the label list that represents all of the faces
            labels_list[i] = int(unique_index_dict[tri])
        
        #print("triangles_labels = " + str(Counter(triangles_labels)))
        #print("labels_list = " + str(Counter(labels_list)))
        

        #print("done with cgal_segmentation")

        #----------------------now return a dictionary of the sdf values like in the older function get_sdf_dictionary
        #get the sdf values and store in sdf_labels
        sdf_labels = np.zeros(len(labels_list)).astype("float")
        with open(sdf_file) as csvfile:

            for i,row in enumerate(csv.reader(csvfile)):
                sdf_labels[i] = float(row[0])

        
        sdf_temp_dict = {}
        for i in range(0,segmentation_length):
            sdf_temp_dict[i] = []
        
        #print("sdf_temp_dict = " + str(sdf_temp_dict))
        #print("sdf_labels = " + str(sdf_labels))
        #iterate through the labels_list
        for i,label in enumerate(labels_list):
            sdf_temp_dict[label].append(sdf_labels[i])
        #print(sdf_temp_dict)

        #now calculate the stats on the sdf values for each label
        sdf_final_dict = {}
        
        for dict_key,value in sdf_temp_dict.items():

            #just want to store the median
            sdf_final_dict[dict_key] = np.median(value)

        self.sdf_final_dict = sdf_final_dict
        self.labels_list = labels_list
        self.labels_list_counter = Counter(labels_list)
    
        adjacency_labels = self.labels_list[self.mesh.face_adjacency]
        
        self.adjacency_labels_col1, self.adjacency_labels_col2 = adjacency_labels.T
        
        return 

    def filter_Stubs_optomized(self,stub_threshold):
        
        #update the adjacency labels graph and counter
        adjacency_labels = self.labels_list[self.mesh.face_adjacency]
        self.labels_list_counter = Counter(self.labels_list)
        
        #feed into the networkx graph generator
        G = nx.Graph()
        G.add_edges_from(adjacency_labels)
        

        #removes the backbone node
        G.remove_node(-1)
        
        #get all of the sub graphs once backbone node is deleted
        sub_graphs = nx.connected_component_subgraphs(G)

        
        labels_to_remove = []
        for i, sg in enumerate(sub_graphs):
            node_sum = sum([self.labels_list_counter[n] for n in sg.nodes() if n != -1])
            if node_sum < stub_threshold:
                labels_to_remove = labels_to_remove + list(sg.nodes())

        print(f"removing {len(labels_to_remove)} labels with stub threshold {stub_threshold}")

        self.labels_list[np.isin(self.labels_list,labels_to_remove)] = -1

    def get_spine_classification(self,labels_file_location,file_name,clusters,smoothness,
                                    smooth_backbone_parameters,stub_threshold=50): 
        
        max_backbone_threshold = smooth_backbone_parameters.pop("max_backbone_threshold",200) #the absolute size if it is greater than this then labeled as a possible backbone
        backbone_threshold=smooth_backbone_parameters.pop("backbone_threshold",40) #if the label meets the width requirements, these are the size requirements as well in order to be considered possible backbone
        shared_vert_threshold=smooth_backbone_parameters.pop("shared_vert_threshold",10) #raw number of backbone verts that need to be shared in order for label to possibly be a backbone
        shared_vert_threshold_new = smooth_backbone_parameters.pop("shared_vert_threshAold_new",5)
        backbone_width_threshold = smooth_backbone_parameters.pop("backbone_width_threshold",0.10)  #the median sdf/width value the segment has to have in order to be considered a possible backbone 
        backbone_neighbor_min=smooth_backbone_parameters.pop("smooth_backbone_parameters",10) # number of backbones in chain in order for label to keep backbone status
       
        print("\nbackbone Parameters")
        print(f"max_backbone_threshold = {max_backbone_threshold}, \
                            backbone_threshold = {backbone_threshold}, \
                            shared_vert_threshold = {shared_vert_threshold}, \
                            shared_vert_threshold_new = {shared_vert_threshold_new} \
                             backbone_width_threshold = {backbone_width_threshold}, \
                             backbone_neighbor_min = {backbone_neighbor_min}")
        
        print("\nstub_threshold = " + str(stub_threshold))
        
        original_start_time = time.time()    
        start_time = time.time()

        faces_raw = self.mesh.faces        
        file_name = file_name[:-4]

        labels_file = str(Path(labels_file_location) / Path(file_name + "-cgal_" + str(clusters) + "_" + str(smoothness) + ".csv" ))  
        sdf_file = str(Path(labels_file_location) / Path(file_name + "-cgal_" + str(clusters) + "_" + str(smoothness) + "_sdf.csv" ))  
        
        #check to make sure thatcgal files were generated:
        #clean up the cgal files 
        #clean up the cgal files 
        for f in [labels_file,sdf_file]:
            if not os.path.isfile(f):
                print("CGAL segmentation files weren't generated")
                raise ValueError("CGAL segmentation files weren't generated")
                return "Failure"
        

        self.get_cgal_data_and_label_local_optomized(file_name,labels_file,sdf_file)
        
        
        
        if(self.sdf_final_dict == [] and labels_list == []):
            print("NO CGAL DATA FOR " + str(neuron_ID))

            return

        print("getting cgal data--- %s seconds ---" % (np.round(time.time() - start_time,5)))
        start_time = time.time()
        
        self.smooth_backbone_vp4_optomized(backbone_width_threshold,max_backbone_threshold = max_backbone_threshold,backbone_threshold=backbone_threshold,
                shared_vert_threshold=shared_vert_threshold,
                shared_vert_threshold_new = shared_vert_threshold_new,
                 backbone_neighbor_min=backbone_neighbor_min)

        #check and make sure that there exists a backbone, and if not then return that whole thing is error:
        if -1 not in self.labels_list:
            self.labels_list = np.ones(len(self.labels_list))*10
            return "No Backbone"
            
            
        
        print("smoothing backbone--- %s seconds ---" % (np.round(time.time() - start_time,5)))
        start_time = time.time()
        
        self.filter_Stubs_optomized(stub_threshold)
        print("---removing stubs: %s seconds ---" % (np.round(time.time() - start_time,5)))
        
        #clean up the cgal files 
        for f in [labels_file,sdf_file]:
            os.remove(f)
            
        
        #print("finished")
        print("Total spine extraction --- %s seconds ---" % (np.round(time.time() - original_start_time,5)))
        
        status = "Success"
        
        return status
    
    
    def extract_spines(self,labels_file_location,file_name,clusters,smoothness,
                                       split_up_spines=True,shaft_mesh=False,**kwargs):
        
        
        smooth_backbone_parameters = kwargs.pop('smooth_backbone_parameters', dict())
        stub_threshold = kwargs.pop('stub_threshold', 50)
        
        
        status = self.get_spine_classification(labels_file_location,file_name,clusters,
                                      smoothness,smooth_backbone_parameters,stub_threshold)
        
        if status != "Success":
            print(f"spine classification did not execute properly with status {status}")
            return None
        
        spine_indexes = np.where(np.array(self.labels_list) != -1)
        spine_meshes_whole = self.mesh.submesh(spine_indexes,append=True)
        
        #decides if passing back spines as one whole mesh or seperate meshes
        if split_up_spines==True:
            individual_spines = []
            temp_spines = spine_meshes_whole.split(only_watertight=False)
            for spine in temp_spines:
                if len(spine.faces) >= stub_threshold:
                    individual_spines.append(spine)
        else:
            individual_spines = spine_meshes_whole
        
        #will also pass back the shaft of the mesh with the extracted spines
        if shaft_mesh==False:
            return individual_spines
        else:
            shaft_indexes = np.where(np.array(self.labels_list) == -1) 
            shaft_mesh_whole = self.mesh.submesh(shaft_indexes,append=True)
            return individual_spines,shaft_mesh_whole
            
        
        
        
        
        #divide into disconnected meshes and return this array
        return individual_spines
    
    ###------------------- Part that deal with classifying the parts of the spine --- ###
    """
    1) Things that need to be made sure that are updated
    self.adjacency_labels_col1 
    self.adjacency_labels_col2  
    
    self.labels_list
    self.labels_list_counter
    
    
    
    
    """
    def update_label_list_dependencies(self):
        
        self.labels_list_counter = Counter(self.labels_list)
    
        adjacency_labels = self.labels_list[self.mesh.face_adjacency]
        
        self.adjacency_labels_col1, self.adjacency_labels_col2 = adjacency_labels.T
        
    
    def get_split_heads_vp2(self,current_label,current_index, path,connections,shared_vertices,mesh_number,sdf_final_dict,absolute_head_threshold):
        final_split_heads = [current_label]

        split_head_threshold = 0.35
        #underneath_threshold = 0.20

        #the only solid number threshold
        split_head_absolute_threshold = 8

        heads_to_check = True
        while heads_to_check:
            #1) go to the next label below it
            if(current_index < (len(path)-1)):
                next_index = current_index + 1
                next_label = path[next_index]

            if(next_label == -1):
                #no_more_split_head_Flag = True
                break

            #ask if this next satisfies  1) enough shared verts?  2) SDF head possible?
            verts_sharing_index = connections[current_label].index(next_label)
            verts_sharing = shared_vertices[current_label][verts_sharing_index]

            #print("split share for faces " + str(current_label) + " " +str(next_label) + "="+str(verts_sharing/mesh_number[current_label]))
            sdf_guess = self.sdf_likely_category(next_label,next_index,path,True,self.sdf_final_dict,connections,mesh_number,absolute_head_threshold)
            if verts_sharing/mesh_number[current_label] > split_head_threshold and  sdf_guess == "head" and mesh_number[next_label] > split_head_absolute_threshold:
                #add next label to the list
                final_split_heads.append(next_label)
                current_index = next_index
                current_label = next_label

            else:
                heads_to_check = False

        return final_split_heads      



    def sdf_likely_category(self,current_label,current_index,path,head_flag,sdf_final_dict,connections,mesh_number,absolute_head_threshold):
        """
        Returns the most likely category of a certain label: as neck or head label
        
        
        """
        
        #width thresholding constants
        width_thresholds = {"base":0.04, "item_top_threshold":1.5} 
        #if size is smaller than the max threshold for a head then return neck
        if mesh_number[current_label] < absolute_head_threshold:
            return "neck"

        #get the mean, max, and median
        median_width = sdf_final_dict[current_label]

        
        neck_near_base_threshold = 0.16
        close_neck_call_threshold = 0.09

        #common characteristics of neck:
        #1) median width Less than neck_cuttoff_threshold
        #2) if larger item on top and that item is not a head
        #3) if larger item on top with more then 50% heads but less width
        #4) connected to backbone



        #1) median width Less than neck_cuttoff_threshold, return as neck
        if median_width < width_thresholds["base"]:
            return "neck"

        #2) if larger item on top and that item is not a head or #3) if larger item on top with more then 50% heads but less width
        #width_on_top = []
        #face_number_on_top = []

        for i in range(0,current_index):
            face_number_on_top = mesh_number[path[i]]
            width_on_top = sdf_final_dict[path[i]]

            if face_number_on_top > mesh_number[current_label]:
                if head_flag == False:
                    return "neck"

                if median_width > width_thresholds["item_top_threshold"]*width_on_top:
                    return "neck"

        #4) connected to backbone
        if -1 in connections[current_label]:
            return "neck"


        ######check for head based on if there is significantly smaller neck underneath it (because can be very close to 0.04 cuttoff sometimes

        #get the mean, median and max

        #will return head or neck
        return "head"      

    
    def find_endpoints(self,G,mesh_number):
        #will first calculate all the shortest paths for each of the nodes
        
        
        #removes the backbone from the node list but not remove it from the nodes
        node_list = list(G.nodes)
        if(-1 in node_list):
            node_list.remove(-1)
        else:
            return [],[] 

        #gets the shortest path from every node to the backbone
        shortest_paths = {}
        for node in node_list:
            shortest_paths[node] = [k for k in nx.all_shortest_paths(G,node,-1)]

        #identify the nodes that are not a subset of other nodes --> these called endpoints
        endpoints = []
        
        for node in node_list:
            other_nodes = [k for k in node_list if k != node ]
            not_unique = 0
            for path in shortest_paths[node]:
                not_unique_Flag = False
                for o_node in other_nodes:
                    for o_shortest_path in shortest_paths[o_node]:
                        if set(path) <= set(o_shortest_path): #only if shortest path is subset of other path
                            not_unique_Flag = True

                if not_unique_Flag == True: #counts the number of the unique paths that are not unique
                    not_unique = not_unique + 1

            """#decide if unique endpoint, because not_unique measures the number of paths out of all 
            of the shortest paths that are not unique, so if have one that is still unique then
            not_unique will be less than number of total shortest paths, which means it is an endpoint
            """
            if not_unique < len(shortest_paths[node]):   # this means there is a unique path

                #if not_unique != 0:
                    #print(node + "-some unique and some non-unique paths for endpoint")
                endpoints.append(node)

        #gets the number of most possible faces between the endpoint and the bakcbone
        #out of all of the shortest paths for each endpoint
        ##Result: have a list that for each endpoint has the shortest path faces length
        longest_paths_list = []
        for end_node in endpoints:
            longest_path = 0
            for path in shortest_paths[end_node]:
                path_length = 0
                for point in path:
                    path_length = path_length + mesh_number[point]
                if path_length > longest_path:
                    longest_path = path_length

            longest_paths_list.append((end_node,longest_path))

        
        
        #sorts the list so that the node with the greatest path length is first
        longest_paths_list.sort(key=lambda pair: pair[1], reverse=True)
        
        ranked_endpoints = [x for x,i in longest_paths_list]
        endpoint_paths_lengths = [i for x,i in longest_paths_list]

        #creates dictionary that maps the endpoints to the shortest path face number
        enpoint_path_list = {}
        for endpt in ranked_endpoints:
            enpoint_path_list[endpt] = shortest_paths[endpt]


        #ranked_endpoints, longest_paths_list = (list(t) for t in zip(*sorted(zip(endpoints, longest_paths_list))))

        #returns ranked list of endpoints by greatest number of faces along shortest path
        #and the dictionary that mapes endpoints to all of the shortest paths
        return ranked_endpoints, enpoint_path_list 
    
    def classify_spine_vp2(self,connections,shared_vertices,mesh_number,
                           absolute_head_threshold,
                            stub_threshold,
                            path_threshold):


        #set this as variable so don't get errors when porting over from blender to trimesh
        sdf_final_dict = self.sdf_final_dict
        
        #make a new dictionary to hold the final labels of the spine for that group
        end_labels = {k:"none" for k in mesh_number.keys()}


        #only one segment so label it as a spine
        if len(connections.keys()) <= 1:
            end_labels[list(connections.keys())[0]] = "spine_one_seg"


        total_mesh_faces_outer = sum([k for i,k in mesh_number.items()])
        #print("total_mesh_faces = " + str( total_mesh_faces_outer))

        #create the graph from the connections
        G=nx.Graph(connections)

        #find the endpoints of the graph and all of the corresponding shortest paths
        endpoint_labels,shortest_paths = self.find_endpoints(G,mesh_number)

        if endpoint_labels == []:
            for jk in end_labels.keys():
                end_labels[jk] = "backbone"
                return end_labels

        #print("endpoint_labels = "+str(endpoint_labels))
        #print("shortest_paths = "+str(shortest_paths))

        #make a new dictionary to hold the final labels  of the spine segmentations
        end_labels = {k:"none" for k in mesh_number.keys()}
        end_labels[-1] = "backbone"


        #iterates through all of the endpoints
        for endpoint in endpoint_labels:
            
            #get the shortest path lists
            endpoint_short_paths = shortest_paths[endpoint]
            #iterates through all of the shortest paths
            for path in endpoint_short_paths:
                path.remove(-1)
                #gets total number of faces along the path
                path_total_mesh_faces = sum([k for i,k in mesh_number.items() if i in path])
                
                travel_index = 0
                head_found = False
                label_everything_above_as_head = False
                #travels up the path until find a head or reached the end of the path
                while (head_found == False ) and travel_index < len(path):
                    current_face = path[travel_index]
                    sdf_guess = self.sdf_likely_category(current_face,travel_index,path,False,self.sdf_final_dict,connections,mesh_number,absolute_head_threshold)
                    if  sdf_guess != "head" or mesh_number[current_face] < absolute_head_threshold:
                        #then not of any significance BUT ONLY REASSIGN IF NOT HAVE ASSIGNMENT***
                        if end_labels[current_face] == "none":
                            end_labels[current_face] = "no_significance"
                        travel_index = travel_index + 1
                    else:
                        #end_labels[current_face] = "head_reg" WAIT TO ASSIGN TILL LATER
                        if "neck" != end_labels[current_face][0:4] and "spine" !=  end_labels[current_face][0:5] :   #if not already labeled as neck or spine
                            head_found = True
                            label_everything_above_as_head = True
                        else:
                            travel_index = travel_index + 1


                #print("end of first while loop, travel_index = "+ str(travel_index) + " head_found = "+ str(head_found))
                ############Added new threshold that makes it so path length can't be really small
                if travel_index < len(path):
                    travel_face = path[travel_index]
                else:
                    travel_face = path[travel_index-1]
                    travel_index = travel_index-1
                
                
                if (path[travel_index] == -1) or (-1 in connections[path[travel_index]]):
                    head_found = False
                    label_everything_above_as_head = True

                if path_total_mesh_faces<path_threshold:
                    head_found = False
                    label_everything_above_as_head = True


                ####do the head splitting####
                #see if there are any labels that border it that also share a high percentage of faces
                if head_found == True:
                    ##will return the names of the faces that have unusually high verts sharing
                    split_head_labels = self.get_split_heads_vp2(path[travel_index],travel_index,path,connections,shared_vertices,mesh_number,sdf_final_dict,absolute_head_threshold)
                    #print("split_head_labels = " + str(split_head_labels))

                    #if two or more split heads
                    if len(split_head_labels) >= 2:
                        #print("adding the split head labels")
                        for split_label in split_head_labels:
                            #######may need to add in CHECK FOR ALREADY LABELED
                            if ("head" == end_labels[split_label][0:4] or end_labels[split_label] == "none"):
                                end_labels[split_label] = "head_split"

                        label_everything_above_as_head = True


                ###if no head was found
                if head_found == False:
                    #print("no head found so labeling as neck")
                    #######WILL NOT OVERWRITE UNLESS LABELED AS NO SIGNIFICANCE
                    for i in path: 

                        if end_labels[i] == "no_significance" or end_labels[i] == "none" or end_labels[i][0:4] == "head":
                            end_labels[i] = "neck_no_head_on_path_head_false"

                    label_everything_above_as_head = False



                #print("label_everything_above_as_head = " + str(label_everything_above_as_head))
                #need to label any of those above it in the chain labeled as insignificant to heads
                if label_everything_above_as_head == True and head_found == True:
                    if end_labels[travel_face] == "none":
                        #print("labeled as head reg")
                        end_labels[travel_face] = "head_reg"
                    #else:               ########don't need this because don't want to overwrite already written spine neck
                        #if "head" not in end_labels[travel_index]:
                            #end_labels[travel_index] = "spine_head_disagree"


                    #will label everything above it as a head and then everything below it as neck
                    #####need to account for special case where not overwrite the head_split####
                    if "head" == end_labels[travel_face][0:4]:
                        #print('labeling all no_significance above as head hats')
                        for i in range(0,travel_index):
                            current_label = path[i]
                            if end_labels[current_label] == "no_significance":
                                end_labels[current_label] = "head_hat"
                            else:
                                if "head" != end_labels[current_label][0:4]:
                                    end_labels[current_label] = "spine_head_disagree_above_head"
                        #print('labeling all below head as necks')
                        for i in range(travel_index+1,len(path)):
                            current_label = path[i]
                            if current_label not in split_head_labels and end_labels[current_label] != "head_split":
                                end_labels[current_label] = "neck_under_head"
                    else: ###not sure when this will be activated but maybe?
                        #print("head not present so labeling everything above as neck_hat")
                        for i in range(0,travel_index):
                            current_label = path[i]
                            #####need to account for special case where not overwrite the head_split####
                            if end_labels[current_label] == "no_significance":
                                end_labels[current_label] == "neck_hats_no_head"

                #print("at end of one cycle of big loop")
                #print("end_labels = " + str(end_labels))

                #what about a head being accidentally written under another head? 
                #####you should not write a head to a spine that has already been labeled as under a head
                #####you should overwrite all labels under a head as spine_under_head

        #print("outside of big loop")
        #print("end_labels = " + str(end_labels))

        #if no heads present at all label as spines
        spine_flag_no_head = False

        for face,label in end_labels.items():
            if "head" == label[0:4]:
                spine_flag_no_head = True

        if spine_flag_no_head == False:
            #print("no face detected in all of spine")
            for label_name in end_labels.keys():
                end_labels[label_name] = "spine_no_head_at_all"


        ###### TO DO: can put in a piece of logic that seekss and labels the ones we know are necks for sure based on width


        #once done all of the paths go through and label things as stubs
        if total_mesh_faces_outer < stub_threshold:
            #print("stub threshold triggered")
            for label_name in end_labels.keys():
                if "head" == end_labels[label_name][0:4]:
                    end_labels[label_name] = "stub_head"

                elif "neck" == end_labels[label_name][0:4]:
                    end_labels[label_name] = "stub_neck"
                else:
                    end_labels[label_name] = "stub_spine"



        end_labels[-1] = "backbone"

        ###To Do: replace where look only in 1st four indexes
        return end_labels



    def export_connection(self,label_name):

        #print("hello from export_connection with label_name = " + str(label_name) )
        #find all the neighbors of the label

        
        total_labels_list = []
        faces_checked = []
        faces_to_check = [label_name]

        still_checking_faces = True

        connections = {}
        shared_vertices = {}
        mesh_number = {}

        #print("about to start checking faces")

        #will iterate through all of the labels with the label name until find all of the neighbors (until hitting the backbone) of the label
        while still_checking_faces:
            #will exit if no more faces to check
            if not faces_to_check:
                still_checking_faces = False
                break

            for facey in faces_to_check:
                if facey != -1:
                    neighbors_list,neighbors_shared_vert,number_of_faces = self.find_neighbors(facey)



                    #reduce the shared vertices with a face and the backbone to 0 so doesn't mess up the shared vertices percentage
                    pairs = list(neighbors_shared_vert.items())
                    pre_connections = [k for k,i in pairs]
                    pre_shared_vertices = [i for k,i in pairs]




                    if (-1 in pre_connections):
                        back_index = pre_connections.index(-1)
                        pre_shared_vertices[back_index] = 0


                    connections[facey] = pre_connections
                    shared_vertices[facey] = pre_shared_vertices
                    mesh_number[facey] = number_of_faces


                    for neighbors in neighbors_list:
                        if (neighbors != -1) and (neighbors not in faces_to_check) and (neighbors not in faces_checked):
                            faces_to_check.append(neighbors)

                    faces_to_check.remove(facey)
                    faces_checked.append(facey)

            #append the backbone to the graph structure
            mesh_number[-1] = 0

        return connections,shared_vertices,mesh_number

    def relabel_segments(self,labels_list,current_label,new_label):

        for i,x in enumerate(labels_list):
            if x == current_label:
                labels_list[i] = new_label

        return labels_list

    def automatic_spine_classification_vp3(self,
                                          absolute_head_threshold = 30,
                                           stub_threshold = 40,
                                           path_threshold = 40
                                          
                                          ):

        #process of labeling
        """1) Get a list of all of the labels
        2) Iterate through the labels and for each:
            a. Get the connections, verts_shared and mesh_sizes for all labels connected to said label 
            b. Run the automatic spine classification to get the categories for each label
            c. Create a new list that stores the categories for each label processed
            d. repeat until all labels have been processed
        3) Delete all the old colors and then setup the global colors with the regular labels
        4) Change the material index for all labels based on the categorical classification"""

        """
        Pseudo code: 
        1) Takes in a list of the faces values containing either the CGAL segmentation group or
            the word "backbone" if it was determined to be that by the smoothing_backbone algorithm
        2) Iterates through each of the CGAL segmentation groups left:
            a. gets the connections and shared vertices data for that spine cluster
            b. Sends the above to a spine classfier to get the spine_head,spine_neck or spine classification
            c. rewrites the copy of the labels list for all of those CGAL segmentation groups with head/neck/spine
            d. Marks all those CGAL segmentation labels as already being processed
            e. increments the spine head/neck counter
        3) Creates an entire new copy of the label list and relabels the head/neck/spine or backbone index
        4) Translates the face labels to the vertices labels
        5) Returns both lists

        """
        
        print("inside auto_spine_classification")
        
        #update the list data based on the new backbone labels being placed
        self.update_label_list_dependencies()

        #but now they have the backbone labels as the label if it was changed in the smoothing backbone
        final_spine_labels = self.labels_list.copy()

        processed_labels = []

        myCounter = Counter(self.labels_list)
        complete_labels =  [label for label,times in myCounter.items()] #OPTOMIZE BY USING KEYS

        head_counter = 0
        spine_counter = 0
        neck_counter = 0
        stub_counter = 0
        print("About to iterate through labels")
        start_time = time.time()
        for i in range(0,len(complete_labels)):
            if complete_labels[i] != -1 and complete_labels[i] not in processed_labels:
                #print(f"working on label {complete_labels[i]}")
                #get the conenections, shared vertices and mesh sizes for the whole spine segment in which label is connected to
                connections,shared_vertices,mesh_number = self.export_connection(complete_labels[i])
                
        
                #send that graph data to the spine classifier to get labels for that group
                #final_labels is dictionary matching the segmentation number to the english label
                final_labels = self.classify_spine_vp2(connections,shared_vertices,mesh_number,
                           absolute_head_threshold,
                            stub_threshold,
                            path_threshold)

                #print(f"final_labels for {complete_labels[i]} =  {final_labels}")
                head_Flag = False
                spine_Flag = False
                stub_Flag = False
                neck_Flag = False
                #relabel the list accordingly
                ############could speed this up where they return the number of types of labels instead of having to search for them############
                #print("about to find number of heads/spines/stubs/necks PLUS RELABEL AND append them to list")
                for key,value in final_labels.items():
                    
                    if value[0:4] == "head":
                        new_label = -2
                        head_Flag = True
                    elif value[0:4] == "spin":
                        new_label = -4
                        spine_Flag = True
                    elif value[0:4] == "stub":
                        new_label = -5
                        stub_Flag = True
                    elif value[0:4] == "neck":
                        new_label = -3
                        neck_Flag = True
                    else:
                        new_label = -1
#                         if value == "backbone":
#                             new_label = -1

                    self.relabel_segments(final_spine_labels,key,new_label)
                    #add them to the list of processed labels
                    processed_labels.append(key)
                    #print("str(-1 in final_spine_labels) = " + str(-1 in final_spine_labels))
                #print("about to find number of heads/spines/stubs/necks PLUS RELABEL AND append them to list")

                if head_Flag == True:
                    head_counter += 1
                if spine_Flag == True:
                    spine_counter += 1
                if stub_Flag == True:
                    stub_counter += 1
                if neck_Flag == True:
                    neck_counter += 1

        print("str(-1 in final_spine_labels) = " + str(-1 in final_spine_labels))
        print(f"done classifying labels: {time.time() - start_time}")

        #current mapping of labels:
        """
        -1 --> backbone
        -2 --> head
        -3 --> neck
        -4 --> spine
        -5 --> stub
        
        """
        datajoint_Flag = False
        
        if datajoint_Flag == True:
            #get the indexes for the labeling from the datajoint table
            label_data = ta3p100.LabelKey().fetch("numeric","description")
            #print(label_data)

            label_names = label_data[1].tolist()
            label_indexes = label_data[0].tolist()
            #print(label_names)

            spine_head_index = label_indexes[label_names.index("Spine Head")]
            spine_neck_index = label_indexes[label_names.index("Spine Neck")]
            spine_reg_index = label_indexes[label_names.index("Spine")]
        else:
            spine_head_index = 13
            spine_neck_index = 15
            spine_reg_index = 14


        final_faces_labels_list = np.zeros(len(self.mesh.faces))
        final_verts_labels_list = np.zeros(len(self.mesh.vertices))
        print("Starting Relabeling final faces and vertices")
        start_time = time.time()

        print("Counter(final_spine_labels) = " + str(Counter(final_spine_labels)))
        #assign the labels to the correct faces
        for i,fi in enumerate(final_spine_labels):
            if fi == -2:
                #fac.material_index = 2
                final_faces_labels_list[i] = spine_head_index
            elif fi == -3:
                #fac.material_index = 3
                final_faces_labels_list[i] = spine_neck_index
            elif fi == -4:
                #fac.material_index = 4
                final_faces_labels_list[i] = spine_reg_index
            else:
                #fac.material_index = 0
                final_faces_labels_list[i] = 0

            #assign the vertices an index
            for vert in self.mesh.faces[i]:
                if final_verts_labels_list[vert] == 0:
                    final_verts_labels_list[vert] = final_faces_labels_list[i]


        print(f"Done relabeling final faces: {time.time() - start_time}")
        return head_counter,neck_counter, spine_counter, stub_counter, final_verts_labels_list, final_faces_labels_list

    
    
def complete_spine_extraction(mesh_file_location,
                              file_name,
                              **kwargs):
    
    """
    Extracts the spine meshes from a given dendritic mesh and returns either 
    just the spine meshes or the spine meshes and the dendritic shaft with the spines removed. 
  

    Parameters: 
    mesh_file_location (str): location of the dendritic mesh on computer
    file_name (str): file name of dendritic mesh on computer
    
    Optional Parameters:
    ---configuring cgal segmentation ---
    
    clusters (int) : number of clusters to use for CGAL surface mesh segmentation (default = 12)
    smoothness (int) : smoothness parameter use for CGAL surface mesh segmentation (default = 0.04)
    
    ---configuring output---
    
    split_up_spines (bool): if True will return array of trimesh objects representing each spine
                         if False will return all spines as one mesh (default = True)
    shaft_mesh (bool) : if True then returns the shaft mesh with the spines stripped out as well (default=False)
    
    --- configuring spine extraction ---
    stub_threshold (int) : number of faces (size) that a spine mesh must include in order to be considered spine (default=50)
                            
    smooth_backbone_parameters (dict) : dict containing parameters for backbone extraction after cgal segmentation
        ---- dictionary can contain the following parameters: ---
        max_backbone_threshold (int) :the absolute size if it is greater than this then labeled as a possible backbone
        (default = 200)
        backbone_threshold (int) :if the label meets the width requirements, these are the size requirements as well in order to be considered possible backbone
        (default = 40)
        shared_vert_threshold (int): raw number of backbone verts that need to be shared in order for label to possibly be a backbone
        (default = 10)
        shared_vert_threshold_new (int): raw number of backbone verts that need to be shared in order for label to possibly be a backbone in phase 2
        (default = 5)
        backbone_width_threshold (float) :#the median sdf/width value the segment has to have in order to be considered a possible backbone 
        (default = 0.1)
        backbone_neighbor_min (int): number of backbones in chain in order for label to keep backbone status
        (default = 10)
    -------------------------------------
  
    Returns: 
    1 or 2 trimesh.mesh objects/lists of objects depending on settings
    
    if split_up_spines == True (default)
        list of trimesh.Mesh: each element in list is trimesh.mesh object representing a single spine
    else:
        trimesh.Mesh: trimesh.mesh object representing all spines
    
    if shaft_mesh == False (default):
         No mesh object 
    else:
        Trimesh.mesh object: representing shaft mesh with all of the spines filtered away
        
    
    Examples:
    #returns the spine meshes as one entire mesh
    
    list_of_spine_meshes = complete_spine_extraction(file_location,file_name)
    list_of_spine_meshes,shaft_mesh = complete_spine_extraction(file_location,file_name,shaft_mesh=True)
    merged_spine_meshes = complete_spine_extraction(file_location,file_name,split_up_spines=False)
    merged_spine_meshes,shaft_mesh = complete_spine_extraction(file_location,file_name,split_up_spines=False,shaft_mesh=True)
    
    
    """

    
    clusters = kwargs.pop('clusters', 12)
    smoothness = kwargs.pop('smoothness', 0.04)
    smooth_backbone_parameters = kwargs.pop('smooth_backbone_parameters', dict())
    stub_threshold = kwargs.pop('stub_threshold', 50)
    split_up_spines = kwargs.pop('split_up_spines', True)
    shaft_mesh = kwargs.pop('shaft_mesh', False)
    
    
    #making sure there is no more keyword arguments left that you weren't expecting
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    

    #check to see if file exists and if it is an off file
    if file_name[-3:] != "off":
        raise TypeError("input file must be a .off ")
        return None
    if not os.path.isfile(str(Path(mesh_file_location) / Path(file_name))):
        raise TypeError(str(Path(mesh_file_location) / Path(file_name)) + " cannot be found")
        return None
    
    total_time = time.time()
    print(f"Starting spine extraction for {file_name} with clusters={clusters} and smoothness={smoothness}")
    start_time = time.time()
    myClassifier = ClassifyMesh(mesh_file_location,file_name)
    print(f"Step 1: Trimesh mesh build total time ---- {np.round(time.time() - start_time,5)} seconds")
    #make sure a cgal folder is created, and if not make one
    
    
#     if (os.path.isdir(str(Path(os.getcwd()) / Path("cgal")))) == False:
#         os.chdir(str(Path(os.getcwd()) / Path("cgal")))
#         os.mkdir("cgal")
    
    start_time = time.time()
    print("\nStarting CGAL segmentation")
    full_file_path = str(Path(mesh_file_location) / Path(file_name))[:-4]
    csm.cgal_segmentation(full_file_path,clusters,smoothness)
    print(f"Step 2: CGAL segmentation total time ---- {np.round(time.time() - start_time,5)} seconds")
    
    
    #do the cgal processing
    #labels_file_location = str(Path(os.getcwd()) / Path("cgal"))
    start_time = time.time()
    print("\nStarting Spine Extraction")
    individual_spines = myClassifier.extract_spines(mesh_file_location,file_name,
                                                    clusters,
                                                    smoothness,
                                                    split_up_spines,
                                                    shaft_mesh,
                                                   smooth_backbone_parameters=smooth_backbone_parameters,
                                                   stub_threshold=stub_threshold
                                                   )
    print(f"Step 3: Spine extraction total time ---- {np.round(time.time() - start_time,5)} seconds")
    
    if individual_spines == None:
        print("no spines were extracted so returning")
        return None
    
    #function call that will retrieve the spine head and neck labels
    start_time = time.time()
    print("\nStep 3: Starting Spine Classification")
    head_counter,neck_counter, spine_counter, stub_counter,final_verts_labels_list, final_faces_labels_list = myClassifier.automatic_spine_classification_vp3(
                                            absolute_head_threshold = 30,
                                           stub_threshold = 40,
                                           path_threshold = 40)
    print(f"\nStep 3: Finshed Spine Classification: {time.time()-start_time}")
    

    print("head_counter = " + str(head_counter))
    print("neck_counter = " + str(neck_counter))
    print("spine_counter = " + str(spine_counter))
    print("stub_counter = " + str(stub_counter))

    
    
    print(f"Total time ---- {np.round(time.time() - total_time,5)} seconds")
    return individual_spines,final_faces_labels_list
    
    
    