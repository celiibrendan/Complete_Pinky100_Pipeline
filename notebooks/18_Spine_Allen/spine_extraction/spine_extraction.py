
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
            print("spine classification did not execute properly")
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
    
    #clean of the cgal files from the computer
    
    
    print(f"Total time ---- {np.round(time.time() - total_time,5)} seconds")
    return individual_spines
    