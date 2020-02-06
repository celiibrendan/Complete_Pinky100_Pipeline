import datajoint as dj
import numpy as np
import time
import pymeshfix
import os
import trimesh
from pathlib import Path

import cgal_Segmentation_Module as csm

#for supressing the output
import contextlib

#for fixing the space issue with the CGAL:readoff
import csv

from collections import Counter

import networkx as nx

import pathlib
def write_Whole_Neuron_Off_file(neuron_ID,vertices=[], triangles=[]):
    #primary_key = dict(segmentation=1, segment_id=segment_id, decimation_ratio=0.35)
    #vertices, triangles = (mesh_Table_35 & primary_key).fetch1('vertices', 'triangles')
    
    num_vertices = (len(vertices))
    num_faces = len(triangles)
    
    #get the current file location
    
    file_loc = pathlib.Path.cwd() / "temp"
    filename = str(neuron_ID)
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
    
    return str(path_and_filename)#,str(filename),str(file_loc)

#class that will handle the whole neuron segmentation:
class WholeNeuronClassifier(object):
    def __init__(self,mesh_file_location="",file_name="",import_Off_Flag=False,
                                 pymeshfix_Flag=True,
                                 joincomp=True,
                                 remove_smallest_components=False,
                                 vertices=[],triangles=[],segment_id=-1):
        """
        imports mesh from off file and runs the pymeshfix algorithm to get of any unwanted portions of mesh
        (particularly used to get rid of basketball like debris that is sometimes inside soma)
        
        """
        if import_Off_Flag == True:
            
            full_path = str(Path(mesh_file_location) / Path(file_name))
            self.mesh = trimesh.load_mesh(full_path)

            #get the vertices to faces lookup table

            original_start_time = time.time() 
            start_time = time.time()
            if os.path.isfile(full_path):
                print("Loading mesh from " + str(full_path))

                my_mesh = trimesh.load_mesh(full_path,process=False) #BC_change: so doesn't process the mesh
                vertices = my_mesh.vertices
                faces = my_mesh.faces
            else:
                print(str(full_path) + " was not a valid file")
                raise Exception("Import Off file flag was set but path was invalid file path")

        elif vertices != [] and triangles != [] and segment_id > -1:
            print("loading mesh from vertices and triangles array")
            #check that neither are not empty




            self.segment_id = segment_id
            self.mesh = trimesh.Trimesh()
            self.mesh.vertices = vertices
            self.mesh.faces = triangles


            vertices = vertices
            faces = triangles



            """  How you would load from datajoint 
            print("Loading mesh from datajoint- id: " + str(key["segment_id"]))

            segment_id = key["segment_id"]
            decimation_ratio = key.pop("decimation_ratio",0.35)
            segmentation = key.pop("segmentation",2)

            primary_key = dict(segmentation=segmentation,decimation_ratio=decimation_ratio,segment_id=segment_id)
            neuron_data = (ta3p100.CleansedMesh & primary_key).fetch1()

            print(neuron_data)
            vertices = neuron_data['vertices']#.astype(dtype=np.int32)
            faces = neuron_data['triangles']#.astype(dtype=np.uint32)

            """
            


        else:
            print("No valid key or filename given")
            print(" VERTICES AND/OR TRIANGLES ARRAY WAS EMPTY")
            raise Exception("Import Off file flag was NOT set but arrays passed for verts and faces were empty")
            #raise Exception("No valid key or filename given")
        
        
        
        self.vertices = vertices
        self.faces = faces
        
        
        #performs the pymeshfix process if it flag is set (WARNING COULD TAKE A WHILE IF UNDECIMATED)
        if pymeshfix_Flag == True:
            #pymeshfix step
            start_time = time.time()
            print("Starting pymeshfix algorithm")
            meshfix = pymeshfix.MeshFix(vertices,faces)
            verbose=False
            meshfix.repair(verbose,joincomp,remove_smallest_components)
            print(f"Finished pymeshfix algorithm: {time.time() - start_time}")



            self.vertices = meshfix.v
            self.faces = meshfix.f
        
#         #------To load local pymeshfix mesh so don't have to wait for it to do it again----#
#         print("Loading local pymeshfixed mesh")
#         temp_mesh = trimesh.load_mesh("temp/neuron_" + str(648518346341393609) + "_fixed.off")
#         self.vertices = temp_mesh.vertices
#         self.faces = temp_mesh.faces
#         #------End of local pymesh import----#
        
        

        trimesh_object = trimesh.Trimesh()
        trimesh_object.faces = self.faces
        trimesh_object.vertices = self.vertices
        self.mesh = trimesh_object
        
        self.mesh_file_location = mesh_file_location
        self.file_name = file_name
    
    def generate_verts_to_face_dictionary(self,labels_list=[]):
        """
        Generates 2 dictionary mapping for vertices:
        1) verts_to_Face: maps each vertex to all the faces it touches
        2) verts_to_Label: maps each vertex to all the unique face labels it touches
        
        
        """
        if len(labels_list) <= 1:
            #print("len(labels_list) <= 1")
            labels_list = self.labels_list
        
        verts_to_Face = {i:[] for i,vertex in enumerate(self.vertices)}
        verts_to_Label = {i:[] for i,vertex in enumerate(self.vertices)}


        for i,verts in enumerate(self.faces):
            
            for vertex in verts:
                verts_to_Face[vertex].append(i)

        #use the verts to face to create the verts to label dictionary
        for vert,face_list in verts_to_Face.items():
            diff_labels = [labels_list[fc] for fc in face_list]
            #print(list(set(diff_labels)))
            verts_to_Label[vert] = list(set(diff_labels))
            
        self.verts_to_Face = verts_to_Face
        self.verts_to_Label = verts_to_Label
        
        #print("inside generate verts_to_face")

        return
    
    
    
    #write the faces and vertices to an off file
    def export_self_mesh(self,file_path_and_name):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.mesh.export(file_path_and_name)
    
    #Step 2
    def load_cgal_segmentation(self,clusters=3,smoothness=0.20,import_CGAL_Flag=False,import_CGAL_paths=[[""],[""]]):
        """
        Runs the cgal surface mesh segmentation on the mesh object and writes it to a temporary file
        
        """
        
        #have to write the new mesh to an off file
#         new_mesh_file_path_and_name = str(Path(self.mesh_file_location) /
#                                             Path(self.file_name[:-4] + "_fixed.off"))
        
#         self.export_self_mesh(new_mesh_file_path_and_name)
#         #add an extra end line to the off file
#         with open(new_mesh_file_path_and_name,'a') as fd:
#             fd.write("\n")
        
        if import_CGAL_Flag == False:
        
            if self.file_name != "":
                path_and_filename = write_Whole_Neuron_Off_file(self.file_name[:-4] + "_fixed",self.vertices,self.faces)
            elif self.segment_id != -1:
                path_and_filename = write_Whole_Neuron_Off_file(str(self.segment_id) + "_fixed",self.vertices,self.faces)
            else:
                raise Exception("Neither File name nor Segment Id set by time reaches load_cgal_segmentation")



            """skip the segmentation for now"""
            start_time = time.time()
            cgal_Flag = True

            print("\nStarting CGAL segmentation")
            if cgal_Flag == True:
                print(f"Right before cgal segmentation, clusters = {clusters}, smoothness = {smoothness}")
                result = csm.cgal_segmentation(path_and_filename,clusters,smoothness)
                print(result)
            print(f"Finished CGAL segmentation algorithm: {time.time() - start_time}")
            
            self.labels_file = path_and_filename + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" 
            self.sdf_file = path_and_filename + "-cgal_" + str(clusters) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv"
        
        else:
            #check that the paths exist in the import
            for path_c in import_CGAL_paths:
                if not os.path.isfile(str(path_c)):
                    raise Exception(str(path_c) + " is not a valid path for cgal import")

            #import the cgal
            self.labels_file = import_CGAL_paths[0]
            self.sdf_file = import_CGAL_paths[1]
            
        
        self.clusters = clusters
        self.smoothness = smoothness
#         self.labels_file = str(Path(self.mesh_file_location) / Path(self.file_name[:-4] + "_fixed" + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" ))  
#         self.sdf_file = str(Path(self.mesh_file_location) / Path(self.file_name[:-4] + "_fixed" + "-cgal_" + str(clusters) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv" ))  
#         #print(f"Step 2: CGAL segmentation total time ---- {np.round(time.time() - start_time,5)} seconds")
 
        return
        
        
    #used for when not pulling from datajoint
    def get_cgal_data_and_label_local_optomized(self):
        """
        Loads the cgal segmentation stored in the temporary file into the object
        And remaps the labels from CGAL file to numerical order
        
        """
        
        labels_file = self.labels_file
        sdf_file = self.sdf_file
        
        #reads int the cgal labels for all of the faces
        triangles_labels = np.zeros(len(self.mesh.faces)).astype("int64")
        with open(labels_file) as csvfile:

            for i,row in enumerate(csv.reader(csvfile)):
                triangles_labels[i] = int(row[0])


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

        #write thses new labels to a file
        with open(labels_file[:-4] + "_revised.csv",mode="w") as csvfile:
            csv_writer = csv.writer(csvfile,delimiter=',')
            for i in labels_list:
                csv_writer.writerow([i])

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
        
        
        #iterate through the labels_list
        for i,label in enumerate(labels_list):
            sdf_temp_dict[label].append(sdf_labels[i])
        #print(sdf_temp_dict)

        #now calculate the stats on the sdf values for each label
        sdf_final_dict = {}
        
        for dict_key,value in sdf_temp_dict.items():

            sdf_final_dict[dict_key] = dict(median=np.median(value),mean=np.mean(value),max=np.amax(value))


        self.sdf_final_dict = sdf_final_dict
        self.labels_list = labels_list
        self.labels_list_counter = Counter(labels_list)
    
#         adjacency_labels = self.labels_list[self.mesh.face_adjacency]
        
#         self.adjacency_labels_col1, self.adjacency_labels_col2 = adjacency_labels.T
        #generate the vertices labels
        self.generate_verts_to_face_dictionary(labels_list)
        
        return 
        
    
    #Step 3
    def get_highest_sdf_part(self,size_threshold=3000,exclude_label=None):
        """
        Based ont the sdf data and the labels data,
        Finds the label with the highest median,
            label with highest max,
            label with highest mean sdf value
        
        *** but only for those that meet the certain threshold ***
        
        
        """
        high_median_val = 0
        high_median = -1
        high_mean_val = 0
        high_mean = -1
        high_max_val = 0
        high_max = -1



        #gets all of the labels
        my_list = Counter(self.labels_list)
        my_list_keys = list(my_list.keys())
        if exclude_label != None:
            my_list_keys.remove(exclude_label)

        #OPTOMIZE
        
        for x in my_list_keys:
            
            if self.sdf_final_dict[x]["median"] > high_median_val and my_list[x] > size_threshold:
                high_median = x
                high_median_val = self.sdf_final_dict[x]["median"]
            if self.sdf_final_dict[x]["mean"] > high_mean_val  and my_list[x] > size_threshold:
                high_mean = x
                high_mean_val = self.sdf_final_dict[x]["mean"]
            if self.sdf_final_dict[x]["max"] > high_max_val  and my_list[x] > size_threshold:
                high_max = x
                high_max_val = self.sdf_final_dict[x]["max"]


        self.highest_vals= [high_median,high_median_val,high_mean,high_mean_val,high_max,high_max_val]
        
        self.high_median = self.highest_vals[0]
        return self.high_median
    

    #Step 3
    def get_graph_structure(self):
        """
        For each unique label gets:
        1) all neighbors
        2) number of faces belonging to that label

        """

        connections = {label_name:[] for label_name in self.labels_list_counter.keys()}
        mesh_Number = {label_name:number for label_name,number in self.labels_list_counter.items()}
        #label_vert_stats = {label_name:[300000,-300000] for label_name in Counter(labels_list).keys()}

        #verts to label curently is the has every vertex and the labels it is toughing in a list
        for verts,total_labels in self.verts_to_Label.items():
            if len(total_labels) > 1: #if more than one label
                for face in total_labels:
                    for fc in [v for v in total_labels if v != face]:
                        if fc not in connections[face]:
                            connections[face].append(fc)

        self.connections = connections
        self.mesh_Number = mesh_Number
        

        return 
    
    #Step 4: this is only used if the number of clusters for cgal value is greater than 4
    def find_Soma_Caps(self,soma_index,min_width=0.23,max_faces=6000,max_n_connection=6,large_extension_size=1500,large_extension_convex_max=3):
        """
        Will identify and relabel soma extensions that are created when using clusters of size 4 or higher
        
        """
        #get the soma neighbors
        soma_neighbors = self.connections[soma_index]
        
        total_soma_caps = []
        for i in soma_neighbors:
            soma_cap = True
            
            #collect the mesh of the cap
            submesh = self.mesh.submesh(np.where(self.labels_list == i))[0]

            mean_convex = abs(np.mean(trimesh.convex.adjacency_projections(submesh)))
            n_faces = len(submesh.faces)
            width_data = self.sdf_final_dict[i]
            width_data_median = self.sdf_final_dict[i]["median"]
            n_connections = len(self.connections[i])
          
            
            if width_data["median"] < min_width or n_faces>max_faces or n_connections>max_n_connection: 
                soma_cap = False
            
            #use the convex data if size is really big:
            if n_faces > large_extension_size:
                if mean_convex > 5:
                    #print(f" {i} Doesn't meet second pass")
                    soma_cap = False
            
            if soma_cap == True:
                total_soma_caps.append(i)
            

        #for all the soma caps replace the labels list with soma_index and recompute neighbors and connections:
        if len(total_soma_caps) > 0:
            print(f"Found {len(total_soma_caps)} soma caps and replacing labels: {total_soma_caps}")
            start_time = time.time()
            self.labels_list[np.where(np.isin(self.labels_list,total_soma_caps))] = soma_index
            
            
#             #write thses new labels to a file
#             with open(self.labels_file[:-4] + "_revised.csv",mode="w") as csvfile:
#                 csv_writer = csv.writer(csvfile,delimiter=',')
#                 for i in self.labels_list:
#                     csv_writer.writerow([i])
            #call the functions to recompute the connections/neighbors and others (but don't need to generate SDF labels again)
            self.labels_list_counter = Counter(self.labels_list)

            #generate the vertices labels
            self.generate_verts_to_face_dictionary(self.labels_list)

            self.get_graph_structure()
            print(f"done replacing soma cap labels : {time.time() - start_time}")
            
    
    #Step 5B OPTIMIZED VERION
    """************** BEGINNING OF OPTIMIZED APICAL FUNCTION *****************************"""
    def find_Apical_optimized(self,soma_index,apical_mesh_threshold=2000,
                                        apical_height_threshold=5000,
                                           apical_sdf_threshold = 0.09):
        """Returns the index of the most likely apical 
        1) calculate the height of 70% up the soma
        2) find all the neighbors of the soma using verts_to_Label
        3) filter out the neighbors that go below that
        4) filter away the neighbors that don't meet minimum number of face, height change and sdf median
        5) If multiple, pick the one that has the most number of neighbors


        """
        print("Soma Index = " + str(soma_index))
        print("Soma Connections = " + str(self.connections[soma_index]))
        mesh_Threshold = apical_mesh_threshold
        height_Threshold =apical_height_threshold
        sdf_Threshold = apical_sdf_threshold
        #1) calculate the height of 70% up the soma (but have to adjust because the negative direction of y is 
        #direction of the apical), this new method gets the height of the first 30% of the somae which is actually
        # the top 30% of the soma once it is flipped in the right orientation
        soma_faces = np.where(self.labels_list == soma_index)
        soma_verts = self.vertices[self.faces[soma_faces].ravel()][:,1]
        #gets the minimum and maximum of the soma mesh
        soma_y_min = np.min(soma_verts)
        soma_y_max = np.max(soma_verts)
        self.soma_y_min = soma_y_min
        self.soma_y_max = soma_y_max
#         print("soma_y_max ="  + str(soma_y_max))
#         print("soma_y_min ="  + str(soma_y_min))
        
        #gets the 70% height of the soma
        soma_80_percent = (soma_y_max - soma_y_min)*0.20 +  soma_y_min
        print("soma_80_percent = " + str(soma_80_percent))
        
        
        """New optimized way of creating the possible Axon filters that will use the vertices at connection"""
        possible_apical_connection_highest_y = self.get_connectors(soma_index,min_max="max")
        
#         col0 = self.mesh.face_adjacency[:,0]
#         col1 = self.mesh.face_adjacency[:,1]

#         col0_soma_boolean = np.isin(col0,soma_faces)
#         col1_soma_boolean = np.isin(col1,soma_faces)
        
#         xor_soma_columns = np.logical_xor(col0_soma_boolean,col1_soma_boolean)
        
#         """ get the indexes we want form each list
#         Want thos where logical xor is true but col0 boolean is false to be true

#         """

#         col0_indexes = col0[np.logical_and(np.logical_not(col0_soma_boolean),xor_soma_columns)]
#         col1_indexes = col1[np.logical_and(np.logical_not(col1_soma_boolean),xor_soma_columns)]
        
#         total_non_soma_indexes = np.hstack([col1_indexes,col0_indexes])

#         #now assemble them into dictionary lists:
#         connection_label_dict = dict([(key,total_non_soma_indexes[self.labels_list[total_non_soma_indexes] == key]) 
#                                      for key in list(Counter(self.labels_list[total_non_soma_indexes]))])
#         connection_label_dict
        
#         """ now get the bounding box of all of those faces"""
#         possible_apical_connection_lowest_y = dict()

#         for possible_apical,connection_list in connection_label_dict.items():
#             #get the vertices that correspond to these indices
#             #verts = current_mesh.vertices[current_mesh.faces[connection_list]]
#             verts_y_values = self.mesh.vertices[self.mesh.faces[connection_list].ravel()][:,1]

#         #     verts = current_mesh.vertices[current_mesh.faces[total_non_soma_indexes].ravel()]
#         #     print(np.min(total_non_soma_vertices[:,1]),np.max(total_non_soma_vertices[:,1]))


#             possible_apical_connection_lowest_y[possible_apical] = np.min(list(verts_y_values))

        possible_Axons_filter_1 = []
        for ap,max_value in possible_apical_connection_highest_y.items():
            if max_value < soma_80_percent:
                possible_Axons_filter_1.append(ap)
        
        
        
        """
        Current way identifying the Apical, only picking those  with:
        1) High enough median sdf value
        2) High enough face count
        3) Bounding box that doesn't go below a certain level
        
        """
        
        
        
        
        #4) filter away the neighbors that don't meet minimum number of face, height change and sdf median
        print("possible_Axons_filter_1 = " + str(possible_Axons_filter_1))
        #see affects of the filters
        size_threshold_filter = [lab for lab in possible_Axons_filter_1 if 
                                        self.mesh_Number[lab] > mesh_Threshold]
        print("size_threshold_filter = " + str(size_threshold_filter))
        
        height_threshold_filter = [lab for lab in possible_Axons_filter_1 if 
                                        possible_apical_connection_highest_y[lab] - np.min(self.vertices[self.faces[np.where(self.labels_list == lab)].ravel()][:,1]) > height_Threshold]
        print("height_threshold_filter = " + str(height_threshold_filter))
        
        sdf_threshold_filter = [lab for lab in possible_Axons_filter_1 if 
                                        self.sdf_final_dict[lab]["median"] > sdf_Threshold]
        print("sdf_threshold_filter = " + str(sdf_threshold_filter))
    
        
        
        possible_Axons_filter_2 = [lab for lab in possible_Axons_filter_1 if 
                                        self.mesh_Number[lab] > mesh_Threshold and 
        possible_apical_connection_highest_y[lab] - np.min(self.vertices[self.faces[np.where(self.labels_list == lab)].ravel()][:,1]) > height_Threshold and
                                        self.sdf_final_dict[lab]["median"] > sdf_Threshold]
        print("possible_Axons_filter_2 = " + str(possible_Axons_filter_2))
        if len(possible_Axons_filter_2) <= 0:
            return "None"
        elif len(possible_Axons_filter_2) == 1:
            picked_apical = possible_Axons_filter_2[0]
            return picked_apical
        else:
#             #find the one with the most neighbors
#             current_apical = possible_Axons_filter_2[0]
#             current_apical_neighbors = len(self.connections[possible_Axons_filter_2[0]])
#             for i in range(1,len(possible_Axons_filter_2)):
#                 if len(self.connections[possible_Axons_filter_2[i]]) > current_apical_neighbors:
#                     current_apical = possible_Axons_filter_2[i]
#                     current_apical_neighbors = len(self.connections[possible_Axons_filter_2[i]])
                    
            """ old way where picked the one with the greatest width
            #--> revised now does the one with the highest thickness
            ##### MIGHT WANT TO ADD IN WHERE FINDS THE THICKEST WIDTH !
            
            
            current_apical = possible_Axons_filter_2[0]
            current_apical_width = self.sdf_final_dict[current_apical]["median"]
            
            
            for i in range(1,len(possible_Axons_filter_2)):
                if self.sdf_final_dict[possible_Axons_filter_2[i]]["median"] > current_apical_width:
                    current_apical = possible_Axons_filter_2[i]
                    current_apical_width = self.sdf_final_dict[possible_Axons_filter_2[i]]["median"]

            """
                        #want to return the apical with the highest connecting point
            picked_apical = possible_Axons_filter_2[0]
            highest_connection = possible_apical_connection_highest_y[picked_apical]
            
            print(f"STARTER picked_apical = {picked_apical} with highest connection {highest_connection}")
            
            for i,ap in enumerate(possible_Axons_filter_2):
                if i == 0:
                    continue
                
                current_highest = possible_apical_connection_highest_y[ap]
                print(f"Next possible apical {ap} with highest y {current_highest}")
                if current_highest < highest_connection:
                    print(f"changing apical to {ap}")
                    picked_apical = ap
                    highest_connection = current_highest
                    
            return picked_apical
    
    
    """************** END OF OPTIMIZED FUNCTION *****************************"""
    #Step 5
    def find_Apical(self,soma_index,apical_mesh_threshold=2000,
                                        apical_height_threshold=5000,
                                           apical_sdf_threshold = 0.09):
        """Returns the index of the most likely apical 
        1) calculate the height of 70% up the soma
        2) find all the neighbors of the soma using verts_to_Label
        3) filter out the neighbors that go below that
        4) filter away the neighbors that don't meet minimum number of face, height change and sdf median
        5) If multiple, pick the one that has the most number of neighbors


        """
        print("Soma Index = " + str(soma_index))
        print("Soma Connections = " + str(self.connections[soma_index]))
        mesh_Threshold = apical_mesh_threshold
        height_Threshold =apical_height_threshold
        sdf_Threshold = apical_sdf_threshold
        #1) calculate the height of 70% up the soma (but have to adjust because the negative direction of y is 
        #direction of the apical), this new method gets the height of the first 30% of the somae which is actually
        # the top 30% of the soma once it is flipped in the right orientation

        soma_verts = self.vertices[self.faces[np.where(self.labels_list == soma_index)].ravel()][:,1]
        #gets the minimum and maximum of the soma mesh
        soma_y_min = np.min(soma_verts)
        soma_y_max = np.max(soma_verts)
        self.soma_y_min = soma_y_min
        self.soma_y_max = soma_y_max
#         print("soma_y_max ="  + str(soma_y_max))
#         print("soma_y_min ="  + str(soma_y_min))
        
        #gets the 70% height of the soma
        soma_80_percent = (soma_y_max - soma_y_min)*0.3 +  soma_y_min
        print("soma_80_percent = " + str(soma_80_percent))
        
        #2) find all the neighbors of the soma using verts_to_Label
        soma_neighbors = self.connections[soma_index]
        
        #3) filter out the neighbors that go below that

        print("Debugging the axon filter")
        print([(label,np.max(self.vertices[self.faces[np.where(self.labels_list == label)].ravel()][:,1])) for label in soma_neighbors])
        possible_Axons_filter_1 = [label for label in soma_neighbors 
                            if np.max(self.vertices[self.faces[np.where(self.labels_list == label)].ravel()][:,1]) < soma_80_percent]

        
        """
        Current way identifying the Apical, only picking those  with:
        1) High enough median sdf value
        2) High enough face count
        3) Bounding box that doesn't go below a certain level
        
        """
        
        #4) filter away the neighbors that don't meet minimum number of face, height change and sdf median
        print("possible_Axons_filter_1 = " + str(possible_Axons_filter_1))
        possible_Axons_filter_2 = [lab for lab in possible_Axons_filter_1 if 
                                        self.mesh_Number[lab] > mesh_Threshold and 
        np.max(self.vertices[self.faces[np.where(self.labels_list == lab)].ravel()][:,1]) - np.min(self.vertices[self.faces[np.where(self.labels_list == lab)].ravel()][:,1]) > height_Threshold and
                                        self.sdf_final_dict[lab]["median"] > sdf_Threshold]
        print("possible_Axons_filter_2 = " + str(possible_Axons_filter_2))
        if len(possible_Axons_filter_2) <= 0:
            return "None"
        elif len(possible_Axons_filter_2) == 1:
            return possible_Axons_filter_2[0]
        else:
#             #find the one with the most neighbors
#             current_apical = possible_Axons_filter_2[0]
#             current_apical_neighbors = len(self.connections[possible_Axons_filter_2[0]])
#             for i in range(1,len(possible_Axons_filter_2)):
#                 if len(self.connections[possible_Axons_filter_2[i]]) > current_apical_neighbors:
#                     current_apical = possible_Axons_filter_2[i]
#                     current_apical_neighbors = len(self.connections[possible_Axons_filter_2[i]])
                    
            #--> revised now does the one with the highest thickness
            ##### MIGHT WANT TO ADD IN WHERE FINDS THE THICKEST WIDTH !
            
            current_apical = possible_Axons_filter_2[0]
            current_apical_width = self.sdf_final_dict[current_apical]["median"]
            
            
            for i in range(1,len(possible_Axons_filter_2)):
                if self.sdf_final_dict[possible_Axons_filter_2[i]]["median"] > current_apical_width:
                    current_apical = possible_Axons_filter_2[i]
                    current_apical_width = self.sdf_final_dict[possible_Axons_filter_2[i]]["median"]

            return current_apical
    
    
    #step 6
    def get_connectors(self,interest_label,mesh=None,labels_list=None,min_max="None"):
        """
        Fill get the connecting pieces to a face group and return either the indices of these connecting faces
        or the min or max values of the each of these connectors based on the parameters asked for
        """
        
        if mesh == None:
            mesh=self.mesh
        if labels_list == None:
            labels_list=self.labels_list
        
        interest_faces = np.where(labels_list==interest_label)
        
        """New optimized way of creating the possible Axon filters that will use the vertices at connection"""
        col0 = mesh.face_adjacency[:,0]
        col1 = mesh.face_adjacency[:,1]

        col0_soma_boolean = np.isin(col0,interest_faces)
        col1_soma_boolean = np.isin(col1,interest_faces)

        xor_soma_columns = np.logical_xor(col0_soma_boolean,col1_soma_boolean)

        """ get the indexes we want form each list
        Want thos where logical xor is true but col0 boolean is false to be true

        """

        col0_indexes = col0[np.logical_and(np.logical_not(col0_soma_boolean),xor_soma_columns)]
        col1_indexes = col1[np.logical_and(np.logical_not(col1_soma_boolean),xor_soma_columns)]

        total_non_soma_indexes = np.hstack([col1_indexes,col0_indexes])

        #now assemble them into dictionary lists:
        connection_label_dict = dict([(key,total_non_soma_indexes[self.labels_list[total_non_soma_indexes] == key]) 
                                     for key in list(Counter(self.labels_list[total_non_soma_indexes]))])
        connection_label_dict

        if min_max=="None":
            return connection_label_dict
        
        """ now get the bounding box of all of those faces"""
        possible_apical_connection_lowest_y = dict()
        
        if min_max=="min" or min_max=="both":
            possible_apical_connection_lowest_y = dict()
            for possible_apical,connection_list in connection_label_dict.items():
                #get the vertices that correspond to these indices
                #verts = current_mesh.vertices[current_mesh.faces[connection_list]]
                verts_y_values = self.mesh.vertices[self.mesh.faces[connection_list].ravel()][:,1]

            #     verts = current_mesh.vertices[current_mesh.faces[total_non_soma_indexes].ravel()]
            #     print(np.min(total_non_soma_vertices[:,1]),np.max(total_non_soma_vertices[:,1]))


                possible_apical_connection_lowest_y[possible_apical] = np.max(list(verts_y_values))
        
        if min_max=="max" or min_max=="both":
            possible_apical_connection_highest_y = dict()
            for possible_apical,connection_list in connection_label_dict.items():
                #get the vertices that correspond to these indices
                #verts = current_mesh.vertices[current_mesh.faces[connection_list]]
                verts_y_values = self.mesh.vertices[self.mesh.faces[connection_list].ravel()][:,1]

            #     verts = current_mesh.vertices[current_mesh.faces[total_non_soma_indexes].ravel()]
            #     print(np.min(total_non_soma_vertices[:,1]),np.max(total_non_soma_vertices[:,1]))


                possible_apical_connection_highest_y[possible_apical] = np.min(list(verts_y_values))
                                                                              
        if min_max=="max":
            return possible_apical_connection_highest_y
        elif min_max=="min":
            return possible_apical_connection_lowest_y
        elif min_max=="both":
            return possible_apical_connection_lowest_y,possible_apical_connection_highest_y
        else:
            return connection_label_dict


    
    
    def classify_whole_neuron(self,possible_Apical,soma_index,
                             classifier_cilia_threshold=1000,
                             classifier_stub_threshold=200,
                             classifier_non_dendrite_convex_threshold = 27.5,
                             classifier_axon_std_dev_threshold = 69,
                             classifier_stub_threshold_apical = 700):
        """
        Will use the soma index and apical index to label the rest of the segmentation portions
        with the appropriate category: Apical, Soma stub, cilia, basal, dendrite, axon, etc.
        
        Parameteres:
        classifier_cilia_threshold #maximum size of cilia
        classifier_stub_threshold # minimum size of appndage of soma to not be considered stub and merged with the soma
        classifier_non_dendrite_convex_threshold #must be above this value to be axon, cilia or error
        
        classifier_stub_threshold_apical #the minimum size threshold for apical appendage not to be merged with apical
        """
        
        #check to see if no soma index
        if soma_index < 0:
            self.whole_neuron_labels ={lb:"unsure" for lb in self.connections.keys()}
            return
        
        #creates dictionary with unique labels whose value will store their final label
        whole_neuron_labels ={lb:"unsure" for lb in self.connections.keys()}
        whole_neuron_labels[soma_index] = "soma"

        #create a networkx graph based on connections
        G=nx.Graph(self.connections)

        
        #removes the soma from the list of nodes, but not actually remove it from the graph
        node_list = list(G.nodes)
        if(soma_index in node_list):
            node_list.remove(soma_index)
        else:
            #didn't find soma
            return []

        
        #finds the shortest path from any label to the soma and stores it in the shortest path dictonary
        shortest_paths = {}
        for node in node_list:
            shortest_paths[node] = [k for k in nx.shortest_path(G,node,soma_index)]

        #find the direct neighbors of the soma
        soma_branches = dict()
        soma_neighbors = self.connections[soma_index]
        
        #get the neighbor connector to use later for determining axon or not:
        soma_connections = self.get_connectors(soma_index,min_max="max")
        
        
        
        #print("soma_neighbors = " + str(soma_neighbors))
        
        #assemble each of these compartments into groups
        for node,path in shortest_paths.items():
            if possible_Apical not in path:  #if the possible apical is not in the path
                #gets the soma neighbor that it passes through
                specific_soma_neighbor = (set(path).intersection(set(soma_neighbors))).pop()
                
                #first add the soma neighbor as a key in the soma branches dictionary and then
                #adds this node to the soma neighbor that it passes through
                if specific_soma_neighbor not in soma_branches.keys():
                    soma_branches[specific_soma_neighbor] = []
                soma_branches[specific_soma_neighbor].append(node)
        

        #print("soma_branches = " + str(soma_branches))
        
        #have groups of branches (NOT INCLUDING THE APICAL) and assmble them into trimesh objects
        branches_submeshes = {}
        for group,group_list in soma_branches.items():
            total_indices = []
            for g in group_list:
                face_indices = np.where(self.labels_list == g)
                total_indices += face_indices
            
            #create a trimesh submshesh
            branches_submeshes[group] = self.mesh.submesh(total_indices,append=True)
        
        
        #iterate through meshes and assign certain labels to these guys
        ## define certain thresholds for determining label
        cilia_threshold = classifier_cilia_threshold #maximum size of cilia
        stub_threshold = classifier_stub_threshold # minimum size of appndage of soma to not be considered stub and merged with the soma
        non_dendrite_convex_threshold = classifier_non_dendrite_convex_threshold #must be above this value to be axon, cilia or error

        
        #Calculate the soma 30% that axon must be lower than

        
        soma_height = self.soma_y_max - self.soma_y_min
        
        soma_lower_30 = self.soma_y_max - 0.3*soma_height
#         print("self.soma_y_max = " + str(self.soma_y_max))
#         print("self.soma_y_min = " + str(self.soma_y_min))
#         print("soma_lower_30 = " + str(soma_lower_30))
        
        
        """
        Labels Soma stubs, cilia or axons (used to do errors but got rid of that)
        """
        
        axon_check = 60
            
            
        
        for neighbor,submesh in branches_submeshes.items():
            #printing_out the thresholds
            if neighbor == axon_check:
                print("Must be greater than non_dendrite_convex_threshold =" + str(non_dendrite_convex_threshold))
                print("Must be less than classifier_axon_std_dev_threshold =  " + str(classifier_axon_std_dev_threshold))
            
            #get the number of faces
            total_faces = len(submesh.faces)
            #print(f"total_faces  = {total_faces}")
            
            if total_faces < stub_threshold: #if the size of a stub then just label as soma
                print(f"{neighbor} = stub soma")
                for x in soma_branches[neighbor]:
                    
                    whole_neuron_labels[x] = "soma"
            else:
            
                mean_convex = abs(np.mean(trimesh.convex.adjacency_projections(submesh)))
                #print(f"total_faces  = {mean_convex}")
                
                #if neighbor == axon_check:
                if neighbor >= 0:
                    print("Neighbor = " + str(neighbor))
                    print("mean_convex == " + str(mean_convex))
                    print("std_dev_conves == " + str(np.std((trimesh.convex.adjacency_projections(submesh)))))
                
                if mean_convex > non_dendrite_convex_threshold:
                    #print("neighbor inside cilia check = " + str(neighbor))
                    #classify according to size
                    if neighbor == axon_check:
                        print("inside cilia checking")
                    
                    
                    if total_faces < cilia_threshold:
                        print(f"{neighbor} = cilia")
                        for x in soma_branches[neighbor]:
                            whole_neuron_labels[x] = "cilia"
                    else:
                        print(f"{neighbor} = axon")
                        for x in soma_branches[neighbor]:
                            whole_neuron_labels[x] = "axon"
                else: #try to see if there is any axon
                    #calculate the standard deviation
                    #print("neighbor inside axon check = " + str(neighbor))
                    std_dev_convex = np.std((trimesh.convex.adjacency_projections(submesh)))
                    
                    if neighbor == axon_check:
                        print("checking std dev")
                    
                    if std_dev_convex < classifier_axon_std_dev_threshold:
                        
                        if neighbor == axon_check:
                            print("inside axon checking")
                        
                        #find the minimum y heght of neighbor --> OLD WAY OF FINDING MINIMUM
                        ####neighbor_y_min = np.min(self.vertices[self.faces[np.where(self.labels_list == neighbor)].ravel()][:,1])
                        neighbor_y_min = soma_connections[neighbor]
                        
                        #now finding minimum using connection point
                        
                        
                        
                        ####*************** need to change this where uses the y amount ****###
                        if neighbor_y_min > soma_lower_30:
                            if neighbor == axon_check:
                                print("inside axon qualification")
                            #make sure that it doesn't go higher than 40% soma height
                            print(f"{neighbor} = axon")
                            for x in soma_branches[neighbor]:
                                whole_neuron_labels[x] = "axon"
                                
                        else:
                            print(f"MET AXON THRESHOLD CRITERIA but not low enough on soma for neighbor = {neighbor}")
                            
        
        # checks if apical is present or not, and if not then just labels everything else basal
        if possible_Apical == "None":
            #label everything as basal if don't know
            for k,vals in whole_neuron_labels.items():
                if k != soma_index and vals == "unsure":
                    whole_neuron_labels[k] = "basal"  #BC_change: BY DEFAULT JUST LABEL AS BASAL
                    #pass
            self.whole_neuron_labels = whole_neuron_labels
            
            return 
                    
            
        #return branches_submeshes
        
        """ 4-29 added edition that will prevent small spines off of apical 
        from being considered oblique branches
        
    
        """
        #find the direct neighbors of the soma
        apical_branches = dict()
        
        apical_neighbors = self.connections[possible_Apical]
        apical_neighbors.remove(soma_index)
        
        
        #assemble each of these compartments into groups
        for node,path in shortest_paths.items():
            if possible_Apical in path and node != possible_Apical: #make sure only those obliques and not actual apical
                
                #find out which apical neighbor this node passes through and then adds it to that specific neighbors list
                specific_apical_neighbor = (set(path).intersection(set(apical_neighbors))).pop()
                
                if specific_apical_neighbor not in apical_branches.keys():
                    apical_branches[specific_apical_neighbor] = []
                apical_branches[specific_apical_neighbor].append(node)
        

        #print("apical_branches = " + str(apical_branches))
        #have groups of branches and assmble them into trimesh objects
        branches_submeshes_apical = {}
        for group,group_list in apical_branches.items():
            total_indices = []
            for g in group_list:
                face_indices = np.where(self.labels_list == g)
                total_indices += face_indices
            
            #create a trimesh submshesh
            
            branches_submeshes_apical[group] = self.mesh.submesh(total_indices,append=True)
        #return branches_submeshes_apical
        
        #iterate through meshes and assign certain labels to these guys
        
       
        """
        Apical filtering: 
        1) if it is too small then will just be absorbed by the apical label
            
        The place to make the cutoff for the oblique is to where it will be absorbed by apical filter: 
        1) If ever reaches the top → apical
        2) If it branches off from the apical above 162000 then apical
            """
        
        top_box = 160000
        oblique_cutoff = 195000
        
        print(f"top_box = {top_box}")
        for neighbor,submesh in branches_submeshes_apical.items():
            
            #get the number of faces
            total_faces = len(submesh.faces)
            #print(f"total_faces  = {total_faces}")
            
            if total_faces < classifier_stub_threshold_apical:
                print(f"{neighbor} = stub apical")
                for x in apical_branches[neighbor]:
                    
                    whole_neuron_labels[x] = "apical"
            
            #check for the highest
            verts_max = np.min(submesh.vertices[:,1])
            #if reaches near the top of the box then assume not oblique
            
            #print(f"{neighbor} has verts_max of {verts_max}")
            if verts_max < top_box: 
                print(f"{neighbor} = top box touching oblique --> change to apical")
                for x in apical_branches[neighbor]:
                    whole_neuron_labels[x] = "apical"
                
                    
        """
        NO LONGER CHECKING FOR APICAL ERRORS

        #####******** can add the smart oblique labeling here ******** ########
        """

        print(f"oblique_cutoff = {oblique_cutoff}")
        
        possible_apical_connection_highest_y = self.get_connectors(possible_Apical,min_max="min")
        
        if soma_index in possible_apical_connection_highest_y.keys():
            possible_apical_connection_highest_y.pop(soma_index, None)
        
        
        
        
        
        for oblique_neighbor,min_y in possible_apical_connection_highest_y.items():
            #print(f"neighbor {oblique_neighbor} has min_y = {min_y}")
            if min_y < oblique_cutoff:
                print(f"neighbor {oblique_neighbor} had too high of a connector point --> being labeled as apical")
                for x in apical_branches[oblique_neighbor]:
                    whole_neuron_labels[x] = "apical"
                
        
            
#             else:
            
#                 mean_convex = abs(np.mean(trimesh.convex.adjacency_projections(submesh)))
#                 #print(f"total_faces  = {mean_convex}")
#                 if mean_convex > non_dendrite_convex_threshold:
#                     #classify according to size

#                     print(f"{neighbor} = error")
#                     for x in apical_branches[neighbor]:
#                         whole_neuron_labels[x] = "error"
#                 else: #try to see if there is any axon like objects off of apical --> if so then error
#                     #calculate the standard deviation
#                     std_dev_convex = np.std((trimesh.convex.adjacency_projections(submesh)))
#                     if std_dev_convex < classifier_axon_std_dev_threshold:
#                         for x in apical_branches[neighbor]:
#                             whole_neuron_labels[x] = "error"


        for label_name, path in shortest_paths.items():
            if label_name == possible_Apical: #labels the possible apical as apical
                whole_neuron_labels[label_name] = "apical"
            else:
                if possible_Apical in path:
                    #if has apical on path and not the apical itself, soma or other label --> label oblique
                    for jj in path: 
                        if jj != possible_Apical and jj != soma_index and whole_neuron_labels[jj] == "unsure":
                            whole_neuron_labels[jj] = "oblique" 
                else:
                    #if NO apical on path and not the apical itself, soma or other label --> label oblique
                    for jj in path:
                        if jj != possible_Apical and jj != soma_index and whole_neuron_labels[jj] == "unsure":
                            whole_neuron_labels[jj] = "basal" 

        #return the final list of labels:
        self.whole_neuron_labels = whole_neuron_labels
        return
    
    #Step 7
    def label_whole_neuron(self):
        
        """
        iterates through all of faces and labels them accoring
        to the labels assigned to the cgal generic labels
        
        """
        
        #instead of going to datajoint for labels
        #this just have it locally so don't rely on datajoint

        apical_index = 2
        basal_index = 3
        oblique_index = 4
        soma_index = 5
        cilia_index = 12
        error_index = 10 
        axon_index=6


        self.final_faces_labels_list = np.zeros(len(self.faces))
        

        unknown_counter = 0

        for i,lab in enumerate(self.labels_list):
            #get the category according to the dictionary
            cat = self.whole_neuron_labels[lab]
            if cat == "apical":
                self.final_faces_labels_list[i] = apical_index
            elif cat == "basal":
                self.final_faces_labels_list[i] = basal_index
            elif cat == "oblique":
                self.final_faces_labels_list[i] = oblique_index
            elif cat == "soma":
                self.final_faces_labels_list[i] = soma_index
            elif cat == "cilia":
                self.final_faces_labels_list[i] = cilia_index
            elif cat == "axon":
                self.final_faces_labels_list[i] = axon_index
            elif cat == "error":
                self.final_faces_labels_list[i] = error_index
            else:
                #if wasn't labeled anything just assing it a random color based on cgal assignment
                self.final_faces_labels_list[i] = 18 + (int(lab))

        
    def generate_output_lists(self):
        """
        Will generate the final faces and vertices labels for the classification
        """

        output_faces_list = self.final_faces_labels_list
        

        #generate the vertices labels
        self.generate_verts_to_face_dictionary(output_faces_list)

        output_verts_list = [int(self.verts_to_Label[v][0]) for v in self.verts_to_Label]

        self.output_verts_labels_list = output_verts_list
        return self.final_faces_labels_list, self.output_verts_labels_list 
    


    def return_branches(self,return_cilia=False,
                        return_soma=False,
                        return_axon=False,
                        return_error=False
                        ,return_size_threshold=200):
        all_components = dict()
        
        apical_index = 2
        basal_index = 3
        oblique_index = 4
        soma_index = 5
        cilia_index = 12
        error_index = 10 
        axon_index=6
        
        basal_indexes = np.where(self.final_faces_labels_list == basal_index)[0]
        oblique_indexes = np.where(self.final_faces_labels_list == oblique_index)[0]
        apical_indexes = np.where(self.final_faces_labels_list == apical_index)[0]
        #axon_indexes = np.where(self.final_faces_labels_list == axon_index)
        spine_indexes = [np.concatenate([basal_indexes,oblique_indexes,apical_indexes])]
        
        if spine_indexes[0].size > 0:
            #gets all of the dendritic branches
            spine_meshes_whole = self.mesh.submesh(spine_indexes,append=True)

            split_up_spines = True
            #decides if passing back spines as one whole mesh or seperate meshes
            if split_up_spines==True:
                individual_spines = []
                temp_spines = spine_meshes_whole.split(only_watertight=False)
                for spine in temp_spines:
                    if len(spine.faces) >= return_size_threshold:
                        individual_spines.append(spine)
            else:
                individual_spines = spine_meshes_whole
                    
        else:
            
            individual_spines = None
        
        if individual_spines == []:
            individual_spines = None
        
        
        all_components["dendrites"] = individual_spines
        
        
        #will also pass back the cilia,axon or soma based on the parameters of the mesh with the extracted spines
        if return_cilia==True:
            shaft_indexes = np.where(np.array(self.final_faces_labels_list) == cilia_index) 
            if shaft_indexes[0].size > 0:
                shaft_mesh_whole = self.mesh.submesh(shaft_indexes,append=True)
                all_components["cilia"] = shaft_mesh_whole
            else:
                all_components["cilia"] = None
        
        #will also pass back the cilia,axon or soma based on the parameters of the mesh with the extracted spines
        if return_soma==True:
            shaft_indexes = np.where(np.array(self.final_faces_labels_list) == soma_index)
            if shaft_indexes[0].size > 0:
                shaft_mesh_whole = self.mesh.submesh(shaft_indexes,append=True)
                all_components["soma"] = shaft_mesh_whole
            else:
                all_components["soma"] = None
            
        #will also pass back the cilia,axon or soma based on the parameters of the mesh with the extracted spines
        if return_axon==True:
            shaft_indexes = np.where(np.array(self.final_faces_labels_list) == axon_index) 
            if shaft_indexes[0].size > 0:
                shaft_mesh_whole = self.mesh.submesh(shaft_indexes,append=True)
                all_components["axon"] = shaft_mesh_whole
            else:
                all_components["axon"] = None
        
        
        if return_error==True:
            shaft_indexes = np.where(np.array(self.final_faces_labels_list) == error_index) 
            if shaft_indexes[0].size > 0:
                #gets all of the dendritic branches
                spine_meshes_whole = self.mesh.submesh(shaft_indexes,append=True)

                split_up_spines = True
                #decides if passing back spines as one whole mesh or seperate meshes
                if split_up_spines==True:
                    individual_error = []
                    temp_spines = spine_meshes_whole.split(only_watertight=False)
                    for spine in temp_spines:
                        individual_error.append(spine)
                else:
                    individual_error = spine_meshes_whole

            else:

                individual_error = None

            if individual_error == []:
                individual_error = None
        
        
            all_components["error"] = individual_error
        
        return all_components
            

    def clean_files(self):
        #clean the files 
        
        #1) new mesh file
        #2) cgal files (sdf and labels)
        
        files_to_delete = [self.file_name[:-4] + "_fixed",
                          self.labels_file[:-4] + "_revised.csv",
                          self.labels_file,
                          self.sdf_file]
        
        for myfile in files_to_delete:
            if os.path.isfile(myfile):
                os.remove(myfile)

        
        return
        
        
        
def extract_branches_whole_neuron(import_Off_Flag,
                              **kwargs):
    """
    Extracts the meshes of all dendritic branches (optionally soma, axon, cilia meshes)
    from a full neuron mesh  (Assumes meshes have been decimated to 35% original size but if not then scaling 
    can be adjusted using size_multiplier argument)
    
    Parameters:
    mesh_file_location (str): location of the dendritic mesh on computer
    file_name (str): file name of dendritic mesh on computer
    size_multiplier (float): multiplying factor to help scale all size thresholds in case of up/downsampling of faces (default = 1)
    
        Option kwargs parameters
        
    --- Step 1: Mesh importing and Pymeshfix parameters ---
    
    joincomp : bool, optional (default = True)
       Attempts to join nearby open components.

    remove_smallest_components : bool, optional (default = False)
        Remove all but the largest isolated component from the mesh
        before beginning the repair process.  Default True
        
    --- Step 2: CGAL segmentation parameters ---

    clusters (int) : number of clusters to use for CGAL surface mesh segmentation (default = 4)
    smoothness (int) : smoothness parameter use for CGAL surface mesh segmentation (default = 0.30)
    
    --- Step 3: Soma identification parameters ---
    
    soma_size_threshold (int) : Minimum number of faces (multiplied by size_multipler) of segment to be classified as soma (default = 3000)
    
    --- Step 4: Findin Soma extensions parameters --- 
    #if clusters > 3, then will try to relabel small stubs off of soma as soma (helps with identifying axons)
    soma_cap_min_width (float): Minimum width size to be categorized as soma extension (default = 0.23) 
    soma_cap_max_faces (int): Maximum number of faces (multiplied by size_multipler) to be categorized as soma extension (default = 6000)
    soma_cap_max_n_connections (int): Maximum number of neighbors to be considered soma extension(default = 6)
    large_extension_size (int): Maximum number of faces (multiplied by size_multipler) to be considered a possible large soma extension segment
    large_extension_convex_max (float): Maximum value for the mean of the convex adjacency projections for large segments to be considered soma extension (default = 3.0) 
    
    --- Step 5: Apical Identifying Parameters --- 
    apical_mesh_threshold (int) : Minimum size of segment (multiplied by size_multipler) to be considered possible apical (default = 2000)
    apical_height_threshold (int) : Minimum height of bounding box of segment to be considered possible apical (default = 5000) 
    apical_sdf_threshold (float) : Minimum width of segment to be considered possible apical (default = 0.09)
    
    --- Step 6: Classifying Entire Mesh Parameters ---
    classifier_cilia_threshold (int): Maximum size of segment (multiplied by size_multipler) to be considered possible cilia (default = 1000) 
    classifier_stub_threshold (int): minimum size of appndage of soma (multiplied by size_multipler) to not be considered stub and merged with the soma (default = 200) 
    classifier_non_dendrite_convex_threshold (float) : Segment must be above this mean convex value to be considered a possible axon, cilia or error(default = 26.5) 
    classifier_axon_std_dev_threshold (float): standard deviation of convex measurements for which axon branches are under this threshold (default = 69.0) 
    classifier_stub_threshold_apical (int) = the minimum size threshold (multiplied by size_multipler) for apical appendage not to be merged with apical(default = 700) 
    
    
    ---Step 9: Output Configuration Parameters ---
    if return_Only_Labels is set to true then will only return the vertex_labels,face_labels
    
    * if any of the below settings are set to true then will return a dictionary storing 
    the lists for each mesh category (dendrite,cilia,soma,axon) only for those present that flag is set True
    The dendritic branches will always be returned
    
    return_cilia (bool) : if true will return cilia mesh inside returned dictionary (default = False)
    return_soma (bool) : if true will return soma mesh inside returned dictionary (default = False)
    return_axon (bool) : if true will return axon mesh inside returned dictionary (default = False)
    return_error (bool) : if true will return error mesh inside returned dictionary (default = False)
    return_size_threshold (int): Minimum size (multiplied by size_multipler) of dendrite piece to be returned (default = 200)
    
    --- Step 10: Cleaning up temporary files parameters ---
    clean_temp_files (bool) : if true, will delete all the temporary segmentation and pymeshfix files (default = True)
    
        -------------------------------------
  
    Returns: 
    if return_cilia,return_soma,return_axon,return_error are all set to false: 
        return  lists of trimesh.mesh/None based on the number of dendrite branches found
    if Any of the return_cilia,return_soma,return_axon,return_error are set to true: 
        returns dictionary containing 4 keys: dendrites,soma,cilia,axon
        For each value will return  lists of object (for dendrtiess), trimesh.mesh objects (for other compartments) or None based on the number of that compartment found

    Examples:
    #returns just simple list of dendrite meshes
    list_of_dendrite_meshes = extract_branches_whole_neuron(file_location,file_name)
    
    #returns dendrite meshes and an available soma mesh
    compartment_meshes= complete_spine_extraction(file_location,file_name,return_soma=True)
    soma_mesh = compartment_meshes["soma"]
    dendrite_mesh_list = compartment_meshes["dendrites"]
    
    #retruns dendrite meshes but adjusts for not downsampling meshes to 35% original as default settings assume
    list_of_dendrite_meshes = extract_branches_whole_neuron(file_location,file_name,size_multiplier=1/0.35)
    
    """
    
    
    

    
    global_start = time.time()
   
    # Step 0: Where to import from
    
    #if import_Off_Flag == True:
    #if loading from an off file
    mesh_file_location = kwargs.pop('mesh_file_location', "")
    file_name = kwargs.pop('file_name', "")
    #else:
    #if loading from datajoint
    vertices = kwargs.pop('vertices', -1)
    triangles = kwargs.pop('triangles', -1)
    segment_id = kwargs.pop("segment_id",-1)
        
    #Step 1: Mesh importing and Pymeshfix parameters
    pymeshfix_Flag = kwargs.pop('pymeshfix_Flag', True)
    
    joincomp = kwargs.pop('joincomp', False)
    remove_smallest_components = kwargs.pop('remove_smallest_components', True)
    
    #Step 2: CGAL segmentation parameters
    
    import_CGAL_Flag = kwargs.pop('import_CGAL_Flag', False)
    import_CGAL_paths = kwargs.pop('import_CGAL_paths', [[""],[""]])
    
    clusters = kwargs.pop('clusters', 4)
    smoothness = kwargs.pop('smoothness', 0.30)
    
    #step 3: Soma identification parameters
    size_multiplier = kwargs.pop('size_multiplier', 1)
    soma_size_threshold = kwargs.pop("soma_size_threshold",3000)
    
    #step 4: finding soma extensions parameters
    soma_cap_min_width= kwargs.pop('soma_cap_min_width', 0.23) 
    soma_cap_max_faces= kwargs.pop('soma_cap_max_faces', 6000) 
    soma_cap_max_n_connections= kwargs.pop('soma_cap_max_n_connections', 6) 
    large_extension_size = kwargs.pop('large_extension_size', 1500) 
    large_extension_convex_max= kwargs.pop('soma_cap_conex_threshold', 3) 
    
    
    
    #Step 5: Apical Identifying Parameters
    apical_mesh_threshold= kwargs.pop('apical_mesh_threshold', 2000)
    apical_height_threshold= kwargs.pop('apical_height_threshold', 5000) 
    apical_sdf_threshold = kwargs.pop('apical_sdf_threshold', 0.05)
    
    #Step 6: Classifying Entire Mesh parameters
    classifier_cilia_threshold=kwargs.pop('classifier_cilia_threshold', 1000) #maximum size of cilia
    classifier_stub_threshold=kwargs.pop('classifier_stub_threshold', 200) # minimum size of appndage of soma to not be considered stub and merged with the soma
    classifier_non_dendrite_convex_threshold = kwargs.pop('classifier_non_dendrite_convex_threshold', 27.5) #must be above this value to be axon, cilia or error
    classifier_axon_std_dev_threshold = kwargs.pop('classifier_axon_std_dev_threshold', 69) #standard deviation of convex measurements for which axon branches are under this threshold
    classifier_stub_threshold_apical = kwargs.pop('classifier_stub_threshold_apical', 700) #the minimum size threshold for apical appendage not to be merged with apical
    
    #Step 9: Output Configuration Parameters
    return_Only_Labels = kwargs.pop("return_Only_Labels",False)
    
    return_cilia=kwargs.pop('return_cilia', False)
    return_soma=kwargs.pop('return_soma', False)
    return_axon=kwargs.pop('return_axon', False)
    return_error=kwargs.pop('return_error', False)
    return_size_threshold=kwargs.pop('return_size_threshold', 200)
    
    clean_temp_files=kwargs.pop('clean_temp_files', True)
    
    
    
    #making sure there is no more keyword arguments left that you weren't expecting
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    

    
    #check to see if file exists and if it is an off file
    if import_Off_Flag == True: 
        if file_name[-3:] != "off":
            raise TypeError("input file must be a .off ")
            return None
        if not os.path.isfile(str(Path(mesh_file_location) / Path(file_name))):
            raise TypeError(str(Path(mesh_file_location) / Path(file_name)) + " cannot be found")
            return None
    


    
    start_time = time.time()
    print("1) Starting: Mesh importing and Pymesh fix")
    classifier = WholeNeuronClassifier(mesh_file_location,file_name,import_Off_Flag,pymeshfix_Flag,joincomp,remove_smallest_components,
                                      vertices,triangles,segment_id)
    print(f"1) Finished: Mesh importing and Pymesh fix: {time.time() - start_time}")
    
    
    start_time = time.time()
    print("2) Staring: Generating CGAL segmentation for neuron")
    classifier.load_cgal_segmentation(clusters,smoothness,import_CGAL_Flag,import_CGAL_paths)
    #retrieves the cgal data from the file
    

    #check to see if files exist
    for f in [classifier.labels_file,classifier.sdf_file]:
            if not os.path.isfile(f):
                print("CGAL segmentation files weren't generated")
                raise ValueError("CGAL segmentation files weren't generated")
                return "Failure"
    
    classifier.get_cgal_data_and_label_local_optomized()
    print(f"2) Finished: Generating CGAL segmentation for neuron: {time.time() - start_time}")
    

    #get the highest values of sdf
    start_time = time.time()
    print("3) Staring: Generating Graph Structure and Identifying Soma")
    soma_index = classifier.get_highest_sdf_part(size_multiplier*soma_size_threshold)
    print("soma_index = " + str(soma_index))

    #create a graph structure and stats for the whole neuron based on the classifications
    classifier.get_graph_structure()
    print(f"3) Finished: Generating Graph Structure and Identifying Soma: {time.time() - start_time}")


    #gets the caps of the somas created from segmenting into 4 clusters
    if clusters > 3:
        start_time = time.time()
        print("4) Staring: Finding Soma Extensions")
        classifier.find_Soma_Caps(soma_index,soma_cap_min_width,
                                  soma_cap_max_faces*size_multiplier,
                                  soma_cap_max_n_connections,
                                  large_extension_size=large_extension_size*size_multiplier,
                                  large_extension_convex_max=large_extension_convex_max
                                 )

        print(f"4) Finished: Finding Soma Extensions: {time.time() - start_time}")


    
    start_time = time.time()
    print("5) Staring: Finding Apical Index")
    #send data to function that will find the Apical
    possible_Apical = classifier.find_Apical_optimized(soma_index,apical_mesh_threshold*size_multiplier,
                                            apical_height_threshold,
                                            apical_sdf_threshold)
    print("possible_Apical = " + str(possible_Apical))
    print(f"5) Finished: Finding Apical Index: {time.time() - start_time}")

    #use the apical label and the soma label to classify the rest as basal or oblique and return a dictionary that has the mapping of label to compartment type
    #but only classifies the cgal labels and not each individual face
    start_time = time.time()
    print("6) Staring: Classifying Entire Neuron")
    classifier.classify_whole_neuron(possible_Apical,soma_index,
                                                classifier_cilia_threshold*size_multiplier,
                                                classifier_stub_threshold*size_multiplier,
                                                classifier_non_dendrite_convex_threshold,
                                                classifier_axon_std_dev_threshold,
                                                classifier_stub_threshold_apical*size_multiplier
                                               )
    
    #print unique list of labels found
    print("Total Labels found = " + str(set(classifier.whole_neuron_labels.values())))
    print(f"6) Finished: Classifying Entire Neuron: {time.time() - start_time}")


    #label the neurons according to classification
    #############NEED TO ADD STEP THAT CALCULATES THE LABELS OF THE VERTICES ##################
    start_time = time.time()
    print("7) Staring: Transfering Segmentation Labels to Face Labels")
    classifier.label_whole_neuron()
    print(f"7) Finished: Transfering Segmentation Labels to Face Labels: {time.time() - start_time}")



    #####need to map the final_faces_labels_list to all successive numbers and get vertices
    start_time = time.time()
    print("8) Staring: Generating final Vertex and Face Labels")
    output_faces_list, output_verts_list = classifier.generate_output_lists()
    print(f"8) Finished: Generating final Vertex and Face Labels: {time.time() - start_time}")

    if return_Only_Labels == True:
        return output_verts_list, output_faces_list
    
    
    start_time = time.time()
    print("9) Staring: Generating Returning Branches")
    dendritic_branches = classifier.return_branches(return_cilia,
                                                    return_soma,
                                                    return_axon,
                                                    return_error,
                                                    return_size_threshold*size_multiplier)
    print(f"9) Finished: Generating Returning Branches: {time.time() - start_time}")
    
    
    dendrites_segments = dendritic_branches.pop("dendrites",None)
    cilia_segments = dendritic_branches.pop("cilia",None)
    soma_segments = dendritic_branches.pop("soma",None)
    axon_segments = dendritic_branches.pop("axon",None)
    error_segments = dendritic_branches.pop("error",None)
                                      
    size_one = np.array(5).shape
    
    if dendrites_segments == None:
        dendrites_number = 0
    elif np.asarray(dendrites_segments).shape == size_one:
        dendrites_number = 1
    else:
        dendrites_number = len(dendrites_segments)
    print(f"Returning: \n{dendrites_number} dendritic branches")
    
    if return_cilia == True:
        if cilia_segments == None:
            cilia_number = 0
        elif np.asarray(cilia_segments).shape == size_one:
            cilia_number = 1
        else:
            cilia_number = len(cilia_segments)
        print(f" {cilia_number} cilia")
    if return_soma == True:
        if soma_segments == None:
            soma_number = 0
        elif np.asarray(soma_segments).shape == size_one:
            soma_number = 1
        else:
            soma_number = len(soma_segments)
        print(f" {soma_number} soma")
    if return_axon == True:
        if axon_segments == None:
            axon_number = 0
        elif np.asarray(axon_segments).shape == size_one:
            axon_number = 1
        else:
            axon_number = len(axon_segments)
        print(f" {axon_number} axon")
    if return_error == True:
        if error_segments == None:
            error_number = 0
        elif np.asarray(error_segments).shape == size_one:
            axon_number = 1
        else:
            error_number = len(error_segments)
        print(f" {error_number} errors")
    
          
    print(f"Total time: {time.time() - global_start }")

    dendritic_branches["dendrites"] =  dendrites_segments
    dendritic_branches["cilia"] =  cilia_segments
    dendritic_branches["soma"] =  soma_segments
    dendritic_branches["axon"] =  axon_segments
    dendritic_branches["error"] =  error_segments

    if clean_temp_files == True:
        classifier.clean_files()
    
    if return_cilia == False and return_soma == False and return_axon == False and return_error == False:
          return dendritic_branches["dendrites"]
    else:
          return dendritic_branches

        
"""
New additions adding: 
1) Can take in arrays of vertices and triangles instead of an off file: Done
2) Can take in optional argument to load CGAL data with the path to it: Done
3) Made pymeshfix optional: Done
4) Added option where can skip the outputting in trimesh objects and will just output the vertice and face labels
    using the return_Only_Labels flag

Need to investigate what these guys are also used for: 
self.mesh_file_location = mesh_file_location --> doesn't do anything else
self.file_name = file_name
    




"""
    
    