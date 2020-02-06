def get_split_heads_vp2(current_label,current_index, path,connections,shared_vertices,mesh_number,sdf_final_dict,absolute_head_threshold):
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
        
        if(next_label == "backbone"):
            #no_more_split_head_Flag = True
            break
        
        #ask if this next satisfies  1) enough shared verts?  2) SDF head possible?
        verts_sharing_index = connections[current_label].index(next_label)
        verts_sharing = shared_vertices[current_label][verts_sharing_index]
        
        #print("split share for faces " + str(current_label) + " " +str(next_label) + "="+str(verts_sharing/mesh_number[current_label]))
        sdf_guess = sdf_likely_category(next_label,next_index,path,True,sdf_final_dict,connections,mesh_number,absolute_head_threshold)
        if verts_sharing/mesh_number[current_label] > split_head_threshold and  sdf_guess == "head" and mesh_number[next_label] > split_head_absolute_threshold:
            #add next label to the list
            final_split_heads.append(next_label)
            current_index = next_index
            current_label = next_label
            
        else:
            heads_to_check = False
                     
    return final_split_heads      



def sdf_likely_category(current_label,current_index,path,head_flag,sdf_final_dict,connections,mesh_number,absolute_head_threshold):
    #width thresholding constants
    width_thresholds = {"base":0.04, "item_top_threshold":1.5} 
    #if size is smaller than the max threshold for a head then return neck
    if mesh_number[current_label] < absolute_head_threshold:
        return "neck"
    
    #get the mean, max, and median
    median_width = sdf_final_dict[current_label]
    
    #if the median is above a certain size and the total number of traingles is above a threshold then return as head
    """sdf_head_threshold = 50
    over_median_threshold  = 0.12
    if label_mesh_number > sdf_head_threshold and median > over_median_threshold:
        return "head"
    """
    
    
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
    if "backbone" in connections[current_label]:
        return "neck"
    

    ######check for head based on if there is significantly smaller neck underneath it (because can be very close to 0.04 cuttoff sometimes
    
    #get the mean, median and max
    
    #will return head or neck
    return "head"      

####For automatic spine labeling
def find_endpoints(G,mesh_number):
    #will first calculate all the shortest paths for each of the nodes
    
    node_list = list(G.nodes)
    if("backbone" in node_list):
        node_list.remove("backbone")
    else:
        return [],[] 
    
    shortest_paths = {}
    for node in node_list:
        shortest_paths[node] = [k for k in nx.all_shortest_paths(G,node,"backbone")]
    
    endpoints = []
    #identify the nodes that are not a subset of other nodes
    for node in node_list:
        other_nodes = [k for k in node_list if k != node ]
        not_unique = 0
        for path in shortest_paths[node]:
            not_unique_Flag = False
            for o_node in other_nodes:
                for o_shortest_path in shortest_paths[o_node]:
                    if set(path) <= set(o_shortest_path):
                        not_unique_Flag = True
                        
            if not_unique_Flag == True:
                not_unique = not_unique + 1
                
        #decide if unique endpoint
        if not_unique < len(shortest_paths[node]):   # this means there is a unique path
            
            #if not_unique != 0:
                #print(node + "-some unique and some non-unique paths for endpoint")
            endpoints.append(node)
        
    ##print(endpoints)  
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
        
    #print(longest_paths_list)
    longest_paths_list.sort(key=lambda pair: pair[1], reverse=True)
    #print(longest_paths_list)
    ranked_endpoints = [x for x,i in longest_paths_list]
    endpoint_paths_lengths = [i for x,i in longest_paths_list]
    
    enpoint_path_list = {}
    for endpt in ranked_endpoints:
        enpoint_path_list[endpt] = shortest_paths[endpt]
        
    
    #ranked_endpoints, longest_paths_list = (list(t) for t in zip(*sorted(zip(endpoints, longest_paths_list))))
    
    
    return ranked_endpoints, enpoint_path_list 
def classify_spine_vp2(connections,shared_vertices,mesh_number,sdf_final_dict):
    
    
    absolute_head_threshold = 30
    stub_threshold = 40
    path_threshold = 40
    
    

    #make a new dictionary to hold the final labels of the spine
    end_labels = {k:"none" for k in mesh_number.keys()}


    #only one segment so label it as a spine
    if len(connections.keys()) <= 1:
        end_labels[list(connections.keys())[0]] = "spine_one_seg"


    total_mesh_faces_outer = sum([k for i,k in mesh_number.items()])
    #print("total_mesh_faces = " + str( total_mesh_faces_outer))

    #create the graph from these
    G=nx.Graph(connections)

    endpoint_labels,shortest_paths = find_endpoints(G,mesh_number)
    
    if endpoint_labels == []:
        for jk in end_labels.keys():
            end_labels[jk] = "backbone"
            return end_labels

    #print("endpoint_labels = "+str(endpoint_labels))
    #print("shortest_paths = "+str(shortest_paths))

    #make a new dictionary to hold the final labels of the spine
    end_labels = {k:"none" for k in mesh_number.keys()}
    end_labels["backbone"] = "backbone"

    #print("end_labels at beginning")
    #print(end_labels)



    for endpoint in endpoint_labels:
        #print("at beginning of endpoint loop with label = "+ str(endpoint))
        #get the shortest path lists
        endpoint_short_paths = shortest_paths[endpoint]
        for path in endpoint_short_paths:
            path.remove("backbone")
            path_total_mesh_faces = sum([k for i,k in mesh_number.items() if i in path])
            #print("path_total_mesh_faces = "+str(path_total_mesh_faces))
            #print("at beginning of path loop with path = "+ str(path))
            travel_index = 0
            head_found = False
            label_everything_above_as_head = False
            while (head_found == False ) and travel_index < len(path):
                current_face = path[travel_index]
                sdf_guess = sdf_likely_category(current_face,travel_index,path,False,sdf_final_dict,connections,mesh_number,absolute_head_threshold)
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
            
            if (path[travel_index] == "backbone") or ("backbone" in connections[path[travel_index]]):
                head_found = False
                label_everything_above_as_head = True
            
            if path_total_mesh_faces<path_threshold:
                head_found = False
                label_everything_above_as_head = True
            
            
            ####do the head splitting####
            #see if there are any labels that border it that also share a high percentage of faces
            if head_found == True:
                ##will return the names of the faces that have unusually high verts sharing
                split_head_labels = get_split_heads_vp2(path[travel_index],travel_index,path,connections,shared_vertices,mesh_number,sdf_final_dict,absolute_head_threshold)
                #print("split_head_labels = " + str(split_head_labels))


                if len(split_head_labels) >= 2:
                    #print("adding the split head labels")
                    for split_label in split_head_labels:
                        #######may need to add in CHECK FOR ALREADY LABELED
                        if ("head" == end_labels[split_label][0:4] or end_labels[split_label] == "none"):
                            end_labels[split_label] = "head_split"
                        #else:      THINK LABELING IT AS SPINE IS NOT WHAT WE WANT
                        #    end_labels[split_label] = "spine_head_disagree_split_head"

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
            
    
    
    end_labels["backbone"] = "backbone"

    ###To Do: replace where look only in 1st four indexes
    return end_labels



def export_connection(labels_list,label_name, verts_to_Face,outputFlag="False",file_name="None"):
    
    #print("hello from export_connection with label_name = " + str(label_name) )
    #find all the neighbors of the label
    
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    #print("object_name = " + bpy.context.object.name)
    me = ob.data
    
    faces_raw = me.polygons
    verts_raw = me.vertices
    
    #print("generating list in export connections")
    #labels_list = generate_labels_list(faces_raw)
    #print("done generating list in export connections")
    
        
    #need to assemble a dictionary that relates vertices to faces
    #*****making into a list if the speed is too slow*******#
    #print("about to making verts_to_Face")
    #verts_to_Face = generate_verts_to_face_dictionary(faces_raw,verts_raw)
    #print("DONE about to making verts_to_Face")
    
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
            if facey != "backbone":
                neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,facey,verts_to_Face,faces_raw,verts_raw)
                
                
                
                #reduce the shared vertices with a face and the backbone to 0 so doesn't mess up the shared vertices percentage
                pairs = list(neighbors_shared_vert.items())
                pre_connections = [k for k,i in pairs]
                pre_shared_vertices = [i for k,i in pairs]
                
                
                
                
                if ("backbone" in pre_connections):
                    back_index = pre_connections.index("backbone")
                    pre_shared_vertices[back_index] = 0
         
                
                connections[facey] = pre_connections
                shared_vertices[facey] = pre_shared_vertices
                mesh_number[facey] = number_of_faces

                
                for neighbors in neighbors_list:
                    if (neighbors != "backbone") and (neighbors not in faces_to_check) and (neighbors not in faces_checked):
                        faces_to_check.append(neighbors)
                
                faces_to_check.remove(facey)
                faces_checked.append(facey)
        
        #append the backbone to the graph structure
        mesh_number["backbone"] = 0
    
    #print("faces_checked = " + str(faces_checked))
    #print("DONE about to start checking faces")
    
    #save off the file to an npz file
    
    
    if(outputFlag == True):
        complete_path = str("/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/Automatic_Labelers/spine_graphs/"+file_name)
        
        
        
        #package up the data that would go to the database and save it locally name of the file will look something like this "4_bcelii_2018-10-01_12-12-34"
    #    np.savez("/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/local_neurons_saved/"+segment_ID+"_"+author+"_"+
    #        date_time[0:9]+"_"+date_time[11:].replace(":","-")+".npz",segment_ID=segment_ID,author=author,
    #					date_time=date_time,vertices=vertices,triangles=triangles,edges=edges,status=status)
        np.savez(complete_path,connections=connections,shared_vertices=shared_vertices,mesh_number=mesh_number ) 
    
    return connections,shared_vertices,mesh_number
   
def relabel_segments(labels_list,current_label,new_label):
    for i,x in enumerate(labels_list):
        if x == current_label:
            labels_list[i] = new_label
            
    return labels_list

def automatic_spine_classification_vp3(labels_list,verts_to_Face,sdf_final_dict):
    
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
    
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    #print("object_name = " + bpy.context.object.name)
    me = ob.data
    
    faces_raw = me.polygons
    verts_raw = me.vertices
    
    #labels_list = generate_labels_list(faces_raw)
    
    #making a copy of the total segmented labels (label list is from cgal), 
    #but now they have the backbone labels as the label if it was changed in the smoothing backbone
    final_spine_labels = labels_list.copy()
    
    processed_labels = []
    
    myCounter = Counter(labels_list)
    complete_labels =  [label for label,times in myCounter.items()] #OPTOMIZE BY USING KEYS
    
    head_counter = 0
    spine_counter = 0
    neck_counter = 0
    stub_counter = 0
    for i in range(0,len(complete_labels)):
        if complete_labels[i] != "backbone" and complete_labels[i] not in processed_labels:
            
            #get the conenections, shared vertices and mesh sizes for the whole spine segment in which label is connected to
            connections,shared_vertices,mesh_number = export_connection(labels_list,complete_labels[i], verts_to_Face,outputFlag="False",file_name="None")
            
            #send that graph data to the spine classifier to get labels for that
            final_labels = classify_spine_vp2(connections,shared_vertices,mesh_number,sdf_final_dict)
            
            head_Flag = False
            spine_Flag = False
            stub_Flag = False
            neck_Flag = False
            #relabel the list accordingly
            ############could speed this up where they return the number of types of labels instead of having to search for them############
            #print("about to find number of heads/spines/stubs/necks PLUS RELABEL AND append them to list")
            for key,value in final_labels.items():
                if value[0:4] == "head":
                    head_Flag = True
                if value[0:4] == "spin":
                    spine_Flag = True
                if value[0:4] == "stub":
                    stub_Flag = True
                if value[0:4] == "neck":
                    neck_Flag = True
                
                
                
                relabel_segments(final_spine_labels,key,value)
                #add them to the list of processed labels
                processed_labels.append(key)
            #print("about to find number of heads/spines/stubs/necks PLUS RELABEL AND append them to list")
                
            if head_Flag == True:
                head_counter += 1
            if spine_Flag == True:
                spine_counter += 1
            if stub_Flag == True:
                stub_counter += 1
            if neck_Flag == True:
                neck_counter += 1
            
            
   
    
    #get the indexes for the labeling from the datajoint table
    label_data = ta3p100.LabelKey().fetch("numeric","description")
    #print(label_data)

    label_names = label_data[1].tolist()
    label_indexes = label_data[0].tolist()
    #print(label_names)

    spine_head_index = label_indexes[label_names.index("Spine Head")]
    spine_neck_index = label_indexes[label_names.index("Spine Neck")]
    spine_reg_index = label_indexes[label_names.index("Spine")]

    
    final_faces_labels_list = np.zeros(len(faces_raw))
    final_verts_labels_list = np.zeros(len(verts_raw))

    
    #assign the labels to the correct faces
    for i,fi in enumerate(final_spine_labels):
        if fi[0:4] == "head":
            #fac.material_index = 2
            final_faces_labels_list[i] = spine_head_index
        elif fi[0:4] == "neck":
            #fac.material_index = 3
            final_faces_labels_list[i] = spine_neck_index
        elif fi[0:4] == "spin":
            #fac.material_index = 4
            final_faces_labels_list[i] = spine_reg_index
        else:
            #fac.material_index = 0
            final_faces_labels_list[i] = 0
            
        #assign the vertices an index
        for vert in faces_raw[i].vertices:
            if final_verts_labels_list[vert] == 0:
                final_verts_labels_list[vert] = final_faces_labels_list[i]
     
    
    return head_counter,neck_counter, spine_counter, stub_counter, final_verts_labels_list, final_faces_labels_list

#function call that will retrieve the spine head and neck labels
head_counter,neck_counter, spine_counter, stub_counter,final_verts_labels_list, final_faces_labels_list = automatic_spine_classification_vp3(labels_list,verts_to_Face,sdf_final_dict)
print("classifying spine--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("head_counter = " + str(head_counter))
print("neck_counter = " + str(neck_counter))
print("spine_counter = " + str(spine_counter))
print("stub_counter = " + str(stub_counter))

