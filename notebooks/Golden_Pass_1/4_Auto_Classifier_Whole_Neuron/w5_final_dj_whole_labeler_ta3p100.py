import bpy
import datajoint as dj
import numpy as np
import datetime
import math
from mathutils import Vector
from Auto_Proofreader_Tab_without_fast_label import create_local_colors , set_View, create_bounding_box
import time
from collections import Counter
import os


#---------------------------DATAJOINT SOLUTION WAY OF GETTING COLORS---------------------#

def get_diffuse_colors_list():
    diff_list = ta3.LabelKey2().fetch("blender_colors")
    diffuse_colors_list = []
    for df in diff_list:
        diffuse_colors_list.append(tuple(df))

    return diffuse_colors_list

def get_Bevel_Weights():
    num_list = ta3.LabelKey2().fetch("numeric")
    num_list[1] = 0
    final_bevel_weights = []
    for n in num_list:
        final_bevel_weights.append(np.round(float(n)/100,2))
    return final_bevel_weights

def getLabels():
    description_list = ta3.LabelKey2().fetch("description")
    description_list[0] = "None"
    description_list[1] = "None"
    final_description_list = []
    for n in description_list:
        final_description_list.append(n)
    return final_description_list

def getColors():
    description_list,color_list = ta3.LabelKey2().fetch("description","color")
    description_list = description_list.tolist()
    color_list = color_list.tolist()
    final_color_list = []
    final_color_list.append(color_list.pop(0))
    final_color_list.append(color_list.pop(0))
    description_list.pop(0)
    description_list.pop(0)
    for i in range(0,len(description_list)):
        final_color_list.append(description_list[i] + " (" + color_list[i] +")")
    
    return final_color_list

def accepted_color_length():
    numeric_list = ta3.LabelKey2().fetch("numeric")
    return len(numeric_list)

#------------------------------------------------#


def create_global_colors():  

    
    #get the list of colors already available in the bpy.data 
    list = bpy.data.materials.keys()  
    
    #adds all of the colors to the master list in blender file if they are not there already
    colors = getColors()
    #print(colors)
    """diffuse_colors_list = [(0.800, 0.800, 0.800),(0.800, 0.800, 0.800),
    (0.0, 0.0, 0.800),(0.800, 0.800, 0.0), #blue, "yellow
    (0.0, 0.800, 0.0),(0.800, 0.0, 0.0),   #green, red
    (0.0, 0.800, 0.527),(0.049, 0.458, 0.800),(0.200, 0.0, 0.800), #aqua, off blue, purple
    (0.800, 0.0, 0.400),(0.250, 0.120, 0.059), #pink, brown
    (0.800, 0.379, 0.232),(0.144,0.193,0.250),(0.800, 0.019, 0.093), #tan, soft, blue, rose
    (0.800, 0.486, 0.459),(0.309,0.689,0.170),(0.800, 0.181, 0.013)] #light_pink, #light green orange"""
    
    diffuse_colors_list = get_diffuse_colors_list()
    
    
    for i in range(0,len(colors)):
        if not(colors[i] in list):
            mat = bpy.data.materials.new(name=colors[i]);
            mat.diffuse_color = diffuse_colors_list[i]
        else:  #if it already exists make sure to set colors list right
            bpy.data.materials[colors[i]].diffuse_color = diffuse_colors_list[i]

def create_local_colors(ob=bpy.context.object):              
    #make sure all of the global colors are set correctly
    create_global_colors()
    #get current object
    #ob = bpy.context.object
    
    #get the colors list
    colors = getColors()
    
    #makes sure that length of color list matches the number of labels/colrs needed + 1
    if(ob.data != None):
        difference = len(ob.data.materials) - len(colors)
    else:
        print("materials was none")
        difference = -len(colors)
    
    #if less than 6 colors already then add the spots there
    if(difference < 0):
        for i in range(0,-difference):
            ob.data.materials.append(None)
            
    #print(len(ob.data.materials))
    
    #make sure the colors are in the correct order for the object
    
    for i in range(0,len(colors)):
        ob.data.materials[i] = bpy.data.materials[colors[i]]



def load_Neuron_automatic_spine(neuron_ID,decimation_ratio,clusters,smoothness):
    ID = str(neuron_ID)
    print("inside load Neuron")
 
    #neuron_data = ((mesh_Table & "segment_ID="+ID).fetch(as_dict=True))[0]
    primary_key = dict(segmentation=2,decimation_ratio=decimation_ratio)
    neuron_data = ((ta3p100.CleansedMesh & primary_key & "segment_ID="+str(ID)).fetch(as_dict=True))[0]


    
    verts = neuron_data['vertices'].astype(dtype=np.int32).tolist()
    faces = neuron_data['triangles'].astype(dtype=np.uint32).tolist()
    
    mymesh = bpy.data.meshes.new("neuron-"+ID + "_" + str(decimation_ratio) + "_" + str(clusters) + "_" + str(smoothness))
    mymesh.from_pydata(verts, [], faces)
 
    mymesh.update(calc_edges=True)
    mymesh.calc_normals()

    object = bpy.data.objects.new("neuron-"+ID + "_" + str(decimation_ratio) + "_" + str(clusters) + "_" + str(smoothness), mymesh)
    #object.location = bpy.context.scene.cursor_location
    object.location = Vector((0,0,0))
    bpy.context.scene.objects.link(object)
    
    object.lock_location[0] = True
    object.lock_location[1] = True
    object.lock_location[2] = True
    object.lock_scale[0] = True
    object.lock_scale[1] = True
    object.lock_scale[2] = True

    object.rotation_euler[0] = 4.7124
    object.rotation_euler[1] = 0
    object.rotation_euler[2] = 0

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(rotation=True)
    bpy.ops.object.select_all(action='DESELECT')
    

    object.lock_rotation[0] = True
    object.lock_rotation[1] = True
    object.lock_rotation[2] = True


    #set view back to normal:
    set_View()

    #run the setup color command
    #bpy.ops.object.select_all(action='TOGGLE')
    
    #create_local_colors(object)

    #make sure in solid mode
    for area in bpy.context.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.viewport_shade = 'SOLID' # set the viewport shading to rendered
    
    return object.name

def select_Neuron():
    # deselect all
    bpy.ops.object.select_all(action='DESELECT')

    # selection
    for obj in bpy.data.objects:
        if "neuron" in obj.name:
            obj.select = True
            bpy.context.scene.objects.active = obj
            print("object was found and active")
            break
        
        
import random

def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [round(random.random(),3) for i in range(3)]
    return (r, g, b)


def segment_random_colors(ob,unique_segments,extraFlag):
    errorFlag = 0
    color_keys = bpy.data.materials.keys()
    '''for i in range(0,len(bpy.data.materials.keys())-16):
        print(" material " + color_keys[i])
        bpy.data.materials.remove(bpy.data.materials[color_keys[i]])'''
    
    #makes sure that nothing but the accepted colors are there    
    accepted_colors = getColors()
    #print("inside segment_random_colors and getColors() = " + str(getColors()))
    
    #ob.user_clear()
    
    for i in range(0,len(bpy.data.materials.keys())):
        if(color_keys[i] not in accepted_colors):
            #print("deleting material " + color_keys[i])
            bpy.data.materials[color_keys[i]].user_clear()
            bpy.data.materials.remove(bpy.data.materials[color_keys[i]])
    
    
    
    if errorFlag == 1:
        for i in range(0,len(unique_segments)-1):
            #add the new color
            mat = bpy.data.materials.new(name=str(i));
            mat.diffuse_color = get_random_color()
            #assign it to the object
            ob.data.materials.append(mat)
    
        #add one more label so that the unlabelables get a label
        mat = bpy.data.materials.new("errors")
        mat.diffuse_color = get_random_color()
        ob.data.materials.append(mat)
    else:
        """mat = bpy.data.materials.new(name=str("base"))
        mat.diffuse_color = [0.800,0.800,0.800]
        if(len(ob.data.materials.keys()) > 0):
            ob.data.materials[0] = mat
        else:
            ob.data.materials.append(mat)"""
            
        for i in range(0,len(unique_segments)):
            #add the new color
            mat = bpy.data.materials.new(name=str(i));
            mat.diffuse_color = get_random_color()
            #assign it to the object
            ob.data.materials.append(mat)
        if extraFlag == True:
            mat = bpy.data.materials.new(name=str(len(unique_segments)+1));
            mat.diffuse_color = get_random_color()
            #assign it to the object
            ob.data.materials.append(mat)

def get_cgal_data_and_label(neuron_ID, decimation_ratio, clusters,smoothness):
       
    #store the group_segmentation in the traingle labels from datajoint
    
    
    #clusters=clusters,smoothness=smoothness
    comp_dict = dict(segmentation=2,
                            segment_id=neuron_ID,
                            decimation_ratio=decimation_ratio,
                            clusters=clusters,
                            smoothness=smoothness
                            )
                            
    component_data = (ta3p100.ComponentAutoSegmentWhole() & [comp_dict]).fetch(as_dict=True)[0]
    triangles_labels = component_data["seg_group"].tolist()
    #activate the current object
    select_Neuron()
    ob = bpy.context.object
    
    
    me = ob.data
    
    #print("starting to hide everything")
    #iterate through all of the vertices
    verts_raw = ob.data.vertices
    #print(len(active_verts_raw))
    
    edges_raw = ob.data.edges
    
    #print(len(active_edges_raw))
    
    faces_raw = ob.data.polygons
    
    #gets a list of the unique labels
    unique_segments = list(Counter(triangles_labels).keys())
    
    
    segmentation_length = len(unique_segments) # equals to list(set(words))
    #print(segmentation_length)

    #makes a dictionary that maps the unique segments to a number from range(0,len(unique_seg))
    unique_index_dict = {unique_segments[x]:x for x in range(0,segmentation_length)}
    
    
    #print("unique_index_dict = " + str(len(unique_index_dict)))
    #print("triangle_labels = " + str(len(triangles_labels)))
    #adds all of the labels to the faces
    max_length = len(triangles_labels)
    
    #just iterate and add them to the faces
    #here is where need to get stats for sdf numbers
    
    
    labels_list = []
    labels_list_index = []
    for i,k in enumerate(faces_raw):
        #gives the material id for that face
        k.material_index = int(unique_index_dict[triangles_labels[i]]) 
        #assembles the label list that represents all of the faces
        labels_list.append(str(unique_index_dict[triangles_labels[i]])) 
        #labels_list_index.append(
    
    select_Neuron()
    
    
    #make sure in solid mode
    for area in bpy.context.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.viewport_shade = 'SOLID' # set the viewport shading to rendered
    
    bpy.ops.object.mode_set(mode='OBJECT')
    


    ######--------setting up the colors -------------###########
    ###assign the color to the object
    current_Material_length = len(bpy.data.materials)
    
    #remove all of the previous numbered colors
    color_keys = bpy.data.materials.keys()

    
    #generate new colors for the bpy.data global

    segment_random_colors(ob=ob,unique_segments=unique_segments,extraFlag=False)
    
    #these variables are set in order to keep the functions the same as FINAL_importing_auto_seg.py
    newname = ob.name
    print("done with cgal_segmentation")
    
    #----------------------now return a dictionary of the sdf values like in the older function get_sdf_dictionary
    #get the sdf values and store in sdf_labels
    sdf_labels = component_data["sdf"].tolist()
        
    sdf_temp_dict = {}
    labels_seen = []
    #iterate through the labels_list
    for i,label in enumerate(labels_list):
        if label not in labels_seen:
            labels_seen.append(label)
            sdf_temp_dict[label] = []
        
        sdf_temp_dict[label].append(sdf_labels[i])
    #print(sdf_temp_dict)
    
    #now calculate the stats on the sdf values for each label
    sdf_final_dict = {}
    for key,value in sdf_temp_dict.items():
        """
        #calculate the average
        mean = np.mean(value)
        #calculate the median
        median = np.median(value)
        #calculate the max
        max = np.amax(value)
        #calculate minimum
        min = np.amin(value)
        
        temp_dict = {"mean":mean,"median":median,"max":max,"min":min}
        
        #assign them 
        sdf_final_dict[key] = temp_dict.copy()
        """
        
        #just want to store the median
        sdf_final_dict[key] = dict(median=np.median(value),mean=np.mean(value),max=np.amax(value))

    return sdf_final_dict, labels_list

def get_highest_sdf_part(sdf_final_dict, labels_list,size_threshold=3000,exclude_label=None):
    high_median_val = 0
    high_median = -1
    high_mean_val = 0
    high_mean = -1
    high_max_val = 0
    high_max = -1
    
    
    
    
    my_list = Counter(labels_list)
    my_list_keys = list(my_list.keys())
    if exclude_label != None:
        my_list_keys.remove(exclude_label)
    
    for x in my_list_keys:
        #print("x = " + str(x))
        #print("high_median_val = " + str(high_median_val))
        #print('sdf_final_dict[x]["median"] = ' + str(sdf_final_dict[x]["median"]))
        if sdf_final_dict[x]["median"] > high_median_val and my_list[x] > size_threshold:
            high_median = x
            high_median_val = sdf_final_dict[x]["median"]
        if sdf_final_dict[x]["mean"] > high_mean_val  and my_list[x] > size_threshold:
            high_mean = x
            high_mean_val = sdf_final_dict[x]["mean"]
        if sdf_final_dict[x]["max"] > high_max_val  and my_list[x] > size_threshold:
            high_max = x
            high_max_val = sdf_final_dict[x]["max"]
    
    
    return high_median,high_median_val,high_mean,high_mean_val,high_max,high_max_val

#export the neuron
def export_neuron(ob_name,destination_folder="whole_neuron_testing"):
    
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)
    #destination_folder = "whole_neuron_testing"
    final_directory = Path(directory) / destination_folder / (ob_name + ".obj")
    #target_file = os.path.join(final_directory, ob_name + ".obj")

    bpy.ops.export_scene.obj(filepath=final_directory.as_posix())
    return
    
#def generate_label_list(path_to_check=""):
def generate_label_list():
    
    #have to go into object mode to do some editing
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    me = ob.data
    faces_raw = me.polygons
    verts_raw = me.vertices 
    
    labels_list = [fac.material_index for fac in faces_raw]
    return labels_list
    
    """
    if path_to_check != "":
        try:
            label_data = np.load(path_to_check + ob.name + ".npz")
        except:
            print("having to generate the list myself")
            #have to generate the labels list manually
            
        else:
            print("loading the labels list from a file")
            labels_list = label_data["labels_list"]
            return labels_list
    """
       

def rewrite_label(i,x,number_to_replace,verts_to_Face,faces_raw,verts_raw,big_labels,soma_index,cilia_threshold=300):
    #won't reassign the same big label to all of those with the same little label
    assign_All_Same = True
   
    #print("x[i] =" + str(x[i]))
    #print(len(x))
    #print(x)
    
    remaining_Faces = []
    
    """for i,j in enumerate(x):
        if j == x[i]:
            print(str(i) + ":" + str(j))
            #print(j)
            remaining_Faces.append(i)"""
            
    remaining_Faces = [z for z,j in enumerate(x) if j == x[i]]
    
    
    #print("remaining_Faces len = " + str(len(remaining_Faces)))
    #print("number to replace = " + str(number_to_replace))
    
    if number_to_replace != len(remaining_Faces):
        raise ValueError("ERROR: faces to replace don't match the number of indexes found for that face")
        return []
    
    #0) Initialize replacement label
    #1) Find the face to relabel
    #2) ADD THIS FACE TO THE CHECKED FACES
    #2) FIND THE VERTICES ASSOCIATED WITH THIS FACE AND ADD TO VERTS TO CHECK
    #2) START another while loop that only breaks when there has been a replacement label found
    #4) For each vertices IN VERTS TO CHECK, find the faces associated with each, and add it to the FACES TO CHECK (IF IT HAS NOT ALREADY BEEN CHECKED)
        #add the vertices to checked verts so we don't end up redoing them
    #5) FOR each of the faces to check, find its label, and see if the label is in the big list
    #6)     If is in big list --> save the replacement label and break from loop
    #7)     If not --> add vertices associated with the face(that have not already been checked) to the verts to check list
    #OUTSIDE OF #2 WHILE LOOP
    #8) if the assign_All_Same flag
    #           is set to true --> assign the label to all of the faces with the same label to be replaced and pop all of them
    #           is false --> only assign the new label to the one being checked
    
    
    
    
    while remaining_Faces:
        counter = 0
        replacement_label = "k"
        face_to_relabel = remaining_Faces[0]
        
        #print("face_to_relabel = " + str(face_to_relabel))
        
        checked_Faces = []
        checked_Verts = []
        verts_to_check = []
        faces_to_check = []
        
        checked_Faces.append(face_to_relabel)
        
        #get the vertices of the face
        face_vertices = faces_raw[face_to_relabel].vertices
        for vertex in face_vertices:
            verts_to_check.append(vertex)
        
        #loop that goes until replacement label is found
        while replacement_label == "k":
            counter = counter + 1
            if counter > 1000:
                print("didn't find big face yet, going through pass: " + str(counter))
            #iterates through each vertex to get all of the faces to check
            for vertices_checking in verts_to_check:
                #gets all of the faces associated with the vertex
                faces_with_vertex = verts_to_Face[vertices_checking]
                
                #puts all the faces to check into a list
                for fc in faces_with_vertex:
                    if (fc not in faces_to_check) and (fc not in checked_Faces):
                        faces_to_check.append(fc)
                if(vertices_checking not in checked_Verts):
                    checked_Verts.append(vertices_checking)
            
            verts_to_check = []
            
            
            
            #iterate through all faces to find until a label from the big list is found
            for face in faces_to_check:
                if x[face] in big_labels:
                    replacement_label = x[face]
                    #print("found replacement label = " + str(replacement_label))
                    break
                else:
                    checked_Faces.append(face)
                    #add their vertices to the needs to be checked list if they haven't already been added
                    for vertex in faces_raw[face].vertices:
                        if (vertex not in checked_Verts) and (vertex not in verts_to_check):
                            verts_to_check.append(vertex)
            
            
                    
            
            faces_to_check = []
        #print(" counter = " + str(counter))
        #print("replacement_label = " + str(replacement_label))
        counter = 0
        #replace that faces label with the new label
        if soma_index == "-1":
            x[face_to_relabel] = replacement_label
            remaining_Faces.pop(0)
            
            if assign_All_Same != False:
                #print("assigning all the same")
                for faces in remaining_Faces:
                    x[faces] = replacement_label   
            
                remaining_Faces = []
        else:
            if replacement_label == soma_index and number_to_replace > cilia_threshold:
                #so don't do any replacing
                remaining_Faces = []
            else:
                x[face_to_relabel] = replacement_label
                remaining_Faces.pop(0)
                
                if assign_All_Same != False:
                    #print("assigning all the same")
                    for faces in remaining_Faces:
                        x[faces] = replacement_label   
                
                    remaining_Faces = []        
    
    #####need to add in a part that watches out for the soma so we don't smooth away the cilia
    
    
    
        
    return x

##x is the labels list
##threshold is how big in size the labels need to be in order to qualify as a big label
##number_Flag: will make it so the number of items left after the smoothing is only a certain number
##seg_numbers: if the number_Flag is true, will make sure smoothing only leaves (seg_numbers) amount of colors at the end


####can possibly change to make faster
def merge_labels(x,ob_name,threshold=50,soma_index=-1,cilia_threshold=300,number_Flag = False, seg_numbers=1):
    #make sure to select the correct base object
    for obj in bpy.data.objects:
        obj.select = False
    
    ob = bpy.data.objects[ob_name]
    ob.select = True
    bpy.context.scene.objects.active = ob
    
    #have to go into object mode to do some editing
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    me = ob.data
    faces_raw = me.polygons
    verts_raw = me.vertices
    
    #create a list of all the labels and which ones are the biggest ones
    from collections import Counter
    
    myCounter = Counter(x)

    big_labels = []
    for label,times in myCounter.items():
        if(times >= threshold):
            #print(str(label) + ":" + str(times))
            big_labels.append([label,times])
    
    #big labels has the index and the number of times it occurs in a list together
    big_labels.sort(key=lambda tup: tup[1], reverse=True)
    #At this point have all of the big labels we want to smooth our neuron to
    print("BIG LABELS = " + str(big_labels))
    
    #reduce the number of items in the list if the number_Flag is set
    if(number_Flag == True):
        big_labels = big_labels[:seg_numbers]
        
    
    
    #need to assemble a dictionary that relates vertices to faces
    #*****making into a list if the speed is too slow*******#
    
    verts_to_Face = {}
    
    #initialize the lookup dictionary as empty lists
    for pre_vertex in verts_raw:
        verts_to_Face[pre_vertex.index] = []
        
    #print(len(verts_raw))
    #print(len(verts_to_Face))
    #print(verts_to_Face[1])
    
    for face in faces_raw:
        #get the vertices
        verts = face.vertices
        #add the index to the list for each of the vertices
        for vertex in verts:
            verts_to_Face[vertex].append(face.index)
            
    
    
    
    
    #just extracts the labels of the big labels (because previously had the labels and number of occurances packaged together)
    big_labels_indexes = [x for x,i in big_labels]
    print("big_labels_indexes="+str(big_labels_indexes))
    
    
    #now need to change the labels
    for i in range(len(x)):
        if x[i] not in big_labels_indexes:
            #print("working on label " + str(x[i]))
            #print("myCounter = " + str(myCounter))
            number_to_replace = myCounter[x[i]]
            #print("number_to_replace = " + str(number_to_replace))
            #i: the index of the label that we want to replace
            #x: the entire list of labels
            #number_to_replace: the number of labels with that specific label that we want to replace
            #verts_to_Face: the lookup dictionary mapping vertices to faces
            #faces_raw/verts_raw: pointers to faces of object
            #big_labels_indexes: the list of possible big labels that we want to rewrite the smaller label as 
            x = rewrite_label(i,x,number_to_replace,verts_to_Face,faces_raw,verts_raw,big_labels_indexes,soma_index,cilia_threshold)
    
    for jj,fac in enumerate(faces_raw):
        fac.material_index = ob.data.materials.keys().index(x[jj])
    
    #need to redo the colors of the object
    from collections import Counter
    
    myCounter = Counter(x)

    
    big_labels = []
    for label,times in myCounter.items():
        if(times >= threshold):
            #print(str(label) + ":" + str(times))
            big_labels.append([label,times])
    
    return x, verts_to_Face




def rewrite_label_vp2(labels_list,small_label,connections, mesh_Number,faces_raw,verts_raw,big_labels,soma_index,cilia_threshold=300):
    #won't reassign the same big label to all of those with the same little label
    assign_All_Same = True
    
    
    replacement_label = "k"
    neighbors_seen = [small_label]
    neighbors_to_check = []
    neighbors_checked = []
    current_label = small_label
    
    #print("small_label in rewrite_label = " + str(small_label))
    
    while replacement_label == "k":
        #Get neighbors of current label using new function
        current_neighbors = connections[current_label]
        
        neighbors_seen = list(set(neighbors_seen + current_neighbors))
        for cn in current_neighbors:
            if cn in big_labels:
                replacement_label = cn
                break
        
        if replacement_label != "k":
            break
        
        
        neighbors_checked.append(current_label)
        neighbors_to_check = list(set(neighbors_to_check + [neigh for neigh in current_neighbors if neigh not in neighbors_checked]))
        
        current_label = neighbors_to_check.pop(0)
    
    if replacement_label == "k":
        print("ERROR SOMETHING WENT WRONG AND NO REPLACEMENT LABEL FOUND")
    
    new_ignore_list = []
    #have the replacement label:
    if soma_index == "-1":
        for i,lab in enumerate(labels_list):
            if lab in neighbors_seen and lab not in big_labels:
                labels_list[i] = replacement_label
    else:
        #get a list of the neighbors sizes that are not part of big_labels
        size_lengths_exceeds = [mesh_Number[labelz]>cilia_threshold for labelz in neighbors_seen if labelz not in big_labels]
        
        
        if replacement_label == soma_index and True in size_lengths_exceeds:
            #label them as cilia
            print("found cilia in " + str(neighbors_seen))
            new_ignore_list = [lb for lb in neighbors_seen if lb not in big_labels]
            """for i,lab in enumerate(labels_list):
                if lab in neighbors_seen and lab not in big_labels:
                    labels_list[i] = "Cilia"""
        else:
            for i,lab in enumerate(labels_list):
                if lab in neighbors_seen and lab not in big_labels:
                    labels_list[i] = replacement_label
                    
    return labels_list,new_ignore_list

####can possibly change to make faster
def merge_labels_vp2(labels_list,ob_name,threshold=50,soma_index=-1,cilia_threshold=80):
    #make sure to select the correct base object
    for obj in bpy.data.objects:
        obj.select = False
    
    ob = bpy.data.objects[ob_name]
    ob.select = True
    bpy.context.scene.objects.active = ob
    
    #have to go into object mode to do some editing
    bpy.ops.object.mode_set(mode='OBJECT')
    ob.update_from_editmode()
    
    me = ob.data
    faces_raw = me.polygons
    verts_raw = me.vertices
    
    #create a list of all the labels and which ones are the biggest ones
    from collections import Counter
    
    myCounter = Counter(labels_list)

    big_labels = [label_name for label_name,times in myCounter.items() if times > threshold]
    
    #At this point have all of the big labels we want to smooth our neuron to
    #print("BIG LABELS = " + str(big_labels))
   
    #need to assemble a dictionary that relates vertices to faces
    #*****making into a list if the speed is too slow*******#
    verts_to_Face,verts_to_Label = generate_verts_to_face_dictionary(labels_list,faces_raw,verts_raw)
    
    ####get the list of connections
    connections, mesh_Number = get_graph_structure(verts_to_Label,labels_list,faces_raw,verts_raw)
    ignore_list = []
    print("about to start merging")
    #now need to change the labels
    for i in range(len(labels_list)):
        if labels_list[i] not in big_labels and labels_list[i] not in ignore_list:
            labels_list,new_ignore_list = rewrite_label_vp2(labels_list,labels_list[i],connections, mesh_Number,faces_raw,verts_raw,big_labels,soma_index,cilia_threshold)
            ignore_list = ignore_list + new_ignore_list
            
    print("DONE merging")
    """#need to add cilia to materials
    cilia_color = (0.450,0.254,0.800)
    mat = bpy.data.materials.new(name="Cilia");
    mat.diffuse_color = cilia_color
    #assign it to the object
    ob.data.materials.append(mat)"""
    
    for jj,fac in enumerate(faces_raw):
        fac.material_index = ob.data.materials.keys().index(labels_list[jj])
    
    return labels_list,verts_to_Face




from collections import Counter

def generate_verts_to_face_dictionary(labels_list,faces_raw,verts_raw):
    verts_to_Face = {pre_vertex.index:[] for pre_vertex in verts_raw}
    verts_to_Label = {pre_vertex.index:[] for pre_vertex in verts_raw}
    
    
    for face in faces_raw:
        #get the vertices
        verts = face.vertices
        #add the index to the list for each of the vertices
        for vertex in verts:
            verts_to_Face[vertex].append(face.index)
    
    #use the verts to face to create the verts to label dictionary
    for vert,face_list in verts_to_Face.items():
        diff_labels = [labels_list[fc] for fc in face_list]
        #print(list(set(diff_labels)))
        verts_to_Label[vert] = list(set(diff_labels))
    
        
            
    return verts_to_Face,verts_to_Label


def get_graph_structure(verts_to_Label,labels_list,faces_raw,verts_raw):
    connections = {label_name:[] for label_name in Counter(labels_list).keys()}
    mesh_Number = {label_name:number for label_name,number in Counter(labels_list).items()}
    #label_vert_stats = {label_name:[300000,-300000] for label_name in Counter(labels_list).keys()}
    
    for verts,total_labels in verts_to_Label.items():
        if len(total_labels) > 1:
            for face in total_labels:
                for fc in [v for v in total_labels if v != face]:
                    if fc not in connections[face]:
                        connections[face].append(fc)
        """#get the verts stats:
        real_vert = verts_raw[verts]
        if real_vert.co[2] < label_vert_stats[verts][0]:
            label_vert_stats[verts][0] = real_vert.co[2]
        if real_vert.co[2] > label_vert_stats[verts][1]:
            label_vert_stats[verts][1] = real_vert.co[2]"""
                        
    
    return connections, mesh_Number#,label_vert_stats
    
def find_max_min_z_vals(neighbors_list,labels_list,faces_raw,verts_raw):
    min_max = {ni:[300000,-300000] for ni in neighbors_list}
    
    for i,ll in enumerate(labels_list):
        if ll in neighbors_list:
            verts_from_faces = faces_raw[i].vertices
            for vert in verts_from_faces:
                real_vert = verts_raw[vert]
                if real_vert.co[2] < min_max[ll][0]:
                    min_max[ll][0] = real_vert.co[2]
                if real_vert.co[2] > min_max[ll][1]:
                    min_max[ll][1] = real_vert.co[2]
    return min_max


"""#stats found in research:
number of faces = 44596
min max of vertices = [-15.0, 20876.0]
sdf median of Apical = 0.14563350000000003

"""

"""pseudocode for find_Apical
1) calculate the height of 70% up the soma
2) find all the neighbors of the soma using verts_to_Label
3) filter out the neighbors that go below that
4) filter away the neighbors that don't meet minimum number of face, vertex change and sdf median
5) If multiple, pick the one that has the most number of neighbors


"""
def find_Apical(min_max,connections,mesh_Number,soma_index,sdf_final_dict):
    mesh_Threshold = 2000
    height_Threshold =5000
    sdf_Threshold = 0.09
    #1) calculate the height of 70% up the soma
    soma_70_percent = (min_max[soma_index][1] - min_max[soma_index][0])*0.7 +  min_max[soma_index][0]
    #2) find all the neighbors of the soma using verts_to_Label
    soma_neighbors = connections[soma_index]
    #3) filter out the neighbors that go below that
     
    
    possible_Axons_filter_1 = [label for label in soma_neighbors if min_max[label][0] > soma_70_percent]
    
    #4) filter away the neighbors that don't meet minimum number of face, vertex change and sdf median
    print("possible_Axons_filter_1 = " + str(possible_Axons_filter_1))
    possible_Axons_filter_2 = [lab for lab in possible_Axons_filter_1 if 
                                    mesh_Number[lab] > mesh_Threshold and 
                                    (min_max[lab][1] - min_max[lab][0]) > height_Threshold and
                                    sdf_final_dict[lab]["median"] > sdf_Threshold]
    print("possible_Axons_filter_2 = " + str(possible_Axons_filter_2))
    if len(possible_Axons_filter_2) <= 0:
        return "None"
    elif len(possible_Axons_filter_2) == 1:
        return possible_Axons_filter_2[0]
    else:
        #find the one with the most neighbors
        current_apical = possible_Axons_filter_2[0]
        current_apical_neighbors = len(connections[possible_Axons_filter_2[0]])
        for i in range(1,len(possible_Axons_filter_2)):
            if len(connections[possible_Axons_filter_2[i]]) > current_apical_neighbors:
                current_apical = possible_Axons_filter_2[i]
                current_apical_neighbors = len(connections[possible_Axons_filter_2[i]])
        
        return current_apical

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



def classify_whole_neuron(possible_Apical,soma_index,connections,mesh_Number,sdf_final_dict,threshold=700):
    cilia_Width_Threshold = 0.05
    whole_neuron_labels ={lb:"unsure" for lb in connections.keys()}
    whole_neuron_labels[soma_index] = "soma"
    
    G=nx.Graph(connections)
    
    node_list = list(G.nodes)
    if(soma_index in node_list):
        node_list.remove(soma_index)
    else:
        return [],[] 
    
    shortest_paths = {}
    for node in node_list:
        shortest_paths[node] = [k for k in nx.shortest_path(G,node,soma_index)]
    
    
    """sdf median of Apical = 0.14563350000000003
    sdf median of Axon = 0.07818095
    sdf median of Axon segment 2 = 0.0502779
    sdf median of Cilia = 0.0102388
    sdf median of Cilia 2 = 0.176036
    
    Rules: if find a segment piece that is less than 0.10 that does not pass through apical and is less than threshold (700) --> everything along path to soma is cilia
    --> if doesn't have component less than that sdf value then label as soma
    """
    cilia_Flag = 0
    #identify if there is any cilia  0.0102388
    for label_key,label_size in mesh_Number.items():
        if label_size < threshold:
            if sdf_final_dict[label_key]["median"] < cilia_Width_Threshold:
                #label the label and everything to the soma as cilia
                for lb in shortest_paths[label_key]:
                    if lb != soma_index:
                        if cilia_Flag == 0:
                            whole_neuron_labels[lb] = "cilia"
                            cilia_Flag= 1
                        else:
                            whole_neuron_labels[lb] = "error"
            else: #label as part of the soma if too high of width
                for lb in shortest_paths[label_key]:
                    if lb != soma_index:
                        whole_neuron_labels[lb] = "soma"
    
    #if possible apical is None then just return
    if possible_Apical == "None":
        return whole_neuron_labels
    
    
    for label_name, path in shortest_paths.items():
        if label_name == possible_Apical:
            whole_neuron_labels[label_name] = "apical"
        else:
            if possible_Apical in path:
                for jj in path:
                    if jj != possible_Apical and jj != soma_index and whole_neuron_labels[jj] == "unsure":
                       whole_neuron_labels[jj] = "oblique" 
            else:
                for jj in path:
                    if jj != possible_Apical and jj != soma_index and whole_neuron_labels[jj] == "unsure":
                       whole_neuron_labels[jj] = "basal" 
    
    #return the final list of labels:
    return whole_neuron_labels

    #CAN'T DETERMINE IF THERE IS AN AXON
    
    
    #endpoint_labels,shortest_paths = find_endpoints(G,mesh_number)
def create_whole_neuron_colors(ob):
    
    color_keys = bpy.data.materials.keys()
    #delete all colors that are not part of the basic colors:
    #print("inside spine color and keys = " + str(color_keys))
    accepted_colors = getColors()
    
    """for i in range(0,len(bpy.data.materials.keys())):
        if(color_keys[i] not in accepted_colors):
            #print("deleting material " + color_keys[i])
            bpy.data.materials.remove(bpy.data.materials[color_keys[i]])"""
        
        
    create_global_colors()
    
    #ob.data.materials[0]           
    
                    
    colors_to_add = ["Apical (blue)",
    "Basal (yellow)",
    "Oblique (green)",
    "Soma (red)",
    "Cilia (light purple)",
    "Error (brown)"
    ]
    
    
    #ob.data.materials[0] = bpy.data.materials[colors_to_add[0]]
    
    #for i in range(1,len(colors_to_add)):
    #    ob.data.materials.append(None)
    
    current_length = len(ob.data.materials.keys())
    print("current_length = " + str(current_length))
    
    color_indexes = []
    for i in range(0,len(colors_to_add)):
        ob.data.materials.append(bpy.data.materials[colors_to_add[i]])
        color_indexes.append(current_length + i)
    
    return color_indexes
        


def label_whole_neuron(labels_list,whole_neuron_labels,ob_name):
    ob = bpy.data.objects[ob_name]
    
    faces_raw = ob.data.polygons
    verts_raw = ob.data.vertices
    
    #delete all the old colors for the object and then add back the local regular colors
    color_indexes = create_whole_neuron_colors(ob)
    
    '''COLORS FOR THE LABELING
    colors_to_add = ["Error (brown)",
    "Dendrite (purple)",
    "Spine Head (rose)",
    "Spine Neck (light green)",
    "Spine (light pink)"]'''
    
    #get the indexes for the labeling from the datajoint table
    label_data = ta3.LabelKey2().fetch("numeric","description")
    #print(label_data)

    label_names = label_data[1].tolist()
    label_indexes = label_data[0].tolist()
    #print(label_names)

    apical_index = label_indexes[label_names.index("Apical")]
    basal_index = label_indexes[label_names.index("Basal")]
    oblique_index = label_indexes[label_names.index("Oblique")]
    soma_index = label_indexes[label_names.index("Soma")]
    cilia_index = label_indexes[label_names.index("Cilia")]
    error_index = label_indexes[label_names.index("Error")]
    print("error_index = " + str(error_index))

    
    final_faces_labels_list = np.zeros(len(faces_raw))
    final_verts_labels_list = np.zeros(len(verts_raw))
    
    unknown_counter = 0
    
    for i,lab in enumerate(labels_list):
        #get the category according to the dictionary
        cat = whole_neuron_labels[lab]
        if cat == "apical":
            faces_raw[i].material_index = color_indexes[0]
            final_faces_labels_list[i] = apical_index
        elif cat == "basal":
            faces_raw[i].material_index = color_indexes[1]
            final_faces_labels_list[i] = basal_index
        elif cat == "oblique":
            faces_raw[i].material_index = color_indexes[2]
            final_faces_labels_list[i] = oblique_index
        elif cat == "soma":
            faces_raw[i].material_index = color_indexes[3]
            final_faces_labels_list[i] = soma_index
        elif cat == "cilia":
            faces_raw[i].material_index = color_indexes[4]
            final_faces_labels_list[i] = cilia_index
        elif cat == "error":
            faces_raw[i].material_index = color_indexes[5]
            final_faces_labels_list[i] = error_index
            #print("labeling error")
        else:
            #faces_raw[i].material_index = color_indexes[4] --don't assign new color
            final_faces_labels_list[i] = len(label_indexes) + (int(lab))
    
    return final_faces_labels_list

def generate_output_lists(final_faces_labels_list,ob_name):
    face_Counter = Counter(final_faces_labels_list)
    #print(face_Counter)
    
    random_labels = {int(l):int(accepted_color_length()+i) for i,l in enumerate(face_Counter.keys()) if l >= accepted_color_length()}
    color_length = accepted_color_length()
    for i in range(0,color_length):
        random_labels[i] = i
        
    #print(random_labels)
    
    #output_faces_list = np.zeros(len(final_faces_labels_list))
    #create list for vertices
    ob = bpy.data.objects[ob_name]
    verts_raw = ob.data.vertices
    faces_raw = ob.data.polygons
    
    start_time_2 = time.time()
    #print("about to do output faces")
    output_faces_list = [random_labels[int(ll)] for ll in final_faces_labels_list]
    #print("done output faces")
    #print("-----output faces = %s---"%(time.time()-start_time_2))
    start_time_2 = time.time()
    
    #generate the vertices
    #print("about to do generate")
    #print("output_faces_list = " + str(output_faces_list))
    verts_to_Face,verts_to_Label = generate_verts_to_face_dictionary(output_faces_list,faces_raw,verts_raw)
    #print("verts_to_Label = " + str(verts_to_Label))
    #print("-----verts_to faces = %s---"%(time.time()-start_time_2))
    start_time_2 = time.time()
    
    output_verts_list = np.zeros(len(verts_raw))
    output_verts_list = [int(verts_to_Label[v][0]) for v in verts_to_Label]
    
    """for v in verts_to_Label:
        output_verts_list[v] = int(verts_to_Label[v][0])
        output_verts_list[v] = int(output_verts_list[v])"""
    #print("-----output_verts = %s---"%(time.time()-start_time_2))
    #print(output_verts_list)
    #print("output_verts_list in function = " + str(Counter(output_verts_list)))
    
    return output_faces_list, output_verts_list 




from pathlib import Path
import w2_smooth_whole_neuron 
import networkx as nx

if __name__ == "__main__":
    
    """bpy.ops.object.mode_set(mode='OBJECT')
    # deselect all
    bpy.ops.object.select_all(action='DESELECT')

    # selection
    #for ob in bpy.data.objects
    #bpy.data.objects[ob_name].select = True
    
    for obj in bpy.data.objects:
        if "neuron" in obj.name or "bound" in obj.name:
            obj.select = True
            
   
    
    # remove it
    bpy.ops.object.delete() 
    #file_loc = "/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/Automatic_Labelers/auto_segmented_big_segments/"
    """
    try:
        print("neuron tab labeler started new")
        #setting the address and the username
        print("about to connect to database")
        dj.config['database.host'] = '10.28.0.34'
        dj.config['database.user'] = 'celiib'
        dj.config['database.password'] = 'newceliipass'
        #will state whether words are shown or not
        dj.config['safemode']=True
        print(dj.conn(reset=True))
    except:
        #Shows a message box with a specific message 
        print("Make sure connected to bcm-wifi!!")
        print("ERROR: Make sure connected to bcm-wifi!!")
        raise ValueError("ERROR: Make sure connected to bcm-wifi!!")
    
    else:
        #connect_to_Databases()
        #create the database inside the server
        schema = dj.schema('microns_ta3p100',create_tables=False)
        ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')
        ta3 = dj.create_virtual_module('ta3', 'microns_ta3')
        #reset_Scene_Variables()
    
        @schema
        class Annotation(dj.Computed):
            definition = """
            # creates the labels for the mesh table
            -> ta3p100.ComponentAutoSegmentWhole
            date_time  : timestamp   #the last time it was edited
            ---
            vertices   : longblob     # label data for the vertices
            triangles  : longblob     # label data for the faces
            """
        

            #key_source = ta3.ComponentAutoSegment #& 'n_triangle_indices>100' & [dict(compartment_type=comp) for comp in ['Basal', 'Apical', 'Oblique', 'Dendrite']]
            
            
            def make(self, key):  
            
                original_start_time = time.time() 
                #create_bounding_box()
                
                start_time = time.time()
                
                
                #[331199,421208,481423]
                neuron_ID = key["segment_id"]
                decimation_ratio = key["decimation_ratio"]
                clusters= key["clusters"]
                smoothness=key["smoothness"]
                
                print("Neuron ID = " + str(neuron_ID))
                #import the object and create the box
                ob_name = load_Neuron_automatic_spine(neuron_ID,decimation_ratio,clusters,smoothness)
                #create_bounding_box()
                
                #assign the colors to the neuron based onthe segmentation data
                start_time = time.time()
                
                #what I will need to get from datajoint acces 1) sdf_final_dict 2) labels_list, might need to make the object active
                sdf_final_dict, labels_list = get_cgal_data_and_label(neuron_ID,decimation_ratio, clusters,smoothness)
                
                #print('df_final_dict["1"]["median"] = ' + str(sdf_final_dict["1"]["median"]))
                
                print("getting cgal data--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                #find the neuron part that has the highest average sdf value
                #high_median, high_median_sdf= get_highest_sdf_part(sdf_final_dict, labels_list )
                #need to look for 2nd highest option and see if split soma
                #high_median_2nd, high_median_2nd_sdf= get_highest_sdf_part(sdf_final_dict, labels_list,high_median)
                
                highest_vals= get_highest_sdf_part(sdf_final_dict, labels_list,size_threshold=3000,)
                #print(highest_vals)
                #highest_vals_2nd= get_highest_sdf_part(sdf_final_dict, labels_list,highest_vals[0])
                #print(highest_vals_2nd)
                
                #print( "large apical = " + str(sdf_final_dict["76"]["max"]))
                
                #can do soma merging here
                
                high_median = highest_vals[0]
                #print("high_medain = " + str(high_median ) + " value = " + str(high_median_sdf))
                #print("high_medain 2 = " + str(high_median_2nd)+ " value = " + str(high_median_2nd_sdf))
                #print("high_mean = " + str(high_mean ))
                #print("high_max = " + str(high_max ))
                 
                print("got highest part--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
                #export the neuron
                #export_neuron(ob_name,destination_folder="whole_neuron_testing")
                
                create_bounding_box()
                
                
                #now do the smoothing
                labels_list,verts_to_Face = merge_labels_vp2(labels_list,ob_name,threshold=3000,soma_index=high_median,cilia_threshold=100)
                
                print("done merging labels-- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
                #need to find the neighbors of the soma
                
                ob = bpy.data.objects[ob_name]
                #w2_smooth_whole_neuron
                faces_raw = ob.data.polygons
                verts_raw = ob.data.vertices
                
                #print("labels_list = " + str(labels_list))
                verts_to_Face,verts_to_Label = generate_verts_to_face_dictionary(labels_list,faces_raw,verts_raw)
                #pprint("done finding verts stats-- %s seconds ---" % (time.time() - start_time))
                print("done creating dictionaries -- %s seconds --" %(time.time() - start_time))
                
                #print("verts_to_Label = " + str(verts_to_Label))
                
                start_time = time.time()
                
                #create a graph structure and stats for the whole neuron
                connections, mesh_Number = get_graph_structure(verts_to_Label,labels_list,faces_raw,verts_raw)
                print("mesh_Number = " + str(mesh_Number))
                print("connections = " + str(connections))
                
                print("done creating connections -- %s seconds --" %(time.time() - start_time))
                
                start_time = time.time()
                soma_index=high_median
                neighbors_list = connections[soma_index].copy()
                
                neighbors_list.append(soma_index)
                min_max = find_max_min_z_vals(neighbors_list,labels_list,faces_raw,verts_raw)
                print("done FINDING NEIGHBOR STATS -- %s seconds --" %(time.time() - start_time))
                print("min_max = " + str(min_max))
                
                """
                print( "sdf median of Apical = " + str(sdf_final_dict["76"]["median"]))
                print( "sdf median of Axon = " + str(sdf_final_dict["72"]["median"]))
                print( "sdf median of Axon segment 2 = " + str(sdf_final_dict["69"]["median"]))
                print( "sdf median of Cilia = " + str(sdf_final_dict["0"]["median"]))
                print( "sdf median of Cilia 2 = " + str(sdf_final_dict["78"]["median"]))
                print( "Basal Piece = " + str(sdf_final_dict["58"]["median"]))"""
                
                
                #send data to function that will find the Apical
                possible_Apical = find_Apical(min_max,connections,mesh_Number,soma_index,sdf_final_dict)
                print(possible_Apical)
                
                #use the apical label and the soma label to classify the rest as basal or oblique and return a dictionary that has the mapping of label to compartment type
                whole_neuron_labels = classify_whole_neuron(possible_Apical,soma_index,connections,mesh_Number,sdf_final_dict,threshold=500)
                print("whole_neuron_labels = " + str(whole_neuron_labels))
                
                #label the neurons according to classification
                #############NEED TO ADD STEP THAT CALCULATES THE LABELS OF THE VERTICES ##################
                final_faces_labels_list = label_whole_neuron(labels_list,whole_neuron_labels,ob_name)
                print("done labeling whole neuron")
                
                #####need to map the final_faces_labels_list to all successive numbers and get vertices
                output_faces_list, output_verts_list = generate_output_lists(final_faces_labels_list,ob_name)
                print("done generating output list whole neuron")
                
                #print(Counter(output_faces_list))
                #print(Counter(output_verts_list))
                
                #Things you have to write to datajoint for primary keys: 
                
                #now write them to the datajoint table  
                comp_dict = dict(key,
                                    date_time = str(datetime.datetime.now())[0:19],
                                    vertices = output_verts_list,
                                    triangles = output_faces_list)

                #print("comp_dict = " + str(comp_dict))
                
                print("about to write row to datajoint")
                self.insert1(comp_dict,skip_duplicates=True)
                print("writing label data to datajoint--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
                #delete the object after this
                #delete the object
                            
                # deselect all
                bpy.ops.object.select_all(action='DESELECT')

                # selection
                #for ob in bpy.data.objects
                #bpy.data.objects[ob_name].select = True
                
                for obj in bpy.data.objects:
                    if "neuron" in obj.name:
                        obj.select = True
                        
                        
                # remove it
                bpy.ops.object.delete()
                
                print("deleting object--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
                # deselect all
                bpy.ops.object.select_all(action='DESELECT')

                # selection
                #for ob in bpy.data.objects
                #bpy.data.objects[ob_name].select = True
                    
                
                object_counter = 0
                for obj in bpy.data.objects:
                    if "neuron" in obj.name:
                        object_counter += 1
                
                if object_counter>1:
                    raise ValueError("THE NUMBER OF OBJECTS ARE MORE THAN 1")
                        
                
                print("finished")
                print("--- %s seconds ---" % (time.time() - original_start_time))
                
                
                """neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,high_median,verts_to_Face,faces_raw,verts_raw)
                print("neighbors_list = " + str(neighbors_list))
                print("done finding soma neighbors-- %s seconds ---" % (time.time() - start_time))
                neighbors_list_max_min = find_max_min_z_vals(neighbors_list + [high_median],labels_list,faces_raw,verts_raw)
                print("neighbors_list_max_min = " + str(neighbors_list_max_min))
                print("done finding verts stats-- %s seconds ---" % (time.time() - start_time))
                
                #crawl and make graph structure:"""
                
populate_start = time.time()
Annotation.populate(reserve_jobs=True)
print("\npopulate:", time.time() - populate_start)
