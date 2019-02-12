import bpy
import numpy as np


#function_description

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
    ob.data.materials[0].name
    
    labels_list = [ob.data.materials[fac.material_index].name for fac in faces_raw]
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
            #if replacement_label == soma_index:
                #print("number_to_replace = " + str(number_to_replace))
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

def merge_labels(x,threshold=50,soma_index=-1,cilia_threshold=300,number_Flag = False, seg_numbers=1,):
    
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
    
    return x




if __name__ == "__main__":
    
   
    
    #x = generate_label_list("/Users/brendancelii/Google Drive/Xaq Lab/Final_Blender/whole_neuron_auto_label/saved_off_labels_list/")
    x = generate_label_list()
    print("done generating labels list")
    high_median = "84.005"
    merge_labels(x,threshold=700,soma_index=high_median,cilia_threshold=300,number_Flag = False, seg_numbers=1)
    
    
