import bpy
import bmesh

import numpy as np
from bpy.types import Operator
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector
import os
import time

from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Operator,
                       AddonPreferences,
                       PropertyGroup,
                       )
                       
import datetime
import math
from pathlib import Path
from collections import Counter

#####where will define all the bpy.context.scene attributes##########
#registration
bpy.types.Scene.username = bpy.props.StringProperty()
bpy.types.Scene.firstname = bpy.props.StringProperty()
bpy.types.Scene.lastname = bpy.props.StringProperty()
bpy.types.Scene.register_Status = bpy.props.StringProperty()



#login
bpy.types.Scene.username_Login = bpy.props.StringProperty()
bpy.types.Scene.login_Status = bpy.props.StringProperty()
bpy.types.Scene.login_Flag = bpy.props.BoolProperty()

#neuron ID picker
bpy.types.Scene.neuron_ID = bpy.props.StringProperty()
bpy.types.Scene.neuron_ID_Status = bpy.props.StringProperty()
bpy.types.Scene.labeled_Flag_ID = bpy.props.BoolProperty()
bpy.types.Scene.import_Neuron_Flag = bpy.props.BoolProperty()

#neuron username picker
bpy.types.Scene.status_picked = bpy.props.StringProperty()
bpy.types.Scene.neuron_username_Status = bpy.props.StringProperty()
bpy.types.Scene.username_neuron_ID = bpy.props.StringProperty()

#next available neuron picker
bpy.types.Scene.next_available_neuron_ID = bpy.props.StringProperty()
bpy.types.Scene.next_available_status = bpy.props.StringProperty()

#for the editing properties
bpy.types.Scene.last_Edited = bpy.props.StringProperty()
bpy.types.Scene.picked_Neuron_ID = bpy.props.StringProperty()
bpy.types.Scene.continue_edit_Status = bpy.props.StringProperty()
bpy.types.Scene.last_Status = bpy.props.StringProperty()
bpy.types.Scene.last_Edited_User = bpy.props.StringProperty()
bpy.types.Scene.load_local_status = bpy.props.StringProperty()



#for saving off the neuron:
bpy.types.Scene.status_To_Save = bpy.props.StringProperty()
bpy.types.Scene.percent_labeled = bpy.props.StringProperty()
bpy.types.Scene.percent_labeled_faces = bpy.props.StringProperty()
bpy.types.Scene.complete_100_check = bpy.props.StringProperty()
bpy.types.Scene.complete_100_check_2 = bpy.props.StringProperty()
bpy.types.Scene.complete_100_check_save_flag = bpy.props.BoolProperty()

#for deleting
bpy.types.Scene.delete_Flag = bpy.props.BoolProperty()
bpy.types.Scene.delete_ID = bpy.props.StringProperty()
bpy.types.Scene.delete_status = bpy.props.StringProperty()

#for editing the automatically labeled neurons
bpy.types.Scene.label_show = bpy.props.StringProperty()
bpy.types.Scene.label_show_status = bpy.props.StringProperty()
bpy.types.Scene.label_show_select = bpy.props.StringProperty()
bpy.types.Scene.label_show_status_select = bpy.props.StringProperty()

#for quick label
bpy.types.Scene.quick_label_2 = bpy.props.BoolProperty()




#for proofreading
#bpy.types.Scene.author_original = bpy.props.StringProperty()

def check_Verts_Match():
    print("inside check verts match")
    ID = bpy.context.scene.picked_Neuron_ID
    
    print((ID != ""))
    print((("neuron-"+str(ID)) in bpy.data.objects.keys()))
    if (ID != "") & (("neuron-"+str(ID)) in bpy.data.objects.keys()):
        print("checking neurons not changed")
        ob = bpy.data.objects["neuron-"+ID]

        verts_raw = ob.data.vertices
        #print(len(active_verts_raw))
        
        edges_raw = ob.data.edges
        
        #print(len(active_edges_raw))
        
        faces_raw = ob.data.polygons
        
            
        n_edges = len(edges_raw)
        n_vertices = len(verts_raw)
        n_triangles = len(faces_raw)

        primary_key = dict(segmentation=2,decimation_ratio=0.35)
        
        mesh_Dict = (ta3p100.CleansedMeshOrphan()  & primary_key & "segment_id="+ID).fetch(as_dict=True)[0]

        #mesh_Dict = (mesh_Table & "segment_id="+ID).fetch(as_dict=True)[0] old way
        n_vertices_check = mesh_Dict['n_vertices']
        n_traingles_check = mesh_Dict['n_triangles']
        
        if (n_vertices_check != n_vertices) or (n_traingles_check != n_triangles):
            print("n_vertices = " + str(n_vertices))
            print("n_vertices_check = " + str(n_vertices_check))
            print("n_triangles = " + str(n_triangles))
            print("n_traingles_check = " + str(n_traingles_check))
            
            print("ERROR: vertices and traingles do NOT match")
            raise ValueError("ERROR: vertices and traingles do NOT match")
            return

def reset_Scene_Variables(login_Flag=False):
    #registration
    bpy.context.scene.username = ""
    bpy.context.scene.firstname =""
    bpy.context.scene.lastname = ""
    bpy.context.scene.register_Status = ""



    #login
    
    bpy.context.scene.login_Status = ""
    if login_Flag == False:
        bpy.context.scene.login_Flag = False
        bpy.context.scene.username_Login = ""

    #neuron ID picker
    bpy.context.scene.neuron_ID = ""
    bpy.context.scene.neuron_ID_Status = ""
    bpy.context.scene.labeled_Flag_ID = False
    bpy.context.scene.import_Neuron_Flag = False
    
    #neuron username picker
    bpy.context.scene.status_picked = ""
    bpy.context.scene.neuron_username_Status = ""
    bpy.context.scene.username_neuron_ID = ""
    
    #next available picker
    bpy.context.scene.next_available_neuron_ID = ""
    bpy.context.scene.next_available_status = ""
    
    #for editing
    bpy.context.scene.last_Edited = ""
    bpy.context.scene.picked_Neuron_ID = ""
    bpy.context.scene.continue_edit_Status = ""
    bpy.context.scene.last_Status = ""
    bpy.context.scene.last_Edited_User  = ""
    
    #for saving off the neuron
    bpy.context.scene.status_To_Save = ""
    bpy.context.scene.percent_labeled = ""
    bpy.context.scene.percent_labeled_faces = ""
    bpy.context.scene.complete_100_check = ""
    bpy.context.scene.complete_100_check_2 = ""
    bpy.context.scene.complete_100_check_save_flag = False

	#for loading neurons
    bpy.context.scene.load_local_status = ""
    
    #for deleting
    bpy.context.scene.delete_Flag = False
    bpy.context.scene.delete_ID = ""
    bpy.context.scene.delete_status = ""
    
    #for editing automatically labeled neurons
    bpy.context.scene.label_show = ""
    bpy.context.scene.label_show_status = ""
    bpy.context.scene.label_show_select = ""
    bpy.context.scene.label_show_status_select = ""
    
    #for quick label
    bpy.context.scene.quick_label_2 = False
    
def reset_Status_Flags():
    bpy.context.scene.register_Status = ""    
    bpy.context.scene.neuron_ID_Status = ""
    bpy.context.scene.neuron_username_Status = ""
    bpy.context.scene.status_picked = ""
    bpy.context.scene.next_available_status = ""
    bpy.context.scene.delete_status = ""
    
    


#-------------------FUNCTIONS ADDED ON 12/19 FOR REVIEW OF AUTOMATIC LABELING--------------#
def show_only_label(accepted_labels):
#put into object mode so that the changes will be persisted    
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    me = ob.data
    """Don't need to hide the edges and vertices
    #print("starting to hide everything")
    #iterate through all of the vertices
    verts_raw = ob.data.vertices
    active_verts_raw = [k for k in verts_raw if k.bevel_weight > 0.0]
    #print(len(active_verts_raw))
    
    edges_raw = ob.data.edges
    
    active_edges_raw = [k for k in edges_raw if k.bevel_weight > 0.0]
    #print(len(active_edges_raw))
    
    """
    
    bpy.context.object.data.materials

    #create lookup dictionary
    bpy_keys = bpy.context.object.data.materials.keys()
    
    #gives you the index number of the material

    lookup = {bpy_keys[x]:x for x in range(0,len(bpy_keys))}
    
    if accepted_labels[0] not in lookup.keys():
        bpy.context.scene.label_show_status = "Label not correct"
        print("done hiding")
        ob.data.update() 
        
        bpy.ops.object.mode_set(mode='EDIT')
        return
    
    accepted_labels_index = []
    
    for label in accepted_labels:
        accepted_labels_index.append(lookup[str(label)])
    
    
    print(accepted_labels_index)
    
    
    
    
    faces_raw = ob.data.polygons
    active_faces_raw = [k for k in faces_raw if k.material_index not in accepted_labels_index]
    show_faces_raw = [k for k in faces_raw if k.material_index in accepted_labels_index]
    
    print("faces active = " + str(len(show_faces_raw)))
    
    print("accepted_labels_index = " + str(accepted_labels_index))
    print("len(active_faces_raw) = " + str(len(active_faces_raw)))
    

    #iterate through all of the face
    
    
    #print(active_faces_raw)
    for k in active_faces_raw:
        k.select = False
        k.hide = True
        
    print("show_faces_raw number = " + str(len(show_faces_raw)))
    for k in show_faces_raw:
        k.select = True
        k.hide = False


    verts_raw = ob.data.vertices
    edges_raw = ob.data.edges
    
    
    for k in verts_raw:
        k.hide = True

        
    for k in edges_raw:
        k.hide = True
        

    if(currentMode == 'EDIT'):
        bpy.ops.object.mode_set(mode='EDIT')
    else:
        bpy.ops.object.mode_set(mode='OBJECT')
        
    print("done hiding")
    ob.data.update() 
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.scene.label_show_status = "Showing " + str(accepted_labels[0])

def show_only_label_select(accepted_labels):
#put into object mode so that the changes will be persisted    
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    me = ob.data
    """Don't need to hide the edges and vertices
    #print("starting to hide everything")
    #iterate through all of the vertices
    verts_raw = ob.data.vertices
    active_verts_raw = [k for k in verts_raw if k.bevel_weight > 0.0]
    #print(len(active_verts_raw))
    
    edges_raw = ob.data.edges
    
    active_edges_raw = [k for k in edges_raw if k.bevel_weight > 0.0]
    #print(len(active_edges_raw))
    
    """
    
    bpy.context.object.data.materials

    #create lookup dictionary
    bpy_keys = bpy.context.object.data.materials.keys()
    
    #gives you the index number of the material

    lookup = {bpy_keys[x]:x for x in range(0,len(bpy_keys))}
    
    if accepted_labels[0] not in lookup.keys():
        bpy.context.scene.label_show_status_select = "Label not correct"
        print("done hiding")
        ob.data.update() 
        
        bpy.ops.object.mode_set(mode='EDIT')
        return
    
    accepted_labels_index = []
    
    for label in accepted_labels:
        accepted_labels_index.append(lookup[str(label)])
    
    
    print(accepted_labels_index)
    
    
    
    
    faces_raw = ob.data.polygons
    active_faces_raw = [k for k in faces_raw if k.material_index not in accepted_labels_index]
    show_faces_raw = [k for k in faces_raw if k.material_index in accepted_labels_index]
    
    print("faces active = " + str(len(show_faces_raw)))
    
    print("accepted_labels_index = " + str(accepted_labels_index))
    print("len(active_faces_raw) = " + str(len(active_faces_raw)))
    

    #iterate through all of the face
    
    
    #print(active_faces_raw)
    for k in active_faces_raw:
        k.select = False
        k.hide = False
        
    print("show_faces_raw number = " + str(len(show_faces_raw)))
    for k in show_faces_raw:
        k.select = True
        k.hide = False


    verts_raw = ob.data.vertices
    edges_raw = ob.data.edges
    
    
    for k in verts_raw:
        k.hide = False

        
    for k in edges_raw:
        k.hide = False
        

    if(currentMode == 'EDIT'):
        bpy.ops.object.mode_set(mode='EDIT')
    else:
        bpy.ops.object.mode_set(mode='OBJECT')
        
    print("done hiding")
    ob.data.update() 
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.scene.label_show_status_select = "Selected " + str(accepted_labels[0])




def main_edit_auto(context,myVar):
    if myVar == "only_show_label":
        if bpy.context.scene.label_show != "":
            #get a list of the labels and see if variable is a substring of any of them
            label_show_var = bpy.context.scene.label_show
            picked_label= ""
            color_list = bpy.context.object.data.materials.keys()
            for cl in color_list:
                if label_show_var == cl[:len(label_show_var)]:
                    picked_label = cl
                    break
                    
            
            show_only_label([picked_label])
    elif myVar == "unhide_labels":
        bpy.ops.mesh.reveal()
        bpy.context.scene.label_show_status = ""
    elif myVar == "only_show_label_select":
        if bpy.context.scene.label_show_select != "":
            #get a list of the labels and see if variable is a substring of any of them
            label_show_var = bpy.context.scene.label_show_select
            picked_label= ""
            color_list = bpy.context.object.data.materials.keys()
            for cl in color_list:
                if label_show_var == cl[:len(label_show_var)]:
                    picked_label = cl
                    break
            show_only_label_select([picked_label])
            #unhide the rest of the labels without having them selected
            
    else:
        return




class edit_auto_neuron(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.edit_auto_neuron"
    bl_label = "Nedit_auto_neuron"
    
    myVar = bpy.props.StringProperty(name="myVar")

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        main_edit_auto(context,self.myVar)
        
        return {'FINISHED'}







############-----------------------END FOR for manual editing of Automatic Labeler-------------###############



    
def set_View():
    print("setting view back to original")
    
    check_Verts_Match()
    
    
    #are that I want to make an allowance for: bpy.data.screens['Scripting'].areas

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = {'area': area, 'region': area.regions[-1]}
            #bpy.ops.view3d.view_pan(override, type='PANRIGHT')
            bpy.ops.view3d.view_lock_to_active(override)
            bpy.ops.view3d.view_lock_clear(override)
            
        
            #FOR SOME REASON ADDING THIS BEFORE IT HELPS GET IT CENTERED
            bpy.ops.view3d.view_all(override, center=False)


            #make sure can view all of the neuron
            bpy.ops.view3d.view_all(override, center=True)
            #way to set it:
            #1) User Persp
            bpy.ops.view3d.view_persportho(override)
            #2) Top Persp
            bpy.ops.view3d.viewnumpad(override,type='FRONT')
            #3) Top Ortho
            bpy.ops.view3d.view_persportho(override)
            
            #putting at end just to make sure it works
            bpy.ops.view3d.view_all(override, center=False)
            
            check_Verts_Match()

def edit_active_neuron():
    #check that there is an active object
    if(bpy.context.scene.objects.active == "None"):
        #set the status variable 
        bpy.context.scene.continue_edit_Status = "No object is currently active!"
        return
    
    #get the name of the active neuron (if not then print out that no neuron is
    #currently selected
    name_of_active = bpy.context.scene.objects.active.name
    
    if("neuron" not in name_of_active):
        bpy.context.scene.continue_edit_Status = "Object selected is not neuron!"
        return
    
    bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.select_all(action='TOGGLE')
    bpy.ops.mesh.select_all(action='DESELECT')
    
    #set the appropriate variables for the editing session
    ID = name_of_active[7:]
    bpy.context.scene.picked_Neuron_ID = name_of_active[7:]
    
    #go pull down the last time that the object was edited 
    #(SHOULD ALWAYS BE IN THE LABELS LIST)
    
    
    
    reset_Status_Flags()
    print("end of edit_active_neuron")



#receives the ID as a string of the neuron it wants 
#AND MAKES SURE YOU ONLY CAN PULL DOWN YOUR OWN
def load_Neuron(ID):
    print("inside load Neuron")
    #create an object to the labels and the mesh_Table
    #already exist in labels_Table and mesh_Table
    author_proofreader = bpy.context.scene.username_Login
    #get list of ID's from labeled to check if there
    username_proof_Table = (proof_Table & "author_proofreader='"+ author_proofreader + "'")
    proof_list = username_proof_Table.fetch("segment_id").tolist()
    
    from_Mesh_Flag = 0
    from_Proof_Flag = 0
    
    #convert ints to Strings
    if int(ID) not in proof_list:
        #don't need to get the labes because there are none
        from_Mesh_Flag = 1
        from_Proof_Flag = 0
        print("not in proof_table")
                       
    else:
        #don't need to push to labels table
        from_Mesh_Flag = 0
        from_Proof_Flag = 1
    
    #GOES THROUGH THE WHOLE PROCESS OF CREATING THE OBJECT AND IMPORTING IT
    
    
    #neuron_data = ((mesh_Table & "segment_ID="+ID).fetch(as_dict=True))[0]
    primary_key = dict(segmentation=2,decimation_ratio=0.35)
    neuron_data = ((ta3p100.CleansedMeshOrphan()  & primary_key & "segment_ID="+ID).fetch(as_dict=True))[0]


    
    verts = neuron_data['vertices'].astype(dtype=np.int32).tolist()
    faces = neuron_data['triangles'].astype(dtype=np.uint32).tolist()
    
    #*********Need to add in my own scale*****************#
    """scale = 0.001
    
    #Makes the vertices as voxels and applies a scale to them
    not_centered_verts = [(x[0], x[2], x[1]) for x in (verts * scale).tolist()]
    #don't need the vertical count being added to the faces like in his because because 
    new_faces = [(x[0], x[1], x[2]) for x in (faces).tolist()]

    offset = np.median(np.array(not_centered_verts), axis=0)
    print("offset = " + str(offset))
    new_verts = [(x[0] - offset[0], x[1] - offset[1], x[2] - offset[2]) for x in not_centered_verts]
 
 
    print("Length of vertices = " + str(len(new_verts)))
    print("Length of vertices = " + str(len(new_faces)))"""
    #-----------------END OF new way of importing using james cotton way------------------#
    mymesh = bpy.data.meshes.new("neuron-"+ID)
    mymesh.from_pydata(verts, [], faces)
    
    
    
    #object = bpy.data.objects.new(optional_Name, mesh)
    
    #uses the bmesh library to import:
    
    
    """ DOES THE SIMPLIFYING THAT I DON'T WANT ANYMORE
    bm = bmesh.new()
    bm.from_mesh(mymesh)
    print("Simplifying {}".format(len(bm.verts)))
    # bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0)
    #for i in range(5):
    #    bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5, use_axis_x=True, use_axis_y=True,
    #                          use_axis_z=True)  # , mirror_clip_x, mirror_clip_y, mirror_clip_z, clip_dist, use_axis_x, use_axis_y, use_axis_z)¶
    
    #finds groups of vertices closer than dist and merges them together 
    bmesh.ops.automerge(bm, verts=bm.verts, dist=1e-6)
    
    for f in bm.faces:
        f.smooth = True
    
    #here it applies all the changes to the mesh
    bm.to_mesh(mymesh)
    print("Done {}".format(len(bm.verts)))
    bm.free()
    """
    #nothing has changed for the vertices with the triangles
    #print(new_verts)
    #print(new_faces)
    
    #************with these array of tuples can't just import 
    #it because will give you ther error:
    #The truth value of an array with more than one element 
    #is ambiguous. Use a.any() or a.all()
    #*************#
    #calculating the edges and the normals
    mymesh.update(calc_edges=True)
    mymesh.calc_normals()


    #mymesh.validate()
    
    #for i in range(0,10):  
    #    print(mymesh.vertices[i].co)


    
    #print("filename right before list = " + filename)
    #objects_Matching_filename = [x for x in object_List if "neuron_mesh_36706215" in x]
    
    #print(objects_Matching_filename)
    object = bpy.data.objects.new("neuron-"+ID, mymesh)
    #object.location = bpy.context.scene.cursor_location
    object.location = Vector((0,0,0))
    bpy.context.scene.objects.link(object)
    
    object.lock_location[0] = True
    object.lock_location[1] = True
    object.lock_location[2] = True
    object.lock_scale[0] = True
    object.lock_scale[1] = True
    object.lock_scale[2] = True

    #rotate the z direction by 90 degrees so point correct way
    
    #object.rotation_euler[2] = 1.5708

    
    object.rotation_euler[0] = 4.7124
    object.rotation_euler[1] = 0
    object.rotation_euler[2] = 0



    
    object.lock_rotation[0] = True
    object.lock_rotation[1] = True
    object.lock_rotation[2] = True


    #set view back to normal:
    set_View()
    
    
    
    
    #run the setup color command
    #bpy.ops.object.select_all(action='TOGGLE')
    create_local_colors(object)
    
    
    
    
    ###**********WILL GO THROUGH AND PUSH UNLABELED DATA TO DATABASE 
    ####SO NO ONE ELSE WILL BE ABLE TO PULL THE SAME THING
    
    if from_Mesh_Flag == 1:
        print("from Mesh Flag option --> meaning hasn't been proof read yet")
        #create array for faces, verts and edges with values of 0 for all
        #and then push to the labels database
        
        ob = bpy.data.objects["neuron-"+ID]
        
        me = ob.data
        edges_raw = ob.data.edges
        
        n_edges = len(edges_raw)
        n_vertices = neuron_data['n_vertices']
        n_traingles = neuron_data['n_triangles']
        
        print("inside mesh")
        print('n_vert = ' + str(n_vertices))
        print('n_tri = ' + str(n_traingles))
        
        
        ####need to pull down the labels from the Mesh table
        author_proofreader
        
        #create empty ______ for them
        
        
        
        verts_labels = (8*np.ones(n_vertices)).tolist()
        triangles_labels = (8*np.ones(n_traingles)).tolist()
        
        
        #do the actual labeling
        me = ob.data
        me.use_customdata_edge_bevel = True
        me.use_customdata_vertex_bevel = True
        
        #print("starting to hide everything")
        #iterate through all of the vertices
        verts_raw = ob.data.vertices
        #print(len(active_verts_raw))
        
        #edges_raw = ob.data.edges
        
        #print(len(active_edges_raw))
        
        faces_raw = ob.data.polygons
        
                    
        """if len(edges_raw) != len(edges_labels):
            edges_labels = edges_labels = np.zeros(len(edges_raw),dtype=np.uint8)
            print("edges imported don't match the edges in neuron")"""
            
                
        print("inside mesh")
        print('verts_raw = ' + str(len(verts_raw)))
        print('faces_raw = ' + str(len(faces_raw)))
        
        #verts_labels = neuron_labels['vertices']
        #triangles_labels = neuron_labels['traingles']
        #edges_labels = neuron_labels['edges']
        
        bevel_Weights = get_Bevel_Weights()
        
        #iterate through all of the 
        """for i,k in enumerate(verts_raw):
            #set the bevel weight
            k.bevel_weight = bevel_Weights[int(verts_labels[i])]"""
        
        """for i,k in enumerate(edges_raw):
            #set the bevel weight
            k.bevel_weight = bevel_Weights[int(edges_labels[i])]"""
        
        
        #iterate through all of the face

        for i,k in enumerate(faces_raw):
            k.material_index = int(triangles_labels[i])
        
        
        
        #push to the labels database
        
        timestamp = str(datetime.datetime.now())
        print(timestamp[0:19])
        
        #need to get the original author
        
        #author_original = ((labels_Table & "segment_id="+str(ID)).fetch("author"))[0]
        #bpy.context.scene.author_original = author_original
        author_proofreader = bpy.context.scene.username_Login
        
        dateTime_stored = str(datetime.datetime.now())[0:19]
        
        segmentation=2
        decimation_ratio=0.35
        
        author_original="computer_Auto"
        
        
        #now try making write
        proof_Table.insert1((segmentation,int(ID),decimation_ratio,author_original,author_proofreader,dateTime_stored,
                        verts_labels,triangles_labels,[],"partial"))
                        
        
        
        print("just stored pulled neuron in the labels table")
    
    elif from_Proof_Flag == 1:
        print("from Proof Flag option")
        #get the labels
        #labels_list
        
        #get neuron info from the mesh table
        neuron_labels = ((proof_Table & "segment_ID="+ID & "author_proofreader='"+ author_proofreader + "'").fetch(as_dict=True))[0]
        #bpy.context.scene.author_original = neuron_labels["author_original"]
        verts_labels = neuron_labels['vertices']
        triangles_labels = neuron_labels['triangles']
        edges_labels = neuron_labels['edges']
        
        print("inside mesh")
        print('verts_labels = ' + str(len(verts_labels)))
        print('triangles_labels = ' + str(len(triangles_labels)))
        
        
        #need to add the labels to the newly created object
        ob = bpy.data.objects["neuron-"+ID]
        
        
        me = ob.data
        me.use_customdata_edge_bevel = True
        me.use_customdata_vertex_bevel = True
        
        #print("starting to hide everything")
        #iterate through all of the vertices
        verts_raw = ob.data.vertices
        #print(len(active_verts_raw))
        
        edges_raw = ob.data.edges
        
        #print(len(active_edges_raw))
        
        faces_raw = ob.data.polygons
        
                    
        if len(edges_raw) != len(edges_labels):
            edges_labels = np.zeros(len(edges_raw),dtype=np.uint8)
            print("edges imported don't match the edges in neuron")
            
                
        print("inside mesh")
        print('verts_raw = ' + str(len(verts_raw)))
        print('faces_raw = ' + str(len(faces_raw)))
        
        #verts_labels = neuron_labels['vertices']
        #triangles_labels = neuron_labels['traingles']
        #edges_labels = neuron_labels['edges']
        
        bevel_Weights = get_Bevel_Weights()
        
        #iterate through all of the 
        """for i,k in enumerate(verts_raw):
            #set the bevel weight
            k.bevel_weight = bevel_Weights[int(verts_labels[i])]
        
        for i,k in enumerate(edges_raw):
            #set the bevel weight
            k.bevel_weight = bevel_Weights[int(edges_labels[i])]"""
        
        
        #iterate through all of the face

        for i,k in enumerate(faces_raw):
            k.material_index = int(triangles_labels[i])

        
    
    else:
        print("ERROR: neither the labels flag or the mesh flag are active")
        return
                
    
    #creates the extra labels that are needed for neuron that isn't completely categorized
    create_extra_local_colors(triangles_labels,object)   

    
    
    #reset the flags
    from_Mesh_Flag = 0
    from_Proof_Flag = 0
    
    #go into edit mode
    print("trying to select the neuron")
    
    bpy.context.scene.objects.active = bpy.context.scene.objects["neuron-"+ID]
    #bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.select_all(action='TOGGLE')
    edit_active_neuron()
    
    #does the label setting
    last_edited, last_status, last_user= (proof_Table & "segment_id="+ID & "author_proofreader='"+ author_proofreader + "'").fetch("date_time", "status","author_proofreader")
    bpy.context.scene.last_Edited = str(last_edited[0])
    last_status_value = last_status[0]

    #get the dictionary of the status key
    
    bpy.context.scene.last_Status = last_status_value

    #set the user who was last to edit it
    bpy.context.scene.last_Edited_User = last_user[0]
    
    is_label_hidden = get_Hide_Flag()
    
    #make sure the faces are hidden if they should be
    if (is_label_hidden == True):
        hide_Labeled(mode=0,waitTime=0)
    else:  #show all of the faces
        #print ("Property Disabled")
        hide_Labeled(mode=1,waitTime=0)

    

    #make sure in solid mode
    for area in bpy.context.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.viewport_shade = 'SOLID' # set the viewport shading to rendered
                    space.lock_cursor = True
    bpy.ops.object.mode_set(mode='OBJECT')
    
                    
    #were for debugging purposes where was checking if there were vertices/faces lost
    """ob = bpy.data.objects["neuron-"+ID]


    me = ob.data
    me.use_customdata_edge_bevel = True
    me.use_customdata_vertex_bevel = True

    #print("starting to hide everything")
    #iterate through all of the vertices
    verts_raw = ob.data.vertices
    #print(len(active_verts_raw))

    edges_raw = ob.data.edges

    #print(len(active_edges_raw))

    faces_raw = ob.data.polygons

                

            
    print("AT THE END OF THE IMPORT FUNCTION")
    print('verts_raw = ' + str(len(verts_raw)))
    print('faces_raw = ' + str(len(faces_raw)))"""
    
    set_View()

    bpy.data.materials["Dendrite (purple)"].diffuse_color = (0.163,0.800,0.425)
    return

#continue_editing
class continue_editing(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.continue_editing"
    bl_label = "continue_editing"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        edit_active_neuron()
        #if not(print_Message == ""):
            #self.report({'INFO'}, print_Message)
        return {'FINISHED'}

#the npz file will look like this
#[segment_ID,author,date_time,vertices,triangles,edges,status]

#-----------------------------------------------importing local saved off files------------------------------------------------------------------------
def import_local_Neuron(filepath):
	
    try:
        neuron_labels = np.load(filepath)
    except:
        print("there is no file with that name in the same directory as the Blender project!")
        return
    	
    	#just need to unpack the label variables and then pull the neuron from the database and label it
    ID = str(neuron_labels["segment_ID"])
    bpy.context.scene.picked_Neuron_ID = ID
    #set the username for the last status
    bpy.context.scene.last_Edited_User = str(neuron_labels["author_proofreader"])
    #bpy.context.scene.author_original = str(neuron_labels["author_original"])
    #set the time for the last status
    bpy.context.scene.last_Edited = str(neuron_labels["date_time"])
    bpy.context.scene.last_Status = str(neuron_labels["status"])
    	
    verts_labels = neuron_labels['vertices'].astype(dtype=np.int32)
    triangles_labels = neuron_labels['triangles'].astype(dtype=np.int32)
    edges_labels = neuron_labels['edges'].astype(dtype=np.int32)


    #GOES THROUGH THE WHOLE PROCESS OF CREATING THE OBJECT AND IMPORTING IT
    primary_key = dict(segmentation=2,decimation_ratio=0.35)
    neuron_data = ((ta3p100.CleansedMeshOrphan()  & primary_key & "segment_ID="+ID).fetch(as_dict=True))[0]

    
    verts = neuron_data['vertices'].astype(dtype=np.int32)
    faces = neuron_data['triangles'].astype(dtype=np.uint32)

        #*********Need to add in my own scale*****************#
    """scale = 0.001
    
    #Makes the vertices as voxels and applies a scale to them
    not_centered_verts = [(x[0], x[2], x[1]) for x in (verts * scale).tolist()]
    #don't need the vertical count being added to the faces like in his because because 
    new_faces = [(x[0], x[1], x[2]) for x in (faces).tolist()]

    offset = np.median(np.array(not_centered_verts), axis=0)
    print("offset = " + str(offset))
    new_verts = [(x[0] - offset[0], x[1] - offset[1], x[2] - offset[2]) for x in not_centered_verts]
 
 
    print("Length of vertices = " + str(len(new_verts)))
    print("Length of vertices = " + str(len(new_faces)))"""
    #-----------------END OF new way of importing using james cotton way------------------#
    mymesh = bpy.data.meshes.new("neuron-"+ID)
    mymesh.from_pydata(verts, [], faces)
    
    
    
    #object = bpy.data.objects.new(optional_Name, mesh)
    
    #uses the bmesh library to import:
    
    
    """ DOES THE SIMPLIFYING THAT I DON'T WANT ANYMORE
    bm = bmesh.new()
    bm.from_mesh(mymesh)
    print("Simplifying {}".format(len(bm.verts)))
    # bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0)
    #for i in range(5):
    #    bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5, use_axis_x=True, use_axis_y=True,
    #                          use_axis_z=True)  # , mirror_clip_x, mirror_clip_y, mirror_clip_z, clip_dist, use_axis_x, use_axis_y, use_axis_z)¶
    
    #finds groups of vertices closer than dist and merges them together 
    bmesh.ops.automerge(bm, verts=bm.verts, dist=1e-6)
    
    for f in bm.faces:
        f.smooth = True
    
    #here it applies all the changes to the mesh
    bm.to_mesh(mymesh)
    print("Done {}".format(len(bm.verts)))
    bm.free()
    """
    #nothing has changed for the vertices with the triangles
    #print(new_verts)
    #print(new_faces)
    
    #************with these array of tuples can't just import 
    #it because will give you ther error:
    #The truth value of an array with more than one element 
    #is ambiguous. Use a.any() or a.all()
    #*************#
    #calculating the edges and the normals
    mymesh.update(calc_edges=True)
    mymesh.calc_normals()


    #mymesh.validate()
    
    #for i in range(0,10):  
    #    print(mymesh.vertices[i].co)


    
    #print("filename right before list = " + filename)
    #objects_Matching_filename = [x for x in object_List if "neuron_mesh_36706215" in x]
    
    #print(objects_Matching_filename)
    object = bpy.data.objects.new("neuron-"+ID, mymesh)
    #object.location = bpy.context.scene.cursor_location
    object.location = Vector((0,0,0))
    bpy.context.scene.objects.link(object)
    
    object.lock_location[0] = True
    object.lock_location[1] = True
    object.lock_location[2] = True
    object.lock_scale[0] = True
    object.lock_scale[1] = True
    object.lock_scale[2] = True

    #rotate the z direction by 90 degrees so point correct way
    
    #object.rotation_euler[2] = 1.5708

    
    object.rotation_euler[0] = 1.5708
    object.rotation_euler[1] = 0
    object.rotation_euler[2] = 0



    
    object.lock_rotation[0] = True
    object.lock_rotation[1] = True
    object.lock_rotation[2] = True


    #set view back to normal:
    set_View()
    
    
    
    
    #run the setup color command
    #bpy.ops.object.select_all(action='TOGGLE')
    create_local_colors(object)
    
    
    


    #need to add the labels to the newly created object
    ob = bpy.data.objects["neuron-"+ID]


    me = ob.data
    me.use_customdata_edge_bevel = True
    me.use_customdata_vertex_bevel = True

    #print("starting to hide everything")
    #iterate through all of the vertices
    verts_raw = ob.data.vertices
    #print(len(active_verts_raw))

    edges_raw = ob.data.edges

    #print(len(active_edges_raw))

    faces_raw = ob.data.polygons

    #verts_labels = neuron_labels['vertices']
    #triangles_labels = neuron_labels['traingles']
    #edges_labels = neuron_labels['edges']

    bevel_Weights = get_Bevel_Weights()

    #iterate through all of the 
    """for i,k in enumerate(verts_raw):
        #set the bevel weight
        k.bevel_weight = bevel_Weights[int(verts_labels[i])]

    for i,k in enumerate(edges_raw):
        #set the bevel weight
        k.bevel_weight = bevel_Weights[int(edges_labels[i])]"""


    #iterate through all of the face

    for i,k in enumerate(faces_raw):
        k.material_index = int(triangles_labels[i])
        

    #go into edit mode
    print("trying to select the neuron")

    bpy.context.scene.objects.active = bpy.context.scene.objects["neuron-"+ID]
    #bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.select_all(action='TOGGLE')
    edit_active_neuron()


    is_label_hidden = get_Hide_Flag()

    #make sure the faces are hidden if they should be
    if (is_label_hidden == True):
        hide_Labeled(mode=0,waitTime=0)
    else:  #show all of the faces
        #print ("Property Disabled")
        hide_Labeled(mode=1,waitTime=0)
        
    #setup the colors for the neuron
    
    #make sure in solidDID NOT Find ID in labeles table mode
    for area in bpy.context.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.viewport_shade = 'SOLID' # set the viewport shading to rendered
    
    bpy.ops.object.mode_set(mode='OBJECT')


    return






#-----------------------------------------------importing local saved off files------------------------------------------------------------------------









def importMesh(filepath, filename = "neuron"):
    #'C:/Users/svc_atlab/Documents/Celii/Blender_Plugin/neuron_mesh_36706215_with_edges.npz'
    #script that will add the neuron as mesh
    try:
        mesh_data = np.load(filepath)
    except:
        print("there is no file with that name in the same directory as the Blender project!")
        return
    """
        #####My way of importing that worked for the simpler meshes###########
        #print("inside loop")
        verts_old = mesh_data['vertices'].tolist()
        faces = mesh_data['triangles'].tolist()

        verts = [Vector(l) for l in verts_old]

    """
    """for i in range(0,10):
      print(verts_old[i])
      print(verts[i])"""
          
      
    
    #-----------------new way of importing using james cotton way------------------#
    verts = mesh_data['vertices'].astype(dtype=np.int32)
    faces = mesh_data['triangles'].astype(dtype=np.uint32)
    
    #*********Need to add in my own scale*****************#
    scale = 0.001
    
    #Makes the vertices as voxels and applies a scale to them
    not_centered_verts = [(x[0], x[2], x[1]) for x in (verts * scale).tolist()]
    #don't need the vertical count being added to the faces like in his because because 
    new_faces = [(x[0], x[1], x[2]) for x in (faces).tolist()]

    offset = np.median(np.array(not_centered_verts), axis=0)
    print("offset = " + str(offset))
    new_verts = [(x[0] - offset[0], x[1] - offset[1], x[2] - offset[2]) for x in not_centered_verts]
 
 
    #print(new_verts)
    #print(new_faces)
    #-----------------END OF new way of importing using james cotton way------------------#
    mymesh = bpy.data.meshes.new(filename)
    mymesh.from_pydata(new_verts, [], new_faces)
    #object = bpy.data.objects.new(optional_Name, mesh)
    
    #uses the bmesh library to import:
    bm = bmesh.new()
    bm.from_mesh(mymesh)
    print("Simplifying {}".format(len(bm.verts)))
    # bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0)
    #for i in range(5):
    #    bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5, use_axis_x=True, use_axis_y=True,
    #                          use_axis_z=True)  # , mirror_clip_x, mirror_clip_y, mirror_clip_z, clip_dist, use_axis_x, use_axis_y, use_axis_z)¶
    
    #finds groups of vertices closer than dist and merges them together 
    #bmesh.ops.automerge(bm, verts=bm.verts, dist=1e-6)
    
    for f in bm.faces:
        f.smooth = True
    
    #here it applies all the changes to the mesh
    bm.to_mesh(mymesh)
    print("Done {}".format(len(bm.verts)))
    bm.free()
    
    #nothing has changed for the vertices with the triangles
    #print(new_verts)
    #print(new_faces)
    
    #************with these array of tuples can't just import 
    #it because will give you ther error:
    #The truth value of an array with more than one element 
    #is ambiguous. Use a.any() or a.all()
    #*************#
    #calculating the edges and the normals
    mymesh.update(calc_edges=True)
    mymesh.calc_normals()


    #mymesh.validate()
    
    #for i in range(0,10):  
    #    print(mymesh.vertices[i].co)


    
    #print("filename right before list = " + filename)
    #objects_Matching_filename = [x for x in object_List if "neuron_mesh_36706215" in x]
    
    #print(objects_Matching_filename)
    object = bpy.data.objects.new(filename, mymesh)
    #object.location = bpy.context.scene.cursor_location
    object.location = Vector((0,0,0))
    bpy.context.scene.objects.link(object)
    
    object.lock_location[0] = True
    object.lock_location[1] = True
    object.lock_location[2] = True
    object.lock_scale[0] = True
    object.lock_scale[1] = True
    object.lock_scale[2] = True

    #rotate the z direction by 90 degrees so point correct way
    
    #object.rotation_euler[2] = 1.5708
    object.rotation_euler[0] = 4.53786
    object.rotation_euler[1] = 0.698132
    object.rotation_euler[2] = 0


    
    object.lock_rotation[0] = True
    object.lock_rotation[1] = True
    object.lock_rotation[2] = True


    #set view back to normal:
    set_View()
    
    
    #run the setup color command
    #bpy.ops.object.select_all(action='TOGGLE')
    create_local_colors(object)
    
    
    """bpy.context.space_data.cursor_location[0] = 0
    bpy.context.space_data.cursor_location[1] = 0
    bpy.context.space_data.cursor_location[2] = 0
    
    
    for i in objects_Matching_filename:
        bbpy.data.objects[i].select = True
    
    bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)"""
    
    #now deselect them if want
#will go through ever vertex and face and either hide or show it if it is labeled (based on what mode it is in)
#mode 0 = HIDE EVERY VERTEX/POLYGON THAT IS LABELED
#mode 1 = SHOW EVERY VERTEX/POLYGON THAT IS LABELED  
def hide_Labeled(mode=0,waitTime=0):
    #if going to hide everything, can optionally make it weight 2 seconds before doing it:
    
    """ THIS IS JUST EXECUTING SCRIPT AND THEN WAITING 2 SECONDS,
    CANT GET BLENDER TO WAIT WITHOUT USING A MODULE, NOT WORTH IT....`
    if(mode == 0 and waitTime > 0):
        time.sleep(2)
    """
        
    #put into object mode so that the changes will be persisted    
    currentMode = bpy.context.object.mode
    


    if(mode ==0): #then we will hide everything
        bpy.ops.object.mode_set(mode='OBJECT')
        ob = bpy.context.object
        ob.update_from_editmode()
        
        me = ob.data
        me.use_customdata_edge_bevel = True
        me.use_customdata_vertex_bevel = True
        
        #print("starting to hide everything")
        #iterate through all of the vertices
        verts_raw = ob.data.vertices
        edges_raw = ob.data.edges
        """active_verts_raw = [k for k in verts_raw if k.bevel_weight > 0.0]
        #print(len(active_verts_raw))
        
        edges_raw = ob.data.edges
        
        active_edges_raw = [k for k in edges_raw if k.bevel_weight > 0.0]"""
        #print(len(active_edges_raw))
        
        #
        faces_raw = ob.data.polygons
        
        max_length = accepted_color_length()
        active_faces_raw = [k for k in faces_raw if (k.material_index > 1 and k.material_index < max_length)]
        not_accepted_faces = np.unique([ob.data.materials[k.material_index].name for k in faces_raw if (k.material_index <= 1 or k.material_index >= max_length)]).tolist()
        
        #active_faces_raw = [k for k in faces_raw if k.material_index > 1]
        
        for k in verts_raw:
            k.select = False
            k.hide = True
        
        for k in edges_raw:
            k.select = False
            k.hide = True
        
        
        
        #iterate through all of the face
        
        #print(active_faces_raw)
        for k in active_faces_raw:
            k.select = False
            k.hide = True
        
        
        
        
        """
        #go through all of the edges and hide them as well
        edges = ob.data.edges
        face_edge_map = {ek: edges[i] for i, ek in enumerate(ob.data.edge_keys)}
        
        
        #might be making a list
        e = []
        for i in active_faces_raw:
            for ed in i.edge_keys:
                if not(face_edge_map[ed] in e):
                    e.append(face_edge_map[ed])
        
        indexes = [jk.index for jk in e]
        
        for k in edges:
            if k.index in indexes:
                k.select = False
                k.hide = True

        """



        """
        print(e)        
        for kk in e:
            print(kk.index)
            k.select = False
            k.hide = True
            print(k.hide)
        
        for yy in e:
            print(yy.hide)
        """

        
        
        """
        #go through using bmesh to make sure they are actually hidden
        mesh = bmesh.from_edit_mesh(ob.data)
        
        
        active_verts = [k for k in mesh.verts if k.hide == True]
        #print(active_verts)

        #go through and unhighlight the verts
        for k in active_verts:
            k.hide_set(True) 


        #get the list of active faces
        active_faces = [k for k in mesh.faces if k.hide == True]
        #print(active_faces)
        print(active_faces_raw)
        #go through and unhighlight the verts
        for k in active_faces:
            k.hide_set(True) 

        """
        if(currentMode == 'EDIT'):
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
            
        print("done hiding")
        ob.data.update()   
    elif(mode ==1): #then we will show everything, so just show  every face and vertex
        print("starting to UN-hide everything")
        
        bpy.ops.object.mode_set(mode='OBJECT')
        ob = bpy.context.object
        ob.update_from_editmode()
        
        verts_raw = ob.data.vertices
        faces_raw = ob.data.polygons
        edges_raw = ob.data.edges
        
        for k in verts_raw:
            k.hide = False
        
        #iterate through all of the face

        for k in faces_raw:
            k.hide = False
        
        for k in edges_raw:
            k.hide = False
        
        
        #GO BACK INTO whatever mode you were on
        if(currentMode == 'EDIT'):
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
        
        #OLD WAY OF GOING IT WITH BMESH
        """mesh = bmesh.from_edit_mesh(ob.data)
        
        for k in mesh.verts :
            k.hide_set(False) 


        #get the list of active faces
        for k in mesh.faces:
            k.hide_set(False) 

        print("done UN-hiding")
        ob.data.update()"""
       
        
    else:
        print("incorrect mode entered")
        return
    ob.data.update()

def get_Hide_Flag():
    #insert code here
    return bpy.context.scene.my_tool.hide_Labels

#---------------------------NON DATAJOINT WAY OF GETTING COLORS---------------------#
'''
def getColors():
    return ["no_color","no_color","Apical (blue)","Basal (yellow)","Oblique (green)"
                    ,"Soma (red)","Axon-Soma (aqua)","Axon-Dendr (off blue)","Dendrite (purple)","Distal (pink)",
                    "Error (brown)","Unlabelable (tan)","Cilia (soft blue)","Spine Head (rose)",
                    "Spine (light pink)","Spine Neck (light green)","Bouton (aqua)"]
def getLabels():
    return ["None","None","Apical","Basal","Oblique","Soma","Axon-Soma","Axon_Dendr","Dendrite","Distal","Error","Unlabelable","Cilia","Spine Head","Spine","Spine Neck","Bouton"]

def get_Bevel_Weights():
    return [0.00,0.00,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16]
'''
#------------------------------------------------#


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

#new way to get color names through datajoint
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




from collections import Counter

def get_quick_Label():
    #insert code here
    
    #return bpy.context.scene.my_tool.quick_Label
    return bpy.context.scene.quick_label_2
    #hide_Labels

def generate_verts_to_face_dictionary(faces_raw,verts_raw):
    verts_to_Face = {pre_vertex.index:[] for pre_vertex in verts_raw}
    #verts_to_Label = {pre_vertex.index:[] for pre_vertex in verts_raw}
    
    
    for face in faces_raw:
        #get the vertices
        verts = face.vertices
        #add the index to the list for each of the vertices
        for vertex in verts:
            verts_to_Face[vertex].append(face.index)
    
    #use the verts to face to create the verts to label dictionary
    """for vert,face_list in verts_to_Face.items():
        diff_labels = [labels_list[fc] for fc in face_list]
        #print(list(set(diff_labels)))
        verts_to_Label[vert] = list(set(diff_labels))"""
    
        
            
    return verts_to_Face#,verts_to_Label

#function that does the labeling: 
def main(context, var):
    print("labeling neuron")
    print("button pressed was " + var)
    #bpy.context.object.active_material=bpy.data.materials["blue"]
    
    colors = getColors()
    
    #get the index and set the color equal to that

    #might need to update who is getting selected so use this

    
    if(var in colors):
        print("var in colors")
        index_number = colors.index(var)
        if(var == 'no_color'):
            index_number = 1
            
        #get all of the vertices and set bevel_weight and select = False
        bpy.ops.object.mode_set(mode='OBJECT')
        ob = bpy.context.object
        me = ob.data
        ob.update_from_editmode()
        
        #print("inside labeling edges raw length = " + str(len(active_edges_raw)))
        verts_raw = ob.data.vertices
        faces_raw = ob.data.polygons
        active_faces_raw_indexes = [i for i,k in enumerate(faces_raw) if k.select > 0]
        print("active_faces_raw_indexes = " + str(active_faces_raw_indexes))
        ####Algorithm for crawling along faces####
        """
        1) for each active face:
            a. get the material index
            b. crawl from that face to all other nearby faces with the same material index and relabel them with desired label

        How to crawl to each face:
            a. Pre-work: Make faces to vert dictionary
            b. Get the verts of the current face and add to verts_to_be_checked
            c. While verts_to_be_checked is not empty 
                1. Pop off the next verts_to_be_checked and put in verts_checked
                2. Get the faces associated with the vert
                3. For each face associated with the vert
                    a. if face has the same index
                        a. if has not in final faces:
                            1. Add the face to the final faces
                            2. Add all vertices that have not been checked or in verts_to_be_checked to the verts_to_be_checked list
        For all faces in final faces list:
            Change material index

            

        """
        verts_to_Face = generate_verts_to_face_dictionary(faces_raw,verts_raw)
        #print("verts_to_Face = " + str(verts_to_Face.values()))
        quick_Label_Status = get_quick_Label()
        if quick_Label_Status == True:
            print("inside_quick_label")
            counter = 1
            while len(active_faces_raw_indexes) > 0:
                print("Round " + str(counter) + " in active_faces loop")
                verts_to_check = set()
                verts_checked = set()
                faces_to_change = set()
                
                first_face_index = active_faces_raw_indexes.pop(0)
                first_face = faces_raw[first_face_index]
                current_material = first_face.material_index
                for v in first_face.vertices:
                    verts_to_check.add(v)
                #print("len(verts_to_check )= " + str(len(verts_to_check)))
                while len(verts_to_check) > 0:
                    print("len(verts_to_check) = " + str(len(verts_to_check)))
                    current_vert = verts_to_check.pop()
                    verts_checked.add(current_vert)
                    faces_associated_vert = verts_to_Face[current_vert]
                    #print("faces_associated_vert = " + str(faces_associated_vert))
                    #print("first_face_index = " + str(first_face_index))
                    for fac in faces_associated_vert:
                        if faces_raw[fac].material_index == current_material:
                            #print("equal to first index")
                            
                            faces_to_change.add(fac)
                            verts_from_face = faces_raw[fac].vertices
                            for v in verts_from_face:
                                if v not in verts_checked:
                                    verts_to_check.add(v)
                
                #by this end will have all of the faces that we need to change
                print("len(faces_to_change) = " + str(len(faces_to_change)))
                for k in faces_to_change:
                    faces_raw[k].material_index = index_number
                    if(index_number == 1 or index_number == 0):
                        #print("setting hide to 0")
                        #print("setting hide to 0 in vertices")
                        k.hide = False
                
                #take off all of the faces that have been changed
                active_faces_raw_indexes = [k for k in active_faces_raw_indexes if k not in faces_to_change]

        else:
            #iterate through all of the faces and set the color and selection = false
            for k in active_faces_raw_indexes:
                faces_raw[k].material_index = index_number
                if(index_number == 1 or index_number == 0):
                    #print("setting hide to 0")
                    #print("setting hide to 0 in vertices")
                    k.hide = False

        ob.data.update() 
        bpy.ops.object.mode_set(mode='EDIT')
        ######-----old way of setting the color---------######  may need to do this if setting index doesn't work
        ######color wasn't being set so had to use this way
        #bpy.context.object.active_material_index = index_number
        #bpy.ops.object.material_slot_assign()
        
        
        #now need to hide all of the faces if the toggle switch is set to that
        hideFlag = get_Hide_Flag()  #Hasn't been implemented yet
        
        if hideFlag > 0:
            hide_Labeled(mode=0,waitTime=3)
        
        #reset the percent_labeled because there was an adjustment
        bpy.context.scene.percent_labeled = ""
        bpy.context.scene.percent_labeled_faces = ""
        bpy.context.scene.complete_100_check = ""
        bpy.context.scene.complete_100_check_2 = ""
        bpy.context.scene.complete_100_check_save_flag = False
        check_Verts_Match()
        return ""
    
    if var == "return User":
        bpy.ops.view3d.view_persportho()
        return ""
    if var == "Ortho Mode":
        bpy.ops.view3d.view_persportho()
        return ""
    if(var == "print"):
        ob = bpy.context.object
        if ob.type != 'MESH':
            print("Active object is not a Mesh")
            return None
        ob.update_from_editmode()
        me = ob.data

        labels_List = getLabels()
        return ("face index = " + str(me.polygons.active) + ": material = " + str(me.polygons[me.polygons.active].material_index))
    
    if(var == "exit_edit_mode"):
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return ""
    if(var == "delete_Neuron"):
        #check that 
        
        if bpy.data.objects.get("neuron-"+bpy.context.scene.delete_ID) is not None:
            bpy.context.scene.delete_Flag = True
            bpy.context.scene.delete_status = "Delete neuron-" + bpy.context.scene.delete_ID + "?"
        else:
            bpy.context.scene.delete_Flag = False
            bpy.context.scene.delete_status = "neuron-" + bpy.context.scene.delete_ID + " does not exist"
           
    if(var == "delete_No"):
        bpy.context.scene.delete_Flag = False
        bpy.context.scene.delete_status = ""
        bpy.context.scene.delete_ID = ""
        return ""
    if(var == "delete_Yes"):
        # deselect all
        bpy.ops.object.select_all(action='DESELECT')

        ID = bpy.context.scene.delete_ID
            
        if( ID != ""):
            bpy.data.objects['neuron-'+ID].select = True
 
            #delete the object from the scene
            bpy.ops.object.delete() 

            #now need to set all of the settings
            if(bpy.context.scene.delete_ID == bpy.context.scene.picked_Neuron_ID):
                reset_Scene_Variables(login_Flag=True)

            bpy.context.scene.delete_Flag = False
            bpy.context.scene.delete_status = ""
            bpy.context.scene.delete_ID = ""
                
        return ""
    return ""

    #else:
        #print out the labels of all of the active faces -- To be implemented
        



class Neuron_Label_Operator_a(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.neuron_label_operator_a"
    bl_label = "Neuron Label Operator_a"
    
    myVar = bpy.props.StringProperty(name="myVar")

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        print_Message = main(context,self.myVar)
        if not(print_Message == ""):
            self.report({'INFO'}, print_Message)
        return {'FINISHED'}

def reset_view_ortho(context, var):
    if var == "return User":
        bpy.ops.view3d.view_persportho()
        return 
    if var == "Ortho Mode":
        bpy.ops.view3d.view_persportho()
        return 
    if(var == "reset view"):
        set_View()
        return 



class changing_view(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.changing_view"
    bl_label = "changing_view"
    
    myVar = bpy.props.StringProperty(name="myVar")

    #@classmethod
    #def poll(cls, context):

    def execute(self, context):
        reset_view_ortho(context,self.myVar)
        return {'FINISHED'}

    
class setup_colors(bpy.types.Operator):
    bl_idname = "object.setup_colors"
    bl_label = "setup colors"
 

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        create_local_colors()
        return {'FINISHED'}
    
class load_file_operator(bpy.types.Operator):
    bl_idname = "object.load_file_operator"
    bl_label = "load_file_operator"
 
    filepath = bpy.props.StringProperty(subtype="FILE_PATH")

 
    def execute(self, context):
        ##need to access component of panel and set it
        """print("File path = " + self.filepath)
        path, filename = os.path.split(self.filepath)
        print("filename = "+ filename)
        finalName, ext = os.path.splitext(filename)
        print("finalName = "+ finalName)
        
        print("ext = "+ ext)"""
        ##need to access component of panel and set it
        import_local_Neuron(self.filepath)
        return {'FINISHED'}
 
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class submit_registration_login(bpy.types.Operator):
    bl_idname = "object.submit_registration_login"
    bl_label = "submit_registration_login"
    
    myRegVar = bpy.props.StringProperty(name="myRegVar")
 

    def execute(self, context):
        print("inside execute submit registration")
        return {'FINISHED'}
 
    def invoke(self, context, event):
        #where you want to put your code
        print("inside invoke submit registration login")
        
        #create an AuthorsLast object and pull all the usernames
        authors =ta3.AuthorsLast()
        current_User_Names = authors.fetch('username')
        
        if self.myRegVar == "logout":
            #need to reset all of the flags:
            reset_Scene_Variables()
            return {'RUNNING_MODAL'}
            
            
        
        if self.myRegVar == "register":
            username = context.scene.username 
            
            firstname = context.scene.firstname 
            lastname = context.scene.lastname 
            name = firstname + " " + lastname
            
                    
            print(username)
            print(name)
            if username in current_User_Names:
                print("inside register username matched")
                #set the text filed to say that username already exists and to pick new username
                context.scene.register_Status = "Username already exists, pick another"
            else:
                print("inside register username NOT!!! matched")
                authors.insert1((username,name),skip_duplicates=True)
                context.scene.register_Status = "Succss! Please login with username"
                
                #clear out the register button
                context.scene.firstname = ""
                context.scene.lastname = ""
                context.scene.username = ""
                
                
        else:
            username = context.scene.username_Login 
            if username in current_User_Names:
                
                print("inside login user name matched")
                bpy.context.scene.login_Status = "Welcome " + username +": Now Pick Neuron"
                
                
                #should enable all of the other things
                #context.scene.firstname = "hellow"
                context.scene.login_Flag = True
                #bpy.context.screen.areas["VIEW_3D"].region["TOOLS"].tag_redraw()
            else:
                print("inside login user name NOT!!! matched")
                bpy.context.scene.login_Status = "Failure! username not registered"
                context.scene.login_Flag = False
                #make sure the login panel is visible
           
                
            
        
        return {'RUNNING_MODAL'}

def hide_Labels_Func(self, context):  #the call if you do something with button 1
    #if checked then hide all the labeled faceds
    if (self.hide_Labels == True):
        hide_Labeled(mode=0,waitTime=0)
    else:  #show all of the faces
        #print ("Property Disabled")
        hide_Labeled(mode=1,waitTime=0)
            
def visible_only_selection_func(self, context):  #the call if you do something with button 2
    #print("hello from button 2")
    if (self.visible_only_selection == True):
        bpy.context.space_data.use_occlude_geometry = True
    else:
        bpy.context.space_data.use_occlude_geometry = False

def hide_Box_func(self, context):  #the call if you do something with button 2
    #print("hello from button 2")
    #check to see that bounding box exists:
    if "bounding_box" in bpy.data.objects.keys():
        ob = bpy.data.objects['bounding_box']
        
        if (self.hide_Box == True):
            ob.hide = True
        else:
            ob.hide = False

            
class MySettings(PropertyGroup):

    hide_Labels = BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = True,
        update = hide_Labels_Func
        )
    visible_only_selection = BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = False,
        update = visible_only_selection_func
        )
    hide_Box = BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = False,
        update = hide_Box_func
        )
    quick_Label = BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = True,
        update = None
        )
    
    


class RegisterPanel(bpy.types.Panel):
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "Register Layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Register"
    bl_context = "objectmode"

    #register_Status = bpy.props.StringProperty(name="Register Statuts")
    #where to put the file name****************Not implemented yet
    register_Status = "regis. status";
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "OBJECT":
            
            layout.row().prop(context.scene, "username")
            layout.row().prop(context.scene, "firstname")
            layout.row().prop(context.scene, "lastname")
            
            row = layout.row()
            #row.scale_y = 3.0
            row.label(text = "") 
            row.operator("object.submit_registration_login", text="submit").myRegVar = "register"
            row = layout.row()
            #to be edited based on the registration status
            row.label(text = scene.register_Status) 
             
            
            """
            # find the next text
            col = layout.column(align=True)
            row = col.row(align=True)
            row.prop(st, "find_text", text="")
            row.operator("text.find_set_selected", text="", icon='TEXT')
            col.operator("text.find")"""


class LoginPanel(bpy.types.Panel):
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "Login Layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Login"
    bl_context = "objectmode"

    
    #where to put the file name****************Not implemented yet

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "OBJECT":
            
            layout.row().prop(context.scene, "username_Login")
            
            row = layout.row()
            #row.scale_y = 3.0
            row.label(text = "") 
            row.operator("object.submit_registration_login", text="login").myRegVar = "login"
            row = layout.row()
            #to be edited based on the registration status
            row.label(text = bpy.context.scene.login_Status) 
            




################  Main function for picking the neuron you want to load  ##################
def main_Neuron_Picker(context, var):
    print("inside main_Neuron_Picker")
    print("var = "+var)
    
    if var == "ID":
        #retrieve the neuron ID
        ID = bpy.context.scene.neuron_ID
        print("ID = " + ID)
        ID = int(ID)
        
        
        #check edited database first for the neuron looking for
        currentLabels = ta3p100.ProofreadLabelOrphan()
        
        labeled_IDs = (currentLabels & "decimation_ratio=0.35").fetch("segment_id").tolist()
        
        if ID in labeled_IDs:
            print("Found ID in labeles table")
            #print success and the name of the neuron: Hover over Edit 
            bpy.context.scene.neuron_ID_Status = "Found " + str(ID) + ": Click Import"
            bpy.context.scene.labeled_Flag_ID = True
        else:
            print("DID NOT Find ID in labeles table")
            bpy.context.scene.labeled_Flag_ID = False
            
        #check edited database first for the neuron looking for
        #mesh_Table = ta3.Mesh()
        
        
        #mesh_Dict = (ta3.Decimation & primary_key & "segment_id="+ID).fetch(as_dict=True)[0]

        #all_IDs = mesh_Table.fetch("segment_id").tolist()   
        primary_key = dict(segmentation=2,decimation_ratio=0.35) 
        all_IDs = (ta3p100.CleansedMeshOrphan()  & primary_key).fetch("segment_id").tolist()  
        
        if ID in all_IDs:
            print("Found ID in Mesh table")
            bpy.context.scene.labeled_Flag_ID = False
            #print success and the name of the neuron: Click Edit to start Labeling
            bpy.context.scene.neuron_ID_Status = "Found " + str(ID) + ": Click Import"
            bpy.context.scene.import_Neuron_Flag = True
        else:
            print("DID NOT Find ID in the mesh table")
            if bpy.context.scene.labeled_Flag_ID == True:
                print("ERROR: FOUND NEURON IN LABELS BUT NOT MESH TABLE")
            bpy.context.scene.neuron_ID_Status = "Can't Find " + str(ID) + ": Try Another ID"
            
            bpy.context.scene.labeled_Flag_ID = False
            bpy.context.scene.import_Neuron_Flag = False
            return ""
            
    elif var == "next_unlabeled":
        #get all of the list of neurons that have never been pulled
        #use the difference operator in datajoint
        #mesh_Table = ta3.Mesh()
        labels = ta3p100.ProofreadLabelOrphan()
        primary_key = dict(segmentation=2,decimation_ratio=0.35) 
        mesh_list = (ta3p100.CleansedMeshOrphan()  & primary_key).fetch("segment_id").tolist() 
        #mesh_list = mesh_Table.fetch("segment_id").tolist() #old way of doing it without decimated list
        labels_list = labels.fetch("segment_id").tolist()

        difference = [x for x in mesh_list if x not in labels_list]
        #print(difference)
        
        #make the 1st neuron in the list the one you get and then go 
        #straight into editing it so there are no pull issues
        
        if len(difference) > 0:
            bpy.context.scene.next_available_neuron_ID = str(difference[0])
            bpy.context.scene.next_available_status = "Found neuron:" + str(difference[0])
            #call function that will automatically load neuron
            print("Found neuron:" + str(difference[0]))
            load_Neuron(str(difference[0]))
        else:
            bpy.context.scene.next_available_status = "NO MORE NEURONS LEFT"
            bpy.context.scene.next_available_neuron_ID = "-1"
        
        
    elif var == "username_edit":
        #means want to send the neuron picked by the username to be edite
        if bpy.context.scene.status_picked !="":
            print("Want to edit neuron picked by username")
            bpy.context.scene.neuron_username_Status = "LOADING NEURON FOR EDIT..."
            load_Neuron(bpy.context.scene.username_neuron_ID)
            
        else:
            print("ERROR: ALLOWED TO CLICK EDIT WHEN NO NEURON STORED FOR USERNAME!")
    elif var == "ID_edit":
        #means want to send the neuron picked by the username to be edite
        if bpy.context.scene.neuron_ID !="":
            print("Want to edit neuron picked by ID")
            bpy.context.scene.neuron_ID_Status = "LOADING NEURON FOR EDIT..."
            load_Neuron(bpy.context.scene.neuron_ID)
            
        else:
            print("ERROR: ALLOWED TO CLICK EDIT WHEN NO NEURON STORED FOR USERNAME!")
    elif var == "exit_edit":
            bpy.ops.object.mode_set(mode='OBJECT')
            
                       
    else:
        return ""
    return ""

class picking_neuron(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.picking_neuron"
    bl_label = "picking_neuron"
    
    myPickVar = bpy.props.StringProperty(name="myPickVar")

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        print_Message = main_Neuron_Picker(context,self.myPickVar)
        if not(print_Message == ""):
            self.report({'INFO'}, print_Message)
        return {'FINISHED'}


class ID_neuron_picker(bpy.types.Panel):
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "ID_neuron_picker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Add Neuron by ID"
    bl_context = "objectmode"

    
    #where to put the file name****************Not implemented yet
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "OBJECT":
            row = layout.row()
            row.prop(context.scene, "neuron_ID")
            if context.scene.login_Flag != True:
                row.enabled = False
            else:
                row.enabled = True
            
            row = layout.row()
            #row.scale_y = 3.0
            row.label(text = "") 
            row.operator("object.picking_neuron", text="Get Neuron").myPickVar = "ID"
            if context.scene.login_Flag == False or bpy.context.scene.neuron_ID == "":
                row.enabled = False
            else:
                row.enabled = True

            row = layout.row()
            #to be edited based on the registration status
            row.label(text = scene.neuron_ID_Status) 
            
             
            row = layout.row()
            #row.scale_y = 3.0
            row.operator("object.picking_neuron", text="Import Neuron").myPickVar = "ID_edit"
            if context.scene.login_Flag == False or bpy.context.scene.neuron_ID == "" or bpy.context.scene.import_Neuron_Flag == False:
                row.enabled = False
            else:
                row.enabled = True
            

def execute_operator_neuron_picker(self, context):
    print(self.primitive)
    bpy.context.scene.status_picked = self.primitive

class statusProperties(bpy.types.PropertyGroup):
    mode_options = [
        ("partial","partial",""),
        ("complete","complete",""),

    ]

    primitive = bpy.props.EnumProperty(
        items=mode_options,
        description="offers....",
        default="partial",
        update=execute_operator_neuron_picker
    )

#will get the names of the neurons that belong to that user
def get_username_neurons(self, context):
    
    empty = [("None yet","None yet","")]
    if bpy.context.scene.username_Login != "" and bpy.context.scene.status_picked != "":
        #go get the names of the neurons from the database
        labels = ta3p100.ProofreadLabelOrphan()
        username = "'" + bpy.context.scene.username_Login + "'"
        filtered_labels = (labels & "author_proofreader="+username
                                    &"status='"+bpy.context.scene.status_picked+"'")
        neurons_for_user = filtered_labels.fetch("segment_id").tolist()
        
        
        #based on the length of the number of neurons that fit that description
        if len(neurons_for_user) <= 0:
            #return an empty tuple and print out to the user that no neurons fit that 
            context.scene.neuron_username_Status = "None of your neurons fit group"
            empty = [("None yet","None yet","")]
            return mode_options
        else:
            #put the list into an enum
            items = []
            for i in neurons_for_user:
                items.append((str(i),str(i),""))
                
            print("items = ")
            print(items)
            return items
    else:
        return empty
   

#will execute when item is picked
def execute_operator_username_neuron_picked(self, context):
    print("inside function called after picked neuron in username tab")
    bpy.context.scene.username_neuron_ID = self.primitive
    ID = bpy.context.scene.username_neuron_ID
    bpy.context.scene.neuron_username_Status = "Found " + str(ID) + ": Click Import to start labeling"
    
    
    
class myNeuronsProperties(bpy.types.PropertyGroup):
    primitive = bpy.props.EnumProperty(
        items=get_username_neurons,
        description="offers....",
        update=execute_operator_username_neuron_picked
    )




####to be used for saving the neurons
           
       
class username_neuron_picker(bpy.types.Panel):
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "username_neuron_picker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Add Personal Neurons"
    bl_context = "objectmode"

    
    #where to put the file name****************Not implemented yet
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "OBJECT":
            col = layout.column()
            col.label(text="Pick Status:")
            col.prop(context.scene.my_status_properties, "primitive")
            if context.scene.login_Flag != True:
                col.enabled = False
            else:
                col.enabled = True
                
            col = layout.column()
            col.label(text="Pick Neuron:")
            col.prop(context.scene.my_neurons_properties, "primitive")
            if context.scene.login_Flag != True or bpy.context.scene.status_picked == '':
                col.enabled = False
            else:
                col.enabled = True
            """    
            row = layout.row()
            #row.scale_y = 3.0
            row.label(text = "") 
            row.operator("object.picking_neuron", text="Get Neuron").myPickVar = "username"
            if context.scene.login_Flag != True or bpy.context.scene.status_picked == '' or bpy.context.scene.username_neuron_ID:
                col.enabled = False
            else:
                col.enabled = True"""
            
            
            row = layout.row()
            #to be edited based on the registration status
            row.label(text = scene.neuron_username_Status) 
            
            row = layout.row()
            #row.scale_y = 3.0
            row.operator("object.picking_neuron", text="Import Neuron").myPickVar = "username_edit"
            if context.scene.login_Flag != True or bpy.context.scene.status_picked == '' or bpy.context.scene.username_neuron_ID =="":
                row.enabled = False
            else:
                row.enabled = True
                
                        

        
  
class next_unlabeled_neuron_picker(bpy.types.Panel):
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "next_unlabeled_neuron_picker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Next Unlabeled Neuron"
    bl_context = "objectmode"

    neuron_Status = bpy.props.StringProperty(name="next_unlabeled_neuron_picker_Status")
    #where to put the file name****************Not implemented yet
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "OBJECT":
            row = layout.row()
            row.operator("object.picking_neuron", text="Import Neuron").myPickVar = "next_unlabeled"
            if context.scene.login_Flag != True:
                row.enabled = False
            else:
                row.enabled = True
                
            row = layout.row()
            #to be edited based on the registration status
            row.label(text = bpy.context.scene.next_available_status) 

class load_local_neuron(bpy.types.Panel):
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "load_local_neuron"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Load Local Neuron"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "OBJECT":
            row = layout.row()         
            row.operator("object.load_file_operator", text="Load File")
            if context.scene.login_Flag != True:
                row.enabled = False
            else:
                row.enabled = True
                
            row = layout.row()
            #to be edited based on the registration status
            row.label(text = bpy.context.scene.load_local_status) 







"""def execute_operator(self, context):
    eval('bpy.ops.' + self.primitive + '()')
    

class statusSavingProperties(bpy.types.PropertyGroup):
    mode_options = [
        ("mesh.primitive_plane_add", "Plane", '', 'MESH_PLANE', 0),
        ("mesh.primitive_cube_add", "Cube", '', 'MESH_CUBE', 1),
        ("mesh.primitive_circle_add", "Circle", '', 'MESH_CIRCLE', 2),
        ("mesh.primitive_uv_sphere_add", "UV Sphere", '', 'MESH_UVSPHERE', 3),
        ("mesh.primitive_ico_sphere_add", "Ico Sphere", '', 'MESH_ICOSPHERE', 4),
        ("mesh.primitive_cylinder_add", "Cylinder", '', 'MESH_CYLINDER', 5),
        ("mesh.primitive_cone_add", "Cone", '', 'MESH_CONE', 6),
        ("mesh.primitive_torus_add", "Torus", '', 'MESH_TORUS', 7)
    ]

    saving_status = bpy.props.EnumProperty(
        items=mode_options,
        description="offers....",
        default="mesh.primitive_plane_add",
        update=execute_operator
    )

"""

def calculate_Percent_Labeled():
    #going to calculate the number of neurons labeled:
    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    me = ob.data
    faces_raw = ob.data.polygons
    not_accepted_faces = []
    max_length = accepted_color_length()
    active_faces_raw = [k for k in faces_raw if (k.material_index > 1 and k.material_index < max_length)]
    not_accepted_faces = np.unique([ob.data.materials[k.material_index].name for k in faces_raw if (k.material_index <= 1 or k.material_index >= max_length)]).tolist()
    
    #ob.data.materials[4].name)
    
    #648518346349386137
    
    total_faces = len(faces_raw)
    total_labeled_faces = len(active_faces_raw)
    perc = round(float(total_labeled_faces)/float(total_faces)*100,2)
    bpy.context.scene.percent_labeled = str(perc)+"% faces labeled"
    bpy.context.scene.percent_labeled_faces = str(not_accepted_faces)
    
    



def execute_operator_neuron_saver(self, context):
    #set the variable that will be used for saving
    print("inside execute_operator_neuron_saver")
    bpy.context.scene.status_To_Save = self.primitive
    

    
    
class myNeuronsProperties_2(bpy.types.PropertyGroup):
    mode_options = [
        ("partial","partial",""),
        ("complete","complete",""),

    ]

    primitive = bpy.props.EnumProperty(
        items=mode_options,
        description="offers....",
        default="partial",
        update=execute_operator_neuron_saver
    )


def generate_verts_to_label(labels_list,faces_raw,verts_raw):
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
    
        
            
    return verts_to_Label

def generate_output_lists(final_faces_labels_list,faces_raw,verts_raw):
    
    verts_to_Label = generate_verts_to_label(final_faces_labels_list,faces_raw,verts_raw)
    
    
    output_verts_list = np.zeros(len(verts_raw))
    output_verts_list = [int(verts_to_Label[v][0]) for v in verts_to_Label]
    
    """for v in verts_to_Label:
        output_verts_list[v] = int(verts_to_Label[v][0])
        output_verts_list[v] = int(output_verts_list[v])"""
    #print("-----output_verts = %s---"%(time.time()-start_time_2))
    #print(output_verts_list)
    #print("output_verts_list in function = " + str(Counter(output_verts_list)))
    
    return output_verts_list 


def save_off_Neuron():
    #check to see if neuron was marked as complete and give warning if all
    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    me = ob.data
    faces_raw = ob.data.polygons
    max_length = accepted_color_length()
    active_faces_raw = [k for k in faces_raw if (k.material_index > 1 and k.material_index < max_length )]
   
    if len(faces_raw) != len(active_faces_raw) and bpy.context.scene.complete_100_check_save_flag==False and bpy.context.scene.status_To_Save=="complete":
        bpy.context.scene.complete_100_check = "Not all faces labeled!"
        bpy.context.scene.complete_100_check_2 = "If OK, press Send again"
        bpy.context.scene.complete_100_check_save_flag = True
        return
    
    bpy.context.scene.complete_100_check_save_flag = False

    #iterate through and save off all of the labels for vertices, edges, faces
    ID = bpy.context.scene.picked_Neuron_ID
    #need to add the labels to the newly created object
    ob = bpy.data.objects["neuron-"+ID]
    
    
    me = ob.data
    #print("starting to hide everything")
    #iterate through all of the vertices
    verts_raw = ob.data.vertices
    #print(len(active_verts_raw))
    
    edges_raw = ob.data.edges
    
    #print(len(active_edges_raw))
    
    faces_raw = ob.data.polygons
    
    #download the size from the database
    
        
    n_edges = len(edges_raw)
    n_vertices = len(verts_raw)
    n_triangles = len(faces_raw)
    
    print("n_vertices = " + str(n_vertices))
    
    #make sure that length of verts, triangles are 
    #same as in the mesh database
    
    primary_key = dict(segmentation=2,decimation_ratio=0.35)
    
    mesh_Dict = (ta3p100.CleansedMeshOrphan()  & primary_key & "segment_id="+ID).fetch(as_dict=True)[0]

    
    
    #mesh_Dict = (mesh_Table & "segment_id="+ID).fetch(as_dict=True)[0] old way
    n_vertices_check = mesh_Dict['n_vertices']
    n_traingles_check = mesh_Dict['n_triangles']
    
    if (n_vertices_check != n_vertices) or (n_traingles_check != n_triangles):
        print("n_vertices = " + str(n_vertices))
        print("n_vertices_check = " + str(n_vertices_check))
        print("n_triangles = " + str(n_triangles))
        print("n_traingles_check = " + str(n_traingles_check))
        
        print("ERROR: vertices and traingles do NOT match")
        raise ValueError("ERROR: vertices and traingles do NOT match")
        return
    
    
    edges = np.zeros(n_edges,dtype=np.uint8)
    #vertices = np.zeros(n_vertices,dtype=np.uint8)
    triangles = np.zeros(n_triangles,dtype=np.uint8)
    
    
    """#iterate through all of the 
    for i,k in enumerate(verts_raw):
        #set the bevel weight
        vertices[i] = math.ceil(k.bevel_weight*100)
    
    for i,k in enumerate(edges_raw):
        #set the bevel weight
        edges[i] = math.ceil(k.bevel_weight*100)
    """
    
    for i,k in enumerate(faces_raw):
        triangles[i] = k.material_index 
  
    #get ID, username, timestamp and status
    segment_ID = bpy.context.scene.picked_Neuron_ID
    #author_original = bpy.context.scene.author_original
    author_original = "computer_Auto"
    author_proofreader = bpy.context.scene.username_Login
    date_time = str(datetime.datetime.now())[0:19]
    status = bpy.context.scene.status_To_Save

    #insert the new row
    segmentation=2
    decimation_ratio=0.35


    #generate the vertices labels
    vertices = generate_output_lists(triangles, faces_raw, verts_raw)
    
    
    
    proof_Table.insert1([segmentation,int(segment_ID),decimation_ratio,author_original,author_proofreader,date_time,vertices,triangles,edges,status])
    
    #delete the current row with the same ID    #####need to decide if going to delete
    #(labels_Table & 'segment_ID='+ID).delete(verbose=False)
        
    #need to delete the neuron
    # deselect all
    bpy.ops.object.select_all(action='DESELECT')

    # selection
    bpy.data.objects['neuron-'+ID].select = True

    #need to save local copy
    #Name of folder = "local_neurons_saved"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    data_folder = Path(dir_path)
    just_Folder = data_folder.parents[0]
    complete_path_Obj = just_Folder / "local_neurons_saved"
    
    
    added_string = segment_ID+"_"+author_proofreader+"_"+ date_time[0:10]+"_"+date_time[11:].replace(":","-")+".npz"
    complete_path = complete_path_Obj / added_string
    
    complete_path = str(complete_path)
    
    """bpy.context.user_preferences.filepaths.script_directory = str(complete_path)
    
    

    
    
    file_parts = dir_path.split("/")
    file_parts.pop()
    #print(file_parts)
    just_Folder = "/".join(file_parts)
    complete_path = just_Folder + "local_neurons_saved"
    print(complete_path)"""
    
    
    #package up the data that would go to the database and save it locally name of the file will look something like this "4_bcelii_2018-10-01_12-12-34"
#    np.savez("/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/local_neurons_saved/"+segment_ID+"_"+author+"_"+
#        date_time[0:9]+"_"+date_time[11:].replace(":","-")+".npz",segment_ID=segment_ID,author=author,
#					date_time=date_time,vertices=vertices,triangles=triangles,edges=edges,status=status)
    np.savez(complete_path,segment_ID=segment_ID,author_original=author_original,author_proofreader=author_proofreader,
					date_time=date_time,vertices=vertices,triangles=triangles,edges=edges,status=status)
   
    
    print(segment_ID+"_"+author_proofreader+"_"+date_time[0:10]+"_"+date_time[11:].replace(":","-")+".npz")
    # remove it

    
    counter = 0
    #do a check that it was saved correctly 5 times and if not there
    new_Table = (proof_Table & "segment_id="+segment_ID  & "author_proofreader='"+author_proofreader+"'" & "date_time='"+date_time+"'").fetch()
    for i in range(0,5):
        if new_Table.size > 0:
            counter = counter + 1
        else:
            new_Table = (proof_Table & "segment_id="+segment_ID  & "author_proofreader='"+author_proofreader+"'" & "date_time='"+date_time+"'").fetch()
    
    if not(counter>0):
        print("ERROR: neuron data was not sent to database correctly but was saved locally")
        raise ValueError("ERROR: neuron data was not sent to database correctly but was saved locally")
    else:
        #delete the current row with the same ID, username and the last time updated
        old_date_time = bpy.context.scene.last_Edited
        (proof_Table & "segment_id="+segment_ID & "author_proofreader='"+author_proofreader+"'" & "date_time='"+old_date_time+"'").delete(verbose="False")
    
        #delete the object from the scene
        bpy.ops.object.delete() 
        
        #now need to set all of the settings
        reset_Scene_Variables(login_Flag=True)

#operator for the saving button
class finish_editing(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.finish_editing"
    bl_label = "finish_editing"
    

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        save_off_Neuron()
        return {'FINISHED'}


#operator for the saving button
class get_percent_labeled(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.get_percent_labeled"
    bl_label = "get_percent_labeled"
    

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        calculate_Percent_Labeled()
        return {'FINISHED'}


class MyDemoPanel(bpy.types.Panel):
    #Creates a Panel in the scene context of the properties editor
    bl_idname = "Tools_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Neuron Tab"

     
    #where to put the file name****************Not implemented yet
    file_Picked = "";
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        row = layout.row()
        
        
        # display the properties
        layout.prop(mytool, "hide_Box", text="Hide Box")

        # Create a simple row.
        #layout.label(text=" Neuron Labels")
        #for the editing properties
        row = layout.row()
        row.label(text = "Logged in: " + bpy.context.scene.username_Login)
        row = layout.row()
        row.label(text = "Neuron ID: " + bpy.context.scene.picked_Neuron_ID) 
        
        
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                #print("Printing view_perspective")
                #print(str(area.spaces.active.region_3d.view_perspective))
                user_persp = str(area.spaces.active.region_3d.view_perspective)
        
        #[area.spaces.active.region_3d.view_perspective for area in bpy.context.screen.areas if area.type == 'VIEW_3D']  
        
        if user_persp == "ORTHO":
            
            row = layout.row()
            row.label(text = "In Ortho Mode now")
            row = layout.row()
            row.operator("object.changing_view", text="reset view").myVar = "reset view"
            row = layout.row()
            row.operator("object.changing_view", text="Return User").myVar = "return User"
            row = layout.row()
            row.label(text = "Hotkey for User Mode:")
            row = layout.row()
            row.label(text = "1)Press Button above tab(`)")
            row = layout.row()
            row.label(text = "2)Press anywhere")
            row.label(text = "  on label neuron tab")
            
        
            
        else:
                
            
            if bpy.context.mode == "EDIT_MESH":
                # Big render button
                ######currently not using setup colors####
                #row = layout.row()
                #row.operator("object.setup_colors", text="setup colors")
                row = layout.row()
                row.label(text = "Last Edited: " + bpy.context.scene.last_Edited)
                row = layout.row()
                row.label(text = "Last Status: " + bpy.context.scene.last_Status) 
                
                row = layout.row()
                row.label(text = "Last User to Edit: " + bpy.context.scene.last_Edited_User ) 

                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.changing_view", text="Ortho Mode").myVar = "Ortho Mode"
                row = layout.row()
                row.operator("object.changing_view", text="reset view").myVar = "reset view"
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="remove Label").myVar = "no_color"

                # Big render button
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Apical (blue)").myVar = "Apical (blue)"
                
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Basal (yellow)").myVar = "Basal (yellow)"
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a",  text="Oblique (green)").myVar = "Oblique (green)"
                
                # Big render button
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Soma (red)").myVar = "Soma (red)"
                
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Axon-Soma (aqua)").myVar = "Axon-Soma (aqua)"
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Axon-Dendr (off blue)").myVar = "Axon-Dendr (off blue)"
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a",  text="Dendrite (purple)").myVar = "Dendrite (purple)"
                
                
                # Big render button
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Distal (pink)").myVar = "Distal (pink)"
                
                
                # Big render button
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Cilia (light purple)").myVar = "Cilia (light purple)"
                
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Error (brown)").myVar = "Error (brown)"
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Unlabelable (tan)").myVar = "Unlabelable (tan)"
               
                
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="print label").myVar = "print"
                
                #layout.prop(mytool, "quick_Label", text="quick label")
                row = layout.row()
                row = layout.row()
                # display the properties
                layout.prop(mytool, "hide_Labels", text="hide labeled")
                layout.prop(mytool, "visible_only_selection", text="Visible Only Selection")
     
                row = layout.row()
                row = layout.row()
                #row = layout.row()
                #row = layout.row()
                
                layout.row().prop(context.scene, "label_show")
            
                row = layout.row()
                #row.scale_y = 3.0
                row.label(text = "") 
                row.operator("object.edit_auto_neuron", text="only show label").myVar = "only_show_label"
                row = layout.row()
                #bpy.context.scene.label_show_status
                row.label(text = bpy.context.scene.label_show_status) 
                
                row = layout.row()
                row.operator("object.edit_auto_neuron", text="unhide labels").myVar = "unhide_labels" 
                
                row = layout.row()
                row = layout.row()
                layout.row().prop(context.scene, "label_show_select")
            
                row = layout.row()
                row = layout.row()
                #row.scale_y = 3.0
                row.label(text = "") 
                row.operator("object.edit_auto_neuron", text="select only label").myVar = "only_show_label_select"
                row = layout.row()
                #bpy.context.scene.label_show_status
                row.label(text = bpy.context.scene.label_show_status_select) 
                
                row = layout.row()
                row = layout.row()
                
                
                row = layout.row()
                row.operator("object.neuron_label_operator_a", text="Exit Edit Mode").myVar = "exit_edit_mode"
                
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row = layout.row()
                # Create a simple row.
                layout.label(text=" Labels for Later:")
                row = layout.row()
                
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a",  text="Spine Head (rose)").myVar = "Spine Head (rose)"
                
                
                # Big render button
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Spine (light pink)").myVar = "Spine (light pink)"
                
                
                
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.neuron_label_operator_a", text="Bouton (aqua)").myVar = "Bouton (aqua)"
                
                    
            #when the neuron is not being edited
            if bpy.context.mode == "OBJECT":
                       
                row = layout.row()
                row.operator("object.changing_view", text="Ortho Mode").myVar = "Ortho Mode"
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.changing_view", text="reset view").myVar = "reset view"
                
                row = layout.row()
                row = layout.row()
                #row.scale_y = 3.0
                row.operator("object.continue_editing", text="Continue Editing")
            
                col = layout.column()
                col.label(text="Upload Neuron to Database:")
                
                col.prop(context.scene.my_neurons_properties_2, "primitive")
                if context.scene.login_Flag != True or bpy.context.scene.picked_Neuron_ID == '':
                    col.enabled = False
                else:
                    col.enabled = True
                
                row = layout.row()
                row.operator("object.get_percent_labeled", text="Get % Labeled")
                if context.scene.login_Flag != True or bpy.context.scene.picked_Neuron_ID == '' or bpy.context.scene.status_To_Save =="":
                    row.enabled = False
                else:
                    row.enabled = True
                
                row = layout.row()
                row.label(text = bpy.context.scene.percent_labeled)
                row = layout.row()
                row.label(text = bpy.context.scene.percent_labeled_faces)  
                
                
                
                row = layout.row()
                row.operator("object.finish_editing", text="Send to Database")
                if context.scene.login_Flag != True or bpy.context.scene.picked_Neuron_ID == '' or bpy.context.scene.status_To_Save =="" or bpy.context.scene.percent_labeled =="" or bpy.context.scene.percent_labeled_faces =="":
                    row.enabled = False
                else:
                    row.enabled = True
            
                
                row = layout.row()
                row.label(text = bpy.context.scene.complete_100_check ) 
                
                row = layout.row()
                row.label(text = bpy.context.scene.complete_100_check_2 ) 
                
                
                row = layout.row()
                row = layout.row()
                row = layout.row()
                row.operator("object.submit_registration_login", text="Logout").myRegVar = "logout"
                if bpy.context.scene.username_Login == "":
                    row.enabled = False
                else:
                    row.enabled = True
                
                
                row = layout.row()
                row.scale_y = 3.0
                row = layout.row()
                row.scale_y = 3.0
                row = layout.row()

class delete_neuron_without_saving(bpy.types.Panel):
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "delete_neuron_without_saving"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Delete Neuron"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "OBJECT":
            row = layout.row()
            row.prop(context.scene, "delete_ID")
            
            
            row = layout.row() 
            row.label(text = "Delete Neuron " + bpy.context.scene.delete_ID + " without saving" )
            row = layout.row() 
            row.operator("object.neuron_label_operator_a", text="Delete neuron").myVar = "delete_Neuron"
            if (bpy.context.scene.delete_ID == ""):
                row.enabled = False
            else:
                row.enabled = True
            
            #row = layout.row() 
            #row.label(text = "Are you sure?")
            row = layout.row() 
            row.label(text = bpy.context.scene.delete_status)
            row = layout.row() 
            row.operator("object.neuron_label_operator_a", text="Yes").myVar = "delete_Yes"
            if bpy.context.scene.delete_Flag == False or bpy.context.scene.delete_ID == "":
                row.enabled = False
            else:
                row.enabled = True
                
            row.operator("object.neuron_label_operator_a", text="No").myVar = "delete_No"
            if bpy.context.scene.delete_Flag == False or bpy.context.scene.delete_ID == "":
                row.enabled = False
            else:
                row.enabled = True


'''
class Auto_Labeler_Edit(bpy.types.Panel):
    bl_idname = "Tools_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Label Neurons"
    bl_label = "Auto Lab"
    
    
    
    """Creates a Panel that you can open in order to get the register fields"""
    bl_idname = "Auto_Labeler_Edit"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Auto_Labeler_Edit"
    bl_label = "Auto_Labeler_Edit"
    bl_context = "editmode"

    
    #where to put the file name****************Not implemented yet

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        #only want this drawn if in object mode
        if bpy.context.mode == "EDIT_MESH":
            #username_Login
            layout.row().prop(context.scene, "label_show")
            
            row = layout.row()
            #row.scale_y = 3.0
            row.label(text = "") 
            row.operator("object.edit_auto_neuron", text="only show label").myRegVar = "only_show_label"
            row = layout.row()
            row = layout.row()
            row.operator("object.edit_auto_neuron", text="unhide labels").myRegVar = "unhide_labels" 
            

'''

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
    (0.800, 0.379, 0.232),(0.800,0.598,0.713),(0.800, 0.019, 0.093), #tan, light pink, rose
    (0.800, 0.486, 0.459),(0.800, 0.181, 0.013)] #light_pink, orange"""
    
    #New datajoint way
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

import random
def get_random_color(seed):
    ''' generate rgb using a list comprehension '''
    random.seed(seed)
    r, g, b = [round(random.random(),3) for i in range(3)]
    return (r, g, b)


def create_extra_local_colors(triangle_labels,ob=bpy.context.object):              
    
    color_length = accepted_color_length()
    
    triangle_counter = Counter(triangle_labels)
    print("triangle_counter = " + str(triangle_counter))
    extra_labels = [lab for lab in triangle_counter.keys() if int(lab) >= color_length]
    print("extra_labels = " + str(extra_labels))
    
    #create a placeholder object
    mat = bpy.data.materials.new("placeholder");
    mat.diffuse_color = get_random_color(1000)
    ob.data.materials.append(bpy.data.materials["placeholder"])
    
    
    for label in extra_labels:
        #add an extra color to the global colors if need to
        label_str = str(label)
        
        if label_str not in bpy.data.materials.keys():
            mat = bpy.data.materials.new(name=label_str);
            mat.diffuse_color = get_random_color(int(label))
        
        #add an extra color to the local colors
        ob.data.materials.append(bpy.data.materials[label_str])
    
    #check and see if missing an extra label
    
    faces_raw = ob.data.polygons
    missing_labels = np.unique([k.material_index for k in faces_raw if k.material_index >= len(ob.data.materials)])
    
    for ml in missing_labels:
        label_str = str(ml) + "00"
        
        if label_str not in bpy.data.materials.keys():
            mat = bpy.data.materials.new(name=label_str);
            mat.diffuse_color = get_random_color(int(label_str))
        
        #add an extra color to the local colors
        ob.data.materials.append(bpy.data.materials[label_str])
     

#enable hot key for quick label mode: 


class ModalOperator(bpy.types.Operator):
    bl_idname = "object.modal_operator"
    bl_label = "Simple Modal Operator"
    """Process input while Control key is pressed."""
    bl_options = {'REGISTER'}

    def __init__(self):
        print("Start")

    def __del__(self):
        print("End")


    def execute(self, context):
        
        return {'FINISHED'}
    
    def modal(self, context, event):
        if event.alt:
            #self.value = event.mouse_x
            #self.execute(context)
            #set the global variable to active
            #print(True)
            bpy.context.scene.quick_label_2 = True
            #pass # Input processing code.
        else:
            #print(False)
            bpy.context.scene.quick_label_2 = False
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        self.execute(context)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}



def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):

    def draw(self, context):
        self.layout.label(message)

    bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)
    
def register():
    bpy.utils.register_module(__name__)
    #bpy.utils.register_class(ModalOperator)
    bpy.types.Scene.my_tool = PointerProperty(type=MySettings)
    bpy.types.Scene.my_status_properties = bpy.props.PointerProperty(type=statusProperties)
    bpy.types.Scene.my_neurons_properties = bpy.props.PointerProperty(type=myNeuronsProperties)
    bpy.types.Scene.my_neurons_properties_2 = bpy.props.PointerProperty(type=myNeuronsProperties_2)

    #bpy.types.Scene.my_status_saving_properties = bpy.props.PointerProperty(type=statusSavingProperties)

    #bpy.utils.register_class(load_file_operator)
    #bpy.utils.register_class(setup_colors)
    #bpy.utils.register_class(Neuron_Label_Operator_a)
    #bpy.utils.register_class(MyDemoPanel)


def unregister():
    bpy.utils.unregister_class(__name__)
    del bpy.types.Scene.my_tool
    del bpy.types.Scene.my_status_properties
    del bpy.types.Scene.my_neurons_properties
    
    del bpy.types.Scene.my_status_saving_properties
    
    
    
    

#*************Databae Code**************#####

#import sys
#sys.path.append("/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/")
#where you could put them /Users/brendancelii/Google Drive/Xaq Lab
#sys.path.append("/Users/brendancelii/Google Drive/Xaq Lab")

######block of code that will add the working folder to the lookup spots for scripts

"""dir_path = os.path.dirname(os.path.realpath(__file__))
file_parts = dir_path.split("/")
file_parts.pop()
#print(file_parts)
just_Folder = "/".join(file_parts)
complete_path = just_Folder + "/blender_scripts/"
print(complete_path)
bpy.context.user_preferences.filepaths.script_directory = complete_path"""

#set the path in the user preferences

import datajoint as dj

#(min_x = 40960.0, max_x = 260096.0, min_y = 30720.0, max_y = 174912.0, min_z = 0.0, max_z = 40120.0)`
#(40960.0, 260096.0, 30720.0, 174912.0, 0.0, 40120.0)

def create_bounding_box():
    #check and make sure there is not already a bounding box
    if "bounding_box" in bpy.data.objects.keys():
        return
    
    mymesh = bpy.data.meshes.new("bounding_box")
    #verts = [(-700,-700,-700),(700,-700,-700),(-700,-700,700),(700,-700,700),
           #(-700,700,-700),(700,700,-700),(-700,700,700),(700,700,700)]
    #faces = [(1,2,3,4),(5,6,7,8),(4,5,2,6),(3,7,1,5),(4,3,7,8),(2,6,5,1)]
    #edges = [(1,2),(1,3),(3,4),(4,2),(5,6),(5,7),(7,8),(8,6),(3,7),(1,5),(4,8),(2,6)]
    """verts = [[-1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],
            [1,-1,-1],[1,1,-1],[1,-1,1],[1,1,1]]"""
    
    """verts_box = []
    for point in verts:
        new_point = [700*k for k in point]
        verts_box.append(new_point)
    """
    
    min_x = 40960
    max_x = 260096
    min_y = 30720
    max_y = 174912
    min_z = 0
    #max_z = 40120
    max_z = 100120
    
    
    verts_box = [[min_x,min_y,min_z],[min_x,max_y,min_z],[min_x,min_y,max_z],[min_x,max_y,max_z],
                [max_x,min_y,min_z],[max_x,max_y,min_z],[max_x,min_y,max_z],[max_x,max_y,max_z]]
    
    
    #vertices for the sides:
    
    
    
    #verts_box = [[40960,30720,0],[40960,174912,0],[40960,30720,40120],[40960,174912,40120],
    #        [260096,30720,0],[260096,174912,0],[260096,30720,40120],[260096,174912,40120]]
    
    width = 1000
    verts_sides = [[min_x + width,min_y,min_z],[min_x + width,max_y,min_z],[min_x + width,min_y,max_z],[min_x + width,max_y,max_z], #inner left points
                    [min_x,min_y + width,min_z],[min_x,max_y- width,min_z],[min_x,min_y +  width,max_z],[min_x,max_y -  width,max_z], #outer left points
                    [max_x - width,min_y,min_z],[max_x -width,max_y,min_z],[max_x - width,min_y,max_z],[max_x - width,max_y,max_z], #inner right points
                    [max_x,min_y + width,min_z],[max_x,max_y- width,min_z],[max_x,min_y +  width,max_z],[max_x,max_y -  width,max_z], #outer right points
                    [min_x,min_y,min_z + width],[min_x,max_y,min_z + width],[min_x,min_y,max_z - width],[min_x,max_y,max_z - width], #middle left points
                    [max_x,min_y,min_z + width],[max_x,max_y,min_z + width],[max_x,min_y,max_z - width],[max_x,max_y,max_z - width]] #middle left points
                    
                    
                    
    faces_side = [[0,8,10,2],[2,10,11,3],[3,1,9,11],[1,0,8,9], #inner left side
                  [4,16,18,6],[18,6,7,19],[19,7,5,17],[17,5,4,16],   #inner right side
                  [20,4,0,12],[12,0,2,14],[14,2,6,22],[22,6,4,20], #inner front
                  [21,5,1,13],[13,1,3,15],[15,3,7,23],[23,7,5,21], #inner back
                  [28,4,0,24],[24,0,1,25],[25,1,5,29],[29,5,4,28], #inner bottom
                  [30,6,2,26],[26,2,3,27],[27,3,7,31],[31,7,6,30]]
                  
    print(verts_box+verts_sides)
    
    
    faces=[[1,3,7,5]]  
    #edges = [(1,3),(2,3),(0,1),(0,2),(5,7),(4,5),(4,6),(6,7),
               (2,6),(0,4),(1,5),(3,7)]

    mymesh.from_pydata(verts_box + verts_sides, [], faces + faces_side)
   
    #mymesh.update(calc_edges=True)
    mymesh.calc_normals()
    
    
    #same scale that the imported neurons have
    #scale= 
    
    #need to rotate it around the X axis 90 degrees in order to line up correctly
    
    
    
    #print(objects_Matching_filename)
    object = bpy.data.objects.new("bounding_box", mymesh)
    #object.location = bpy.context.scene.cursor_location
    object.location = Vector((0,0,0))
    bpy.context.scene.objects.link(object)
    
    object.lock_location[0] = True
    object.lock_location[1] = True
    object.lock_location[2] = True
    object.lock_scale[0] = True
    object.lock_scale[1] = True
    object.lock_scale[2] = True

    #rotate the z direction by 90 degrees so point correct way
    
    #object.rotation_euler[2] = 1.5708
    
    object.rotation_euler[0] = 0
    object.rotation_euler[1] = 0
    object.rotation_euler[2] = 0



    
    object.lock_rotation[0] = True
    object.lock_rotation[1] = True
    object.lock_rotation[2] = True
    
    #go through and turn all the edges visible
    edges_raw = object.data.edges
    
    for edge in edges_raw:
        edge.hide = False
    
    
    object.show_transparent = True

    mat = bpy.data.materials.new("black");
    mat.diffuse_color = (0, 0, 0)
    
    
    bpy.context.scene.objects.active = object
    object.data.materials.append(None)
    object.data.materials[0] = bpy.data.materials["black"]
    
    object.active_material = bpy.data.materials["black"]
    object.active_material.diffuse_color = (0, 0, 0)
    object.active_material.translucency = 1
    object.active_material.use_shadeless = True
    object.active_material.use_transparency = True
    object.active_material.alpha = 0.2

    
    object.rotation_euler[0] = 1.5708
    object.rotation_euler[2] = 3.1416
    object.hide_select = True
    
    object.location[0] = 500000
    object.location[2] = -320000
    
    bpy.context.scene.objects.active = None
    
    set_View()

    
    
    #for face in edges_raw:
    #    k.material_index = int(triangles_labels[i])

    
    


    
if __name__ == "__main__":
    
    try:
        print("neuron tab labeler started new")
        #setting the address and the username
        print("about to connect to database")
        dj.config['database.host'] = '10.28.0.34'
        dj.config['database.user'] = 'celiib'
        dj.config['database.password'] = 'newceliipass'
        #will state whether words are shown or not
        dj.config['safemode']=False
        print(dj.conn(reset=True))
    except:
        #Shows a message box with a specific message 
        ShowMessageBox("Make sure connected to bcm-wifi!!")
        print("ERROR: Make sure connected to bcm-wifi!!")
        #raise ValueError("ERROR: Make sure connected to bcm-wifi!!")
    
    else:
        #connect_to_Databases()
        #create the database inside the server
        schema = dj.schema('microns_ta3p100',create_tables=False)
        ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')
        ta3 = dj.create_virtual_module('ta3', 'microns_ta3')
        reset_Scene_Variables()
        
        register()
        create_global_colors()
        
        
        '''
        @schema
        class AnnotationLast(dj.Manual):
            definition = """
            # creates the labels for the mesh table
            -> ta3.Decimation
            author     : varchar(20)  # name of last editor
            date_time  : timestamp   #the last time it was edited
            ---
            vertices   : longblob     # label data for the vertices
            triangles  : longblob     # label data for the faces
            edges      : longblob     # label data for the edges
            status     : varchar(16)          # the index of the status descriptor that can be references by the StatusKey
            """
            

        # @schema
        # class Status(dj.Manual):
        #     definition = """
        #     -> AnnotationLast
        #     ---
        #     (status) -> StatusKey
        #     """

    
        @schema
        class ProofreadLabels(dj.Manual):
            definition = """
            # creates the labels for the mesh table
            -> ta3.Decimation
            author_original     : varchar(20)  # name of last editor
            author_proofreader     : varchar(20)  # name of last editor
            date_time  : timestamp   #the last time it was edited
            ---
            vertices   : longblob     # label data for the vertices
            triangles  : longblob     # label data for the faces
            edges      : longblob     # label data for the edges
            status     : varchar(16)          # the index of the status descriptor that can be references by the StatusKey
            """
        
        @schema
        class AuthorsLast(dj.Manual):
            definition = """
            # maps numeric labels to descriptive labels
            username     : varchar(20)       # username the person pcisk
            ---
            real_name   : varchar(40)   #the real name of that corresponds to the username
            """
        '''
        
        
        #######DONE SETTING UP ALL OF THE TABLES#########
        
        #create reference to all of the tables
        #labels_Table = ta3p100.Annotation()
        proof_Table = ta3p100.ProofreadLabelOrphan()
        
        
        #status_key_table = StatusKey()
        label_key_table = ta3.LabelKey2()
        author_table = ta3.AuthorsLast()
        
        create_bounding_box()
        print("done initializing")
        
        bpy.ops.object.modal_operator('INVOKE_DEFAULT')
    