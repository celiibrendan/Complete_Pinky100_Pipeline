
# coding: utf-8

# In[8]:


import datajoint as dj
import numpy as np
import time
from pykdtree.kdtree import KDTree


# In[2]:


schema = dj.schema('microns_pinky')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')
dj.config["display.limit"] = 30


# In[3]:


pinky.LeftoverCompartmentFinal()


# In[36]:


pinky.CoarseLabelOrphan & "segment_id=648518346341352891"


# In[37]:




@schema
class UndecimatedNeuronLabels(dj.Computed):
    definition = """
    # Undecimated Mesh Coarse and Overlayed Labels
    -> pinky.Mesh
    ---
    coarse_vertices_labels             : longblob # vertex labels for the base compartments                   
    coarse_triangles_labels            : longblob # traingle labels for the base compartments
    overlay_vertices_labels             : longblob # vertex labels for the spine compartments overlayed over the base compartments                  
    overlay_triangles_labels            : longblob # triangles labels for the spine compartments overlayed over the base compartments
    """
    
    key_source = ((dj.U("segmentation","segment_id") & pinky.CoarseLabelFinal.proj()) 
     + (dj.U("segmentation","segment_id") & pinky.CoarseLabelOrphan.proj()))

    
    def make(self, key):
        print("\n\n------------- Working on " + str(key["segment_id"]) +"----------------------------------------------")
        #figure out if from the orphan list or excitatory list
        full_time = time.time()
        
        search_key = dict(key,decimation=0.35)
        new_key = key
        
        table = ""
        if len(pinky.CoarseLabelFinal & key) > 0:
            print("classified as Excitatory")
            table="Excitatory"
            dec_vert_labels,dec_tri_labels = (pinky.OverlayedSpineLabel & search_key).fetch1("vertices","triangles")
            #get the decimated mesh
            dec_mesh_table = pinky.PymeshfixDecimatedExcitatoryStitchedMesh & search_key
            dec_vertices, dec_triangles = dec_mesh_table.fetch1("vertices","triangles")


            #make sure that the labels match up:
            print((len(dec_vert_labels),len(dec_vertices)))
            #len(dec_tri_labels),len(dec_triangles)
            
        elif len(pinky.CoarseLabelOrphan & key) > 0:
            print("classified as Orphan")
            table="Orphan"
            dec_vert_labels,dec_tri_labels = (pinky.OverlayedSpineLabelOrphan & search_key).fetch1("vertices","triangles")
            #get the decimated mesh
            dec_mesh_table = pinky.Decimation35OrphanStitched & search_key
            dec_vertices, dec_triangles = dec_mesh_table.fetch1("vertices","triangles")


            #make sure that the labels match up:
            print((len(dec_vert_labels),len(dec_vertices)))
            #len(dec_tri_labels),len(dec_triangles)
        else:
            raise Exception("Segment id was not in Excitatory or Orphan list")
        
        #based on what table it is, get the decimated and label arrays
        
        #get the undecimated mesh
        undec_mesh_table = pinky.Mesh & new_key
        undec_vertices, undec_triangles = undec_mesh_table.fetch1("vertices","triangles")
        print(len(undec_triangles))
        
        from pykdtree.kdtree import KDTree
        start_time = time.time()
        dec_KDTree = KDTree(dec_vertices)
        distances, nearest_nodes = dec_KDTree.query(undec_vertices)
        print(f"Total time for main KDTree creation and queries = {time.time() - start_time}")

        #get the labels for the undecimated mesh
        undecimated_vert_labels = dec_vert_labels[nearest_nodes]

        #get the final labels by combining leftover with regular
        
        if table=="Orphan" or len(pinky.LeftoverCompartmentFinal & key) <= 0:
            print("No leftover meshes to add or just orphan")
            
            final_undec_overlay_verts_labels = np.zeros(len(undecimated_vert_labels))

            error_distance_threshold = 200

            for i in range(0,len(final_undec_overlay_verts_labels)):
                if distances[i] > error_distance_threshold:
                    final_undec_overlay_verts_labels[i] = 10
                else:
                    final_undec_overlay_verts_labels[i] = undecimated_vert_labels[i]
            
            #get the triangle labels for the overlay
            triangle_overlay_labels = final_undec_overlay_verts_labels[undec_triangles[:,0]]

            #get the coarse labels 
            
            if table == "Excitatory":
                dec_vert_labels_coarse,dec_tri_labels_coarse= (pinky.CoarseLabelFinal & search_key).fetch1("vertices","triangles")
                undecimated_vert_labels_coarse = dec_vert_labels_coarse[nearest_nodes]
            else:
                dec_vert_labels_coarse,dec_tri_labels_coarse= (pinky.CoarseLabelOrphan & search_key).fetch1("vertices","triangles")
                undecimated_vert_labels_coarse = dec_vert_labels_coarse[nearest_nodes]
            
            
            #get the final labels by combining leftover with regular
            final_undec_coarse_verts_labels = np.zeros(len(undec_vertices))
            
            for i in range(0,len(final_undec_coarse_verts_labels)):
                if distances[i] > error_distance_threshold:
                    final_undec_coarse_verts_labels[i] = 10
                else:
                    final_undec_coarse_verts_labels[i] = undecimated_vert_labels_coarse[i]
            
            triangle_overlay_labels_coarse = final_undec_coarse_verts_labels[undec_triangles[:,0]]
            #Counter(triangle_overlay_labels_coarse)
            
            #write this to the database
            self.insert1(dict(key,
                        coarse_vertices_labels = final_undec_coarse_verts_labels,                               
                        coarse_triangles_labels= triangle_overlay_labels_coarse,           
                        overlay_vertices_labels = final_undec_overlay_verts_labels,                         
                        overlay_triangles_labels = triangle_overlay_labels ))
            print(f"Total time for mapping: {time.time() - full_time}")

        
        else:
            print("Going to add in the leftover meshes labels as well")
            
            #get the decimated mesh
            dec_vert_labels_leftover,dec_tri_labels_leftover = (pinky.LeftoverOverlayedSpineLabel & search_key).fetch1("vertices","triangles")

            #get the decimated mesh
            dec_mesh_table_leftover = pinky.ExcitatoryLeftoverMeshes & search_key
            dec_vertices_leftover, dec_triangles_leftover = dec_mesh_table_leftover.fetch1("vertices","triangles")


            #make sure that the labels match up:
            print((len(dec_vert_labels_leftover),len(dec_vertices_leftover)))
            #len(dec_tri_labels),len(dec_triangles)

            from pykdtree.kdtree import KDTree
            dec_KDTree_leftover = KDTree(dec_vertices_leftover)

            start_time = time.time()
            distances_leftover, nearest_nodes_leftover = dec_KDTree_leftover.query(undec_vertices)
            print(f"Total time = {time.time() - start_time}")

            #get the labels for the undecimated mesh
            undecimated_vert_labels_leftover = dec_vert_labels_leftover[nearest_nodes_leftover]


            
            
            #get the final labels by combining leftover with regular
            final_undec_overlay_verts_labels = np.zeros(len(undec_vertices))

            error_distance_threshold = 200

            for i in range(0,len(final_undec_overlay_verts_labels)):
                if distances_leftover[i]>error_distance_threshold and distances[i] > error_distance_threshold:
                    final_undec_overlay_verts_labels[i] = 10
                else:
                    if distances[i] < distances_leftover[i]:
                        final_undec_overlay_verts_labels[i] = undecimated_vert_labels[i]
                    else:
                        final_undec_overlay_verts_labels[i] = undecimated_vert_labels_leftover[i]


            triangle_overlay_labels = final_undec_overlay_verts_labels[undec_triangles[:,0]]
            #Counter(triangle_overlay_labels)

            #---------------- Done with getting the overalyed vertices ---------------- ##

            #---------------- Started getting just the coarse labels ---------------- ##

            #get the labels for the undecimated mesh
            #get the decimated mesh
            dec_vert_labels_coarse,dec_tri_labels_coarse= (pinky.CoarseLabelFinal & search_key).fetch1("vertices","triangles")
            undecimated_vert_labels_coarse = dec_vert_labels_coarse[nearest_nodes]


            dec_vert_labels_leftover_coarse,dec_tri_labels_leftover_coarse_coarse = (pinky.LeftoverCoarseLabelFinal & search_key).fetch1("vertices","triangles")
            undecimated_vert_labels_leftover_coarse = dec_vert_labels_leftover_coarse[nearest_nodes_leftover]


            #get the final labels by combining leftover with regular
            final_undec_coarse_verts_labels = np.zeros(len(undec_vertices))

            error_distance_threshold = 200

            for i in range(0,len(final_undec_coarse_verts_labels)):
                if distances_leftover[i]>error_distance_threshold and distances[i] > error_distance_threshold:
                    final_undec_coarse_verts_labels[i] = 10
                else:
                    if distances[i] < distances_leftover[i]:
                        final_undec_coarse_verts_labels[i] = undecimated_vert_labels_coarse[i]
                    else:
                        final_undec_coarse_verts_labels[i] = undecimated_vert_labels_leftover_coarse[i]


            triangle_overlay_labels_coarse = final_undec_coarse_verts_labels[undec_triangles[:,0]]

            
            #write this to the database
            self.insert1(dict(key,
                        coarse_vertices_labels = final_undec_coarse_verts_labels,                               
                        coarse_triangles_labels= triangle_overlay_labels_coarse,           
                        overlay_vertices_labels = final_undec_overlay_verts_labels,                         
                        overlay_triangles_labels = triangle_overlay_labels ))
        
            print(f"Total time for mapping: {time.time() - full_time}")

            


# In[38]:


#(schema.jobs & "table_name='__undecimated_neuron_labels'").delete()


# In[ ]:


start = time.time()
UndecimatedNeuronLabels.populate(reserve_jobs=True)
print(time.time() - start)

#648518346341352891


# In[28]:


key_source = ((dj.U("segmentation","segment_id") & pinky.CoarseLabelFinal.proj()) 
 + (dj.U("segmentation","segment_id") & pinky.CoarseLabelOrphan.proj()))


# In[31]:


pinky.CoarseLabelOrphan & "segment_id=648518346341352891"

