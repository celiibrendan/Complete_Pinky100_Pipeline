
# coding: utf-8

# In[1]:


import datajoint as dj
import numpy as np
import time


# In[2]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

schema = dj.schema('microns_pinky')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')


# In[ ]:


#schema.jobs     & 'table_name = "__compartment_final"'      


# In[ ]:


pinky.Decimation35OrphanStitched & pinky.CurrentSegmentation & 'decimation_ratio=0.35' & pinky.CoarseLabelOrphan.proj()
    


# In[ ]:


#############################################################################################################

def generate_neighborhood(triangles, num_vertices):
    neighborhood = dict()
    for i in range(num_vertices):
        neighborhood[i] = set()
    for node1, node2, node3 in triangles:
        neighborhood[node1].update([node2, node3])
        neighborhood[node2].update([node1, node3])
        neighborhood[node3].update([node1, node2])
    return neighborhood

def compress_compartments(neighborhood, vertex_labels):
    boundary_clusters = dict()
    for unique_label in np.unique(vertex_labels):
        boundary_clusters[unique_label] = dict()#list()

    starting_node = 0 # This assumes that there are no disconnected portions... I should actually figure out exactly what's going on here.
    visited_nodes = set()
    temp_stack = set()
    temp_stack.add(starting_node)    
    while len(temp_stack) > 0:
        starting_node = temp_stack.pop()
        if starting_node not in visited_nodes:
            same_label_neighbors = set()
            node_label = vertex_labels[starting_node]
            is_on_boundary = False
            for neighboring_node in neighborhood[starting_node]: # Think about if I truly need the same labeled neighbors...
                                                                 # Only way for it to be truly self contained right?
                if node_label == vertex_labels[neighboring_node]:
                    same_label_neighbors.add(neighboring_node)
                else:
                    is_on_boundary = True
            if is_on_boundary:
#                 boundary_clusters[node_label].append((starting_node, same_label_neighbors))
                boundary_clusters[node_label][starting_node] = same_label_neighbors
                
            visited_nodes.add(starting_node)
            temp_stack.update(neighborhood[starting_node])
    return boundary_clusters

def _separate_compartment(neighborhood, cluster, boundary_points):
    components = dict()
    compartment_index = 0
    while len(cluster) > 0:
        visited_nodes = set()
        temp_stack = set()
        temp_stack.add(next(iter(cluster)))
        boundaries_hit = set()
        while len(temp_stack) > 0:
            starting_node = temp_stack.pop()
            if starting_node not in visited_nodes:
                visited_nodes.add(starting_node)
                if starting_node in boundary_points:
                    boundaries_hit.add(starting_node)
                    temp_stack.update(cluster[starting_node])
                else:
                    temp_stack.update(neighborhood[starting_node])
        [cluster.pop(boundary_hit) for boundary_hit in boundaries_hit]        
        components[compartment_index] = visited_nodes
        compartment_index += 1
    return components

def separate_compartments(neighborhood, boundary_clusters):
    compartment_components = dict()
    boundary_clusters_copy = boundary_clusters.copy()
    for label, boundary_cluster in boundary_clusters_copy.items():
        cluster = dict()
        boundary_points = set()
        for node, neighbors in boundary_cluster.items():
            boundary_points.add(node)
            cluster[node] = neighbors
        components = _separate_compartment(neighborhood, cluster, boundary_points)
        compartment_components[label] = components
    return compartment_components
        
############################################################################################################# For Below

@schema
class CompartmentOrphan(dj.Computed):
    definition = """
    -> pinky.Decimation35OrphanStitched
    ---
    """

    class ComponentOrphan(dj.Part):
        definition = """
        -> CompartmentOrphan
        compartment_type   : varchar(16)        # Basal, Apical, spine head, etc.
        component_index    : smallint unsigned  # Which sub-compartment of a certain label this is.
        ---
        n_vertex_indices   : bigint
        n_triangle_indices : bigint
        vertex_indices     : longblob           # preserved indices of each vertex of this sub-compartment
        triangle_indices   : longblob           # preserved indices of each triangle of this sub-compartment
        """
    
    key_source = pinky.Decimation35OrphanStitched & pinky.CurrentSegmentation & 'decimation_ratio=0.35' & pinky.CoarseLabelOrphan.proj()
    
    def make(self, key):
        def generate_triangle_neighborhood(triangles):
            """
            Maps each vertex node to every triangle they appear in.
            """
            triangle_neighborhood = dict()
            for i in range(len(triangles)):
                triangle_neighborhood[i] = set()
            for i, (node1, node2, node3) in enumerate(triangles):
                triangle_neighborhood[node1].add(i)
                triangle_neighborhood[node2].add(i)
                triangle_neighborhood[node3].add(i)
            return triangle_neighborhood
        
        def generate_component_keys(key, components, triangles, triangle_neighborhood, labeled_triangles):
            for label_key, compartment in components.items():
                for component_index, component in compartment.items():
                    try:
                        label_name = (pinky.LabelKey & dict(numeric=label_key)).fetch1('description')
                    except:
                        label_name = str(label_key)
                        
                    vertex_indices = np.array(list(component))
                    triangle_indices = np.unique([triangle_index for node in component
                                                  for triangle_index in triangle_neighborhood[node]
                                                  if labeled_triangles[triangle_index] == label_key])
                    set_vertex_indices = set(vertex_indices)
                    true_triangle_indices = []
                    for triangle_index in triangle_indices:
                        node1, node2, node3 = triangles[triangle_index]
                        if node1 in set_vertex_indices:
                            if node2 in set_vertex_indices:
                                if node3 in set_vertex_indices:
                                    true_triangle_indices.append(triangle_index)                        
                    triangle_indices = np.array(true_triangle_indices)
                    yield dict(key,
                               compartment_type=label_name,
                               component_index=component_index,
                               n_vertex_indices=len(vertex_indices),
                               n_triangle_indices=len(triangle_indices),
                               vertex_indices=vertex_indices,
                               triangle_indices=triangle_indices)
        
        start = time.time()
        #print("hello")
        mesh = (pinky.Decimation35OrphanStitched & key).fetch1()
        labels = (pinky.CoarseLabelOrphan & key).fetch1()
        #print("something")
        if len(np.unique(labels['triangles'])) == 1:
            #print("heyo")
            self.insert1(key)
            label_name = (pinky.LabelKey & dict(numeric=np.unique(labels['triangles'])[0])).fetch1('description')
            vertex_indices = np.arange(len(labels['vertices']), dtype=np.uint32)
            triangle_indices = np.arange(len(labels['triangles']), dtype=np.uint32)
            new_dict= dict(key,
                                                compartment_type=label_name,
                                                component_index=0,
                                                n_vertex_indices=len(vertex_indices),
                                                n_triangle_indices=len(triangle_indices),
                                                vertex_indices=vertex_indices,
                                                triangle_indices=triangle_indices)
            
            CompartmentOrphan.ComponentOrphan().insert1(dict(key,
                                                compartment_type=label_name,
                                                component_index=0,
                                                n_vertex_indices=len(vertex_indices),
                                                n_triangle_indices=len(triangle_indices),
                                                vertex_indices=vertex_indices,
                                                triangle_indices=triangle_indices))
            return
        
        neighborhood = generate_neighborhood(mesh['triangles'], len(mesh['vertices']))
        boundary_clusters = compress_compartments(neighborhood, labels['vertices'])
        components = separate_compartments(neighborhood, boundary_clusters)
        triangle_neighborhood = generate_triangle_neighborhood(mesh['triangles'])

        self.insert1(key)
        CompartmentOrphan.ComponentOrphan().insert(generate_component_keys(key, components, mesh['triangles'],
                                                               triangle_neighborhood, labels['triangles']))

        print(key['segment_id'], "finished separating components:", time.time() - start)


# In[3]:


#(schema.jobs & "table_name='__compartment_orphan'").delete()


# In[ ]:


start_time = time.time()
CompartmentOrphan.populate(reserve_jobs=True)
print(f"Total time = {time.time() - start_time}")


# # check that all neurons have components

# In[ ]:


#(schema.jobs & "table_name='__compartment_final'").delete()


# In[ ]:


#check that there are all components in there
pinky.CompartmentOrphan.ComponentOrphan()

