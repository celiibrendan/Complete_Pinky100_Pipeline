import time
import numpy as np
import datajoint as dj
import networkx as nx

import matplotlib.pyplot as plt #for graph visualization
from scipy.spatial import distance_matrix
from tqdm import tqdm

def stitch_skeleton(staring_edges):
    

    #unpacks so just list of vertices
    vertices_unpacked  = staring_edges.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = [unique_rows_list.index(a) for a in vertices_unpacked.tolist()]

    #reshapes the vertex list to become an edge list
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)
    
    #create the graph from the edges
    B = nx.Graph()
    B.add_nodes_from([(x,{"coordinates":y}) for x,y in enumerate(unique_rows_list)])
    B.add_edges_from(edges_with_coefficients)

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix

    UG = B.to_undirected()

    # extract subgraphs
    sub_graphs = nx.connected_component_subgraphs(UG)

    subgraphs_list = []
    for i, sg in enumerate(sub_graphs):
        #print("subgraph {} has {} nodes".format(i, sg.number_of_nodes()))
        #print("\tNodes:", sg.nodes(data=True))
        #print("\tEdges:", sg.edges())
        subgraphs_list.append(sg)

    print("len_subgraphs AT BEGINNING = " + str(len(subgraphs_list)))

    while len(subgraphs_list) > 1:
        current_sub_graph = subgraphs_list[0]
        #get all of the coordinates of this subgraph 
        coord = nx.get_node_attributes(current_sub_graph,'coordinates')
        current_nodes = np.array(list(current_sub_graph.nodes))
        current_coordinates = np.array(list(coord.values()))

        #store the minimum distance
        min_dist = np.inf
        min_dist_subgraph_index = -1
        min_dist_edge = [-1,-1]
        min_dist_edge_index = [-1,-1]

        #i = 0
        for i, sg in enumerate(subgraphs_list):
            if i == 0:
                continue


            #get all of the coordinates
            new_coord = nx.get_node_attributes(sg,'coordinates')
            new_coordinates = np.array(list(new_coord.values()))
            new_nodes = np.array(list(sg.nodes))

            #find the closest distance between those points
            a_New = current_coordinates
            b_New = new_coordinates

            # print("starting distance matrix")
            start_time = time.time()

            a_b_distance = distance_matrix(a_New, b_New)
            #print(a_b_distance)

            current_min = np.min(a_b_distance)
            dist_matrix_index = np.where(a_b_distance == current_min)
            min_indexes = dist_matrix_index[0][0],dist_matrix_index[1][0]

            #print("Min indexes = " + str(min_indexes))

            if current_min<min_dist:
                min_dist = current_min
                min_dist_subgraph_index = i
                min_dist_edge_index = [current_nodes[min_indexes[0]],
                                       new_nodes[min_indexes[1]]]
                min_dist_edge = [current_coordinates[min_indexes[0]],
                                        new_coordinates[min_indexes[1]]]
            #i += 1

        #add the shortest connection to the overall edges
        print("min_dist = " + str(min_dist))
        print("min_dist_subgraph_index = " + str(min_dist_subgraph_index))
        print("min_dist_edge_index = " + str(min_dist_edge_index))
        print("min_dist_edge = " + str(min_dist_edge))

        #adds the new edge to the graph
        B.add_edge(*min_dist_edge_index)

        #
        #print("edges after adding")
        #print(B.edges())
        UG = B.to_undirected()

        # extract subgraphs
        sub_graphs = nx.connected_component_subgraphs(UG)
        subgraphs_list = []
        for i, sg in enumerate(sub_graphs):
#             print("subgraph {} has {} nodes".format(i, sg.number_of_nodes()))
#             print("\tNodes:", sg.nodes(data=True))
#             print("\tEdges:", sg.edges())
            subgraphs_list.append(sg)

        print("len_subgraphs AT END= " + str(len(subgraphs_list)))
        
    #add all the new edges to the 

    total_coord = nx.get_node_attributes(B,'coordinates')
    current_coordinates = np.array(list(total_coord.values()))
    return current_coordinates[B.edges()]




def stitch_skeleton_with_degree_check(staring_edges,end_node=False):
    

    #unpacks so just list of vertices
    vertices_unpacked  = staring_edges.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = [unique_rows_list.index(a) for a in vertices_unpacked.tolist()]

    #reshapes the vertex list to become an edge list
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)
    
    #create the graph from the edges
    B = nx.Graph()
    B.add_nodes_from([(x,{"coordinates":y}) for x,y in enumerate(unique_rows_list)])
    B.add_edges_from(edges_with_coefficients)

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix

    UG = B.to_undirected()

    # extract subgraphs
    sub_graphs = nx.connected_component_subgraphs(UG)

    subgraphs_list = []
    for i, sg in enumerate(sub_graphs):
        #print("subgraph {} has {} nodes".format(i, sg.number_of_nodes()))
        #print("\tNodes:", sg.nodes(data=True))
        #print("\tEdges:", sg.edges())
        subgraphs_list.append(sg)

    print("len_subgraphs AT BEGINNING = " + str(len(subgraphs_list)))

    while len(subgraphs_list) > 1:
        current_sub_graph = subgraphs_list[0]
        #get all of the coordinates of this subgraph 
        #get all of the coordinates
        if end_node == True:
            current_nodes = np.array([x for x,n in current_sub_graph.degree() if n <= 1])
            current_coord = nx.get_node_attributes(current_sub_graph,'coordinates')
            current_coord_filter = { k:v for k,v in current_coord.items() if k in  current_nodes}
            current_coordinates = np.array(list(current_coord_filter.values()))
        
        else:
            #get all of the coordinates
            current_nodes = np.array(list(current_sub_graph.nodes))
            current_coord = nx.get_node_attributes(current_sub_graph,'coordinates')
            current_coordinates = np.array(list(current_coord.values()))
            
#         coord = nx.get_node_attributes(current_sub_graph,'coordinates')
#         current_nodes = np.array(list(current_sub_graph.nodes))
#         current_coordinates = np.array(list(coord.values()))

        #store the minimum distance
        min_dist = np.inf
        min_dist_subgraph_index = -1
        min_dist_edge = [-1,-1]
        min_dist_edge_index = [-1,-1]

        #i = 0
        for i, sg in enumerate(subgraphs_list):
            if i == 0:
                continue

            #get all of the coordinates
            if end_node == True:
                new_nodes = np.array([x for x,n in sg.degree() if n <= 1])
                #print("new nodes = " + str(new_nodes))
                new_coord = nx.get_node_attributes(sg,'coordinates')
                #print("new_coord = " + str(new_coord))
                new_coord_filter = { k:v for k,v in new_coord.items() if k in  new_nodes}
                new_coordinates = np.array(list(new_coord_filter.values()))
            else:
                #get all of the coordinates
                new_nodes = np.array(list(sg.nodes))
                new_coord = nx.get_node_attributes(sg,'coordinates')
                new_coordinates = np.array(list(new_coord.values()))
                
            #find the closest distance between those points
            a_New = current_coordinates
            b_New = new_coordinates

            # print("starting distance matrix")
            start_time = time.time()

            a_b_distance = distance_matrix(a_New, b_New)
            #print(a_b_distance)

            current_min = np.min(a_b_distance)
            dist_matrix_index = np.where(a_b_distance == current_min)
            min_indexes = dist_matrix_index[0][0],dist_matrix_index[1][0]

            #print("Min indexes = " + str(min_indexes))

            if current_min<min_dist:
                min_dist = current_min
                min_dist_subgraph_index = i
                min_dist_edge_index = [current_nodes[min_indexes[0]],
                                       new_nodes[min_indexes[1]]]
                min_dist_edge = [current_coordinates[min_indexes[0]],
                                        new_coordinates[min_indexes[1]]]
            #i += 1

        #add the shortest connection to the overall edges
        print("min_dist = " + str(min_dist))
        print("min_dist_subgraph_index = " + str(min_dist_subgraph_index))
        print("min_dist_edge_index = " + str(min_dist_edge_index))
        print("min_dist_edge = " + str(min_dist_edge))

        #adds the new edge to the graph
        B.add_edge(*min_dist_edge_index)

        #
        #print("edges after adding")
        #print(B.edges())
        UG = B.to_undirected()

        # extract subgraphs
        sub_graphs = nx.connected_component_subgraphs(UG)
        subgraphs_list = []
        for i, sg in enumerate(sub_graphs):
#             print("subgraph {} has {} nodes".format(i, sg.number_of_nodes()))
#             print("\tNodes:", sg.nodes(data=True))
#             print("\tEdges:", sg.edges())
            subgraphs_list.append(sg)

        print("len_subgraphs AT END= " + str(len(subgraphs_list)))
        
    #add all the new edges to the 

    total_coord = nx.get_node_attributes(B,'coordinates')
    current_coordinates = np.array(list(total_coord.values()))
    return current_coordinates[B.edges()]


"""
For finding the distances of the skeleton
"""
def find_skeleton_distance(example_edges):
    total_distance = np.sum([np.linalg.norm(a-b) for a,b in example_edges])
    return total_distance

from scipy.spatial import distance

def find_skeleton_distance_scipy(example_edges):
    total_distance = np.sum([distance.euclidean(a, b) for a,b in example_edges])
    return total_distance
