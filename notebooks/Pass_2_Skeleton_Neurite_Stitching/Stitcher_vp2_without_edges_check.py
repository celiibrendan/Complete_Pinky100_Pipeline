"""
Purpose: Only thing changed from Stitcher_vp2
was commented out the part that checked for more or less
than 2 starting edges for facet


"""

import trimesh
import numpy as np
from collections import Counter
import time
import math
from tqdm import tqdm
import os
from pathlib import Path
import pymeshfix

def filter_mesh_significant_outside_pieces(unfiltered_mesh,significance_threshold=2000,n_sample_points=3):
    """
    Purpose; will take in a full, unfiltered mesh and find the biggest mesh piece, and then return a list of that mesh 
    with all of the other mesh fragments that are both above the significance_threshold AND outside of the biggest mesh piece

    Pseudocode: 
    1) split the meshes to unconnected pieces
    2) Filter the meshes for only those above the significance_threshold
    3) find the biggest mesh piece
    4) Iterate through all of the remaining pieces:
        a. Determine if mesh inside or outside main mesh
        b. If outside add to final list to return

    Returns: 
    1) list of significant mesh pieces, including the main one that are not inside of main mesh

    """
    
    print("------Starting the stitcher vp2-------")

    mesh_pieces = unfiltered_mesh.split(only_watertight=False)
    
    print(f"There were {len(mesh_pieces)} pieces after mesh split")

    significant_pieces = [m for m in mesh_pieces if len(m.faces) > significance_threshold]

    print(f"There were {len(significant_pieces)} pieces found after size threshold")
    if len(significant_pieces) <=0:
        print("THERE WERE NO MESH PIECES GREATER THAN THE significance_threshold")
        return []

    #find piece with largest size
    max_index = 0
    max_face_len = len(significant_pieces[max_index].faces)

    for i in range(1,len(significant_pieces)):
        if max_face_len < len(significant_pieces[i].faces):
            max_index = i
            max_face_len = len(significant_pieces[i].faces)

    print("max_index = " + str(max_index))
    print("max_face_len = " + str(max_face_len))

    final_mesh_pieces = []

    main_mesh = significant_pieces[max_index]

    #final_mesh_pieces.append(main_mesh)
    for i,mesh in enumerate(significant_pieces):
        if i != max_index:
            #get a random sample of points
            # points = np.array(mesh.vertices[:n_sample_points,:]) # OLD WAY OF DOING THIS
            idx = np.random.randint(len(mesh.vertices), size=n_sample_points)
            points = mesh.vertices[idx,:]
            
            
            start_time = time.time()
            signed_distance = trimesh.proximity.signed_distance(main_mesh,points)
            #print(f"Total time = {time.time() - start_time}")

            outside_percentage = sum(signed_distance < 0)/n_sample_points
            if outside_percentage > 0.9:
                final_mesh_pieces.append(mesh)
                #print(f"Mesh piece {i} OUTSIDE mesh")
            else:
                #print(f"Mesh piece {i} inside mesh :( ")
                pass
                
    return main_mesh,final_mesh_pieces

import math
def area(vertices):
    """
    Calculates the area of a 3D triangle from it's coordinates
    """
    side_a = np.linalg.norm(vertices[0]-vertices[1])
    side_b = np.linalg.norm(vertices[1]-vertices[2])
    side_c = np.linalg.norm(vertices[2]-vertices[0])
    s = 0.5 * ( side_a + side_b + side_c)
    return math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))

def find_polygon_area(mesh,list_of_faces):
    "Calculates the area of a 3D polygon that is created from connected traingles"
    return(sum([area(mesh.vertices[mesh.faces[r]]) for r in list_of_faces]))


#filters by convexity and generates the facet centers

def filter_final_facets_working(gap_mesh,final_facets,adjacency_threshold =  0.8):
    """
    Gets the facets faces list and the center points of these facets from mesh
    Filters:
    1) Only lets facets greater than first_pass_size_threshold exist
      **** might need to look at this because area is that of facets before expansion
    2) Has to have a high convex border
    
    Expansions:
    1) Expands the facet group to neighbors that are within the normal_closeness
    for their normals when doing the dot product
    
    Order:
    1) Size filtering
    2) Expansion
    3) Convex Border filtering
    """


    # after computing final faces, filter for convexity
    edges = gap_mesh.edges_sorted.reshape((-1, 6)) #groups all the edges belonging to the corresponding face in one row
    final_facets_mean = np.zeros(len(final_facets))
    
    
    #make lookup table for face number to spot in the adjacency edges
    face_adjacency_index_lookup = [[] for i in gap_mesh.faces]
    for i,faces in enumerate(gap_mesh.face_adjacency):
        for f in faces:
            face_adjacency_index_lookup[f].append(i)


    for j,facet in enumerate(final_facets):
        # get the edges for each facet
        edges_facet = [edges[i].reshape((-1, 2)) for i in [facet]][0] #stores all the edges belonging to that face

        #get the indexes of the boundary edges:
        indexes = trimesh.grouping.group_rows(edges_facet, require_count=1)
        edge_0 = edges_facet[indexes]

        #find the faces that correspond to the boundary edges
        edge_0_faces = [facet[int(k/3)] for k in indexes]


        #2) Find the indexes of the edges int he face_adajacency_edges and store the projections
        adjacency_values = []
        for edge,edge_face in zip(edge_0,edge_0_faces):
            possible_adj_indexes = face_adjacency_index_lookup[edge_face]

            for index in possible_adj_indexes:
                if len(set(edge).intersection(set(gap_mesh.face_adjacency_edges[index]))) >= 2:
                    #print(f"adj edge = {e} and boundary edge = {edge}")
                    adjacency_values.append(gap_mesh.face_adjacency_angles[index]) # the metric we actually want to measure
                    break


        final_facets_mean[j] = np.mean(adjacency_values)
        
        

    #filter the final facets and output them so they can be plotted
    
    
#     #need to make sure they are array of arrays
#     thresholld_boolean_results = final_facets_mean > adjacency_threshold
#     for fa,individual_facet in  enumerate(final_facets):
#         if thresholld_boolean_results[fa] == True:
#             np
    
    final_facets_mean_filtered = np.array(final_facets)[final_facets_mean > adjacency_threshold]
    
    if len(final_facets_mean_filtered.shape) > 1:
        total_array = np.empty(final_facets_mean_filtered.shape[0],object)
        total_array[:] = [x for x in final_facets_mean_filtered]
        final_facets_mean_filtered = total_array
        print("Had to restructure the array because was 2D array")

    

    #Compute the centers
    final_facets_centers = []
    
    for filt in final_facets_mean_filtered: 
        #print("filt = " + str(filt))
        unique_vertices = gap_mesh.vertices[np.unique(gap_mesh.faces[filt].ravel())].astype("float")
        final_facets_centers.append((np.mean(unique_vertices[:,0]),
                          np.mean(unique_vertices[:,1]),
                          np.mean(unique_vertices[:,2])))
    
    return final_facets_mean_filtered,final_facets_centers


#main function that generates the facets
def filter_final_facets_optimized_with_checks(example_mesh,min_len=2,
                                  normal_closeness=0.99,
                                  first_pass_size_threshold=6000,
                                  adjacency_threshold =  0.8
                                  ):
    """
    Way of computing facets that uses trimesh grouping function and normals threshold
    instead of the pre-built trimesh.facets
    
    """

    #get the facets for the child mesh
    start_time_total = time.time()

    #switch out with dot product
    
    
    #Old way of computing the dot product manually
    total_normals = example_mesh.face_normals[example_mesh.face_adjacency]
    
    #using manual method
#     print(len(total_normals[:,0]*total_normals[:,1]))
    start_normal = time.time()
    total_normal_dots = np.sum(total_normals[:,0]*total_normals[:,1],axis=1)
    #print(f"Total normals took {time.time() - start_normal}")
    ''' DONT HAVE TO DO DOT PRODUCT BECAUSE JUST WANT TO MULTIPLY THE ROWS OF EACH COLUMN'''
#     print("About to do dot product")
#     dot_start = time.time()
#     a = total_normals[:,0]
#     b = total_normals[:,1].T
#     #new way using the dot product
#     total_normal_dots = np.dot(a,b)
    
    
    #get the True/False value if face adjacency is within normal closeness
    start_normal = time.time()
    total_normal_dots_facet_mask = total_normal_dots > normal_closeness
    #print(f"Boolean mask finished: {time.time() - start_normal}")

    #getting the grouping of the faces into facets within the threshold
    start_normal = time.time()
    components = trimesh.graph.connected_components(example_mesh.face_adjacency[total_normal_dots_facet_mask],
                                          nodes=np.arange(len(example_mesh.faces)),
                                          min_len=min_len,
                                          engine="None")
    #print(f"Grouping took: {time.time() - start_normal}")
    
    

    #print(f"Lenght of facets BEFORE filtering = {len(components)}")
    
    #filter by the size
    
    start_normal = time.time()
    size_filtered_components = [facet_list for facet_list in components 
                                    if find_polygon_area(example_mesh,facet_list) > first_pass_size_threshold]
    print(f"Filtering edges by size finished: {time.time() - start_normal}, facet # = {len(size_filtered_components)}")

#     if facet_index in checking_list:
#         print("size_filtered_components = " + str(size_filtered_components))

    #filter by the convexity
    #final_facets_mean_filtered,final_facets_centers = filter_final_facets(example_mesh,size_filtered_components)
    
    start_normal = time.time()
    final_facets_mean_filtered,final_facets_centers = filter_final_facets_working(example_mesh,size_filtered_components,adjacency_threshold)
    print(f"Filtering by convexity and getting centers took: {time.time() - start_normal}, facet # = {len(final_facets_mean_filtered)}")

#     if facet_index in checking_list:
#         print("final_facets_mean_filtered = " + str(final_facets_mean_filtered))
    
    #print(f"Total Time = {time.time() - start_time_total}")

    return final_facets_mean_filtered,final_facets_centers
   
   
def apply_bbox_filter(child,min_bb_zone,max_bb_zone):
    """
    Determines if child is withing the bounding box zone
    designated by the bounding box corners
    
    """
    #get the min and max of the bounding box for the mesh
    min_bb = np.array(child.bounding_box.vertices).min(0)
    max_bb = np.array(child.bounding_box.vertices).max(0)
    
    #print(min_bb,max_bb)
    #print(min_bb_zone,max_bb_zone)
    
    #if fails any of these checks then return false, else return True
    if min(min_bb[0],max_bb[0])>max_bb_zone[0]:
        print("returning x greater max")
        return False
    
    if max(min_bb[0],max_bb[0])<min_bb_zone[0]:
        print("returning x less min")
        return False
    
    if min(min_bb[1],max_bb[1])>max_bb_zone[1]:
        print("returning y greater max")
        return False
    
    if max(min_bb[1],max_bb[1])<min_bb_zone[1]:
        print("returning y less min")
        return False
        
    if min(min_bb[2],max_bb[2])>max_bb_zone[2]:
        print("returning z greater max")
        return False
    
    if max(min_bb[2],max_bb[2])<min_bb_zone[2]:
        print("returning z less mim")
        return False
    return True

import math
def area(vertices):
    """
    Calculates the area of a 3D triangle from it's coordinates
    """
    side_a = np.linalg.norm(vertices[0]-vertices[1])
    side_b = np.linalg.norm(vertices[1]-vertices[2])
    side_c = np.linalg.norm(vertices[2]-vertices[0])
    s = 0.5 * ( side_a + side_b + side_c)
    return math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))

def find_polygon_area(mesh,list_of_faces):
    "Calculates the area of a 3D polygon that is created from connected traingles"
    return(sum([area(mesh.vertices[mesh.faces[r]]) for r in list_of_faces]))

# restitching functions
import trimesh
import numpy as np
from collections import Counter
import time
import math

#gets the projection of point p onto line a
def ClosestPointOnLine(a, b, p):
    ap = p-a
    ab = b-a
    #base_vector = ab
    result = np.dot(ap,ab)/np.dot(ab,ab) # * ab
    return result



#now have the mesh and the facet faces, can send to function
def stitch_mesh_piece_vp4(new_mesh,facet_1,facet_2,
                          delete_facets=False,
                         return_added_mesh = False,
                         fix_normals = False):

    """
    Changed since last version: 
    1) parameter for deleting facets at end or not
    2) parameter for returning added mesh or not 
    3) Changed normals check to print statement and not exception


    """
    #how to find the normals of facet groups:
    facet_group_1_normal = new_mesh.face_normals[facet_1[0]] #main_mesh_normals
    facet_group_2_normal = new_mesh.face_normals[facet_2[0]] #child_mesh_normals


    #get the correct version of the normals: (might need to flip them if going in opposite direction)
    if np.dot(facet_group_1_normal,facet_group_2_normal) > 0.8:
        raise Exception("same direction normals")
    elif np.dot(facet_group_1_normal,facet_group_2_normal) < -0.8:
        print("opposite direction normals")
    else:
        print("Not correct normals")
        #raise Exception("Not correct normals")

    # make each row correspond to a single face 
    edges = new_mesh.edges_sorted.reshape((-1, 6))
    # get the edges for each facet
    edges_facet = [edges[i].reshape((-1, 2)) for i in [facet_1,facet_2]]
    edges_boundary = np.array([i[trimesh.grouping.group_rows(i, require_count=1)]
                               for i in edges_facet])

    #the list of boundary edges and unique points in the boundary edges
    edge_0 = edges_boundary[0]
    edge_1 = edges_boundary[1]

    #gets the unique number of points
    edge_0_points = np.unique(np.hstack(edge_0))
    edge_1_points = np.unique(np.hstack(edge_1))
    print("Found boundary edges")
    """
    get the dot product of all the points
    """

    #get any 2 points on the triangle and make that the reference edge
    edge_0_anchor_points = new_mesh.vertices[[edge_0_points[0],edge_0_points[1]]]

    #gets the starting index for the 1st facet (so that the start of the stitching is close)
    max_index = 0
    max_magnitude = ClosestPointOnLine(edge_0_anchor_points[0],edge_0_anchor_points[1],new_mesh.vertices[edge_0_points[max_index]])

    for i in range(1,len(edge_0_points)):
        current_magnitude = ClosestPointOnLine(edge_0_anchor_points[0],edge_0_anchor_points[1],new_mesh.vertices[edge_0_points[i]])

        if current_magnitude > max_magnitude:
            max_index = i
            max_magnitude = current_magnitude

    edge_0_starting_point = edge_0_points[max_index]

    #gets the starting index for the 2nd facet (so that the start of the stitching is close)
    max_index = 0
    max_magnitude = ClosestPointOnLine(edge_0_anchor_points[0],edge_0_anchor_points[1],new_mesh.vertices[edge_1_points[max_index]])

    for i in range(1,len(edge_1_points)):
        current_magnitude = ClosestPointOnLine(edge_0_anchor_points[0],edge_0_anchor_points[1],new_mesh.vertices[edge_1_points[i]])
        if current_magnitude > max_magnitude:
            max_index = i
            max_magnitude = current_magnitude

    edge_1_starting_point = edge_1_points[max_index]

    print(f"starting edge 1st facet = {edge_0_starting_point}, starting edge 2nd facet= {edge_1_starting_point}, ")
    #print(new_mesh.vertices[edge_0_starting_point],new_mesh.vertices[edge_1_starting_point])

    """
    Need to order the points for restitching

    Pseudocode: 
    1) Get starting piont
    2) Find the two edges corresponding to that point
    3) Need to decide which direction to start....
    - go in direction that make the cross of the (1st and last) point in the same direction of the 
    normal of the first facet
    4) loop through and record the orders of the vertices as you traverse along the edges 
    until you arrive back at the start
    5) Error if:
        a. You arrive back at the start and haven't processed all the edges
        b. Processsed all the edges but haven't arrived back at start

    6) Repeat steps 1 through 5 for 2nd facet group
    """
    start_point_list = [edge_0_starting_point,edge_1_starting_point]
    edge_list = [edge_0,edge_1]
    edge_order_list = []



    #loop that organizes the unique boundary points into the correct order
    for i,start_point in enumerate(start_point_list):
        print(f"Starting Organizing vertices for side {i}")
        #print(f"start_point = {start_point}")
        edge_order = [start_point]
        processed_edges = []

        #find the matching edges to the starting point
        starting_edges_indices = np.where(np.logical_or(edge_list[i][:,0] == start_point,edge_list[i][:,1] == start_point) == True)[0]

        starting_edges = edge_list[i][starting_edges_indices]
        #print(f"starting edges = {starting_edges}") #the list of the two possible edges

        """ REMOVING TO SEE IF CAN STITCH WITHOUT THIS
        if starting_edges.size < 4:
            raise Exception("Not enough edges for 1st facet start point")

        if starting_edges.size > 4:
            raise Exception("Too many edges for 1st facet start point") 
            
        """

        #np.where(starting_edges[1,:] != start_point)[0][0]
        #gets the vectors that will be used for the cross product
        #print("np.where(starting_edges[0,:] != start_point)[0][0] = " + str(np.where(starting_edges[0,:] != start_point)[0][0]))
        #print("np.where(starting_edges[1,:] != start_point)[0][0] = " + str(np.where(starting_edges[1,:] != start_point)[0][0]))

        """*************** where pick the starting edge starts ************
        The way it works: 
        1) Gets the two possible starting edges
        2) Generates the vectors for the edges where origin is the starting point
        3) Gets the cross porduct of both vectors
        4) Chooses the cross product that is in the direction of the face normals

        Why that doesn't work:
        1) they are opposite normals


        """

        """
        Possible other solution:
        1) Get the starting points on the child edge
        2) Pick the first edge as the default edge

        """
        processed_edges.append(starting_edges_indices[0])
        current_vertex = starting_edges[0][np.where(starting_edges[0,:] != start_point)[0][0]]




    #         #gets the possible starting vectors from the two possible edges
    #         possible_starting_vector_1 = new_mesh.vertices[starting_edges[0,:][np.where(starting_edges[0,:] != start_point)[0][0]]] - new_mesh.vertices[start_point]
    #         #just start with a random edge
    #         possible_starting_vector_2 = new_mesh.vertices[starting_edges[1,:][np.where(starting_edges[1,:] != start_point)[0][0]]] - new_mesh.vertices[start_point]


    #         #find the cross product of the starting vectors
    #         starting_edges_cross = np.cross(possible_starting_vector_1,possible_starting_vector_2)

    #         #make sure that the order of the vectors goes so that the cross product is in line with the starting normal
    #         #this ensures the the circular direction of the stitching will be the same
    #         if np.dot(starting_edges_cross,facet_group_1_normal) > 0:
    #             print("Edge 1 picked for direction")
    #             processed_edges.append(starting_edges_indices[0])
    #             current_vertex = starting_edges[0][np.where(starting_edges[0,:] != start_point)[0][0]]
    #         else:
    #             print("Edge 2 picked for direction")
    #             processed_edges.append(starting_edges_indices[1])
    #             #print("np.where(starting_edges[1,:] != start_point) = " + str(np.where(starting_edges[1,:] != start_point)))
    #             current_vertex = starting_edges[1][np.where(starting_edges[1,:] != start_point)[0][0]]

        #print(f"current_vertex = {current_vertex}" )
        #print("edge_list = " + str(edge_list))



        """*************** where pick the starting edge ends ************"""
        #now iterate through number of 
        for z in range(1,edge_list[i][:,0].size):
            #print("edge_order_temp = " + str(edge_order))
            if current_vertex == start_point:
                print("Start vertex reached before processed all of edges")

                """

                These should be ok because the extra loops are created from holes inside and this process should always get the outside loop
                But some cases may not have processed all of the edges

                """
                break

            #get the next edge
            counter = 0
            next_vertex = -1
            for j,edg in enumerate(edge_list[i]):
                #print("edg = " + str(edg))
                #print("processed_edges = " + str(processed_edges))
                if current_vertex in edg and j not in processed_edges:
                    current_edge_index = j
                    if edg[0] != current_vertex:
                        next_vertex = edg[0]
                    else:
                        next_vertex = edg[1]


                    counter += 1
                    if counter >= 2:
                        #raise Exception(f"More than 2 edges possibilities for {current_vertex}")
                        #Don't want to make it an exception anymore put just print out warning
                        print("More than 2 edges possibilities for {current_vertex}") # BAC change

            #make sure the next vertex was found
            if next_vertex <= -1:
                raise Exception(f"No next vertex was found for {current_vertex} ")

            #if found next vertex then add the old vertex and edge index
            #to the processed edges lists and the order of vertices
            processed_edges.append(current_edge_index)
            edge_order.append(current_vertex)

            current_vertex = next_vertex


        edge_order_list.append(edge_order)
        print(f"edge_{i}_order done, len = {len(edge_order)} ")#"= {edge_order}")

    #     #print the edge orders
    #     for e in edge_order_list:
    #         print(type(e))
    lengths_of_boundaries = [len(x) for x in edge_order_list]



    """ ************ PROCESS OF ORDERING THE EDGE PATHS SO THAT *********
    main mesh loop goes in counter clockwise in reference to the gap
    child goes clockwise in reference to gap




    """

    """
    1) Pick the starting point as your P point
    2) For each point in ordered list calculate point - P and store that vector
    3) Do the cross product of list[0:n-2] x list[1:n-1]
    4) Take the sum of these lists of cross products
    5) Do the dot product to compare the direction with the normal vector 
    Result: 
    - if positive --> conuter-clockwise according to normal
    - if negative --> clockwise according to normal

    """

    starter_point = np.array([0,0,0])

    #     list_of_points = [1,2,3,4,0]

    #     list_of_points = [list_of_points[len(list_of_points) - x -1] for x in range(0,len(list_of_points))]
    #     print(list_of_points)

    #print("facet_group_2_normal = " + str(facet_group_2_normal))

    for ed,e_loop in enumerate(edge_order_list):

        #get the vertices according to the points
        vertices_list = new_mesh.vertices[e_loop]

        #get the cross product of the offsetted list
        #vertices_list[0:len(vertices_list)-1,:]
        """ Wrong earlier previous way

        cross_products = np.cross(vertices_list[0:len(vertices_list)-1,:],vertices_list[1:len(vertices_list)])
        """

        cross_products = np.cross(vertices_list[0:len(vertices_list),:],
                             np.vstack([vertices_list[1:len(vertices_list),:],vertices_list[0,:]]))

        sum_cross_products = np.sum(cross_products,axis=0)

        #print("cross_products = " + str(cross_products))
        #print("sum_cross_products = " + str(sum_cross_products))

        #print("Before edge list = " + str(edge_order_list[ed]))
        if ed == 0:
            #get the dot product
            normals_dot = np.dot(sum_cross_products,facet_group_1_normal)
            #print(' normals_dot = ' + str(normals_dot))
            if normals_dot > 0:
                print("Main originally was counter-clockwise --> keeping")


            else:
                print("Main originally was clockwise --> flipping")
                new_loop = [e_loop[len(e_loop) - x -1] for x in range(0,len(e_loop)-1)]
                new_loop.insert(0,e_loop[0])
                                                                                                          
                                                                                                          
                edge_order_list[ed] = new_loop.copy()


        else: 
            # for the children want the cross product to be counter clockwise
            normals_dot = np.dot(sum_cross_products,facet_group_2_normal)
            #print(' normals_dot = ' + str(normals_dot))
            if normals_dot > 0:
                print("Child originally was counter-clockwise --> flipping")
                new_loop = [e_loop[len(e_loop) - x -1] for x in range(0,len(e_loop)-1)]
                new_loop.insert(0,e_loop[0])
                                                                                                          
                                                                                                          
                edge_order_list[ed] = new_loop.copy()
            else:
                print("Child originally was clockwise --> keeping")


        #print("After edge list = " + str(edge_order_list[ed]))
    """  SHOWS THAT DOING THE CROSS PRODUCT THAT WAY WORKS
    #do the cross products manually
    for jj in range(0,len(vertices_list)-1):
        print(np.cross(vertices_list[jj],vertices_list[jj+1]))

    """




    #getting which one is the bigger one
    bigger = lengths_of_boundaries.index(max(lengths_of_boundaries))
    smaller = 1-bigger

    if bigger == 0:
        starter_face = "child" #if the bigger one was the main, then child is the smaller one you start from
    else:
        starter_face = "main" #if the bigger one was the child, then main is the smaller one you start from


    print("smaller_face = " + str(starter_face))    

    """ The rules that have to be following in order for normals to be correctly aligned
    1) if the smaller is the child (will be traveling in clockwise direction
    --> need to stitch points as:
    Regular: other_2, other_1,current_point
    Neighbor: current,other_1,current-1


    1) if the smaller is the main mesh (
    --> need to stitch points as:
    Regular: current_point,other_1,other_2
    Neighbor: current,current-1,other
    """


    #print(f"index of bigger facets = {bigger}\nindex of smaller facets = {smaller}",)

    #calculates the number of vertices will be stitched to each vertices on smaller side
    dividend = int(lengths_of_boundaries[bigger]/lengths_of_boundaries[smaller])
    remainder = lengths_of_boundaries[bigger] - int(lengths_of_boundaries[bigger]/lengths_of_boundaries[smaller])*lengths_of_boundaries[smaller]

    print(f"dividend = {dividend}, remainder = {remainder}")

    #loop that adds the new faces
    print("About to add faces")
    start_time = time.time()
    new_faces = []
    current_bigger = 0

    for i,current_smaller in enumerate(edge_order_list[smaller]):
        #print("current_smaller =" + str(current_smaller))
        #print("current_bigger=" + str(edge_order_list[bigger][current_bigger]))

        #connecting to the neighbor on the shorter side
        """
        if i == 0:

            new_faces.append([current_smaller,edge_order_list[smaller][-1],edge_order_list[bigger][current_bigger]])
        else:
            new_faces.append([current_smaller,edge_order_list[smaller][i-1],edge_order_list[bigger][current_bigger]])
        """


        if starter_face == "main":
            new_faces.append([current_smaller,edge_order_list[bigger][current_bigger],edge_order_list[smaller][i-1]])
        else:
            new_faces.append([current_smaller,edge_order_list[smaller][i-1],edge_order_list[bigger][current_bigger]])


        for j in range(0,dividend + int(i<remainder)):
            if current_bigger > len(edge_order_list[bigger]):
                raise Exception("Somehow rapped around too much")

            if current_bigger >= len(edge_order_list[bigger])-1:
                next_bigger = 0
            else:
                next_bigger = current_bigger+1

            if starter_face == "main":
                new_faces.append([edge_order_list[bigger][next_bigger],
                                  edge_order_list[bigger][current_bigger],
                                  current_smaller,
                                ])
            else:
                new_faces.append([
                                current_smaller,
                                  edge_order_list[bigger][current_bigger],
                                  edge_order_list[bigger][next_bigger],

                                ])

            current_bigger += 1





    #print("new_faces = " + str(new_faces))
    print(f"Finished adding faces: {time.time() - start_time}")


    print("Starting creating stitch mesh")
    start_time = time.time()
    stitch_mesh = trimesh.Trimesh()

    stitch_mesh.vertices = new_mesh.vertices
    stitch_mesh.faces = np.vstack([new_mesh.faces, new_faces])
    print(f"Finished creating stitch mesh: {time.time() - start_time}")




    if delete_facets == True:
        #now take away the original facet faces:
        total_faces = np.linspace(0,len(stitch_mesh.faces)-1,len(stitch_mesh.faces)).astype("int")
        facet_faces = np.hstack([facet_1 ,facet_2])
        faces_to_keep = set(total_faces).difference(set(facet_faces))
        faces_to_keep

        stitch_mesh = stitch_mesh.submesh([list(faces_to_keep)])[0]

    if fix_normals == True:
        trimesh.repair.fix_inversion(stitch_mesh)
        trimesh.repair.fix_winding(stitch_mesh)
        trimesh.repair.fix_normals(stitch_mesh)

    #print("Finished stitching")

    if return_added_mesh == True:
        added_mesh = trimesh.Trimesh()
        added_mesh.vertices = new_mesh.vertices
        added_mesh.faces = new_faces
        trimesh.repair.fix_inversion(added_mesh)
        trimesh.repair.fix_winding(added_mesh)
        trimesh.repair.fix_normals(added_mesh)

        return stitch_mesh,added_mesh

    else:
        return stitch_mesh

# ITERATIVE PROCESS THAT STITCHES TOGETHER MESHES
"""

Pseudocode for loop at the end that will keep everything going:
1) When process a child, will add that index to a list
2) Have no_new_children_processed counter set at end of loop 
    if no children were added to main mesh
    --> this will prompt the expansion of the initial parameters
3) If this gets too high



Things that still need to add:
1) Better way of making sure that the normals are good
- Can sample one of the neighboring points and flip normals if the dot product is negative


Change list: 
1) Reduced the stitch distance
2) Added the copy features that allows this cell to be rerun without rerunning whole notebook
3) added the new tie_breaker for childs that want to connect to same parent is just the one with closest facet
    Uses the added feature of child_meshes_stitch_distances that is saved along the way
4) Iterates through the repeated faces instead of just doing so once which was incorrect before
5) Fixed bug that was adding to current_main_mesh but then was using main mesh also in the loop
6) Made changes to the stitching mechanism that not error if find a vertices with more than 2 edges
--> because did observe some facets with cut out faces along the boundary
7) Changed the consider_same_direction normal as False 
8) Changed the max index error that was used for finding the starting point in stitch meshes

"""

def stitch_iteration(main_mesh,
                     main_mesh_facets_centers,
                     main_mesh_facets,
                     child_meshes,
                     child_meshes_facets,
                     bounding_box_threshold=4000,
                     stitch_distance_threshold=800,
                     size_ratio_threshold=0.15,
                     normal_closeness= 0.95,
                    bbox_expansion_percentage = 0.10,
                    stitch_expansion_percentage = 0.20,
                    size_ratio_expansion_percentage = 0.10,
                    no_new_children_limit = 4,
                    consider_same_direction_normals = False
                     
                     
                     
    ):
    
#     print_dict = dict(bounding_box_threshold=bounding_box_threshold,
#                     stitch_distance_threshold=stitch_distance_threshold,
#                     size_ratio_threshold=size_ratio_threshold,
#                     normal_closeness= normal_closeness,
#                     bbox_expansion_percentage = bbox_expansion_percentage,
#                     stitch_expansion_percentage = stitch_expansion_percentage,
#                     size_ratio_expansion_percentage = size_ratio_expansion_percentage,
#                     no_new_children_limit = no_new_children_limit,
#                     consider_same_direction_normals = consider_same_direction_normals
#                                                                           )
#     print(print_dict)

    no_new_children_multiplier = 0
    total_stitch_processing_time = time.time()

    children_processed = []

    #lists to store the faces that need to be removed later from main mesh
    child_faces_to_remove = []
    main_faces_to_remove = []

    while len(children_processed) < len(child_meshes):
        stitch_loop_time = time.time()
        if no_new_children_multiplier >= no_new_children_limit:
            print("The number of times expanding the thresholds has exceed the limit /n Just returning main mesh")
            print(f"total_stitch_processing_time = {time.time() - total_stitch_processing_time}")
            return main_mesh,children_processed,child_faces_to_remove,main_faces_to_remove

        #update the thresholds
        bounding_box_threshold = bounding_box_threshold*(1 + bbox_expansion_percentage*no_new_children_multiplier)
        stitch_distance_threshold = stitch_distance_threshold*(1 + stitch_expansion_percentage*no_new_children_multiplier)
        #reduces the 
        size_ratio_threshold = size_ratio_threshold*(1 - size_ratio_expansion_percentage*no_new_children_multiplier)

        
        #get the main mesh facets normals
        main_mesh_normals = [main_mesh.face_normals[fac[0]] for fac in main_mesh_facets]
        hit_indexes_list= []


        #dictionary to save the stitch points
        child_meshes_stitch_facets = dict()
        child_meshes_stitch_face_ratios = dict()
        child_meshes_stitch_distances = dict()
        for i,child in enumerate(child_meshes):

            if i in children_processed:
                #print(f"Child {i} already processed")
                continue

            print(f"Starting Child {i}")

            #initialize the stitch index
            #child_meshes_stitch_facets[i] = [-1,-1]

            #two highest points for the bounding box
            min_bb = np.array(main_mesh.bounding_box.vertices).min(0)
            max_bb = np.array(main_mesh.bounding_box.vertices).max(0)

            min_bb_zone = min_bb - bounding_box_threshold
            max_bb_zone = max_bb + bounding_box_threshold



            #then send mesh to function that decides if with
            pass_bbox_filter = apply_bbox_filter(child,min_bb_zone,max_bb_zone)

            if not pass_bbox_filter:
                print("skipped by bounding box filter")
                continue



            """  NEW WAY OF COMPUTING THE PAIRWISE CALCULATIONS  *************************************************************** """

            child_facets,child_facets_centers = child_meshes_facets[i]
            pairs_start_time = time.time()


            if len(child_facets_centers) == 0:
                print("child_facets_centers for child {i} was 0, so skipping")
                continue

            start_time = time.time()
            a_New = np.array(child_facets_centers).astype("float")
            b_New = np.array(main_mesh_facets_centers).astype("float")

            print(len(a_New),len(b_New))
    #         print("starting distance matrix")
            start_time = time.time()
            from scipy.spatial import distance_matrix
            a_b_distance = distance_matrix(a_New, b_New)

            #print(f"Time = {time.time() - start_time}")

            #now get the indexes that are within stitch distance

    #         print("min(a_b_distance) = " + str(np.amin(a_b_distance.shape)))
            indexes_not = np.where(a_b_distance < stitch_distance_threshold)
            print(f"Done distance matrix: {time.time() - start_time}")


    #         print("(indexes_not) = " + str((indexes_not)))
    #         print("starting normals")
            start_time = time.time()

    #         #old way to do it
    #         child_normals = child.face_normals[indexes_not[0]]
    #         main_normals = main_mesh.face_normals[indexes_not[1]]

            if not indexes_not[0].any():
                print(f"Child {i} There were no points close enough")
                continue

            #old way to do it
    #         print(indexes_not[0])
    #         print(child_facets[indexes_not[0]])


            child_normals = child.face_normals[[p[0] for p in child_facets[indexes_not[0]]]]
            main_normals = main_mesh.face_normals[[p[0] for p in main_mesh_facets[indexes_not[1]]]]



    #         print(child_facets[indexes_not[0]][0])
    #         print('child_normals = ' + str(child_normals))
    #         print(main_mesh_facets[indexes_not[1]][0])
    #         print('main_normals = ' + str(main_normals))
            #start_time = time.time()
            #dot_products = np.dot(child_normals,main_normals.T)

            total_normal_dots = np.sum(child_normals*main_normals,axis=1)
    #         print("total_normal_dots = " + str(total_normal_dots))
            total_normal_dots_facet_mask = total_normal_dots < -normal_closeness

    #         print(f"Done Normals generation: {time.time() - start_time}")
    #         print("starting pair generation")


            start_time = time.time()
            final_pairs = (np.array(indexes_not).T)[total_normal_dots_facet_mask]
            #print(final_pairs)


            #get the sizes of all the unique ones
            face_0_unique_facets = np.unique(final_pairs[:,0])
            face_1_unique_facets = np.unique(final_pairs[:,1])
            #print("final_pairs = " + str(final_pairs))
    #         print("face_0_unique_facets = " + str(face_0_unique_facets))
    #         print("face_1_unique_facets = " + str(face_1_unique_facets))


            if len(final_pairs) <= 0:
                print(f"Child {i} There was no possible stitch found after stitch distance and face normal filters")
                continue

            if len(final_pairs[0]) > 0:
                face_0_facet_sizes = dict([(u,find_polygon_area(child,child_facets[u])) for u in face_0_unique_facets])
                face_1_facet_sizes = dict([(u,find_polygon_area(main_mesh,main_mesh_facets[u])) for u in face_1_unique_facets])


            possible_stitch_pairs = final_pairs
            print(f"Done Generating final pairs: {time.time() - pairs_start_time}")


            """  NEW WAY OF COMPUTING THE PAIRWISE  *************************************************************** """





        #     print("possible_stitch_pairs = " + str(possible_stitch_pairs))
        #     print("len(hit_indexes) = " + str(len(hit_indexes)))
        #     print("hit_indexes[0].any() = " + str(hit_indexes[0].any()))
        #     print("hit_indexes = " + str(hit_indexes))
        #     print("hit_indexes[0] = " + str(hit_indexes[0]))
        #     print("len(hit_indexes[0]) = " + str(len(hit_indexes[0])))

            #find the sizes of all of them
            face_pair_sizes = np.zeros(len(possible_stitch_pairs[:,0]))
            face_size_ratios = np.zeros(len(possible_stitch_pairs[:,0]))

            #print("possible_stitch_pairs = " + str(possible_stitch_pairs))
            for numba,pair in enumerate(possible_stitch_pairs):
                #print("pair = " + str(pair))
                sizes = [face_0_facet_sizes[pair[0]],face_1_facet_sizes[pair[1]]]
                min_area = min(sizes)
                max_area = max(sizes)

                ratio = min_area/max_area

                #print(f"ratio = {ratio}")
                #print(f"Total size  = {min_area + max_area}")
                if ratio >= size_ratio_threshold:
                    face_pair_sizes[numba] = min_area + max_area
                    face_size_ratios[numba] = ratio

                    #print(f"face_pair_sizes[numba] = {face_pair_sizes[numba]}, face_size_ratios[numba] = {face_size_ratios[numba]}")


            #check that made it past stitch ratio threshold

            #best possible stitch pair is just the maximum sized matching ones
            best_index = np.where(face_pair_sizes == max(face_pair_sizes))
            best_stitch_pair = possible_stitch_pairs[best_index][0]
            best_stitch_pair_size = face_pair_sizes[best_index][0]
            best_stitch_pair_size_ratio = face_size_ratios[best_index][0]

            #get the distance of the best_stitch_pair
            best_stitch_pair_distance = a_b_distance[possible_stitch_pairs[best_index][0][0],
                                                       possible_stitch_pairs[best_index][0][1]]

            print("best_stitch_pair = " + str(best_stitch_pair))
            print("best_stitch_pair_size = " + str(best_stitch_pair_size))
            print("best_stitch_pair_distance = " + str(best_stitch_pair_distance))
            print("best_stitch_pair_size_ratio = " + str(best_stitch_pair_size_ratio))

            child_meshes_stitch_facets[i] = [best_stitch_pair[0],best_stitch_pair[1]]
            child_meshes_stitch_face_ratios[i] = best_stitch_pair_size_ratio
            child_meshes_stitch_distances[i] = best_stitch_pair_distance

    #         if i == 1:
    #             print("Just processed piece 1")
    #             raise Exception("stoping")


    #     if 1 in child_meshes_stitch_facets.keys():
    #         print("Just processed piece 1")
    #         raise Exception("stoping")


        #if there were no possible stitch points found
        if len(child_meshes_stitch_facets.keys()) == 0:
            #increment the no children flag multiplier

            no_new_children_multiplier += 1
            print(f"no stitch points found IN ALL CHILDREN --> relaxing the parameters time {no_new_children_multiplier}")
            continue

        # makes sure that no two child branches try to connect to the same main branch
        from collections import Counter
        mesh_stitch_counter = Counter(np.array([val for val in child_meshes_stitch_facets.values()])[:,1])

        repeat_main_facets = [key for key,val in mesh_stitch_counter.items() if (key != -1 and val > 1)] #gets the main mesh facet with multiples
        print("repeat_main_facets = " + str(repeat_main_facets))


        #how to fix that some faces are trying to branch to same main facet
        #make it iterate through all of the repeats
        if len(repeat_main_facets)>0:
            for repeat_main in repeat_main_facets:
                child_mesh_double_indexes = [key for key,val in child_meshes_stitch_facets.items() if val[1] == repeat_main]
                print("child_mesh_double_indexes = " + str(child_mesh_double_indexes))


                #decide which one to keep and then ditch all the rest --> pick the CLOSEST ONE, and not the best matching area:

                ###### picks the closest area
                min_distance = math.inf
                min_child = -1


                for child_index in child_mesh_double_indexes:
                    current_distance = child_meshes_stitch_distances[child_index]

                    if current_distance < min_distance:
                        min_child = child_index
                        min_distance = current_distance

                print(f"min_child = {min_child}, max_ratio = {min_distance}")

        #         ###### picks the maximum area
        #         max_ratio = -1
        #         max_child = -1


        #         for child_index in child_mesh_double_indexes:
        #             current_ratio = child_meshes_stitch_face_ratios[child_index]

        #             if current_ratio > max_ratio:
        #                 max_child = child_index
        #                 max_ratio = current_ratio

        #         print(f"max_child = {max_child}, max_ratio = {max_ratio}")



                #remove the others from the stitch facets
                for double_index in child_mesh_double_indexes:
                    if double_index != min_child:
                        del child_meshes_stitch_facets[double_index]

        """
        Pseudocode for stitching:
        1) For each pair in the child_meshes_stitch_facets:
        a. Get the child mesh for that pair
        b. Get the list of faces for the child facet (from the facet number)
        c. Get the list of faces for the main facet (from the main number)

        d. Get the original number of faces and vertices in the main mesh
        d2. Use the orignal number of faces and add to list of faces for child facet to offset them correctly
            - Save this number list in a dictionary (to use for later and creating the submesh)
        e. Add the two meshes together to get big mesh
        f. Send the two meshes and the two facet lists to the restitching function to get a main mesh that is stitched up
         - but send it to function that doesnt delete the original facet faces 
             (because this would remove meshes from original and screw up facet number)
        g. reassign the main_mesh to this newly stitched up mesh
        h. recompute the facets for the main mesh



        """


        current_main_mesh = main_mesh.copy()

        main_mesh_facet_index_to_delete = []


        #if there were no possible stitch points found
        if len(child_meshes_stitch_facets.keys()) == 0:
            #increment the no children flag multiplier

            no_new_children_multiplier += 1
            print(f"no stitch points found IN ALL CHILDREN --> relaxing the parameters time {no_new_children_multiplier}")
            continue
        else: #if there were stitch points found
            print("child_meshes_stitch_facets = " + str(child_meshes_stitch_facets))
            for child_key,pair in child_meshes_stitch_facets.items():
                """
                child_key has the child that is currently being processed

                pair has the facet ids that are to be stitched together

                """

                child_used_facet_index = pair[0]
                main_used_faceet_index = pair[1]

                stitch_time = time.time()
                print(f"---Stitching child {child_key} with pair: {pair}---")
                current_child_mesh = child_meshes[child_key]
                current_child_facet_faces = child_meshes_facets[child_key][0][pair[0]]


                current_main_mesh_facet_faces = main_mesh_facets[pair[1]]

                #Get the original number of faces and vertices in the main mesh
                original_mesh_faces_len = len(current_main_mesh.faces)
                current_child_facet_faces_adjusted = current_child_facet_faces + original_mesh_faces_len

                #Save the faces number for deletion later
                child_faces_to_remove += current_child_facet_faces_adjusted.tolist()
                main_faces_to_remove += current_main_mesh_facet_faces.tolist()

                combined_mesh = current_main_mesh + current_child_mesh

                #how to stitch up the mesh
                start_time = time.time()
                current_main_mesh = stitch_mesh_piece_vp4(new_mesh=combined_mesh,
                                                               facet_1=current_main_mesh_facet_faces,
                                                               facet_2=current_child_facet_faces_adjusted,
                                                              delete_facets=False,
                                                              return_added_mesh=False,
                                                               fix_normals = False)

                print(f"returned from stitch mesh: {time.time() - start_time}")


                #Don't need to do any deletion now 

                #add the child to processed child
                children_processed.append(child_key)
                print(f"Finished stitching child {child_key} : {time.time() - stitch_time}")

                """
                Add all of the child facets that weren't the ones used to the main mesh facets (with adjusted facet numbers)
                """
                original_mesh_faces_len #the original number of faces in the main mesh (including all those stitched before)

                #remove certian rows from array of the child facets and cetners
                current_child_facet_faces_with_deletion = np.delete(child_meshes_facets[child_key][0],(child_used_facet_index),axis=0)
                current_child_facet_centers_with_deletion = np.delete(child_meshes_facets[child_key][1],(child_used_facet_index),axis=0)

                #have to adjust the faces of the child_facets list to account for the faces off set
                current_child_facet_faces_with_deletion = current_child_facet_faces_with_deletion + original_mesh_faces_len

                #append this list to the main mesh list
                main_mesh_facets = np.concatenate([main_mesh_facets,current_child_facet_faces_with_deletion])
                main_mesh_facets_centers = np.concatenate([main_mesh_facets_centers,current_child_facet_centers_with_deletion])

                #save off the facets to delete for the main mesh at the end of the loop
                main_mesh_facet_index_to_delete.append(pair[1])




            #reset the no_new_children_multiplier because there were successful stitching
            no_new_children_multiplier = 0
            """recompute the main mesh facets  ###Not going to recompute anymore
            #main_mesh_facets,main_mesh_facets_centers = filter_final_facets(main_mesh)

            Now we just remove the main mesh facets that are used and reassign
            """
            main_mesh = current_main_mesh


            #at the end of the big loop have to delete the facets used from main mesh
            #remove certian rows from array

            #creating the new facets
            main_mesh_facets = np.delete(main_mesh_facets,(main_mesh_facet_index_to_delete),axis=0)
            main_mesh_facets_centers = np.delete(main_mesh_facets_centers,(main_mesh_facet_index_to_delete),axis=0)


            print(f"***************Finished big stitching iteration: {time.time() - stitch_loop_time}***************")

    #         if 0 in children_processed and 1 in children_processed:
    #             print("Exiting before does piece 2")
    #             break

            if len(np.unique(children_processed)) == len(child_meshes):
                print("All children have been processed")
                print(f"total_stitch_processing_time = {time.time() - total_stitch_processing_time}")
                return main_mesh,children_processed,child_faces_to_remove,main_faces_to_remove


def stitch_neuron(segment_id,
                  vertices,
                  faces,
                **kwargs):
    
    """
    Stitches the portions of disconnected mesh pieces of a given mesh back together and applies a pymeshfix
    cleaning on the mesh (if the pymeshfix_flag is set). 
  

    Parameters: 
    segment_id (int): segment id of the mesh to be stitched and cleaned
    vertices (np.array): list of vertices for mesh
    faces (np.array): list of faces array for the mesh
    
    Returns:
    stitched_vertices (np.array): list of vertices for single largest mesh after stitching
    stitched_faces (np.array): list of faces for single largest mesh after stitching
    
    **** if stitch_and_unstitched_flag is set to True, returns the following instead of stitched_vertices,stitched_faces
    stitched_and_unstitched_vertices (np.array): list of vertices for both stitched and unstitched meshes after stitching
    stitched_and_unstitched_faces (np.array): list of faces for both stitched and unstitched meshes after stitching
    ********************************************************************************************************************
    
    filtered_inside_percentage (float) : percentage of total mesh that was filtered away becuase inside the main mesh
    stitched_percentage (float) : percentage of mesh after inside/outside filtered that was stitched back to the main mesh
    unstitched_percentage (float) : percentage of mesh after inside/outside filtered that was left unstitched back to main mesh
    
    
    Optional Parameters:
    -- importing parameters --
    import_from_off (bool) : if set true then will attempt to import the function mesh a local directory instead of 
                            using the segment id and datajoint database (default = False)
    off_file_path (str) : path to local off file to be loaded (default = "")
    
    -- saving and loading main and child mesh options --
    save_file_location (string) : location for saved/cached files to be stored (default = "./stitch_mesh_saved")
                                    if this does not already exists then creates it
                                    
    save_file_name (string) : filename for saved main and child meshes (must be an .npz file) 
                                (default = str(segment_id) + "_" + str(outisde_size_threshold) + " _main_and_child_meshes_array.npz"
  
    )
    
    save_meshes_flag (bool) : if set true, will save off the main and child meshes in the save_file_location with 
                        the following name (default = False)
    
    
    load_file_location (string) : location for saved/cached files to be loaded from if save_meshes flag set (default = "./stitch_mesh_saved")
                                   
    load_file_name (string) : filename in the save_file_location where a saved copy of the 
                                            main and child meshes .npz file is, if this is specified then does
                                            not do inside/outside filtering but instead uses this saved file
                                            (default = str(segment_id) + "_" + str(outisde_size_threshold) + " _main_and_child_meshes_array.npz"
  
     
    load_meshes_flag (bool) : if set true, will load off the main and child meshes specified in the 
                                load locations and load file name (default = False)
                     
                     
    
    
    --Inside/Outisde Mesh Filtering--
    
    outisde_size_threshold (int): number of faces of a child mesh outside the main piece must be 
                                   in order to be attempted to be restitched (default = 30)
    n_sample_points (int) : The number of points randomly sampled to decide if the child mesh is 
                            inside or outside the main mesh (default = 3)
    
                            
    -- extracting facets --
    length_threshold (int) : size that determines large from small meshes that determines how many minumum faces must be in a facet (default = 60000)
    min_len_large (int) :  minimum number of faces to be included in a facet for large meshes (default = 3)
    min_len_small (int) :  minimum number of faces to be included in a facet for small meshes (default = 2)
    normal_closeness_facets (float) : minimum dot product of normals for adjacent faces to 
                                be considered in the same facet (default = 0.99)
    first_pass_size_threshold (float) : minimum area of facet (default = 6000)
    adjacency_threshold (float ) : threshold for average convexity of a facet edge 
                                in order to be a possible stitch point (default = 0.8)
                                
                                
    -- stitching parameters --
    None
    
    -- finding stitching point parameters --
    
    bounding_box_threshold (int) : number added to dimensions of main mesh bounding
                                    that will help filter the number of child meshes that could possibly be stitched (default=4000)
    stitch_distance_threshold (int) : starting distance that is the maximum distance between 
                                      two possible facets to be stitched together (default=800) 
    size_ratio_threshold (float) : minimum ratio between sizes of facets in order to be considered possible pair (default=0.15)
    normal_closeness: minimum dot product between normals of first faces in each facet to make sure they are
                            pointing in the correct directions(default=0.95)
    no_new_children_limit (int) : maximum number of iterations that can be done without finding any new stitch points,
                                if this number is exceeded then breaks out of loop (default= 4)
    consider_same_direction_normals (bool) : flag that if true allows for normals pointing in the same direction
                                            to be in consideration for possible stitch points(default = False)
    
    - paameter expansion values -
    
    bbox_expansion_percentage (float) : expands bounding box threshold by x percentage after iteration 
                                        with no stitch points found (default = 0.10 )
    stitch_expansion_percentage (float) : expands stitch length threshold by x percentage after iteration 
                                        with no stitch points found (default = 0.20 )
    size_ratio_expansion_percentage (float) : contracts size ratio x percentage after iteration 
                                        with no stitch points found (default = 0.10 )
                                        
    -- pymeshfix clean --
    
    pymeshfix_flag (bool) : if set true, applies a pymeshfix cleaning on stitched mesh at the end
                            before returning the mesh (default = False) 
                            
    -- return flags --
    stitch_and_unstitched_flag (bool) : if set true returns the stitched portions 
                                        with the unstitched portions (default = False)
    

    """
    
    #now get all of the parameters
    
    #-- importing parameters --
    
    global_time = time.time()
    
    import_from_off = kwargs.pop('import_from_off', False)
    off_file_path = kwargs.pop('off_file_path', "")
    
    
        
    #--Inside/Outisde Mesh Filtering-- parameters
    outisde_size_threshold = kwargs.pop("outisde_size_threshold",30)
    n_sample_points = kwargs.pop("n_sample_points",3)
    
    
    
    
    #-- saving and loading main and child mesh options --
    save_file_location = kwargs.pop("save_file_location","./stitch_mesh_saved")
    save_file_name = kwargs.pop("save_file_name",str(segment_id) + "_" + str(outisde_size_threshold) + "_main_and_child_meshes_array.npz")
    save_meshes_flag = kwargs.pop("save_meshes_flag",False)
    
    load_file_location = kwargs.pop("load_file_location","./stitch_mesh_saved")
    load_file_name = kwargs.pop("load_file_name",str(segment_id) + "_" + str(outisde_size_threshold) + "_main_and_child_meshes_array.npz")
    load_meshes_flag = kwargs.pop("load_meshes_flag",False)
    
    
    
    #-- extracting facets -- parameters
    length_threshold = kwargs.pop("length_threshold",60000)
    min_len_large = kwargs.pop("min_len_large",3)
    min_len_small = kwargs.pop("min_len_small",2)
    normal_closeness_facets = kwargs.pop("normal_closeness_facets",0.99)
    first_pass_size_threshold = kwargs.pop("first_pass_size_threshold",6000)
    adjacency_threshold = kwargs.pop("adjacency_threshold",0.8)
    
        
    #-- finding stitching point parameters --
    bounding_box_threshold = kwargs.pop("bounding_box_threshold",4000)
    stitch_distance_threshold = kwargs.pop("stitch_distance_threshold",2000)
    size_ratio_threshold = kwargs.pop("size_ratio_threshold",0.15)
    normal_closeness = kwargs.pop("normal_closeness",0.95)
    no_new_children_limit = kwargs.pop("no_new_children_limit",4)
    consider_same_direction_normals = kwargs.pop("consider_same_direction_normals",False)
    
    
    #-- paameter expansion values -- parameters
    bbox_expansion_percentage = kwargs.pop("bbox_expansion_percentage",0.10)
    stitch_expansion_percentage = kwargs.pop("stitch_expansion_percentage",0.20)
    size_ratio_expansion_percentage = kwargs.pop("size_ratio_expansion_percentage",0.10)
    
    #    -- pymeshfix clean -- parameters
    pymeshfix_flag = kwargs.pop("pymeshfix_flag",False)
 
                            
    #-- return parameters --
    stitch_and_unstitched_flag = kwargs.pop("stitch_and_unstitched_flag",False)

    #######------ finished importing all of the parameters --------
    
    #making sure there is no more keyword arguments left that you weren't expecting
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    
    
    ##### STEP 1) IMPORT THE MESH FROM FACES,VERTICES OR FROM AN OFF FILE
    
    if import_from_off == True:
        #load the path
        unfiltered_mesh = trimesh.load_mesh(file_location + file_name)
    else:
        unfiltered_mesh = trimesh.Trimesh()
        unfiltered_mesh.vertices = vertices
        unfiltered_mesh.faces = faces

    
    
    ##### STEP 2) GENERATE THE CHILD AND MAIN MESH FROM THE WHOLE MESH
    start_time = time.time()
    if load_meshes_flag == False:
        print("generating child and main meshes")


        #load the child meshes from a locally saved file
        #setting thresholds

        #the main mesh is the first mesh in the piece
        main_mesh,child_meshes = filter_mesh_significant_outside_pieces(unfiltered_mesh,
                                    significance_threshold=outisde_size_threshold,
                                        n_sample_points=n_sample_points)

    else: #load them from a saved location
        print("import child and main meshes from" + str(Path(load_file_location) / Path(load_file_name)))
        
        if not os.path.isfile(str(Path(load_file_location) / Path(load_file_name))):
            raise TypeError(str(Path(load_file_location) / Path(load_file_name)) + " cannot be found for loading mesh files")
            return None

        #load in the file
        loaded_meshes = np.load(str(Path(load_file_location) / Path(load_file_name)))

        main_mesh_loaded = loaded_meshes["main_mesh"][0]
        child_meshes_loaded = loaded_meshes["child_meshes"]

        main_mesh = main_mesh_loaded
        child_meshes = child_meshes_loaded

    
    
    
    #get the percentage of the largest mesh piece in relation to the whole
    largest_piece_perc = len(main_mesh.faces)/len(unfiltered_mesh.faces)
    n_pieces = len(child_meshes)
    

    
    print(f"Total time for Mesh Cleansing: {time.time() - start_time}")
    total_faces_retained = len(main_mesh.faces) + sum([len(current_mesh.faces) for current_mesh in child_meshes])
    filtered_inside_percentage = total_faces_retained/len(unfiltered_mesh.faces)
    print(f"Number of outside child meshes =  {str(len(child_meshes))}")
    print(f"Main mesh and outside child meshes make up {filtered_inside_percentage}% of original mesh")
    # get the statistics on the parts of the mesh that was filtered away

    
    
                     
    #check to see if child meshes is empty:
    if len(child_meshes) == 0:
        
        n_stitched=0
        stitched_percentage= 0
        n_unstitched=0
        unstitched_percentage = 0
        
        #just return the current main mesh and all the parameters 
        return (len(main_mesh.vertices),
        len(main_mesh.faces),
        main_mesh.vertices,
        main_mesh.faces,
        n_pieces,
        largest_piece_perc,
        filtered_inside_percentage,
        n_stitched,
        stitched_percentage,
        n_unstitched,
        unstitched_percentage)
        
        
    
    ##### STEP 3) POSSIBLY SAVING THE CHILD AND MAIN MESH

    if save_meshes_flag == True:
        #check that the save location exists:
        try:
            os.listdir(save_file_location)
        except:
            print(f"{save_file_location} didn't exist so making it now")
            os.mkdir(save_file_location)

        #check that the file name is an npz file extension
        if save_file_name[-4:] != ".npz":
            raise Exception(str(save_file_name) + " cannot be saved because isn't a .npz file extension")
            return None


        np.savez(str(Path(save_file_location) / Path(save_file_name)),main_mesh=[main_mesh],child_meshes=child_meshes)

        print("Saved child and main meshes at " + str(Path(save_file_location) / Path(save_file_name)))
    
    ##### STEP 4) COMPUTE FACEETS FOR CHILD MESHES AND MAIN MESHES
    facet_time = time.time()
    start_time = time.time()
    

    if len(main_mesh.faces > length_threshold):
        print(f" face length {len(main_mesh.faces)} using optimized facets with 3 neighbors")
        main_mesh_facets,main_mesh_facets_centers = filter_final_facets_optimized_with_checks(example_mesh=main_mesh,
                                                                                              min_len=min_len_large,
                                                                                              normal_closeness=normal_closeness_facets,
                                                                                              first_pass_size_threshold=first_pass_size_threshold,
                                                                                              adjacency_threshold=adjacency_threshold)

    else:
        print(f" face length {len(main_mesh.faces)} using optimized facets with 2 neighbors")
        main_mesh_facets,main_mesh_facets_centers = filter_final_facets_optimized_with_checks(example_mesh=main_mesh,
                                                                                              min_len=min_len_small,
                                                                                              normal_closeness=normal_closeness_facets,
                                                                                              first_pass_size_threshold=first_pass_size_threshold,
                                                                                              adjacency_threshold=adjacency_threshold)
    if len(main_mesh_facets) > 10000:
            print(f"Finished facets for main mesh: {time.time() - start_time} with facet length = {len(main_mesh_facets)}")
                                                                                          
    

    print(f"Finished { len(main_mesh_facets)} facets for main mesh: {time.time() - start_time}")
    child_meshes_facets= []
    for jj,gap_mesh in enumerate(child_meshes):
        #print("Starting child " + str(jj))
        start_time = time.time()
        if len(gap_mesh.faces) > length_threshold:
            #print(f" face length {len(gap_mesh.faces)} using optimized facets 3")
            child_meshes_facets.append(filter_final_facets_optimized_with_checks(example_mesh=gap_mesh,
                                                                                  min_len=min_len_large,
                                                                                  normal_closeness=normal_closeness_facets,
                                                                                  first_pass_size_threshold=first_pass_size_threshold,
                                                                                  adjacency_threshold=adjacency_threshold))
        else:
            #print(f" face length {len(gap_mesh.faces)} using optimized facets 2")
            child_meshes_facets.append(filter_final_facets_optimized_with_checks(example_mesh=gap_mesh,
                                                                                              min_len=min_len_small,
                                                                                              normal_closeness=normal_closeness_facets,
                                                                                              first_pass_size_threshold=first_pass_size_threshold,
                                                                                              adjacency_threshold=adjacency_threshold))


        if len(child_meshes_facets[jj][0]) > 10000:
            print(f"Finished facets for child {jj} : {time.time() - start_time} with facet length = {len(child_meshes_facets[jj][0])}")


    print(f"Total time for facets: {time.time() - facet_time}")
    #child_meshes_facets = [filter_final_facets(gap_mesh) for gap_mesh in child_meshes]
    
    
    
    # Check the ones that don't have any facets
    zero_faced = []

    for i,ch in enumerate(child_meshes_facets):
        num_facets = len(ch[1])
        if num_facets == 0:
            zero_faced.append(i)
        #print(f"Child {i} has {num_facets} facets")
    print("Zero faceted faces = " + str(zero_faced))
    
    ##### STEP 5) STITCHING THE CHILD MESHES TO THE MAIN MESH
    restitch_time = time.time()
    
    original_main_mesh_face_size = len(main_mesh.faces)
              
    main_mesh,children_processed,child_faces_to_remove,main_faces_to_remove = stitch_iteration(main_mesh=main_mesh,
                                                                                               main_mesh_facets_centers=main_mesh_facets_centers,main_mesh_facets=main_mesh_facets,
                                                                                               child_meshes=child_meshes,
                                                                                                child_meshes_facets=child_meshes_facets,
                                                                                                bounding_box_threshold=bounding_box_threshold,
                                                                                                stitch_distance_threshold=stitch_distance_threshold,
                                                                                                size_ratio_threshold=size_ratio_threshold,
                                                                                                normal_closeness= normal_closeness,
                                                                                                bbox_expansion_percentage = bbox_expansion_percentage,
                                                                                                stitch_expansion_percentage = stitch_expansion_percentage,
                                                                                                size_ratio_expansion_percentage = size_ratio_expansion_percentage,
                                                                                                no_new_children_limit = no_new_children_limit,
                                                                                                consider_same_direction_normals = consider_same_direction_normals
                                                                                                                                                      )

                     
                     
    
    print(f"Total time for restitching = {time.time() - restitch_time}")
    
    
    ##### STEP 6) remove all of the processed facets from the main mesh
    #now take away the original facet faces:
    total_faces = np.linspace(0,len(main_mesh.faces)-1,len(main_mesh.faces)).astype("int")


    #these are the faces that need to be removed
    facet_faces = np.hstack([child_faces_to_remove ,main_faces_to_remove])
    faces_to_keep = set(total_faces).difference(set(facet_faces))


    main_mesh_final = main_mesh.submesh([list(faces_to_keep)])[0]
    
    ##### STEP 7) calculate final statistics for run
    #finds the total children missed and prints them out
    total_children = np.linspace(0,len(child_meshes)-1,len(child_meshes))
    missed_children = list(set(total_children).difference(set(children_processed)))
    print("missed_children = " + str(missed_children))

    #calculate the percentage of mesh faces that were added because of stitching
    stitched_percentage = (len(main_mesh_final.faces) - original_main_mesh_face_size)/ original_main_mesh_face_size
    #calculate the percentage of meshes that still aren't stitched back to the main mesh
    non_stitch_total_size = sum([len(current_mesh.faces) for i,current_mesh in enumerate(child_meshes) if i in missed_children])
    unstitched_percentage = non_stitch_total_size/ original_main_mesh_face_size
    
    """
    Returns:
    stitched_vertices (np.array): list of vertices for single largest mesh after stitching
    stitched_faces (np.array): list of faces for single largest mesh after stitching
    
    **** if stitch_and_unstitched_flag is set to True, returns the following instead of stitched_vertices,stitched_faces
    stitched_and_unstitched_vertices (np.array): list of vertices for both stitched and unstitched meshes after stitching
    stitched_and_unstitched_faces (np.array): list of faces for both stitched and unstitched meshes after stitching
    ********************************************************************************************************************
    
    filtered_inside_percentage (float) : percentage of total mesh that was filtered away becuase inside the main mesh
    stitched_percentage (float) : percentage of mesh after inside/outside filtered that was stitched back to the main mesh
    unstitched_percentage (float) : percentage of mesh after inside/outside filtered that was left unstitched back to main mesh
    """
    
    if pymeshfix_flag == True:
        start_time = time.time()
        #pass the vertices and faces to pymeshfix to become watertight
        meshfix = pymeshfix.MeshFix(main_mesh_final.vertices,main_mesh_final.faces)
        meshfix.repair(verbose=False,joincomp=True,remove_smallest_components=False)
        
        print(f"Pymesh shrinkwrapping: {time.time() - start_time}")
        
        main_mesh_final.vertices = meshfix.v
        main_mesh_final.faces = meshfix.f
    
    
    print(f"Whole stitching function complete: {time.time() - global_time}")
    
    
    
    """
    Goal of what we want it to return to write to datajoint table
    
    n_vertices           : bigint           # number of vertices in this mesh
    n_triangles          : bigint           # number of triangles in this mesh
    vertices             : longblob         # x,y,z coordinates of vertices
    triangles            : longblob         # triangles (triplets of vertices)
    n_pieces             : int              # number of unconnected mesh pieces
    largest_piece_perc   : decimal(6,5)     # number of faces percentage of largest mesh piece in respect to total mesh
    outside_perc         : decimal(6,5)     # number of faces percentage of mesh outside the biggest mesh piece
    n_stitched           : int              # number of mesh pieces stitched back to main mesh
    stitched_addon_perc  : decimal(6,5)     # number of faces percentage of pieces that were stitched back in respect to largest mesh piece
    n_unstitched         : int              # number of mesh pieces remaining unstitched back to main mesh        
    unstitched_perce     : decimal(6,5)     # number of faces percentage of pieces that were not in respect to largest mesh piece
    
    
    """

    
    
    n_stitched = len(children_processed)
    n_unstitched = len(missed_children)


    if stitch_and_unstitched_flag == False:
        return (len(main_mesh_final.vertices),
                len(main_mesh_final.faces),
                main_mesh_final.vertices,
                main_mesh_final.faces,
                n_pieces,
                largest_piece_perc,
                filtered_inside_percentage,
                n_stitched,
                stitched_percentage,
                n_unstitched,
                unstitched_percentage)
        #return main_mesh_final,filtered_inside_percentage,stitched_percentage,unstitched_percentage
    else:
        for child_id in missed_children:
            main_mesh_final = main_mesh_final + child_meshes[child_id]
        return (len(main_mesh_final.vertices),
                len(main_mesh_final.faces),
                main_mesh_final.vertices,
                main_mesh_final.faces,
                n_pieces,
                largest_piece_perc,
                filtered_inside_percentage,
                n_stitched,
                stitched_percentage,
                n_unstitched,
                unstitched_percentage)
    
              