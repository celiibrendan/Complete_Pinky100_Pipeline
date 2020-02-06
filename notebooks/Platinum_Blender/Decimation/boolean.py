from pykdtree.kdtree import KDTree
import time
import trimesh
import numpy as np

def filter_mesh_significant_outside_pieces(unfiltered_mesh,main_mesh,significance_threshold=2000,n_sample_points=3):
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
    1) list of significant mesh pieces that are not inside of main mesh

    """
    
    print("------Starting the mesh filter for significant outside pieces-------")

    mesh_pieces = unfiltered_mesh.split(only_watertight=False)
    
    print(f"There were {len(mesh_pieces)} pieces after mesh split")

    significant_pieces = [m for m in mesh_pieces if len(m.faces) > significance_threshold]

    print(f"There were {len(significant_pieces)} pieces found after size threshold")
    if len(significant_pieces) <=0:
        print("THERE WERE NO MESH PIECES GREATER THAN THE significance_threshold")
        return []



    final_mesh_pieces = []

    #final_mesh_pieces.append(main_mesh)
    for i,mesh in enumerate(significant_pieces):
        #get a random sample of points
        # points = np.array(mesh.vertices[:n_sample_points,:]) # OLD WAY OF DOING THIS
        idx = np.random.randint(len(mesh.vertices), size=n_sample_points)
        points = mesh.vertices[idx,:]


        start_time = time.time()
        signed_distance = trimesh.proximity.signed_distance(main_mesh,points)
        #print(f"Total time = {time.time() - start_time}")

        outside_percentage = sum(signed_distance <= 0)/n_sample_points
        if outside_percentage > 0.9:
            final_mesh_pieces.append(mesh)
            #print(f"Mesh piece {i} OUTSIDE mesh")
        else:
            #print(f"Mesh piece {i} inside mesh :( ")
            pass
                
    return final_mesh_pieces
    
def neuron_boolean_difference(main_mesh_verts,
                             main_mesh_faces,
                             child_mesh_verts,
                             child_mesh_faces,
                             distance_threshold=5,
                             significance_threshold=1000,
                             n_sample_points=3):
    """
    returns the boolean difference of two meshes passed. Only returns meshes pieces that 
    are greater than the size threshold and outside of the main mesh
    
    Function operation: child_mesh - main_mesh
  

    Parameters: 
    main_mesh_verts (np.array): array of the reference mesh vertices
    main_mesh_faces (np.array): array of the reference mesh faces
    child_mesh_verts (np.array): array of the child mesh vertices that is to be compared to the main mesh
    child_mesh_faces (np.array): array of the child mesh faces that is to be compared to the main mesh
    
    Optional parameters:
    -- for controlling the mesh returned --
    distance_threshold (int) : distance away from the reference mesh that vertices from the child
                                mesh will considered distinct and part of the difference mesh 
                                (default=5)
    significance_threshold (int) : number of faces necessary for any distinct/unconnected part of the 
                                    difference mesh to be considered relevant and included in the difference mesh
                                (default=1000)
    n_sample_points (int) : number of vertices to check to see if submesh is a mesh located inside of the main mesh.
                            The less there are the quicker the speed for filtering the difference mesh
                            (default=3)
    
    
    Returns:
    difference_mesh_verts (np.array): array of vertices from the mesh boolean operation of child - main mesh
    difference_mesh_faces (np.array): array of faces from the mesh boolean operation of child - main mesh
    """
    
    #Create the kdtree from main mesh and run the queries
    
    import time
    global_time = time.time()
    start_time = time.time()
    mesh_tree = KDTree(main_mesh_verts)
    distances,closest_node = mesh_tree.query(child_mesh_verts)
    print(f"Total time for KDTree creation and queries: {time.time() - start_time}")
    
    print("Original number vertices in child mesh = " + str(len(child_mesh_verts)))
    vertex_indices = np.where(distances > distance_threshold)[0]
    print("Number of vertices after distance threshold applied =  " + str(len(vertex_indices)))
    
    start_time = time.time()
    set_vertex_indices = set(list(vertex_indices))
    face_indices_lookup = np.linspace(0,len(child_mesh_faces)-1,len(child_mesh_faces)).astype('int')
    face_indices_lookup_bool = [len(set_vertex_indices.intersection(set(tri))) > 0 for tri in child_mesh_faces]
    face_indices_lookup = face_indices_lookup[face_indices_lookup_bool]

    print(f"Total time for finding faces after distance threshold applied: {time.time() - start_time}")
    
    start_time = time.time()
    trimesh_original = trimesh.Trimesh(child_mesh_verts,child_mesh_faces,process=False) 
    new_submesh = trimesh_original.submesh([face_indices_lookup],only_watertight=False,append=True)
    
    pymesh_mesh = trimesh.Trimesh(main_mesh_verts,main_mesh_faces)
    
    #filter the mesh for only significant pieces on the outside
    returned_mesh = filter_mesh_significant_outside_pieces(new_submesh,pymesh_mesh,significance_threshold,n_sample_points=n_sample_points)

    total_returned_mesh = trimesh.Trimesh()
    for r in returned_mesh:
        total_returned_mesh = total_returned_mesh + r

    print(f"Total time for filtering: {time.time() - start_time}")
    return total_returned_mesh.vertices,total_returned_mesh.faces

    print(f"Total time for boolean difference: {time.time() - global_time}")
    
    
    
    
    
    