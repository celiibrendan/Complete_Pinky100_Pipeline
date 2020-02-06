def add_mesh_piece(main_mesh_vertices,main_mesh_faces,sub_mesh_vertices,sub_mesh_faces):
    """
    Purpose: Takes in a large mesh piece and an array of other meshes and 
    returns a large mesh with all meshes appended
    
    Parameters:
    main_mesh_vertices (np.array) : np array store the vertices as rows and the elements as the coordinates
    main_mesh_faces (np.array) : np array store the faces as rows and the elements as the referenced vertices
    sub_mesh_vertices(list of np.arrays) : list of np arrays with the vertices arrays for all subsegments to be added
    sub_mesh_faces(list of np.arrays) : list of np arrays with the faces arrays for all subsegments to be added
    
    Returns:
    mesh_vertices (np.array) : np array store the vertices as rows and the elements as the coordinates for NEW CONCATENATED MESH
    mesh_faces (np.array) : np array store the faces as rows and the elements as the referenced vertices for NEW CONCATENATED MESH
    
    
    Pseudocode: 
    - Checks: 
    a. Make sure there sub_mesh arrays are greater than 0 and of the same length

    1) Count the number of vertices and faces in the main mesh
    2) Iterate through the submesh vertices and faces. In loop:
        a. Count the number of vertices in the submesh and concate the vertices arrays to the main mesh array
        b. Add the vertices_count and add that to every number in the faces array
        c. Concatenate the submesh faces onto the larger mesh face
        d. Save this new vertices and faces as the main_mesh verts and faces
        e. Print out how many new vertices and faces added
    3) Print out number of segments added, total faces/vertices for new mesh
    4) Return the main mesh vertices and faces
    
    """
    #a. Make sure there sub_mesh arrays are greater than 0 and of the same length
    if len(sub_mesh_vertices) <= 0:
        print("There were no vertices in submesh to add, returning main mesh")
        return main_mesh_vertices, main_mesh_faces
    if len(sub_mesh_faces) <= 0:
        print("There were no face in submesh to add, returning main mesh")
        return main_mesh_vertices, main_mesh_faces
    if len(sub_mesh_faces) != len(sub_mesh_vertices):
        raise Exception("The sub_mesh_faces and sub_mesh_vertices length did not match")
        
    
    #1) Count the number of vertices and faces in the main mesh
    n_main_vertices = len(main_mesh_vertices)
    n_main_faces = len(main_mesh_faces)
    
    
    #2) Iterate through the submesh vertices and faces. In loop:
    for i,(sub_verts, sub_faces) in enumerate(zip(sub_mesh_vertices,sub_mesh_faces)):
        #a. Count the number of vertices in the submesh and concate the vertices arrays to the main mesh array
        n_sub_verts = len(sub_verts)
        n_sub_faces = len(sub_faces)
        
        main_mesh_vertices = np.vstack([main_mesh_vertices,sub_verts])

        
        #b. Add the vertices_count of main to every number in the faces array
        sub_faces = sub_faces + n_main_vertices
        
        #c. Concatenate the submesh faces onto the larger mesh face
        main_mesh_faces = np.vstack([main_mesh_faces,sub_faces])
        
        #d. Save this new vertices and faces as the main_mesh verts and faces (DONE)
        
        #e. Print out how many new vertices and faces added
        #print(f"Added subsegment {i} with {n_sub_verts} vertices and {n_sub_faces} faces")
        
        n_main_vertices = len(main_mesh_vertices)
        n_main_faces = len(main_mesh_faces)
    
    #3) Print out number of segments added, total faces/vertices for new mesh  
    #print(f"Added {len(sub_mesh_vertices)} subsegements \n  --> final mesh: {len(main_mesh_vertices)} vertices and {len(main_mesh_faces)} faces")
        
    return main_mesh_vertices,main_mesh_faces 