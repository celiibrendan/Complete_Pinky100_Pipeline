import datajoint as dj
import numpy as np
import time
import bpy


dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
    
schema = dj.schema('microns_pinky')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')
pinky_nda = dj.create_virtual_module('pinky_nda','microns_pinky_nda')

@schema
class Decimation35Orphan(dj.Computed):
    definition="""
    -> pinky.Mesh
    decimation_ratio     : decimal(3,2)                 
    ---
    n_vertices           : bigint                       
    n_triangles          : bigint                       
    vertices             : longblob                     
    triangles            : longblob 
    """
    
    #getting the keys from the following three tables
    ### Old: key_source = (ta3p100.Mesh() & "segmentation=2") - ta3p100.AllenSoma & ( ndap100.Spike() & "segmentation=2")
    
    #getting the keys from the following three tables
    key_source = (pinky.Mesh() & "segmentation=3") - pinky.AllenSoma & ( pinky_nda.Spike() & "segmentation=3")
    
    def make(self, key):
        start = time.time()
        
        # Configurations:
        decimation_ratio = 0.35
        
        # Load mesh
        mesh = (pinky.Mesh & key).fetch1()
        vertices, triangles = mesh['vertices'], mesh['triangles']
        the_vertices = [(x.item(), y.item(), z.item()) for x, y, z in vertices]
        the_triangles = [(node1.item(), node2.item(), node3.item()) for node1, node2, node3 in triangles]
        #this string method below was only added in Python 3.6 and blender uses 3.5
        #print(f'Num before decimation -> vertices: {len(vertices)}, triangles: {len(triangles)}')
        print('Num before decimation -> vertices: ' + str(len(vertices))+ ", triangles: " + str(len(triangles)))

        # Create blender mesh
        name = '{}_{}'.format(key['segment_id'], decimation_ratio)
        blender_mesh = bpy.data.meshes.new(name)
        blender_mesh.from_pydata(the_vertices, [], the_triangles)
        blender_mesh.update(calc_edges=True, calc_tessface=True)
        blender_mesh.calc_normals()
        blender_mesh.validate()

        # Create blender object and link it to the scene
        blender_object = bpy.data.objects.new(name, blender_mesh)
        bpy.context.scene.objects.link(blender_object)

        # Create decimation modifier
        ratio = 0.35 # 0.01
        decimate_modifier = blender_object.modifiers.new('decimate', 'DECIMATE')
        decimate_modifier.ratio = ratio
        decimate_modifier.use_collapse_triangulate = True

        # Apply decimation modifier
        bpy.context.scene.objects.active = blender_object
        bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)
        
        #print(f'Num After decimation -> vertices: {len(blender_object.data.vertices)}, triangles: {len(blender_object.data.polygons)}')
        
        # Save decimated mesh
        new_vertices = np.array([(v.co[0], v.co[1], v.co[2]) for v in blender_object.data.vertices])
        new_triangles = np.array([[v for v in d.vertices] for d in blender_object.data.polygons], dtype=np.uint32)
        
        print('Num before decimation -> vertices: ' + str(len(new_vertices))+ ", triangles: " + str(len(new_triangles)))

        self.insert1(dict(key,
                          decimation_ratio=decimation_ratio,
                          n_vertices=len(new_vertices),
                          n_triangles=len(new_triangles),
                          vertices=new_vertices,
                          triangles=new_triangles))
        
        print(time.time() - start)

start = time.time()
Decimation35Orphan.populate(reserve_jobs=True)
print(time.time() - start)
