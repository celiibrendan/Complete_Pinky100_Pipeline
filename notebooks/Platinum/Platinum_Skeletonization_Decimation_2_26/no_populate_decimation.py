from populate_decimation import *

def concatenated_rel(cls, core_segment=None, version=-1, return_with_meshes=False):
    """
    Returns all. You can restrict by a core_segment first though.
    
    :core_segment: The core segment to restrict by. If left empty will fetch all.
    :version: The default will fetch the highest version for each core segment
        and its subsegments.
    :param return_with_meshes: When set to true or 'Decimation' will default
        to using the Decimation table for the meshes, otherwise 'Mesh' will
        choose the Mesh table with the original meshes.
    """
    
    subsegment_rel = cls.Subsegment.proj()

    if core_segment is not None:
        subsegment_rel &= dict(segment_id=core_segment)

    if version == -1:
        version_rel = dj.U('segment_id').aggr(subsegment_rel, version='max(version)')
    else:
        version_rel = subsegment_rel & dict(version=version)

    a_rel = dj.U('segment_id') & version_rel
    b_rel = dj.U('segment_id') & subsegment_rel.proj(_='segment_id', segment_id='subsegment_id')
    c_rel = a_rel + b_rel

    if return_with_meshes:
        if isinstance(return_with_meshes, str) and return_with_meshes.lower() == 'mesh':
            c_rel = m65.Mesh & c_rel
        else:
            c_rel = m65.Decimation & c_rel
    
    return c_rel


def get_neuromancer_seg_ids(segment_id):
    return concatenated_rel(minnie.FromNeuromancer, segment_id).fetch('segment_id')

core_segments = (dj.U('segment_id') & minnie.FromNeuromancer).fetch('segment_id')

if __name__ == '__main__':
    print('No populate populate start.')
    
    version = 0
    decimation_ratio = 0.25 # Only the value in the .mls can change the ratio though.

    for core_segment in core_segments:
        seg_ids = get_neuromancer_seg_ids(core_segment)
        for segment_id in seg_ids:
            key = {'segment_id': segment_id}
            if len(minnie.Decimation & dict(key, version=version)) < 1:
                mesh = (minnie.Mesh & key).fetch1('mesh')
                new_vertices, new_faces = decimate_mesh(mesh.vertices, mesh.faces, key['segment_id'], folder_name)
                Decimation.make_entry(
                    segment_id=key['segment_id'],
                    version=version,
                    decimation_ratio=decimation_ratio,
                    vertices=new_vertices,
                    faces=new_faces
                )