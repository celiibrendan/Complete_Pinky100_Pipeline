def MoveToExclude(segmentation, segment_list, criteria, ignore_missing_ids=False, skip_duplicates=False):
    segmentation_key = dict(segmentation=segmentation)
    
    # Check if the arguments include any invalid segments (didn't exist or might have already been moved)    
    if not ignore_missing_ids:
        segments_not_in_table = list()
        for segment_id in segment_list:
            if not len(ta3p100.Segment & segmentation_key & dict(segment_id=segment_id)):
                segments_not_in_table.append(segment_id)
        
        if len(segments_not_in_table) > 0:
            raise ValueError('These segments were not in ta3p100.Segment', segments_not_in_table)
    
    # Insert Criteria, Segment, and Synapses into Exclude tables
    all_criteria_desc = ta3p100.ExclusionCriteria.fetch('criteria_desc')
    if any([criteria == fetched for fetched in all_criteria_desc]):
        criteria_id = np.where(all_criteria_desc == criteria)[0][0]
    else:
        criteria_id = len(all_criteria_desc)
        ta3p100.ExclusionCriteria.insert1((criteria_id, criteria))
    
    segment_id_keys = [dict(segment_id=seg) for seg in segment_list]
    
    segment_exclude_rel = ta3p100.Segment & segmentation_key & segment_id_keys
    print(f'Number of segments to be inserted into ta3p100.SegmentExclude: {len(segment_exclude_rel)}')
    segment_exclude_df = pd.DataFrame(segment_exclude_rel.fetch())
    segment_exclude_df['criteria_id'] = criteria_id
    ta3p100.SegmentExclude.insert(segment_exclude_df.to_dict('records'), skip_duplicates=skip_duplicates)
    
    synapse_exclude_rel_presyn = ta3p100.Synapse & segment_exclude_rel.proj(presyn='segment_id')
    synapse_exclude_rel_postsyn = ta3p100.Synapse & segment_exclude_rel.proj(postsyn='segment_id')
    print(f'(Approximate) Number of synapses to be inserted into ta3p100.SynapseExclude: {len(synapse_exclude_rel_presyn)}')
    print(f'(Approximate) Number of synapses to be inserted into ta3p100.SynapseExclude: {len(synapse_exclude_rel_postsyn)}')
    ta3p100.SynapseExclude.insert(synapse_exclude_rel_presyn, skip_duplicates=True)
    ta3p100.SynapseExclude.insert(synapse_exclude_rel_postsyn, skip_duplicates=True)