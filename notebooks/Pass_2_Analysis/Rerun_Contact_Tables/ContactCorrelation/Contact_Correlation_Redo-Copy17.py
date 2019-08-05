#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import datajoint as dj
import time
import os
import datetime

#for supressing the output
import os, contextlib
import pathlib
import subprocess

#for error counting
from collections import Counter

#for reading in the new raw_skeleton files
import csv
import pandas as pd
from tqdm import tqdm

#for filtering
import math


# In[2]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

schema = dj.schema('microns_pinky')
schema_fc = dj.schema('microns_pinky_fc')
pinky = dj.create_virtual_module('pinky', 'microns_pinky')
pinky_nda = dj.create_virtual_module('pinky_nda', 'microns_pinky_nda')
pinky_radtune = dj.create_virtual_module('pinky_radtune', 'microns_pinky_radtune')
pinky_spattune = dj.create_virtual_module('pinky_spattune', 'microns_pinky_spattune')
pinky_fc = dj.create_virtual_module('pinky_fc', 'microns_pinky_fc')


# In[3]:


#calculates the pearson correlation and cosine similarity while accounting for the corner cases
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings

def check_nan(v1,v2):
    if np.isscalar(v1):
        if np.isnan(v1) or v1 == None:
            return True
    else:
        if True in np.isnan(v1) or len(v1) <= 0:
            return True
        
    if np.isscalar(v2):
        if np.isnan(v2) or v2 == None:
            return True
    else:
        if True in np.isnan(v2) or len(v2) <= 0:
            return True
    
    return False

def find_pearson_old(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    print("v1 = " + str(v1))
    print("v2 = " + str(v2))
    if check_nan(v1,v2):
        return np.NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.array_equal(v1,v2):
            return 1
        elif abs(sum(v1 - v2)) >= v1.size:
            return -1
        else:
            #perform the pearson correlation
            corr_conversion, p_value_conversion = pearsonr(v1, v2)
            return corr_conversion

def find_cosine_old(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    if check_nan(v1,v2):
        return np.NaN
    if np.array_equal(v1,v2):
        return 1
    elif abs(sum(v1 - v2)) == v1.size:
        return 0
    else:
        v1 = v1.reshape(1,len(v1))
        v2 = v2.reshape(1,len(v2))
        return cosine_similarity(v1, v2)[0][0]



def find_pearson(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
#     print("v1 = " + str(v1))
#     print("v2 = " + str(v2))
    if check_nan(v1,v2):
        return np.NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #perform the pearson correlation
        if v1.size <= 1 or v2.size <= 1:
            return np.NaN
        if np.array_equal(v1,v2):
            return 1
        elif abs(sum(v1 - v2)) >= v1.size:
            return -1
        else:
            corr_conversion, p_value_conversion = pearsonr(v1, v2)
            return corr_conversion

def find_cosine(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    if check_nan(v1,v2):
        return np.NaN
    if v1.size <= 1 or v2.size <= 1:
            return np.NaN
    if np.array_equal(v1,v2):
        return 1
    elif abs(sum(v1 - v2)) == v1.size:
        return 0
    else:
        v1 = v1.reshape(1,len(v1))
        v2 = v2.reshape(1,len(v2))
        return cosine_similarity(v1, v2)[0][0]

def find_binary_sim(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    if check_nan(v1,v2):
            return np.NaN
    a = np.dot(v1,v2)
    b = np.dot(1-v1,v2)
    c = np.dot(v1,1-v2)
    d = np.dot(1-v1,1-v2)
    
    return (a)/(a + b + c + d),(a + d)/(a + b + c + d)


# In[4]:


@schema_fc
class ContactCorrelation(dj.Computed):
    definition="""
    -> pinky.Segment
    segment_b :bigint unsigned #id of the postsynaptic neuron
    ---
    n_seg_a              :bigint unsigned #n_presyns contacting onto segment_id
    n_seg_b              :bigint unsigned #n_presyns contacting onto segment_b
    n_seg_shared           :bigint unsigned #n_presyns contacting onto both segment_id and segment_b
    n_seg_union            :bigint unsigned #n_presyns contacting either segment_id or segment_b
    n_seg_shared_converted :bigint unsigned #n_presyns contacting onto both and converting on at least 1 postsyn
    n_seg_a_converted      :bigint unsigned #n_presyns contacting onto both and converting on postsyna a
    n_seg_a_converted_prop=null :float           #proportion of n_presyns contacting onto both which convert at least onto postsyna a
    n_seg_b_converted      :bigint unsigned #n_presyns contacting onto both and converting on postsyna b
    n_seg_b_converted_prop=null :float           #proportion of n_presyns contacting onto both which convert at least onto postsyna b
    binary_conversion_pearson=null :float   #pearson correlation for binary n_synapse/n_contact rate
    binary_conversion_cosine=null :float    #cosine similarity correlation for binary n_synapse/n_contact rate
    binary_conv_jaccard_ones_ratio=null :float   #a / (a + b + c  + d) for jaccard similarity of binary conversion rate
    binary_conv_jaccard_matching_ratio=null :float  # ( a + d )/ (a + b + c  + d) for jaccard similarity of binary conversion rate
    conversion_pearson=null :float          #Pearson correlation for n_synapse/n_contact rate
    conversion_cosine=null :float           #cosine similarity for n_synapse/n_contact rate
    density_pearson=null :float             #Pearson correlation for n_synapse/postsyn_length rate
    density_cosine=null :float              #cosine similarity for n_synapse/postsyn_length rate
    synapse_volume_mean_pearson=null :float     #Pearson correlation for mean of synaptic volume
    synapse_volume_mean_cosine=null :float      #cosine similarity for mean of synaptic volume
    synapse_vol_density_pearson=null :float         #Pearson correlation for n_synapses*synapse_sizes_mean/postsyn_length rate
    synapse_vol_density_cosine=null :float          #cosine similarity for n_synapses*synapse_sizes_mean/postsyn_length rate
    binary_conversion_pearson_converted=null :float   #pearson correlation for binary n_synapse/n_contact rate for axon group with at least 1 conversion
    binary_conversion_cosine_converted=null :float    #cosine similarity correlation for binary n_synapse/n_contact rate for axon group with at least 1 conversion
    binary_conv_jaccard_ones_ratio_converted=null :float   #a / (a + b + c  + d) for jaccard similarity of binary conversion rate with at least 1 conversion
    binary_conv_jaccard_matching_ratio_converted=null :float  # ( a + d )/ (a + b + c  + d) for jaccard similarity of binary conversion rate with at least 1 conversion
    conversion_pearson_converted=null :float          #Pearson correlation for n_synapse/n_contact rate for axon group with at least 1 conversion
    conversion_cosine_converted=null :float           #cosine similarity for n_synapse/n_contact rate for axon group with at least 1 conversion
    density_pearson_converted=null :float             #Pearson correlation for n_synapse/postsyn_length rate for axon group with at least 1 conversion
    density_cosine_converted=null :float              #cosine similarity for n_synapse/postsyn_length rate for axon group with at least 1 conversion
    synapse_volume_mean_pearson_converted=null :float     #Pearson correlation for mean of synaptic volume for axon group with at least 1 conversion
    synapse_volume_mean_cosine_converted=null :float      #cosine similarity for mean of synaptic volume for axon group with at least 1 conversion
    synapse_vol_density_pearson_converted=null :float         #Pearson correlation for n_synapses*synapse_sizes_mean/postsyn_length rate for axon group with at least 1 conversion
    synapse_vol_density_cosine_converted=null :float          #cosine similarity for n_synapses*synapse_sizes_mean/postsyn_length rate for axon group with at least 1 conversion
    """

    key_source = pinky.Segmentation & pinky.CurrentSegmentation
    
    def make(self,key):
        #Retrieves the PrePost table that will be using in an all in one insertion (MAY HAVE TO ADJUST FOR BIGGER DATA SETS IN FUTURE)
        prepost_data = pinky_fc.ContactPrePost.proj("postsyn","total_contact_conversion",
                "total_contact_density","total_synapse_sizes_mean",
                syn_density="if(total_postsyn_length=0,null,(total_n_synapses*total_synapse_sizes_mean)/total_postsyn_length)",
                presyn="segment_id").fetch()
        df = pd.DataFrame(prepost_data)

        #gets all the combinations of postsyn-postsyn without any repeats
        targets = (dj.U("postsyn") & pinky.SkeletonContact).proj(segment_id="postsyn") - pinky.SegmentExclude
        info = targets * targets.proj(segment_b='segment_id') & 'segment_id < segment_b'
        segment_pairs = info.fetch()
        
        total_correlations = []
        print("About to start postsyn1,postsyn2")
        
        
        for i,posts in tqdm(enumerate(segment_pairs)):
            index = 17
            multiple = 24231
            if i < index*multiple:
                continue
            if i > (index+1)*multiple:
                break
            
            postsyn1,postsyn2 = posts
            
            start_time = time.time()

            #print("postsyn1 = " + str(postsyn1))
            #print("postsyn2 = " + str(postsyn2))

            #get all of the rows with postsyn 1 and 2 AKA find the number of presyns for each
            df_1 = df[df["postsyn"].to_numpy()==postsyn1]
            df_2 = df[df["postsyn"].to_numpy()==postsyn2]

            #reduce both tables down to common presyns
            df_1_common = df_1[df_1["presyn"].isin(df_2["presyn"].to_numpy())].sort_values(by=['presyn'])
            df_2_common = df_2[df_2["presyn"].isin(df_1["presyn"].to_numpy())].sort_values(by=['presyn'])


            ###########------------------------------------------------------###########
            #need to get the common axons that have at least one converted contact on one of the postsyns
            """pseudocode
            #get the conversion rates for both tables
            #add them up
            #get the indices that are greater than 0
            #get the presyn ids that match those rows
            #further restrict both groups by those ids
            """

            #get both of their conversion rates
            test_1_conv = df_1_common["total_contact_conversion"].to_numpy()
            test_2_conv = df_2_common["total_contact_conversion"].to_numpy()

            test_1_presyn = df_1_common["presyn"].to_numpy()
            new_presyns = test_1_presyn[(test_1_conv + test_2_conv) > 0]

            df_1_common_converted = df_1_common[df_1_common["presyn"].isin(new_presyns)]
            df_2_common_converted = df_2_common[df_2_common["presyn"].isin(new_presyns)]
            ###########------------------------------------------------------###########


            #finds the number of segments, shared_segments and union segments
            n_seg_a = df_1.shape[0]
            n_seg_b = df_2.shape[0]
            n_seg_shared = df_1_common.shape[0]
            n_seg_shared_converted = df_1_common_converted.shape[0]
            n_seg_union = n_seg_a + n_seg_b - n_seg_shared
            
            
            #get the number and proportion on presyns that convert onto each segment inside the converted axon group
            if n_seg_shared_converted > 0:
                test_1_conv[test_1_conv>1] = 1
                n_seg_a_converted = sum(np.ceil(test_1_conv))
                test_2_conv[test_2_conv>1] = 1
                n_seg_b_converted = sum(np.ceil(test_2_conv))

                n_seg_a_converted_prop = n_seg_a_converted/n_seg_shared_converted
                n_seg_b_converted_prop = n_seg_b_converted/n_seg_shared_converted
            else:
                n_seg_a_converted = 0
                n_seg_b_converted = 0
                n_seg_a_converted_prop = np.NaN
                n_seg_b_converted_prop = np.NaN
     
            dict_segmenation=2
            dict_segment_id=postsyn1
            dict_segment_b=postsyn2
            #initialize the dictionary that will be saved:
            corr_dict = dict(segmentation=3,segment_id=postsyn1,
                                          segment_b=postsyn2,
                                          n_seg_a=n_seg_a,
                                            n_seg_b=n_seg_b,
                                            n_seg_shared=n_seg_shared,
                                            n_seg_shared_converted=n_seg_shared_converted,
                                            n_seg_union=n_seg_union,
                                            n_seg_a_converted=n_seg_a_converted,
                                            n_seg_b_converted=n_seg_b_converted,
                                            n_seg_a_converted_prop=n_seg_a_converted_prop,
                                            n_seg_b_converted_prop=n_seg_b_converted_prop)

            #initialize the variables that need to be set in the dictionary

            #ones that are set by 1st group
            binary_conversion_pearson = np.NaN
            binary_conversion_cosine = np.NaN
            binary_conv_jaccard_ones_ratio = np.NaN
            binary_conv_jaccard_matching_ratio = np.NaN
            conversion_pearson = np.NaN
            conversion_cosine = np.NaN
            density_pearson = np.NaN
            density_cosine = np.NaN
            synapse_volume_mean_pearson = np.NaN
            synapse_volume_mean_cosine = np.NaN
            synapse_vol_density_pearson = np.NaN
            synapse_vol_density_cosine = np.NaN

            #ones that are set by 2nd group
            binary_conversion_pearson_converted = np.NaN
            binary_conversion_cosine_converted = np.NaN
            binary_conv_jaccard_ones_ratio_converted = np.NaN
            binary_conv_jaccard_matching_ratio_converted = np.NaN
            conversion_pearson_converted = np.NaN
            conversion_cosine_converted = np.NaN
            density_pearson_converted = np.NaN
            density_cosine_converted = np.NaN
            synapse_volume_mean_pearson_converted = np.NaN
            synapse_volume_mean_cosine_converted = np.NaN
            synapse_vol_density_pearson_converted = np.NaN
            synapse_vol_density_cosine_converted = np.NaN


            if (not df_1_common.to_numpy().any()) or (not df_2_common.to_numpy().any()):
                #total_correlations.append(corr_dict)
                pass

            else:
                #retrieve the conversion rates
                df_1_common_conversion = df_1_common["total_contact_conversion"].to_numpy()
                df_2_common_conversion = df_2_common["total_contact_conversion"].to_numpy()
                
                #calculate the binary conversion rates
                df_1_common_binary_conversion = np.copy(df_1_common_conversion)
                df_2_common_binary_conversion = np.copy(df_2_common_conversion)

                df_1_common_binary_conversion[df_1_common_binary_conversion>0] = 1.0
                df_2_common_binary_conversion[df_2_common_binary_conversion>0] = 1.0
                
                #retrieve the synapse/postsyn_len
                df_1_common_density = df_1_common["total_contact_density"].to_numpy()
                df_2_common_density = df_2_common["total_contact_density"].to_numpy()

                #retrieve mean of synapse_size
                df_1_common_synaptic_size = df_1_common["total_synapse_sizes_mean"].to_numpy()
                df_2_common_synaptic_size = df_2_common["total_synapse_sizes_mean"].to_numpy()

                #retrieve (total_n_synapses*total_synapse_sizes_mean)/total_postsyn_length
                df_1_common_syn_density = df_1_common["syn_density"].to_numpy()
                df_2_common_syn_density = df_2_common["syn_density"].to_numpy()

                binary_conversion_pearson = find_pearson(df_1_common_binary_conversion, df_2_common_binary_conversion)
                binary_conversion_cosine = find_cosine(df_1_common_binary_conversion, df_2_common_binary_conversion)
                
                #new added metric for the binary calculations based on jacard_similarity
                binary_conv_jaccard_ones_ratio,binary_conv_jaccard_matching_ratio = find_binary_sim(df_1_common_binary_conversion,df_2_common_binary_conversion)
                
                conversion_pearson = find_pearson(df_1_common_conversion, df_2_common_conversion)
                conversion_cosine = find_cosine(df_1_common_conversion, df_2_common_conversion)
                density_pearson = find_pearson(df_1_common_density, df_2_common_density)
                density_cosine = find_cosine(df_1_common_density, df_2_common_density)
                synapse_volume_mean_pearson = find_pearson(df_1_common_synaptic_size, df_2_common_synaptic_size)
                synapse_volume_mean_cosine = find_cosine(df_1_common_synaptic_size, df_2_common_synaptic_size)
                synapse_vol_density_pearson = find_pearson(df_1_common_syn_density, df_2_common_syn_density)
                synapse_vol_density_cosine = find_cosine(df_1_common_syn_density, df_2_common_syn_density)

                ####reset the df_1_common and df_1_common to reuse code
                df_1_common = df_1_common_converted
                df_2_common = df_2_common_converted


                if (not df_1_common.to_numpy().any()) or (not df_2_common.to_numpy().any()):
                    #print("none_in_converted")
                    pass
                else:
                    df_1_common_conversion = df_1_common["total_contact_conversion"].to_numpy()
                    df_2_common_conversion = df_2_common["total_contact_conversion"].to_numpy()

                    df_1_common_binary_conversion = np.copy(df_1_common_conversion)
                    df_2_common_binary_conversion = np.copy(df_2_common_conversion)


                    df_1_common_binary_conversion[df_1_common_binary_conversion>0] = 1.0
                    df_2_common_binary_conversion[df_2_common_binary_conversion>0] = 1.0

                    df_1_common_density = df_1_common["total_contact_density"].to_numpy()
                    df_2_common_density = df_2_common["total_contact_density"].to_numpy()


                    df_1_common_synaptic_size = df_1_common["total_synapse_sizes_mean"].to_numpy()
                    df_2_common_synaptic_size = df_2_common["total_synapse_sizes_mean"].to_numpy()

                    df_1_common_syn_density = df_1_common["syn_density"].to_numpy()
                    df_2_common_syn_density = df_2_common["syn_density"].to_numpy()

                    binary_conversion_pearson_converted = find_pearson(df_1_common_binary_conversion, df_2_common_binary_conversion)
                    binary_conversion_cosine_converted = find_cosine(df_1_common_binary_conversion, df_2_common_binary_conversion)
                    
                    #new added metric for the binary calculations based on jacard_similarity
                    binary_conv_jaccard_ones_ratio_converted,binary_conv_jaccard_matching_ratio_converted = find_binary_sim(df_1_common_binary_conversion,df_2_common_binary_conversion)
                
                    conversion_pearson_converted = find_pearson(df_1_common_conversion, df_2_common_conversion)
                    conversion_cosine_converted = find_cosine(df_1_common_conversion, df_2_common_conversion)
                    density_pearson_converted = find_pearson(df_1_common_density, df_2_common_density)
                    density_cosine_converted = find_cosine(df_1_common_density, df_2_common_density)
                    synapse_volume_mean_pearson_converted = find_pearson(df_1_common_synaptic_size, df_2_common_synaptic_size)
                    synapse_volume_mean_cosine_converted = find_cosine(df_1_common_synaptic_size, df_2_common_synaptic_size)
                    synapse_vol_density_pearson_converted = find_pearson(df_1_common_syn_density, df_2_common_syn_density)
                    synapse_vol_density_cosine_converted = find_cosine(df_1_common_syn_density, df_2_common_syn_density)


            corr_dict["binary_conversion_pearson"] = binary_conversion_pearson
            corr_dict["binary_conversion_cosine"] = binary_conversion_cosine
            corr_dict["binary_conv_jaccard_ones_ratio"] = binary_conv_jaccard_ones_ratio
            corr_dict["binary_conv_jaccard_matching_ratio"] = binary_conv_jaccard_matching_ratio
            corr_dict["conversion_pearson"] = conversion_pearson
            corr_dict["conversion_cosine"] = conversion_cosine
            corr_dict["density_pearson"] = density_pearson
            corr_dict["density_cosine"] = density_cosine
            corr_dict["synapse_volume_mean_pearson"] = synapse_volume_mean_pearson
            corr_dict["synapse_volume_mean_cosine"] = synapse_volume_mean_cosine
            corr_dict["synapse_vol_density_pearson"] = synapse_vol_density_pearson
            corr_dict["synapse_vol_density_cosine"] = synapse_vol_density_cosine

            corr_dict["binary_conversion_pearson_converted"] = binary_conversion_pearson_converted
            corr_dict["binary_conversion_cosine_converted"] = binary_conversion_cosine_converted
            corr_dict["binary_conv_jaccard_ones_ratio_converted"] = binary_conv_jaccard_ones_ratio_converted
            corr_dict["binary_conv_jaccard_matching_ratio_converted"] = binary_conv_jaccard_matching_ratio_converted
            corr_dict["conversion_pearson_converted"] = conversion_pearson_converted
            corr_dict["conversion_cosine_converted"] = conversion_cosine_converted
            corr_dict["density_pearson_converted"] = density_pearson_converted
            corr_dict["density_cosine_converted"] = density_cosine_converted
            corr_dict["synapse_volume_mean_pearson_converted"] = synapse_volume_mean_pearson_converted
            corr_dict["synapse_volume_mean_cosine_converted"] = synapse_volume_mean_cosine_converted
            corr_dict["synapse_vol_density_pearson_converted"] = synapse_vol_density_pearson_converted
            corr_dict["synapse_vol_density_cosine_converted"] = synapse_vol_density_cosine_converted

            total_correlations.append(corr_dict)

        #write all of the dictionaries to the database
        self.insert(total_correlations,skip_duplicates=True)


# In[5]:


import time
start_time = time.time()
ContactCorrelation.populate()
print("Total time = " + str(time.time() - start_time))


# In[6]:


484620/20


# In[7]:


"""
What this table measures: 
For each axon and postsyn pairing:
1) Number of contacts
2) postsyn_length
3) contact conversion rate
4) contact densirty
5) total number of synapses
6) synapse sizes mean
-- repeats them for all the categories
"""

pinky_fc.ContactPrePost() 

