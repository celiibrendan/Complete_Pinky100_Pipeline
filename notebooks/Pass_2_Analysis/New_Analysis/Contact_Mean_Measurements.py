#!/usr/bin/env python
# coding: utf-8

# In[24]:


"""
Purpose: To do another analysis where analyze the mean conversion rate
1) Get all the significantly tuned postsyns
2) For each postsyn pair:
a. Get the axons that are contacting both
b. For each axon get the mean conversion rate for the axon onto the two
c. Take the 90 percentile mean conversion rate
- store other percentiles in table


"""


# In[25]:


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


# In[26]:


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


# In[27]:


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


# In[28]:


ori_confidence=0.5
von_p_value=0.05
segment = (pinky.Segment - pinky.SegmentExclude) & pinky.CurrentSegmentation
tuned = 'confidence > ' + str(ori_confidence)
#get the significantly tuned segments
sig_units_op = pinky_radtune.BestVonFit.Unit & 'von_p_value <= ' + str(von_p_value)  & segment & tuned
# print("Number of significanlty orientationally tuned neurons = " + str(len(sig_units_op)))


# #gets the significantly tuned neurons and their differences in combinational pairs 
# sig_unit_pairs_op = (radtune.BestVonCorr() & sig_units_op.proj(segment_id1="segment_id") 
#                  & sig_units_op.proj(segment_id2="segment_id")).proj("diff_pref_ori")

# sig_unit_pairs_op = sig_unit_pairs_op.proj(segment_a="segment_id1",
#                                            segment_b="segment_id2",
#                                            dori="diff_pref_ori")
# sig_unit_pairs_op
# print("Length of pairwise orientation difference table = " + str(len(sig_unit_pairs_op)))
sig_unit_seg_ids = sig_units_op.fetch("segment_id")
sig_unit_seg_ids


# In[29]:


postsyn1 = 648518346341366346
postsyn2 = 64851834634950930

if (postsyn1 not in sig_unit_seg_ids) or (postsyn2 not in sig_unit_seg_ids):
    print("not in")


# In[30]:


prepost_data = pinky_fc.ContactPrePost.proj("postsyn","total_contact_conversion",
                "total_contact_density","total_synapse_sizes_mean",
                syn_density="if(total_postsyn_length=0,null,(total_n_synapses*total_synapse_sizes_mean)/total_postsyn_length)",
                presyn="segment_id")
prepost_data


# In[31]:


x = np.array([5,4,3])
y = np.array([1,10,2])


combined = np.column_stack([np.transpose(x),np.transpose(y)])
mean_conversion = np.mean(combined,axis=1)
print("mean conversion = " + str(mean_conversion))
print(np.percentile(mean_conversion,90))
print(np.percentile(mean_conversion,80))
print(np.percentile(mean_conversion,70))
print(np.mean(mean_conversion))
print(np.median(mean_conversion))


# In[32]:


@schema_fc
class ContactMeanStatistics(dj.Computed):
    definition="""
    -> pinky.Segment
    segment_b :bigint unsigned #id of the postsynaptic neuron
    ---
    n_seg_a              :bigint unsigned #n_presyns contacting onto segment_id
    n_seg_b              :bigint unsigned #n_presyns contacting onto segment_b
    n_seg_shared           :bigint unsigned #n_presyns contacting onto both segment_id and segment_b
    n_seg_shared_converted :bigint unsigned #n_presyns contacting onto both and converting on at least 1 postsyn
    n_seg_a_converted      :bigint unsigned #n_presyns contacting onto both and converting on postsyna a
    n_seg_b_converted      :bigint unsigned #n_presyns contacting onto both and converting on postsyna b
    perc_90_mean_conversion=null :float   #pearson correlation for binary n_synapse/n_contact rate
    perc_80_mean_conversion=null :float   #pearson correlation for binary n_synapse/n_contact rate
    perc_70_mean_conversion=null :float   #pearson correlation for binary n_synapse/n_contact rate
    mean_mean_conversion=null :float   #pearson correlation for binary n_synapse/n_contact rate
    median_mean_conversion=null :float   #pearson correlation for binary n_synapse/n_contact rate
    perc_90_mean_conversion_converted=null :float   #pearson correlation for binary n_synapse/n_contact rate
    perc_80_mean_conversion_converted=null :float   #pearson correlation for binary n_synapse/n_contact rate
    perc_70_mean_conversion_converted=null :float   #pearson correlation for binary n_synapse/n_contact rate
    mean_mean_conversion_converted=null :float   #pearson correlation for binary n_synapse/n_contact rate
    median_mean_conversion_converted=null :float   #pearson correlation for binary n_synapse/n_contact rate
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
        
        ori_confidence=0.5
        von_p_value=0.05
        segment = (pinky.Segment - pinky.SegmentExclude) & pinky.CurrentSegmentation
        tuned = 'confidence > ' + str(ori_confidence)
        #get the significantly tuned segments
        sig_units_op = pinky_radtune.BestVonFit.Unit & 'von_p_value <= ' + str(von_p_value)  & segment & tuned
        # print("Number of significanlty orientationally tuned neurons = " + str(len(sig_units_op)))


        # #gets the significantly tuned neurons and their differences in combinational pairs 
        # sig_unit_pairs_op = (radtune.BestVonCorr() & sig_units_op.proj(segment_id1="segment_id") 
        #                  & sig_units_op.proj(segment_id2="segment_id")).proj("diff_pref_ori")

        # sig_unit_pairs_op = sig_unit_pairs_op.proj(segment_a="segment_id1",
        #                                            segment_b="segment_id2",
        #                                            dori="diff_pref_ori")
        # sig_unit_pairs_op
        # print("Length of pairwise orientation difference table = " + str(len(sig_unit_pairs_op)))
        sig_unit_seg_ids = sig_units_op.fetch("segment_id")
        
        
        
        for i,posts in tqdm(enumerate(segment_pairs)):
#             index = 19
#             multiple = 24231
#             if i < index*multiple:
#                 continue
#             if i > (index+1)*multiple:
#                 break

            
            postsyn1,postsyn2 = posts
            if (postsyn1 not in sig_unit_seg_ids) or (postsyn2 not in sig_unit_seg_ids):
                    continue
            
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
            
            
            #get the number and proportion on presyns that convert onto each segment inside the converted axon group
            if n_seg_shared_converted > 0:
                test_1_conv[test_1_conv>1] = 1
                n_seg_a_converted = sum(np.ceil(test_1_conv))
                test_2_conv[test_2_conv>1] = 1
                n_seg_b_converted = sum(np.ceil(test_2_conv))
            else:
                n_seg_a_converted = 0
                n_seg_b_converted = 0
                
            dict_segmenation=3
            dict_segment_id=postsyn1
            dict_segment_b=postsyn2
            #initialize the dictionary that will be saved:
            corr_dict = dict(segmentation=3,segment_id=postsyn1,
                                          segment_b=postsyn2,
                                          n_seg_a=n_seg_a,
                                            n_seg_b=n_seg_b,
                                            n_seg_shared=n_seg_shared,
                                            n_seg_shared_converted=n_seg_shared_converted,
                                            n_seg_a_converted=n_seg_a_converted,
                                            n_seg_b_converted=n_seg_b_converted)

            #initialize the variables that need to be set in the dictionary
            
            #ones that are set by 1st group
            
            perc_90_mean_conversion=np.NaN
            perc_80_mean_conversion=np.NaN
            perc_70_mean_conversion=np.NaN
            mean_mean_conversion=np.NaN
            median_mean_conversion=np.NaN
            perc_90_mean_conversion_converted=np.NaN
            perc_80_mean_conversion_converted=np.NaN
            perc_70_mean_conversion_converted=np.NaN
            mean_mean_conversion_converted=np.NaN
            median_mean_conversion_converted=np.NaN
            

            if (not df_1_common.to_numpy().any()) or (not df_2_common.to_numpy().any()):
                #total_correlations.append(corr_dict)
                pass

            else:
                #retrieve the conversion rates
                df_1_common_conversion = df_1_common["total_contact_conversion"].to_numpy()
                df_2_common_conversion = df_2_common["total_contact_conversion"].to_numpy()
                
                #calculate the percentiles and the mean
                combined = np.column_stack([np.transpose(df_1_common_conversion),np.transpose(df_2_common_conversion)])
                mean_combined = np.mean(combined,axis=1)
                mean_mean_conversion = np.mean(mean_combined)
                median_mean_conversion = np.median(mean_combined)
                perc_90_mean_conversion = np.percentile(mean_combined,90)
                perc_80_mean_conversion = np.percentile(mean_combined,80)
                perc_70_mean_conversion = np.percentile(mean_combined,70)
                
                
                ####reset the df_1_common and df_1_common to reuse code
                df_1_common = df_1_common_converted
                df_2_common = df_2_common_converted


                if (not df_1_common.to_numpy().any()) or (not df_2_common.to_numpy().any()):
                    #print("none_in_converted")
                    pass
                else:
                    #where does the calculations on the converted
                    
                    df_1_common_conversion = df_1_common["total_contact_conversion"].to_numpy()
                    df_2_common_conversion = df_2_common["total_contact_conversion"].to_numpy()

                    #calculate the percentiles and the mean
                    combined = np.column_stack([np.transpose(df_1_common_conversion),np.transpose(df_2_common_conversion)])
                
                    mean_combined = np.mean(combined,axis=1)
                    mean_mean_conversion_converted = np.mean(mean_combined)
                    median_mean_conversion_converted = np.median(mean_combined)
                    perc_90_mean_conversion_converted = np.percentile(mean_combined,90)
                    perc_80_mean_conversion_converted = np.percentile(mean_combined,80)
                    perc_70_mean_conversion_converted = np.percentile(mean_combined,70)
                
                
            corr_dict["perc_90_mean_conversion"] = perc_90_mean_conversion
            corr_dict["perc_80_mean_conversion"] = perc_80_mean_conversion
            corr_dict["perc_70_mean_conversion"] = perc_70_mean_conversion
            corr_dict["mean_mean_conversion"] = mean_mean_conversion
            corr_dict["median_mean_conversion"] = median_mean_conversion
            
            
            corr_dict["perc_90_mean_conversion_converted"] = perc_90_mean_conversion_converted
            corr_dict["perc_80_mean_conversion_converted"] = perc_80_mean_conversion_converted
            corr_dict["perc_70_mean_conversion_converted"] = perc_70_mean_conversion_converted
            corr_dict["mean_mean_conversion_converted"] = mean_mean_conversion_converted
            corr_dict["median_mean_conversion_converted"] = median_mean_conversion_converted
            
            #print(corr_dict)
            
            total_correlations.append(corr_dict)

        #write all of the dictionaries to the database
        self.insert(total_correlations,skip_duplicates=True)


# In[33]:


import time
start_time = time.time()
ContactMeanStatistics.populate()
print("Total time = " + str(time.time() - start_time))

