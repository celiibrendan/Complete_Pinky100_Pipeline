@schema
class ContactPrePost(dj.Computed):
    definition="""
    -> pinky.Segment
    postsyn :bigint unsigned #id of the postsynaptic neuron
    ---
    total_n_contacts   :bigint unsigned #total number of contacts
    total_postsyn_length   :bigint unsigned #total total postsynaptic contact length
    total_contact_conversion=null   :float #total synapse to contact ratio
    total_contact_density=null   :float #total synapse to contact length ratio
    total_n_synapses   :bigint unsigned #total total number of synapses for neurite
    total_synapse_sizes_mean=null   :float #total average synaptic size
    apical_n_contacts   :bigint unsigned #apical number of contacts
    apical_postsyn_length   :bigint unsigned #apical total postsynaptic contact length
    apical_contact_conversion=null   :float #apical synapse to contact ratio
    apical_contact_density=null   :float #apical synapse to contact length ratio
    apical_n_synapses   :bigint unsigned #apical total number of synapses for neurite
    apical_synapse_sizes_mean=null   :float #apical average synaptic size
    basal_n_contacts   :bigint unsigned #basal number of contacts
    basal_postsyn_length   :bigint unsigned #basal total postsynaptic contact length
    basal_contact_conversion=null   :float #basal synapse to contact ratio
    basal_contact_density=null   :float #basal synapse to contact length ratio
    basal_n_synapses   :bigint unsigned #basal total number of synapses for neurite
    basal_synapse_sizes_mean=null   :float #basal average synaptic size
    oblique_n_contacts   :bigint unsigned #oblique number of contacts
    oblique_postsyn_length   :bigint unsigned #oblique total postsynaptic contact length
    oblique_contact_conversion=null   :float #oblique synapse to contact ratio
    oblique_contact_density=null   :float #oblique synapse to contact length ratio
    oblique_n_synapses   :bigint unsigned #oblique total number of synapses for neurite
    oblique_synapse_sizes_mean=null   :float #oblique average synaptic size
    soma_n_contacts   :bigint unsigned #soma number of contacts
    soma_postsyn_length   :bigint unsigned #soma total postsynaptic contact length
    soma_contact_conversion=null   :float #soma synapse to contact ratio
    soma_contact_density=null   :float #soma synapse to contact length ratio
    soma_n_synapses   :bigint unsigned #soma total number of synapses for neurite
    soma_synapse_sizes_mean=null   :float #soma average synaptic size
    axon_n_contacts   :bigint unsigned #axon number of contacts
    axon_postsyn_length   :bigint unsigned #axon total postsynaptic contact length
    axon_contact_conversion=null   :float #axon synapse to contact ratio
    axon_contact_density=null   :float #axon synapse to contact length ratio
    axon_n_synapses   :bigint unsigned #axon total number of synapses for neurite
    axon_synapse_sizes_mean=null   :float #axon average synaptic size
    dendrites_n_contacts   :bigint unsigned #dendrites number of contacts
    dendrites_postsyn_length   :bigint unsigned #dendrites total postsynaptic contact length
    dendrites_contact_conversion=null   :float #dendrites synapse to contact ratio
    dendrites_contact_density=null   :float #dendrites synapse to contact length ratio
    dendrites_n_synapses   :bigint unsigned #dendrites total number of synapses for neurite
    dendrites_synapse_sizes_mean=null   :float #dendrites average synaptic size
    """
    
    labels_to_keep = [2,3,4,5,6,7,8]

    key_source = pinky.Segmentation & pinky.CurrentSegmentation
    
    def make(self,key):

        skeleton_contact = dj.U("segmentation","segment_id","postsyn").aggr(
            pinky.SkeletonContact.proj("postsyn","postsyn_length","n_synapses","majority_label","synapse_sizes_mean",segment_id="presyn")
            & [dict(majority_label=x) for x in self.labels_to_keep],

            
            total_n_contacts= "count(*)",
            total_postsyn_length= "sum(postsyn_length)",
            total_contact_conversion= "sum( n_synapses )/count(*)",
            total_contact_density= "if(sum(postsyn_length)=0,null,sum( n_synapses )/sum(postsyn_length))",
            total_n_synapses= "sum(n_synapses)",
            total_synapse_sizes_mean= "if(sum(n_synapses)=0,0,sum(n_synapses*synapse_sizes_mean)/sum(n_synapses))",


            apical_n_contacts= "sum(majority_label=2)",
            apical_postsyn_length= "sum( postsyn_length * (majority_label=2))",
            apical_contact_conversion= "if(sum(majority_label=2)=0,null,sum( n_synapses * (majority_label=2))/sum(majority_label=2))",
            apical_contact_density= "if(sum( postsyn_length * (majority_label=2))=0,null,sum( n_synapses * (majority_label=2))/sum( postsyn_length * (majority_label=2)))",
            apical_n_synapses= "sum(n_synapses * (majority_label=2))",
            apical_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=2))=0,0,sum((n_synapses * (majority_label=2))*synapse_sizes_mean*(majority_label=2))/sum(n_synapses * (majority_label=2)))",

            basal_n_contacts= "sum(majority_label=3)",
            basal_postsyn_length= "sum( postsyn_length * (majority_label=3))",
            basal_contact_conversion= "if(sum(majority_label=3)=0,null,sum( n_synapses * (majority_label=3))/sum(majority_label=3))",
            basal_contact_density= "if(sum( postsyn_length * (majority_label=3))=0,null,sum( n_synapses * (majority_label=3))/sum( postsyn_length * (majority_label=3)))",
            basal_n_synapses= "sum(n_synapses * (majority_label=3))",
            basal_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=3))=0,0,sum((n_synapses * (majority_label=3))*synapse_sizes_mean*(majority_label=3))/sum(n_synapses * (majority_label=3)))",

            oblique_n_contacts= "sum(majority_label=4)",
            oblique_postsyn_length= "sum( postsyn_length * (majority_label=4))",
            oblique_contact_conversion= "if(sum(majority_label=4)=0,null,sum( n_synapses * (majority_label=4))/sum(majority_label=4))",
            oblique_contact_density= "if(sum( postsyn_length * (majority_label=4))=0,null,sum( n_synapses * (majority_label=4))/sum( postsyn_length * (majority_label=4)))",
            oblique_n_synapses= "sum(n_synapses * (majority_label=4))",
            oblique_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=4))=0,0,sum((n_synapses * (majority_label=4))*synapse_sizes_mean*(majority_label=4))/sum(n_synapses * (majority_label=4)))",

            soma_n_contacts= "sum(majority_label=5)",
            soma_postsyn_length= "sum( postsyn_length * (majority_label=5))",
            soma_contact_conversion= "if(sum(majority_label=5)=0,null,sum( n_synapses * (majority_label=5))/sum(majority_label=5))",
            soma_contact_density= "if(sum( postsyn_length * (majority_label=5))=0,null,sum( n_synapses * (majority_label=5))/sum( postsyn_length * (majority_label=5)))",
            soma_n_synapses= "sum(n_synapses * (majority_label=5))",
            soma_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=5))=0,0,sum((n_synapses * (majority_label=5))*synapse_sizes_mean*(majority_label=5))/sum(n_synapses * (majority_label=5)))",

            axon_n_contacts= "sum(majority_label=6 OR majority_label=7)",
            axon_postsyn_length= "sum( postsyn_length * (majority_label=6 OR majority_label=7))",
            axon_contact_conversion= "if(sum(majority_label=6 OR majority_label=7)=0,null,sum( n_synapses * (majority_label=6 OR majority_label=7))/sum(majority_label=6 OR majority_label=7))",
            axon_contact_density= "if(sum( postsyn_length * (majority_label=6 OR majority_label=7))=0,null,sum( n_synapses * (majority_label=6 OR majority_label=7))/sum( postsyn_length * (majority_label=6 OR majority_label=7)))",
            axon_n_synapses= "sum(n_synapses * (majority_label=6 OR majority_label=7))",
            axon_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=6 OR majority_label=7))=0,0,sum((n_synapses * (majority_label=6 OR majority_label=7))*synapse_sizes_mean*(majority_label=6 OR majority_label=7))/sum(n_synapses * (majority_label=6 OR majority_label=7)))",

            dendrites_n_contacts= "sum(majority_label=8)",
            dendrites_postsyn_length= "sum( postsyn_length * (majority_label=8))",
            dendrites_contact_conversion= "if(sum(majority_label=8)=0,null,sum( n_synapses * (majority_label=8))/sum(majority_label=8))",
            dendrites_contact_density= "if(sum( postsyn_length * (majority_label=8))=0,null,sum( n_synapses * (majority_label=8))/sum( postsyn_length * (majority_label=8)))",
            dendrites_n_synapses= "sum(n_synapses * (majority_label=8))",
            dendrites_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=8))=0,0,sum((n_synapses * (majority_label=8))*synapse_sizes_mean*(majority_label=8))/sum(n_synapses * (majority_label=8)))",
        )

        self.insert(skeleton_contact,skip_duplicates=True)

@schema
class NeuriteContact(dj.Computed):
    definition="""
    -> pinky.Segment
    ---
    total_n_contacts   :bigint unsigned #total number of contacts
    total_postsyn_length   :bigint unsigned #total total postsynaptic contact length
    total_contact_conversion=null   :float #total synapse to contact ratio
    total_contact_density=null   :float #total synapse to contact length ratio
    total_n_synapses   :bigint unsigned #total total number of synapses for neurite
    total_synapse_sizes_mean=null   :float #total average synaptic size
    apical_n_contacts   :bigint unsigned #apical number of contacts
    apical_n_contacts_prop=null   :float #apical proportion of number of contacts
    apical_postsyn_length   :bigint unsigned #apical total postsynaptic contact length
    apical_postsyn_length_prop=null   :float #apical total postsynaptic contact length
    apical_contact_conversion=null   :float #apical synapse to contact ratio
    apical_contact_density=null   :float #apical synapse to contact length ratio
    apical_n_synapses   :bigint unsigned #apical total number of synapses for neurite
    apical_synapse_sizes_mean=null   :float #apical average synaptic size
    basal_n_contacts   :bigint unsigned #basal number of contacts
    basal_n_contacts_prop=null   :float #basal proportion of number of contacts
    basal_postsyn_length   :bigint unsigned #basal total postsynaptic contact length
    basal_postsyn_length_prop=null   :float #basal total postsynaptic contact length
    basal_contact_conversion=null   :float #basal synapse to contact ratio
    basal_contact_density=null   :float #basal synapse to contact length ratio
    basal_n_synapses   :bigint unsigned #basal total number of synapses for neurite
    basal_synapse_sizes_mean=null   :float #basal average synaptic size
    oblique_n_contacts   :bigint unsigned #oblique number of contacts
    oblique_n_contacts_prop=null   :float #oblique proportion of number of contacts
    oblique_postsyn_length   :bigint unsigned #oblique total postsynaptic contact length
    oblique_postsyn_length_prop=null   :float #oblique total postsynaptic contact length
    oblique_contact_conversion=null   :float #oblique synapse to contact ratio
    oblique_contact_density=null   :float #oblique synapse to contact length ratio
    oblique_n_synapses   :bigint unsigned #oblique total number of synapses for neurite
    oblique_synapse_sizes_mean=null   :float #oblique average synaptic size
    soma_n_contacts   :bigint unsigned #soma number of contacts
    soma_n_contacts_prop=null   :float #soma proportion of number of contacts
    soma_postsyn_length   :bigint unsigned #soma total postsynaptic contact length
    soma_postsyn_length_prop=null   :float #soma total postsynaptic contact length
    soma_contact_conversion=null   :float #soma synapse to contact ratio
    soma_contact_density=null   :float #soma synapse to contact length ratio
    soma_n_synapses   :bigint unsigned #soma total number of synapses for neurite
    soma_synapse_sizes_mean=null   :float #soma average synaptic size
    axon_n_contacts   :bigint unsigned #axon number of contacts
    axon_n_contacts_prop=null   :float #axon proportion of number of contacts
    axon_postsyn_length   :bigint unsigned #axon total postsynaptic contact length
    axon_postsyn_length_prop=null   :float #axon total postsynaptic contact length
    axon_contact_conversion=null   :float #axon synapse to contact ratio
    axon_contact_density=null   :float #axon synapse to contact length ratio
    axon_n_synapses   :bigint unsigned #axon total number of synapses for neurite
    axon_synapse_sizes_mean=null   :float #axon average synaptic size
    dendrites_n_contacts   :bigint unsigned #dendrites number of contacts
    dendrites_n_contacts_prop=null   :float #dendrites proportion of number of contacts
    dendrites_postsyn_length   :bigint unsigned #dendrites total postsynaptic contact length
    dendrites_postsyn_length_prop=null   :float #dendrites total postsynaptic contact length
    dendrites_contact_conversion=null   :float #dendrites synapse to contact ratio
    dendrites_contact_density=null   :float #dendrites synapse to contact length ratio
    dendrites_n_synapses   :bigint unsigned #dendrites total number of synapses for neurite
    dendrites_synapse_sizes_mean=null   :float #dendrites average synaptic size
    """
    
    labels_to_keep = [2,3,4,5,6,7,8]

    key_source = pinky.Segmentation & pinky.CurrentSegmentation
    
    def make(self,key):

        contact_skeleton = dj.U("segmentation","segment_id").aggr(
            pinky.SkeletonContact.proj("postsyn","postsyn_length","n_synapses","majority_label","synapse_sizes_mean",segment_id="presyn")
            & [dict(majority_label=x) for x in self.labels_to_keep],

            
            total_n_contacts= "count(*)",
            total_postsyn_length= "sum(postsyn_length)",
            total_contact_conversion= "sum( n_synapses )/count(*)",
            total_contact_density= "if(sum(postsyn_length)=0,null,sum( n_synapses )/sum(postsyn_length))",
            total_n_synapses= "sum(n_synapses)",
            total_synapse_sizes_mean= "if(sum(n_synapses)=0,0,sum(n_synapses*synapse_sizes_mean)/sum(n_synapses))",


            apical_n_contacts= "sum(majority_label=2)",
            apical_n_contacts_prop= "sum(majority_label=2)/count(*)",
            apical_postsyn_length= "sum( postsyn_length * (majority_label=2))",
            apical_postsyn_length_prop= "if(sum(majority_label=2)=0,0,sum( postsyn_length * (majority_label=2))/sum(postsyn_length))",
            apical_contact_conversion= "if(sum(majority_label=2)=0,null,sum( n_synapses * (majority_label=2))/sum(majority_label=2))",
            apical_contact_density= "if(sum( postsyn_length * (majority_label=2))=0,null,sum( n_synapses * (majority_label=2))/sum( postsyn_length * (majority_label=2)))",
            apical_n_synapses= "sum(n_synapses * (majority_label=2))",
            apical_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=2))=0,0,sum((n_synapses * (majority_label=2))*synapse_sizes_mean*(majority_label=2))/sum(n_synapses * (majority_label=2)))",

            basal_n_contacts= "sum(majority_label=3)",
            basal_n_contacts_prop= "sum(majority_label=3)/count(*)",
            basal_postsyn_length= "sum( postsyn_length * (majority_label=3))",
            basal_postsyn_length_prop= "if(sum(majority_label=3)=0,0,sum( postsyn_length * (majority_label=3))/sum(postsyn_length))",
            basal_contact_conversion= "if(sum(majority_label=3)=0,null,sum( n_synapses * (majority_label=3))/sum(majority_label=3))",
            basal_contact_density= "if(sum( postsyn_length * (majority_label=3))=0,null,sum( n_synapses * (majority_label=3))/sum( postsyn_length * (majority_label=3)))",
            basal_n_synapses= "sum(n_synapses * (majority_label=3))",
            basal_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=3))=0,0,sum((n_synapses * (majority_label=3))*synapse_sizes_mean*(majority_label=3))/sum(n_synapses * (majority_label=3)))",

            oblique_n_contacts= "sum(majority_label=4)",
            oblique_n_contacts_prop= "sum(majority_label=4)/count(*)",
            oblique_postsyn_length= "sum( postsyn_length * (majority_label=4))",
            oblique_postsyn_length_prop= "if(sum(majority_label=4)=0,0,sum( postsyn_length * (majority_label=4))/sum(postsyn_length))",
            oblique_contact_conversion= "if(sum(majority_label=4)=0,null,sum( n_synapses * (majority_label=4))/sum(majority_label=4))",
            oblique_contact_density= "if(sum( postsyn_length * (majority_label=4))=0,null,sum( n_synapses * (majority_label=4))/sum( postsyn_length * (majority_label=4)))",
            oblique_n_synapses= "sum(n_synapses * (majority_label=4))",
            oblique_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=4))=0,0,sum((n_synapses * (majority_label=4))*synapse_sizes_mean*(majority_label=4))/sum(n_synapses * (majority_label=4)))",

            soma_n_contacts= "sum(majority_label=5)",
            soma_n_contacts_prop= "sum(majority_label=5)/count(*)",
            soma_postsyn_length= "sum( postsyn_length * (majority_label=5))",
            soma_postsyn_length_prop= "if(sum(majority_label=5)=0,0,sum( postsyn_length * (majority_label=5))/sum(postsyn_length))",
            soma_contact_conversion= "if(sum(majority_label=5)=0,null,sum( n_synapses * (majority_label=5))/sum(majority_label=5))",
            soma_contact_density= "if(sum( postsyn_length * (majority_label=5))=0,null,sum( n_synapses * (majority_label=5))/sum( postsyn_length * (majority_label=5)))",
            soma_n_synapses= "sum(n_synapses * (majority_label=5))",
            soma_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=5))=0,0,sum((n_synapses * (majority_label=5))*synapse_sizes_mean*(majority_label=5))/sum(n_synapses * (majority_label=5)))",

            axon_n_contacts= "sum(majority_label=6 OR majority_label=7)",
            axon_n_contacts_prop= "sum(majority_label=6 OR majority_label=7)/count(*)",
            axon_postsyn_length= "sum( postsyn_length * (majority_label=6 OR majority_label=7))",
            axon_postsyn_length_prop= "if(sum(majority_label=6 OR majority_label=7)=0,0,sum( postsyn_length * (majority_label=6 OR majority_label=7))/sum(postsyn_length))",
            axon_contact_conversion= "if(sum(majority_label=6 OR majority_label=7)=0,null,sum( n_synapses * (majority_label=6 OR majority_label=7))/sum(majority_label=6 OR majority_label=7))",
            axon_contact_density= "if(sum( postsyn_length * (majority_label=6 OR majority_label=7))=0,null,sum( n_synapses * (majority_label=6 OR majority_label=7))/sum( postsyn_length * (majority_label=6 OR majority_label=7)))",
            axon_n_synapses= "sum(n_synapses * (majority_label=6 OR majority_label=7))",
            axon_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=6 OR majority_label=7))=0,0,sum((n_synapses * (majority_label=6 OR majority_label=7))*synapse_sizes_mean*(majority_label=6 OR majority_label=7))/sum(n_synapses * (majority_label=6 OR majority_label=7)))",

            dendrites_n_contacts= "sum(majority_label=8)",
            dendrites_n_contacts_prop= "sum(majority_label=8)/count(*)",
            dendrites_postsyn_length= "sum( postsyn_length * (majority_label=8))",
            dendrites_postsyn_length_prop= "if(sum(majority_label=8)=0,0,sum( postsyn_length * (majority_label=8))/sum(postsyn_length))",
            dendrites_contact_conversion= "if(sum(majority_label=8)=0,null,sum( n_synapses * (majority_label=8))/sum(majority_label=8))",
            dendrites_contact_density= "if(sum( postsyn_length * (majority_label=8))=0,null,sum( n_synapses * (majority_label=8))/sum( postsyn_length * (majority_label=8)))",
            dendrites_n_synapses= "sum(n_synapses * (majority_label=8))",
            dendrites_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=8))=0,0,sum((n_synapses * (majority_label=8))*synapse_sizes_mean*(majority_label=8))/sum(n_synapses * (majority_label=8)))",    
        )
        
        self.insert(contact_skeleton,skip_duplicates=True)

@schema
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
        prepost_data = ContactPrePost.proj("postsyn","total_contact_conversion",
                "total_contact_density","total_synapse_sizes_mean",
                syn_density="if(total_postsyn_length=0,null,(total_n_synapses*total_synapse_sizes_mean)/total_postsyn_length)",
                presyn="segment_id").fetch()
        df = pd.DataFrame(prepost_data)

        #gets all the combinations of postsyn-postsyn without any repeats
        targets = (dj.U("postsyn") & pinky.SkeletonContact).proj(segment_id="postsyn") - pinky.SegmentExclude
        info = targets * targets.proj(segment_b='segment_id') & 'segment_id < segment_b'
        segment_pairs = info.fetch()
        
        total_correlations = []

        for postsyn1,postsyn2 in tqdm(segment_pairs):

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
            corr_dict = dict(segmentation=2,segment_id=postsyn1,
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

@schema
class PostsynContact(dj.Computed):
    definition="""
    -> pinky.Segment
        ---
    total_n_contacts   :bigint unsigned #total number of contacts
    total_postsyn_length   :bigint unsigned #total total postsynaptic contact length
    total_contact_conversion=null   :float #total synapse to contact ratio
    total_contact_density=null   :float #total synapse to contact length ratio
    total_n_synapses   :bigint unsigned #total total number of synapses for neurite
    total_synapse_sizes_mean=null   :float #total average synaptic size
    apical_n_contacts   :bigint unsigned #apical number of contacts
    apical_n_contacts_prop=null   :float #apical proportion of number of contacts
    apical_postsyn_length   :bigint unsigned #apical total postsynaptic contact length
    apical_postsyn_length_prop=null   :float #apical total postsynaptic contact length
    apical_contact_conversion=null   :float #apical synapse to contact ratio
    apical_contact_density=null   :float #apical synapse to contact length ratio
    apical_n_synapses   :bigint unsigned #apical total number of synapses for neurite
    apical_synapse_sizes_mean=null   :float #apical average synaptic size
    basal_n_contacts   :bigint unsigned #basal number of contacts
    basal_n_contacts_prop=null   :float #basal proportion of number of contacts
    basal_postsyn_length   :bigint unsigned #basal total postsynaptic contact length
    basal_postsyn_length_prop=null   :float #basal total postsynaptic contact length
    basal_contact_conversion=null   :float #basal synapse to contact ratio
    basal_contact_density=null   :float #basal synapse to contact length ratio
    basal_n_synapses   :bigint unsigned #basal total number of synapses for neurite
    basal_synapse_sizes_mean=null   :float #basal average synaptic size
    oblique_n_contacts   :bigint unsigned #oblique number of contacts
    oblique_n_contacts_prop=null   :float #oblique proportion of number of contacts
    oblique_postsyn_length   :bigint unsigned #oblique total postsynaptic contact length
    oblique_postsyn_length_prop=null   :float #oblique total postsynaptic contact length
    oblique_contact_conversion=null   :float #oblique synapse to contact ratio
    oblique_contact_density=null   :float #oblique synapse to contact length ratio
    oblique_n_synapses   :bigint unsigned #oblique total number of synapses for neurite
    oblique_synapse_sizes_mean=null   :float #oblique average synaptic size
    soma_n_contacts   :bigint unsigned #soma number of contacts
    soma_n_contacts_prop=null   :float #soma proportion of number of contacts
    soma_postsyn_length   :bigint unsigned #soma total postsynaptic contact length
    soma_postsyn_length_prop=null   :float #soma total postsynaptic contact length
    soma_contact_conversion=null   :float #soma synapse to contact ratio
    soma_contact_density=null   :float #soma synapse to contact length ratio
    soma_n_synapses   :bigint unsigned #soma total number of synapses for neurite
    soma_synapse_sizes_mean=null   :float #soma average synaptic size
    axon_n_contacts   :bigint unsigned #axon number of contacts
    axon_n_contacts_prop=null   :float #axon proportion of number of contacts
    axon_postsyn_length   :bigint unsigned #axon total postsynaptic contact length
    axon_postsyn_length_prop=null   :float #axon total postsynaptic contact length
    axon_contact_conversion=null   :float #axon synapse to contact ratio
    axon_contact_density=null   :float #axon synapse to contact length ratio
    axon_n_synapses   :bigint unsigned #axon total number of synapses for neurite
    axon_synapse_sizes_mean=null   :float #axon average synaptic size
    dendrites_n_contacts   :bigint unsigned #dendrites number of contacts
    dendrites_n_contacts_prop=null   :float #dendrites proportion of number of contacts
    dendrites_postsyn_length   :bigint unsigned #dendrites total postsynaptic contact length
    dendrites_postsyn_length_prop=null   :float #dendrites total postsynaptic contact length
    dendrites_contact_conversion=null   :float #dendrites synapse to contact ratio
    dendrites_contact_density=null   :float #dendrites synapse to contact length ratio
    dendrites_n_synapses   :bigint unsigned #dendrites total number of synapses for neurite
    dendrites_synapse_sizes_mean=null   :float #dendrites average synaptic size
    """
    
    labels_to_keep = [2,3,4,5,6,7,8]

    key_source = pinky.Segmentation & pinky.CurrentSegmentation
    
    def make(self,key):

        skeleton_contact = dj.U("segmentation","segment_id").aggr(
            pinky.SkeletonContact.proj("presyn","postsyn_length","n_synapses","majority_label","synapse_sizes_mean",segment_id="postsyn")
            & [dict(majority_label=x) for x in self.labels_to_keep],

            
            total_n_contacts= "count(*)",
            total_postsyn_length= "sum(postsyn_length)",
            total_contact_conversion= "sum( n_synapses )/count(*)",
            total_contact_density= "if(sum(postsyn_length)=0,null,sum( n_synapses )/sum(postsyn_length))",
            total_n_synapses= "sum(n_synapses)",
            total_synapse_sizes_mean= "if(sum(n_synapses)=0,0,sum(n_synapses*synapse_sizes_mean)/sum(n_synapses))",


            apical_n_contacts= "sum(majority_label=2)",
            apical_n_contacts_prop= "sum(majority_label=2)/count(*)",
            apical_postsyn_length= "sum( postsyn_length * (majority_label=2))",
            apical_postsyn_length_prop= "if(sum(majority_label=2)=0,0,sum( postsyn_length * (majority_label=2))/sum(postsyn_length))",
            apical_contact_conversion= "if(sum(majority_label=2)=0,null,sum( n_synapses * (majority_label=2))/sum(majority_label=2))",
            apical_contact_density= "if(sum( postsyn_length * (majority_label=2))=0,null,sum( n_synapses * (majority_label=2))/sum( postsyn_length * (majority_label=2)))",
            apical_n_synapses= "sum(n_synapses * (majority_label=2))",
            apical_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=2))=0,0,sum((n_synapses * (majority_label=2))*synapse_sizes_mean*(majority_label=2))/sum(n_synapses * (majority_label=2)))",

            basal_n_contacts= "sum(majority_label=3)",
            basal_n_contacts_prop= "sum(majority_label=3)/count(*)",
            basal_postsyn_length= "sum( postsyn_length * (majority_label=3))",
            basal_postsyn_length_prop= "if(sum(majority_label=3)=0,0,sum( postsyn_length * (majority_label=3))/sum(postsyn_length))",
            basal_contact_conversion= "if(sum(majority_label=3)=0,null,sum( n_synapses * (majority_label=3))/sum(majority_label=3))",
            basal_contact_density= "if(sum( postsyn_length * (majority_label=3))=0,null,sum( n_synapses * (majority_label=3))/sum( postsyn_length * (majority_label=3)))",
            basal_n_synapses= "sum(n_synapses * (majority_label=3))",
            basal_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=3))=0,0,sum((n_synapses * (majority_label=3))*synapse_sizes_mean*(majority_label=3))/sum(n_synapses * (majority_label=3)))",

            oblique_n_contacts= "sum(majority_label=4)",
            oblique_n_contacts_prop= "sum(majority_label=4)/count(*)",
            oblique_postsyn_length= "sum( postsyn_length * (majority_label=4))",
            oblique_postsyn_length_prop= "if(sum(majority_label=4)=0,0,sum( postsyn_length * (majority_label=4))/sum(postsyn_length))",
            oblique_contact_conversion= "if(sum(majority_label=4)=0,null,sum( n_synapses * (majority_label=4))/sum(majority_label=4))",
            oblique_contact_density= "if(sum( postsyn_length * (majority_label=4))=0,null,sum( n_synapses * (majority_label=4))/sum( postsyn_length * (majority_label=4)))",
            oblique_n_synapses= "sum(n_synapses * (majority_label=4))",
            oblique_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=4))=0,0,sum((n_synapses * (majority_label=4))*synapse_sizes_mean*(majority_label=4))/sum(n_synapses * (majority_label=4)))",

            soma_n_contacts= "sum(majority_label=5)",
            soma_n_contacts_prop= "sum(majority_label=5)/count(*)",
            soma_postsyn_length= "sum( postsyn_length * (majority_label=5))",
            soma_postsyn_length_prop= "if(sum(majority_label=5)=0,0,sum( postsyn_length * (majority_label=5))/sum(postsyn_length))",
            soma_contact_conversion= "if(sum(majority_label=5)=0,null,sum( n_synapses * (majority_label=5))/sum(majority_label=5))",
            soma_contact_density= "if(sum( postsyn_length * (majority_label=5))=0,null,sum( n_synapses * (majority_label=5))/sum( postsyn_length * (majority_label=5)))",
            soma_n_synapses= "sum(n_synapses * (majority_label=5))",
            soma_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=5))=0,0,sum((n_synapses * (majority_label=5))*synapse_sizes_mean*(majority_label=5))/sum(n_synapses * (majority_label=5)))",

            axon_n_contacts= "sum(majority_label=6 OR majority_label=7)",
            axon_n_contacts_prop= "sum(majority_label=6 OR majority_label=7)/count(*)",
            axon_postsyn_length= "sum( postsyn_length * (majority_label=6 OR majority_label=7))",
            axon_postsyn_length_prop= "if(sum(majority_label=6 OR majority_label=7)=0,0,sum( postsyn_length * (majority_label=6 OR majority_label=7))/sum(postsyn_length))",
            axon_contact_conversion= "if(sum(majority_label=6 OR majority_label=7)=0,null,sum( n_synapses * (majority_label=6 OR majority_label=7))/sum(majority_label=6 OR majority_label=7))",
            axon_contact_density= "if(sum( postsyn_length * (majority_label=6 OR majority_label=7))=0,null,sum( n_synapses * (majority_label=6 OR majority_label=7))/sum( postsyn_length * (majority_label=6 OR majority_label=7)))",
            axon_n_synapses= "sum(n_synapses * (majority_label=6 OR majority_label=7))",
            axon_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=6 OR majority_label=7))=0,0,sum((n_synapses * (majority_label=6 OR majority_label=7))*synapse_sizes_mean*(majority_label=6 OR majority_label=7))/sum(n_synapses * (majority_label=6 OR majority_label=7)))",

            dendrites_n_contacts= "sum(majority_label=8)",
            dendrites_n_contacts_prop= "sum(majority_label=8)/count(*)",
            dendrites_postsyn_length= "sum( postsyn_length * (majority_label=8))",
            dendrites_postsyn_length_prop= "if(sum(majority_label=8)=0,0,sum( postsyn_length * (majority_label=8))/sum(postsyn_length))",
            dendrites_contact_conversion= "if(sum(majority_label=8)=0,null,sum( n_synapses * (majority_label=8))/sum(majority_label=8))",
            dendrites_contact_density= "if(sum( postsyn_length * (majority_label=8))=0,null,sum( n_synapses * (majority_label=8))/sum( postsyn_length * (majority_label=8)))",
            dendrites_n_synapses= "sum(n_synapses * (majority_label=8))",
            dendrites_synapse_sizes_mean= "if(sum(n_synapses * (majority_label=8))=0,0,sum((n_synapses * (majority_label=8))*synapse_sizes_mean*(majority_label=8))/sum(n_synapses * (majority_label=8)))",    
        )
        
        self.insert(skeleton_contact,skip_duplicates=True)
		