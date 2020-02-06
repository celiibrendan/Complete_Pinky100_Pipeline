sn

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
for i in range(len(rels)):
    radsyncont_df = pd.DataFrame(((radtune.BestVonCorr & spat_unit_pairs) * 
                              rels[i].proj(*attrs, cont_seg_shared = 'n_seg_shared', cont_seg_union = 'n_seg_union', segment_id1 = 'segment_id', segment_id2 = 'segment_b')).fetch())
    radsyncont_df['bin_diff_pref_ori'] = ori_centers[(np.digitize(np.abs(radsyncont_df['diff_pref_ori']), ori_edges)) - 1]
    
    x_coords = ori_centers
    y_coords = radsyncont_df.groupby('bin_diff_pref_ori').mean()['density_pearson_converted']
    
    ax.plot(x_coords, y_coords, label=labels[i], color=colors[i])

    errors = radsyncont_df.groupby('bin_diff_pref_ori').sem()['density_pearson_converted']  # compute SE
    ax.errorbar(x_coords, y_coords, yerr=errors, ecolor=colors[i], fmt=' ', zorder=-1, label=None)
    
ax.legend(frameon=False, loc=0, fontsize=12)
ax.tick_params(labelsize=12)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xticks(ori_centers)
ax.set_xticklabels(['{}°-{}°'.format(*a) for a in ori_labels])
ax.set_xlabel(r'$\Delta \theta$', fontsize = 14)
ax.set_ylabel(r'$\langle$Synapse Density Correlation$\rangle$', fontsize = 14)
fig.savefig('figures/density_four_classes.png', dpi=300)


# In[32]:


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(x_coords, y_coords, label=labels[i], color=colors[i])
errors = spatsyncont_df.groupby('bin_center_dist').sem()['density_pearson_converted']  # compute SE
ax.errorbar(x_coords, y_coords, yerr=errors, ecolor=colors[i], fmt=' ', zorder=-1, label=None)
    


# In[ ]:


dist_labels


# In[ ]:


spat_units = spattune.BestSTA.Loc & 'sta_snr > 1.5' & (segment & (spattune.BestSTA.Confidence() & tuned))
spat_unit_pairs = (spat_units.proj(segment_id1 = 'segment_id') * 
                  spat_units.proj(segment_id2 = 'segment_id')) & 'segment_id1 < segment_id2'


# In[ ]:


# RF distance

rels = [ta3p100.ContactCorrelationShaft, ta3p100.ContactCorrelationHead, ta3p100.ContactCorrelationSoma, ta3p100.ContactCorrelationAxon]
labels = ['Shaft', 'Head', 'Soma', 'Axon']
colors = ['r', 'g', 'b', 'y']

#rels = [ta3p100.ContactCorrelation]
#labels = ['Total']
#colors = ['k']
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
for i in range(len(rels)):
    spatsyncont_df = pd.DataFrame(((spattune.BestSTACorr & sig_unit_pairs) * 
                              rels[i].proj(*attrs, cont_seg_shared = 'n_seg_shared', cont_seg_union = 'n_seg_union', segment_id1 = 'segment_id', segment_id2 = 'segment_b')).fetch())
    
    dist_edges = np.linspace(min(spatsyncont_df['center_dist']), max(spatsyncont_df['center_dist']), 8)
    de = list(['{:.1f}'.format(ee) for ee in dist_edges])
    dist_labels = list(zip(de[:-1], de[1:]))
    dist_centers = np.hstack((np.nan, np.round((np.array(dist_edges[1:]) + np.array(dist_edges[:-1]))/2, decimals=2), np.nan))
    spatsyncont_df['bin_center_dist'] = dist_centers[(np.digitize(spatsyncont_df['center_dist'], dist_edges))]
    
    x_coords = dist_centers[1:-1]
    y_coords = spatsyncont_df.groupby('bin_center_dist').mean()['density_pearson_converted']
    #print(x_coords, y_coords)
    ax.plot(x_coords, y_coords, label=labels[i], color=colors[i])
    #ax.scatter(x_coords, y_coords, label=labels[i], color=colors[i])

    errors = spatsyncont_df.groupby('bin_center_dist').sem()['density_pearson_converted']  # compute SE
    ax.errorbar(x_coords, y_coords, yerr=errors, ecolor=colors[i], fmt=' ', zorder=-1, label=None)
    
ax.legend(frameon=False, loc=0, fontsize=12)
ax.tick_params(labelsize=12)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xticks(dist_centers[1:-1])
ax.set_xticklabels(['[{},{}]'.format(*a) for a in dist_labels], rotation=15)
ax.set_xlabel('RF distance', fontsize = 15)
ax.set_ylabel(r'$\langle$Synapse Density Correlation$\rangle$', fontsize = 15)
fig.savefig('figures/RFdistance_density_four_classes.png', dpi=300)


# In[ ]:


# receptive field

functional = ['bin_center_dist']    
connectomics = ['cont_shared_percent', 'syn_shared_percent', 'binary_conversion_pearson',
       'binary_conversion_cosine', 'binary_conv_jaccard_ones_ratio',
       'binary_conv_jaccard_matching_ratio', 'conversion_pearson',
       'conversion_cosine', 'density_pearson', 'density_cosine',
       'synapse_volume_mean_pearson', 'synapse_volume_mean_cosine',
       'synapse_vol_density_pearson', 'synapse_vol_density_cosine',
       'binary_conversion_pearson_converted',
       'binary_conversion_cosine_converted',
       'binary_conv_jaccard_ones_ratio_converted',
       'binary_conv_jaccard_matching_ratio_converted',
       'conversion_pearson_converted', 'conversion_cosine_converted',
       'density_pearson_converted', 'density_cosine_converted',
       'synapse_volume_mean_pearson_converted',
       'synapse_volume_mean_cosine_converted',
       'synapse_vol_density_pearson_converted',
       'synapse_vol_density_cosine_converted']

with sns.axes_style('ticks'):
    fig, ax = plt.subplots(len(connectomics), len(functional), figsize=(5*len(functional), 3*len(connectomics)))

for i, pair in enumerate(itertools.product(functional, connectomics)):
    #sns.pointplot(pair[0], pair[1], data = spatsyncont_df, ax=ax[i], ci=None) 
    
    x_coords = dist_centers[1:-1]
    y_coords = spatsyncont_df.groupby(pair[0]).mean()[pair[1]] 
    ax[i].plot(x_coords, y_coords)
    
    errors = spatsyncont_df.groupby(pair[0]).sem()[pair[1]]  # compute SE
    ax[i].errorbar(x_coords, y_coords, yerr=errors, ecolor='k', fmt=' ', zorder=-1)
    
    ax[i].set_title(connectomics[i], fontsize=15)

sns.despine(trim=True)
fig.tight_layout()
fig.savefig('figures/Head_union_corr.png', dpi=100)


# In[ ]:


# plot n_seg_shared/n_seg_union for both synapse and contact vs functional differences

functional = ['bin_diff_pref_ori', 'bin_diff_pref_dir', 'bin_von_corr']    
connectomics = ['cont_shared_percent', 'syn_shared_percent', 'binary_conversion_pearson',
       'binary_conversion_cosine', 'binary_conv_jaccard_ones_ratio',
       'binary_conv_jaccard_matching_ratio', 'conversion_pearson',
       'conversion_cosine', 'density_pearson', 'density_cosine',
       'synapse_volume_mean_pearson', 'synapse_volume_mean_cosine',
       'synapse_vol_density_pearson', 'synapse_vol_density_cosine',
       'binary_conversion_pearson_converted',
       'binary_conversion_cosine_converted',
       'binary_conv_jaccard_ones_ratio_converted',
       'binary_conv_jaccard_matching_ratio_converted',
       'conversion_pearson_converted', 'conversion_cosine_converted',
       'density_pearson_converted', 'density_cosine_converted',
       'synapse_volume_mean_pearson_converted',
       'synapse_volume_mean_cosine_converted',
       'synapse_vol_density_pearson_converted',
       'synapse_vol_density_cosine_converted']

with sns.axes_style('ticks'):
    fig, ax = plt.subplots(len(connectomics), len(functional), figsize=(5*len(functional), 3*len(connectomics)))

for i, pair in enumerate(itertools.product(functional, connectomics)):
    sns.pointplot(pair[0], pair[1], data = radsyncont_df, ax=ax[i%len(connectomics), i//len(connectomics)], ci=None) 
    
    # manually plot error bars
    x_coords = []
    y_coords = []
    for point_pair in ax[i%len(connectomics), i//len(connectomics)].collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    errors = radsyncont_df.groupby(pair[0]).sem()[pair[1]]  # compute SE
    
    ax[i%len(connectomics), i//len(connectomics)].errorbar(x_coords, y_coords, yerr=errors, ecolor='k', fmt=' ', zorder=-1)

for i in range(len(connectomics)):
    ax[i, 1].set_title(connectomics[i], fontsize=15)

#sns.pointplot('bin_diff_pref_ori', 'cont_shared_percent', ci=None, data = radsyncont_df, ax=ax[0,0], linestyles='--', color='k')    
#sns.pointplot('bin_diff_pref_dir', 'cont_shared_percent', ci=None, data = radsyncont_df, ax=ax[0,1], linestyles='--', color='k')    
#sns.pointplot('bin_von_corr', 'cont_shared_percent', ci=None, data = radsyncont_df, ax=ax[0,2], linestyles='--', color='k')    
#sns.pointplot('bin_diff_pref_ori', 'syn_shared_percent', ci=None, data = radsyncont_df, ax=ax[1,0], color='k')    
#sns.pointplot('bin_diff_pref_dir', 'syn_shared_percent', ci=None, data = radsyncont_df, ax=ax[1,1], color='k')    
#sns.pointplot('bin_von_corr', 'syn_shared_percent', ci=None, data = radsyncont_df, ax=ax[1,2], color='k')  
'''
l = ['Contact', 'Synapse']
for i in range(2):
    ax[i, 0].set_title('{} percent shared seg vs diff in orientation'.format(l[i]))
    ax[i, 0].set_xticklabels(['{}°-{}°'.format(*a) for a in ori_labels])
    ax[i, 0].set_xlabel(r'$\Delta \theta$')
    ax[i, 0].set_ylabel('$<Shared/Union>$')

    ax[i, 1].set_title('{} percent shared seg vs diff in direction'.format(l[i]))
    ax[i, 1].set_xticklabels(['{}°-{}°'.format(*a) for a in dir_labels])
    ax[i, 1].set_xlabel(r'$\Delta \theta$')
    ax[i, 1].set_ylabel('$<Shared/Union>$')

    ax[i, 2].set_title('{} percent shared seg vs von corr'.format(l[i]))
    ax[i, 2].set_xticklabels(['[{},{}]'.format(*a) for a in vc_labels])
    ax[i, 2].set_xlabel('Von corr')
    ax[i, 2].set_ylabel('$<Shared/Union>$')

'''
sns.despine(trim=True)
fig.tight_layout()
fig.savefig('figures/Head_confidence1.0_percent_shared_seg_by_functional_difference.png', dpi=100)


# In[ ]:




