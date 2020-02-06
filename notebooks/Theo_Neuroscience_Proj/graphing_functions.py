def bin_orientation_data(orientation_preference,
                         n_bins = 20,
                        lower_bin_bound = 0,
                         upper_bin_bound = np.pi):
    """
    Functions that will do binning and plotting
    """
    
    rad2deg = 180/np.pi
    ori_edges = np.linspace(lower_bin_bound, upper_bin_bound, n_bins+1)
    oe = list(['{:.0f}'.format(ee) for ee in [np.round(e * rad2deg) for e in ori_edges]])
    ori_labels = list(zip(oe[:-1], oe[1:]))
    ori_centers = np.round((ori_edges[1:] + ori_edges[:-1])/2 * rad2deg, decimals=2) 

    bin_centers = ori_centers
    ori_bin_edges=ori_edges
    binned_ori_preferences = bin_centers[(np.digitize(orientation_preference, ori_bin_edges))-1]
    raw_orientation_degrees = (np.array(orientation_preference)*rad2deg).astype("int")
    
    from collections import Counter
    my_counter = Counter(binned_ori_preferences)
    #print(my_counter)
    binned_angles = np.array(list(my_counter.keys()))
    binned_angles_histogram = np.array(list(my_counter.values()))
    
    return binned_angles,binned_angles_histogram

def plot_circular_distribution_lite(theta,
                               radii,
                                    ax,
                               width=-1,
                               bottom = 8,
                               max_height = 4):
    N = len(radii)
    #theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    #radii = max_height*np.random.rand(N)
    if width == -1:
        width = (2*np.pi) / N

    #ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=bottom)

    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.8)
    #ax.set_yticks([0,45,90,135,180])
    """
    ax.set_xticks(np.pi/180. * np.linspace(0,  180, 8, endpoint=False))
    ax.set_thetalim(0,np.pi)
    """
    ax.set_xticklabels(np.linspace(0,  180, 8, endpoint=False))
    #plt.show()

def graph_binned_orientation_data(binned_angles,
                                binned_angles_histogram,
                                  neuron_id,
                                  title,
                                  global_binned_angles = [],
                                global_binned_angles_histogram= [],
                                 graphs_to_plot=[],
                                  figure_size = (20,20),
                                 circular_flag = True):
    
    axis_label_size=20
    tick_label_size = 20
    title_size=30
    
    if circular_flag:
        n_subplot = 2*len(graphs_to_plot)
    else:
        n_subplot = len(graphs_to_plot)
    
    print("n_subplot = " + str(n_subplot))
    currnet_subplot = 1
    fig = plt.figure(figsize=(figure_size[0],figure_size[1]*len(graphs_to_plot)))
    fig.tight_layout()
    #fig.set_size_inches(20, 12)
    if "local" in graphs_to_plot:
        if circular_flag:
            ax = plt.subplot(n_subplot,2,currnet_subplot)
            #graphing the linear graph
            plt.bar(binned_angles/rad2deg,binned_angles_histogram,width=ori_edges[1]-ori_edges[0]-0.1)
            
#             ax.set(
#                    title=title + "_linear",
#                    ylabel='Number of Neurons',
#                    xlabel='Radians',
#                     titlesize=20,
#                     labelsize=20)
            ax.set_title(title+"_linear",fontsize=title_size)
            ax.set_xlabel("Degrees",fontsize=axis_label_size)
            ax.set_ylabel('Number of Neurons',fontsize=axis_label_size)
            ax.tick_params(axis="both",which="major",labelsize=tick_label_size)
            
            #graphing the circular graph
            ax = plt.subplot(n_subplot,2,currnet_subplot+1,polar=True)
            plot_circular_distribution_lite(binned_angles/rad2deg*2,
                                            binned_angles_histogram,
                                            ax,
                                            width=(ori_edges[1]-ori_edges[0])*2-0.1)
            
#             ax.set(
#                    title=title + "_circular",
#                    ylabel='Number of Neurons',
#                    xlabel='Degrees',
#                     titlesize=20,
#                     labelsize=20)
            ax.set_title(title+"_linear",fontsize=title_size)
            ax.set_xlabel("Degrees",fontsize=axis_label_size)
            ax.set_ylabel('Number of Neurons',fontsize=axis_label_size)
            ax.tick_params(axis="both",which="major",labelsize=tick_label_size)
#             ax.set_xticklabels(ax.get_xticklabels(),fontsize=tick_label_size)
#             ax.set_yticklabels(ax.get_yticklabels(),fontsize=tick_label_size)
            
            #increment the current_subplot
            currnet_subplot += 2
        else:
            ax = plt.subplot(n_subplot,1,currnet_subplot)
            #graphing the linear graph
            plt.bar(binned_angles/rad2deg,binned_angles_histogram,width=ori_edges[1]-ori_edges[0]-0.1)
            
#             ax.set(
#                    title=title + "_linear",
#                    ylabel='Number of Neurons',
#                    xlabel='Radians',
#                     titlesize=20,
#                     labelsize=20)
            ax.set_title(title+"_linear",fontsize=title_size)
            ax.set_xlabel("Degrees",fontsize=axis_label_size)
            ax.set_ylabel('Number of Neurons',fontsize=axis_label_size)
            ax.tick_params(axis="both",which="major",labelsize=tick_label_size)
            #increment the current_subplot
            currnet_subplot += 1
    
    print("\n\n\n")
    if "global" in graphs_to_plot:
        binned_angles = global_binned_angles
        binned_angles_histogram =  global_binned_angles_histogram
        if circular_flag:
            ax = plt.subplot(n_subplot,2,currnet_subplot)
            #graphing the linear graph
            plt.bar(binned_angles/rad2deg,binned_angles_histogram,width=ori_edges[1]-ori_edges[0]-0.1)
            
#             ax.set(
#                    title=title + "_linear",
#                    ylabel='Number of Neurons',
#                    xlabel='Radians',
#                     titlesize=20,
#                     labelsize=20)
            ax.set_title(title+"_linear",fontsize=title_size)
            ax.set_xlabel("Degrees",fontsize=axis_label_size)
            ax.set_ylabel('Number of Neurons',fontsize=axis_label_size)
            ax.tick_params(axis="both",which="major",labelsize=tick_label_size)
            
            #graphing the circular graph
            ax = plt.subplot(n_subplot,2,currnet_subplot+1,polar=True)
            plot_circular_distribution_lite(binned_angles/rad2deg*2,
                                            binned_angles_histogram,
                                            ax,
                                            width=(ori_edges[1]-ori_edges[0])*2-0.1)
            
#             ax.set(
#                    title=title + "_circular",
#                    ylabel='Number of Neurons',
#                    xlabel='Degrees',
#                     titlesize=20,
#                     labelsize=20)
            ax.set_title(title+"_linear",fontsize=title_size)
            ax.set_xlabel("Degrees",fontsize=axis_label_size)
            ax.set_ylabel('Number of Neurons',fontsize=axis_label_size)
            ax.tick_params(axis="both",which="major",labelsize=tick_label_size)
#             ax.set_xticklabels(ax.get_xticklabels(),fontsize=tick_label_size)
#             ax.set_yticklabels(ax.get_yticklabels(),fontsize=tick_label_size)
            
            #increment the current_subplot
            currnet_subplot += 2
        else:
            ax = plt.subplot(n_subplot,1,currnet_subplot)
            #graphing the linear graph
            plt.bar(binned_angles/rad2deg,binned_angles_histogram,width=ori_edges[1]-ori_edges[0]-0.1)
            
#             ax.set(
#                    title=title + "_linear",
#                    ylabel='Number of Neurons',
#                    xlabel='Radians',
#                     titlesize=20,
#                     labelsize=20)
            ax.set_title(title+"_linear",fontsize=title_size)
            ax.set_xlabel("Degrees",fontsize=axis_label_size)
            ax.set_ylabel('Number of Neurons',fontsize=axis_label_size)
            ax.tick_params(axis="both",which="major",labelsize=tick_label_size)
            #increment the current_subplot
            currnet_subplot += 1
    left = 0.125
    right = 0.9
    bottom = 0.1
    top = 0.9
    wspace = 0.2
    hspace = 0.5
    fig.subplots_adjust(left=left,bottom=bottom,right=right,
                       top=top,wspace=wspace,hspace=hspace)

binned_angles,binned_angles_histogram = bin_orientation_data(orientation_preference,
                                                         n_bins = 4,
                                                        lower_bin_bound = 0,
                                                         upper_bin_bound = np.pi)

graph_binned_orientation_data(binned_angles,
                                binned_angles_histogram,
                                  neuron_id=45,
                                  title="example_type",
                              global_binned_angles = binned_angles,
                                global_binned_angles_histogram= binned_angles_histogram,
                                 graphs_to_plot=["local","global"],
                                 figure_size = (20,20),
                                 circular_flag = True)