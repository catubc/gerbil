import matplotlib
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np
from tqdm import tqdm, trange

from itertools import combinations

names= ['female','male','pup1','pup2']
clrs = ['red','blue','cyan','green']


#############################

#
def plot_matrix_analysis2(data, behavior_name, cmap):

    #
    locs = [[1,0],[2,0],[3,0],
            [2,1],[3,1],
            [3,2]
           ]
    pairs = ['female-male','female-pup1','female-pup2',
             'male-pup1', 'male-pup2',
             'pup1-pup2'
            ]

    vmin = 0
    vmax=200
    shift = 18
    real_start = 109
    real_end = 372

    # Load exact days and times for this datasets from the notebook:  NYU_greene_process_cohort1
    daystacks = np.load('/home/cat/daystacks.txt.npy',allow_pickle=True)
    print (daystacks.shape)

    # grab the day # for each day
    yticks = []
    for k in range(len(daystacks)):
        yticks.append(daystacks[k][0][0])

    yticks2 = []
    for k in range(len(daystacks)):
        yticks2.append("P"+str(daystacks[k][0][0]+9))


    xticks = np.arange(24)

    #
    fig=plt.figure(figsize=(20,10))
    for k in range(len(locs)):
        ax=plt.subplot(3,3,k+1)

        res = data[:,locs[k][0], locs[k][1]]
        res_24 = shift_data(res,
                            shift,
                            real_start,
                            real_end)

        im = plt.imshow(res_24,
                   vmin=vmin,
                   vmax=vmax,
                   aspect='auto',
                   cmap=cmap)


        #print (k)
        plt.title(pairs[k],pad=.95)
        if k%3==0:
            plt.ylabel("Day of study (March 2020)")
        if k>=3:
            plt.xlabel("Time of day (military time)")

        plt.yticks(np.arange(res_24.shape[0]),
                   yticks2,
                   fontsize=8)
        plt.xticks(xticks,
                   fontsize=8,
                  rotation=90
                  )
        plt.ylim(15.5,3.5)

    ax=plt.subplot(3,3,7)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label(behavior_name, fontsize=10)

    #plt.imshow(temp, aspect='auto', interpolation='none')
    plt.suptitle(behavior_name+"(0=min, "+str(vmax)+"=max)")
    if False:
        plt.savefig('/home/cat/cohort1_interactions_per_hour.svg')
        plt.close()
    plt.show()

def shift_data(data,
               shift,
               real_start,
               real_end):
    '''  Correct data for time of day start and padd missing hrs

    '''

    # make front padding
    d2 = np.zeros(shift, 'float32')+np.nan
    data = np.concatenate((d2, data))

    # make end padding
    l = data.shape[0]
    padding = 24-l%24
    d3 = np.zeros(padding, 'float32')+np.nan
    res = np.concatenate((data, d3))
    print (" data: ", res.shape)

    #
    res[:real_start]= np.nan
    res[real_end:]=np.nan

    # split in 24 chunks;

    res_24 = []
    for p in range(0,res.shape[0], 24):
        temp = res[p:p+24]
        if temp.shape[0]<24:
            temp2 = np.zeros(24,'float32')
            temp2[:temp.shape[0]]= temp
            temp = temp2
        res_24.append(temp)
    res_24=np.vstack(res_24)

    return res_24


def plot_matrix_analysis(data, behavior_name, cmap):

    #
    locs = [[1,0],[2,0],[3,0],
            [2,1],[3,1],
            [3,2]
           ]
    pairs = ['female-male','female-pup1','female-pup2',
             'male-pup1', 'male-pup2',
             'pup1-pup2'
            ]

    vmin = 0
    vmax=1500
    shift = 18
    real_start = 109
    real_end = 372

    # Load exact days and times for this datasets from the notebook:  NYU_greene_process_cohort1
    daystacks = np.load('/home/cat/daystacks.txt.npy',allow_pickle=True)
    print (daystacks.shape)

    # grab the day # for each day
    yticks = []
    for k in range(len(daystacks)):
        yticks.append(daystacks[k][0][0])
    xticks = np.arange(24)

    #
    fig=plt.figure()
    for k in range(len(locs)):
        ax=plt.subplot(3,3,k+1)

        res = data[:,locs[k][0], locs[k][1]]
        res_24 = shift_data(res,
                            shift,
                            real_start,
                            real_end)

        im = plt.imshow(res_24,
                   vmin=vmin,
                   vmax=vmax,
                   aspect='auto',
                   cmap=cmap)


        #print (k)
        plt.title(pairs[k],pad=.95)
        if k%3==0:
            plt.ylabel("day of study (March 2020)")
        if k>=3:
            plt.xlabel("time of day")

        plt.yticks(np.arange(res_24.shape[0]),
                   yticks,
                   fontsize=8)
        plt.xticks(xticks,
                   fontsize=8,
                  rotation=90)

    ax=plt.subplot(3,3,7)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label(behavior_name, fontsize=10)

    #plt.imshow(temp, aspect='auto', interpolation='none')
    plt.suptitle(behavior_name+"(0=min, "+str(vmax)+"=max)")
    plt.show()

def plot_pairwise_interactions2(locs_multi_hour):

    #
    names= ['female','male','pup1','pup2']
    clrs = ['red','blue','cyan','green']

    x_ticks=['female','male','pup1','pup2']
    text_clr = 'red'

    #
    distance_threshold_pixels = 200 # number of pixels between animals;
                                    # assume 1 pixel ~= 0.5mm -> 200pixels = 100mm = 10cm
    min_time_together = 5*25 # no of seconds to consider
    smoothing_window = 3
    min_distance = 25 # number of frames window
    #

    # COMPUTE PAIRWISE INTERACTIONS
    pair_interaction_times2 = []
    triples_interaction_time2 = []
    quad_interaction_times2 = []
    multi_animal_durations2 = []
    interactions2 = []
    duration_matrix2 = []

    for locs in tqdm(locs_multi_hour):

        #
        locs = locs.transpose(1,0,2)
        #print ("locs: ", locs.shape)

        #
        traces_23hrs = locs

        animals=np.arange(locs.shape[0])
        interactions = np.zeros((animals.shape[0],animals.shape[0]),'int32') + np.nan
        durations_matrix = np.zeros((animals.shape[0], animals.shape[0]),'int32') + np.nan

        # loop over all pairwise combinations
        pair_interaction_times = []
        pairs1 = list(combinations(animals,2))
        for ctr_x, pair in enumerate(pairs1):
            traces = []

            # smooth out traces;
            for k in pair:
                traces1=traces_23hrs[k].copy()
                traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((smoothing_window,))/smoothing_window, mode='same')
                traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((smoothing_window,))/smoothing_window, mode='same')
                traces1 = traces1
                traces.append(traces1)

            # COMPUTE PAIRWISE DISTANCES AND NEARBY TIMES POINTS
            idx_array = []
            diffs = np.sqrt((traces[0][:,0]-traces[1][:,0])**2+
                            (traces[0][:,1]-traces[1][:,1])**2)

            #
            idx = np.where(diffs<distance_threshold_pixels)[0]

            # COMPUTE TOTAL TIME TOGETHER
            #print ("Pairwise: ", pair, idx.shape)
            durations_matrix[pair[0],pair[1]]=idx.shape[0]/25

            # COMPUTE # OF INTERACTIONS;
            diffs_idx = idx[1:]-idx[:-1]
            idx2 = np.where(diffs_idx>min_time_together)[0]
            interactions[pair[0],pair[1]]=idx2.shape[0]

            # SAVE TIMES OF INTERACTION
            pair_interaction_times.append(idx)

        # SYMMETRIZE MATRICES
        for k in range(durations_matrix.shape[0]):
            for j in range(durations_matrix.shape[1]):
                if np.isnan(durations_matrix[k,j])==False:
                    durations_matrix[j,k]=durations_matrix[k,j]
                    interactions[j,k]=interactions[k,j]


        #############################
        # COMPUTE TRIPLE INTERACTIONS

        multi_animal_durations = [] #np.zeros((animals.shape[0]+4),'int32') + np.nan
        names_multi_animal =[]
        ctr=0

        pairs3 = list(combinations(animals,3))
        triples_interaction_times = []
        for ctr_x, pair in enumerate(pairs3):
            names_multi_animal.append(names[pair[0]]+"\n"+ names[pair[1]]+"\n"+names[pair[2]]+"\n")
            #names_multi_animal.append(names[pair[0]]+" - "+ names[pair[1]]+" - "+names[pair[2]])
            traces = []

            # COMPUTE LOCATIONS
            for k in pair:
                traces1=traces_23hrs[k].copy()
                traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((smoothing_window,))/smoothing_window, mode='same')
                traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((smoothing_window,))/smoothing_window, mode='same')
                traces1 = traces1
                traces.append(traces1)

            # COMPUTE PAIRWISE DISTANCES AND NEARBY TIMES POINTS
            idx_array = []
            pairs2 = list(combinations(np.arange(3),2))
            for pair_ in pairs2:
                #print ("pair_", pair_)
                diffs = np.sqrt((traces[pair_[0]][:,0]-traces[pair_[1]][:,0])**2+
                                (traces[pair_[0]][:,1]-traces[pair_[1]][:,1])**2)
                idx_temp = np.where(diffs<distance_threshold_pixels)[0]
                #print ("pair_: ", pair_, idx_temp.shape)
                idx_array.append(idx_temp)

            # COMPUTE OVERLAP
            idx3 = np.unique(np.hstack(set.intersection(*[set(x) for x in idx_array])))
            #print (pair, "IDX3 4: ", len(idx3))
            multi_animal_durations.append(len(idx3)/25)

            # SAVE TIMES OF INTERACTION
            triples_interaction_times.append(idx3)

        #print ("")

        #################################
        # COMPUTE 4 ANIMAL INTERACTIONS
        pairs = list(combinations(animals,4))
        quad_interaction_times = []
        for ctr_x, pair in enumerate(pairs):
            names_multi_animal.append(names[pair[0]]+"\n"+ names[pair[1]]+"\n"+names[pair[2]]+"\n"+names[pair[3]])
            #names_multi_animal.append(names[pair[0]]+" - "+ names[pair[1]]+" - "+names[pair[2]]+"  - "+names[pair[3]])
            #print (pair)
            traces = []
            for k in pair:
                traces1=traces_23hrs[k].copy()
                traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((smoothing_window,))/smoothing_window, mode='same')
                traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((smoothing_window,))/smoothing_window, mode='same')
                traces1 = traces1
                traces.append(traces1)

            # loop over all combinations and get distances
            pairs2 = list(combinations(animals,2))
            idx_array = []
            for pair2 in pairs2:
                diffs = np.sqrt((traces[pair2[0]][:,0]-traces[pair2[1]][:,0])**2+
                                 (traces[pair2[0]][:,1]-traces[pair2[1]][:,1])**2)
                idx_temp = np.where(diffs<distance_threshold_pixels)[0]
                #print ("pair2: ", pair2, idx_temp.shape)
                idx_array.append(idx_temp)

            # COMPUTE TOTAL TIME TOGETHER
            #print ("idx_array: ", idx_array)
            idx3 = set.intersection(*[set(x) for x in idx_array])
            #print ("4 animals 4: ", len(idx3))
            multi_animal_durations.append(len(idx3)/25)

            # SAVE TIMES OF INTERACTION
            quad_interaction_times.append(list(idx3)  )

            ctr+=1

        pair_interaction_times2.append(pair_interaction_times)
        triples_interaction_time2.append(triples_interaction_times)
        quad_interaction_times2.append(quad_interaction_times)
        multi_animal_durations2.append(multi_animal_durations)
        interactions2.append(interactions)
        duration_matrix2.append(durations_matrix)

    return (pair_interaction_times2,triples_interaction_time2, quad_interaction_times2, multi_animal_durations2, names_multi_animal, interactions2, duration_matrix2)
#
# def plot_multi_interactions(locs):
#
#     # PLOT MULTI ANIMAL DURATIONS
#     fig=plt.figure()
#     ax3=plt.subplot(1,1,1)
#     handles, labels = ax3.get_legend_handles_labels()
#
#     import matplotlib.patches as mpatches
#
#     for k in range(len(names)):
#         plt.bar(k, multi_animal_durations[k], 0.9, color=clrs[k])
#
#         patch = mpatches.Patch(color=clrs[k], label=names_multi_animal[k])
#
#         # handles is a list, so append manual patch
#         handles.append(patch)
#
#     # plot the legend
#     plt.legend(handles=handles,fontsize=20)
#     plt.xticks([])
#     plt.tick_params(labelsize=20)
#     plt.xlim(-0.5,4.5)
#     plt.ylabel("time together (sec)",fontsize=20)
#
#     #ax3.set_title("Duration Multi-animal interactions",fontsize=15)
#     #################################################
#     ######### PLOT INTERACTIONS PAIRWISE ############
#     #################################################
#     fig=plt.figure()
#     ax1=plt.subplot(1,2,1)
#
#     im = plt.imshow(interactions, cmap='viridis')
#     #print(locs.shape[0], x_ticks)
#     plt.xticks(np.arange(locs.shape[0]), x_ticks,rotation=15)
#     plt.yticks(np.arange(locs.shape[0]), x_ticks,rotation=75)
#     plt.tick_params(labelsize=20)
#     ax1.set_title("# of interactions",fontsize=20)
#     plt.tick_params(labelsize=20)
#
#     cbar = plt.colorbar()
#     cbar.set_label("# interactions", fontsize=20)
#
#     # Loop over data dimensions and create text annotations.
#     if False:
#         for i in range(interactions.shape[0]):
#             for j in range(interactions.shape[1]):
#                 if np.isnan(durations_matrix[i, j])==True:
#                     continue
#                 text = ax1.text(j, i, interactions[i, j],
#                                ha="center", va="center", color=text_clr,
#                                fontsize=13)
#
#     ##############################################
#     ############ PLOT PAIRWISE DURATIONS ########
#     #################################################
#     ax2=plt.subplot(1,2,2)
#     im = plt.imshow(durations_matrix, cmap='viridis')
#
#     x_ticks=['female','male','pup1','pup2']
#     plt.xticks(np.arange(locs.shape[0]), x_ticks,rotation=15)
#     plt.yticks(np.arange(locs.shape[0]), x_ticks,rotation=75)
#     plt.tick_params(labelsize=20)
#
#     cbar = plt.colorbar()
#     cbar.set_label("time together (sec)", fontsize=20)
#
#
#     # Loop over data dimensions and create text annotations.
#     if False:
#         for i in range(durations_matrix.shape[0]):
#             for j in range(durations_matrix.shape[1]):
#                 if np.isnan(durations_matrix[i, j])==True:
#                     continue
#                 text = ax2.text(j, i, durations_matrix[i, j],
#                                ha="center", va="center", color=text_clr,
#                                fontsize=13)
#
#     ax2.set_title('time together (sec)',fontsize=20)
#
#     plt.suptitle(suptitle, fontsize=15)
#
#     plt.show()


def plot_pairwise_interactions(locs, suptitle):

    x_ticks=['female','male','pup1','pup2']
    text_clr = 'red'

    distance_threshold = 250 # # of pixels away assume 1 pixel ~= 0.5mm -> 20cm
    time_window = 5*25 # no of seconds to consider
    smoothing_window = 3
    min_distance = 25 # number of frames window

    traces_23hrs = locs
    #print (traces_23hrs[0].shape)

    # COMPUTE PAIRWISE INTERACTIONS
    animals=np.arange(locs.shape[0])
    interactions = np.zeros((animals.shape[0],animals.shape[0]),'int32') + np.nan
    durations_matrix = np.zeros((animals.shape[0], animals.shape[0]),'int32') + np.nan

    # loop over all pairwise combinations
    pair_interaction_times = []
    pairs1 = list(combinations(animals,2))
    for ctr_x, pair in enumerate(pairs1):
        traces = []

        # smooth out traces;
        for k in pair:
            traces1=traces_23hrs[k].copy()
            traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((smoothing_window,))/smoothing_window, mode='same')
            traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((smoothing_window,))/smoothing_window, mode='same')
            traces1 = traces1
            traces.append(traces1)

        # COMPUTE PAIRWISE DISTANCES AND NEARBY TIMES POINTS
        idx_array = []
        diffs = np.sqrt((traces[0][:,0]-traces[1][:,0])**2+
                        (traces[0][:,1]-traces[1][:,1])**2)
        idx = np.where(diffs<distance_threshold)[0]

        # COMPUTE TOTAL TIME TOGETHER
        #print ("Pairwise: ", pair, idx.shape)
        durations_matrix[pair[0],pair[1]]=idx.shape[0]/25

        # COMPUTE # OF INTERACTIONS;
        diffs_idx = idx[1:]-idx[:-1]
        idx2 = np.where(diffs_idx>time_window)[0]
        interactions[pair[0],pair[1]]=idx2.shape[0]

        # SAVE TIMES OF INTERACTION
        pair_interaction_times.append(idx)

    # SYMMETRIZE MATRICES
    for k in range(durations_matrix.shape[0]):
        for j in range(durations_matrix.shape[1]):
            if np.isnan(durations_matrix[k,j])==False:
                durations_matrix[j,k]=durations_matrix[k,j]
                interactions[j,k]=interactions[k,j]


    #############################
    # COMPUTE TRIPLE INTERACTIONS

    multi_animal_durations = [] #np.zeros((animals.shape[0]+4),'int32') + np.nan
    names_multi_animal =[]
    ctr=0

    pairs3 = list(combinations(animals,3))
    triples_interaction_time = []
    for ctr_x, pair in enumerate(pairs3):
        names_multi_animal.append(names[pair[0]]+"\n"+ names[pair[1]]+"\n"+names[pair[2]]+"\n")
        #names_multi_animal.append(names[pair[0]]+" - "+ names[pair[1]]+" - "+names[pair[2]])
        traces = []

        # COMPUTE LOCATIONS
        for k in pair:
            traces1=traces_23hrs[k].copy()
            traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((smoothing_window,))/smoothing_window, mode='same')
            traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((smoothing_window,))/smoothing_window, mode='same')
            traces1 = traces1
            traces.append(traces1)

        # COMPUTE PAIRWISE DISTANCES AND NEARBY TIMES POINTS
        idx_array = []
        pairs2 = list(combinations(np.arange(3),2))
        for pair_ in pairs2:
            #print ("pair_", pair_)
            diffs = np.sqrt((traces[pair_[0]][:,0]-traces[pair_[1]][:,0])**2+
                            (traces[pair_[0]][:,1]-traces[pair_[1]][:,1])**2)
            idx_temp = np.where(diffs<distance_threshold)[0]
            #print ("pair_: ", pair_, idx_temp.shape)
            idx_array.append(idx_temp)

        # COMPUTE OVERLAP
        idx3 = np.unique(np.hstack(set.intersection(*[set(x) for x in idx_array])))
        #print (pair, "IDX3 4: ", len(idx3))
        multi_animal_durations.append(len(idx3)/25)

        # SAVE TIMES OF INTERACTION
        triples_interaction_time.append(idx3)

    print ("")

    #################################
    # COMPUTE 4 ANIMAL INTERACTIONS
    pairs = list(combinations(animals,4))
    quad_interaction_times = []
    for ctr_x, pair in enumerate(pairs):
        names_multi_animal.append(names[pair[0]]+"\n"+ names[pair[1]]+"\n"+names[pair[2]]+"\n"+names[pair[3]])
        #names_multi_animal.append(names[pair[0]]+" - "+ names[pair[1]]+" - "+names[pair[2]]+"  - "+names[pair[3]])
        #print (pair)
        traces = []
        for k in pair:
            traces1=traces_23hrs[k].copy()
            traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((smoothing_window,))/smoothing_window, mode='same')
            traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((smoothing_window,))/smoothing_window, mode='same')
            traces1 = traces1
            traces.append(traces1)

        # loop over all combinations and get distances
        pairs2 = list(combinations(animals,2))
        idx_array = []
        for pair2 in pairs2:
            diffs = np.sqrt((traces[pair2[0]][:,0]-traces[pair2[1]][:,0])**2+
                             (traces[pair2[0]][:,1]-traces[pair2[1]][:,1])**2)
            idx_temp = np.where(diffs<distance_threshold)[0]
            #print ("pair2: ", pair2, idx_temp.shape)
            idx_array.append(idx_temp)

        # COMPUTE TOTAL TIME TOGETHER
        #print ("idx_array: ", idx_array)
        idx3 = set.intersection(*[set(x) for x in idx_array])
        #print ("4 animals 4: ", len(idx3))
        multi_animal_durations.append(len(idx3)/25)

        # SAVE TIMES OF INTERACTION
        quad_interaction_times.append(list(idx3)  )

        ctr+=1
    #print ("")


    #####################################
    ######### PLOT RESULTS  ########
    #####################################



    # PLOT MULTI ANIMAL DURATIONS
    fig=plt.figure()
    ax3=plt.subplot(1,1,1)
    handles, labels = ax3.get_legend_handles_labels()

    import matplotlib.patches as mpatches

    for k in range(len(names)):
        plt.bar(k, multi_animal_durations[k], 0.9, color=clrs[k])

        patch = mpatches.Patch(color=clrs[k], label=names_multi_animal[k])

        # handles is a list, so append manual patch
        handles.append(patch)

    # plot the legend
    plt.legend(handles=handles,fontsize=20)
    plt.xticks([])
    plt.tick_params(labelsize=20)
    plt.xlim(-0.5,4.5)
    plt.ylabel("time together (sec)",fontsize=20)

    #ax3.set_title("Duration Multi-animal interactions",fontsize=15)
    #################################################
    ######### PLOT INTERACTIONS PAIRWISE ############
    #################################################
    fig=plt.figure()
    ax1=plt.subplot(1,2,1)

    im = plt.imshow(interactions, cmap='viridis')
    #print(locs.shape[0], x_ticks)
    plt.xticks(np.arange(locs.shape[0]), x_ticks,rotation=15)
    plt.yticks(np.arange(locs.shape[0]), x_ticks,rotation=75)
    plt.tick_params(labelsize=20)
    ax1.set_title("# of interactions",fontsize=20)
    plt.tick_params(labelsize=20)

    cbar = plt.colorbar()
    cbar.set_label("# interactions", fontsize=20)

    # Loop over data dimensions and create text annotations.
    if False:
        for i in range(interactions.shape[0]):
            for j in range(interactions.shape[1]):
                if np.isnan(durations_matrix[i, j])==True:
                    continue
                text = ax1.text(j, i, interactions[i, j],
                               ha="center", va="center", color=text_clr,
                               fontsize=13)

    ##############################################
    ############ PLOT PAIRWISE DURATIONS ########
    #################################################
    ax2=plt.subplot(1,2,2)
    im = plt.imshow(durations_matrix, cmap='viridis')

    x_ticks=['female','male','pup1','pup2']
    plt.xticks(np.arange(locs.shape[0]), x_ticks,rotation=15)
    plt.yticks(np.arange(locs.shape[0]), x_ticks,rotation=75)
    plt.tick_params(labelsize=20)

    cbar = plt.colorbar()
    cbar.set_label("time together (sec)", fontsize=20)


    # Loop over data dimensions and create text annotations.
    if False:
        for i in range(durations_matrix.shape[0]):
            for j in range(durations_matrix.shape[1]):
                if np.isnan(durations_matrix[i, j])==True:
                    continue
                text = ax2.text(j, i, durations_matrix[i, j],
                               ha="center", va="center", color=text_clr,
                               fontsize=13)

    ax2.set_title('time together (sec)',fontsize=20)

    plt.suptitle(suptitle, fontsize=15)

    plt.show()

    return pair_interaction_times,triples_interaction_time, quad_interaction_times


def ethogram_social_huddling(pair_interaction_times,
                             triples_interaction_time,
                             quad_interaction_times):


    clrs = ['blue','red','cyan','green','magenta', 'black']

    fig=plt.figure()
    ax=plt.subplot(111)
    span = 0.09
    for k in range(len(pair_interaction_times)):

        # multiple lines all full height
        temp = pair_interaction_times[k]
        chunk_start = []
        chunk_end = []
        chunk_start.append(temp[0])
        for p in range(1, temp.shape[0],1):
            if temp[p]-temp[p-1]>1:
                #print (temp[p], temp[p-1])
                chunk_end.append(temp[p-1])
                chunk_start.append(temp[p])
        chunk_end.append(temp[-1])

        chunk_start=np.array(chunk_start)/25.
        chunk_end=np.array(chunk_end)/25.

        for p in range(len(chunk_start)):
            ax.axvspan(chunk_start[p], chunk_end[p], ymin=k*span, ymax=k*span+span, alpha=0.5, color=clrs[0])

    # Interactions triples
    start1 = k*span+span
    for k in trange(len(triples_interaction_time)):

        # multiple lines all full height
        temp = np.unique(triples_interaction_time[k])
        #print (temp)
        chunk_start = []
        chunk_end = []
        chunk_start.append(temp[0])
        for p in range(1, temp.shape[0],1):
            if temp[p]-temp[p-1]>1:
                #print (temp[p], temp[p-1])
                chunk_end.append(temp[p-1])
                chunk_start.append(temp[p])
        chunk_end.append(temp[-1])

        chunk_start=np.array(chunk_start)/25.
        chunk_end=np.array(chunk_end)/25.

    #     print ("# chunks: ", len(chunk_start), clrs[k])
    #     print (chunk_start[:3], chunk_end[:3])

        for p in range(len(chunk_start)):
            ax.axvspan(chunk_start[p], chunk_end[p], ymin=start1+k*span, ymax=start1+k*span+span, alpha=0.5, color=clrs[1])


    # Interactions triples
    start2 = start1+k*span+span
    for k in trange(len(quad_interaction_times)):

        # multiple lines all full height
        temp = np.unique(quad_interaction_times[k])
        chunk_start = []
        chunk_end = []
        chunk_start.append(temp[0])
        for p in range(1, temp.shape[0],1):
            if temp[p]-temp[p-1]>1:
                #print (temp[p], temp[p-1])
                chunk_end.append(temp[p-1])
                chunk_start.append(temp[p])
        chunk_end.append(temp[-1])

        chunk_start=np.array(chunk_start)/25.
        chunk_end=np.array(chunk_end)/25.

    #     print ("# chunks: ", len(chunk_start), clrs[k])
    #     print (chunk_start[:3], chunk_end[:3])

        for p in trange(len(chunk_start)):
            ax.axvspan(chunk_start[p], chunk_end[p], ymin=start2+k*span, ymax=start2+k*span+span, alpha=0.5, color=clrs[3])

    plt.xlim(0,3600.)
    plt.ylim(0,1)

    handles=[]
    patch = mpatches.Patch(color=clrs[0], label='pairs')
    handles.append(patch)
    patch = mpatches.Patch(color=clrs[1], label='triples')
    handles.append(patch)
    patch = mpatches.Patch(color=clrs[3], label='quads')
    handles.append(patch)

    # plot the legend
    plt.legend(handles=handles,fontsize=20)
    plt.tick_params(labelsize=20)
    plt.xlabel("Time (sec)",fontsize=20)
    plt.yticks([])

    plt.show()






