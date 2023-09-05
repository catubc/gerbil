import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import os
from tqdm import trange
import parmap
import glob
from sklearn.decomposition import PCA

import pandas as pd

from tqdm import tqdm

from track import Track

import pandas as pd

#
def plot_me(temp, clr, clr_map, offset):
    
    #plt.scatter(trace1_array[k][0][0],
    #temp = temp - offset
    temp = temp
    plt.plot(temp[:,0],
             temp[:,1],
             clr,
             linewidth=2,
            )
    plt.xlim(0,900)
    plt.ylim(0,700)
    
    plt.scatter(temp[:,0],
         temp[:,1],
                s=100,
         c=np.arange(temp.shape[0]),
         cmap=clr_map,
        )
    
#
def smooth_traces(trace):
    max_jump = 20
    diffs = trace[1:] - trace[:-1]
    dists = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)
    idx = np.where(dists>max_jump)[0]
    trace[idx] = trace[idx+1]

    return trace


def generate_track_interactions_all(id_,
                                   id_pairs,
                                   fnames,
                                   root_dir,
                                   track,
                                   data):

    #
    clr1 = 'red'; clr1_map = "Reds"
    clr2 = 'blue'; clr2_map = "Blues"
    clrs = ['black','blue','red','green','magenta','cyan','brown','pink','orange']

    tracks_names = ['female','male','pup1','pup2','pup3','pup4']

    plt.figure()

    #id_ = 0
    id_pair = id_pairs[id_]
    print ("id pair: ", id_pair)

    #
    times = data[:, id_*2:id_*2+2]

    trace1_array = []
    trace2_array = []
    for ctr,fname in enumerate(fnames):
        #
        track = np.load(os.path.join(root_dir, 
                                   fname+'_compressed_Day_spine.npy'))

        #
        time_seg = times[ctr]
        if time_seg[0]==0:

            trace1_array.append([])
            trace2_array.append([])
            continue

        #
        
        trace1 = track[int(time_seg[0]):int(time_seg[1])+1,id_pair[0]]
    
        trace2 = track[int(time_seg[0]):int(time_seg[1])+1,id_pair[1]]

        # fix nans
        if np.isnan(trace1).sum()>0:
            # fix nans forward
            for k in range(1,trace1.shape[0],1):
                if np.isnan(trace1[k]).sum()>0:
                    trace1[k]=trace1[k-1]

            # fix nans backwards
            for k in range(trace1.shape[0]-2,-1,-1):
                if np.isnan(trace1[k]).sum()>0:
                    trace1[k]=trace1[k+1]

            if np.isnan(trace1).sum()>0:
                print (id_, ctr, "has nans and couldn't fix them", trace1)

        # smooth traces


        #
        if True:
            trace1 = smooth_traces(trace1)
            trace2 = smooth_traces(trace2)

        #
        trace1_array.append(trace1)
        trace2_array.append(trace2)

    ctr3 = 1
    for k in range(len(trace1_array)):

        #
        temp = trace1_array[k]
        if len(temp)>0:
            ax=plt.subplot(5,5,ctr3)
            offset = temp[0]

            #
            plot_me(temp, clr1, clr1_map, offset)

            #
            temp = trace2_array[k]
            #offset = temp[0]
            plot_me(temp, clr2, clr2_map, offset)
            ctr3+=1
            
            # Add grid
            plt.xticks(np.arange(0, 900, 100))
            plt.yticks(np.arange(0, 700, 100))
            plt.grid(True)
        
            plt.title(fnames[k] + ":  " + str(times[k][0])+ " to " + str(times[k][1]),fontsize=6)

        plt.suptitle(tracks_names[id_pair[0]]+ " --> "+tracks_names[id_pair[1]])

    #plt.suptitle("All labeled approaches "+ )
        #plt.show()
        plt.savefig(f"/mnt/b3a68699-495d-4ebb-9ab1-ac74f11c68c5/simba/chasing2/plots/plot_{id_pair[0]}_{id_pair[1]}_all.png") 
        
def generate_track_interactions(id_,
                               id_pairs,
                               fnames,
                               root_dir,
                               track,
                               data):

    #
    clr1 = 'red'; clr1_map = "Reds"
    clr2 = 'blue'; clr2_map = "Blues"
    clrs = ['black','blue','red','green','magenta','cyan','brown','pink','orange']

    tracks_names = ['female','male','pup1','pup2','pup3','pup4']

    #id_ = 0
    id_pair = id_pairs[id_]
    print ("id pair: ", id_pair)

    #
    times = data[:, id_*2:id_*2+2]

    trace1_array = []
    trace2_array = []
    for ctr,fname in enumerate(fnames):
        #
        track = np.load(os.path.join(root_dir, 
                                   fname+'_compressed_Day_spine.npy'))

        #
        time_seg = times[ctr]
        if time_seg[0]==0:

            trace1_array.append([])
            trace2_array.append([])
            continue

        #
        
        trace1 = track[int(time_seg[0]):int(time_seg[1])+1,id_pair[0]]
    
        trace2 = track[int(time_seg[0]):int(time_seg[1])+1,id_pair[1]]

        # fix nans
        if np.isnan(trace1).sum()>0:
            # fix nans forward
            for k in range(1,trace1.shape[0],1):
                if np.isnan(trace1[k]).sum()>0:
                    trace1[k]=trace1[k-1]

            # fix nans backwards
            for k in range(trace1.shape[0]-2,-1,-1):
                if np.isnan(trace1[k]).sum()>0:
                    trace1[k]=trace1[k+1]

            if np.isnan(trace1).sum()>0:
                print (id_, ctr, "has nans and couldn't fix them", trace1)

        # smooth traces
        if True:
            trace1 = smooth_traces(trace1)
            trace2 = smooth_traces(trace2)

        #
        trace1_array.append(trace1)
        trace2_array.append(trace2)

    for k in range(len(trace1_array)):
        plt.figure()  # Move plt.figure() into the loop
        #
        temp = trace1_array[k]
        if len(temp)>0:
            offset = temp[0]

            #
            plot_me(temp, clr1, clr1_map, offset)

            #
            temp = trace2_array[k]
            #offset = temp[0]
            plot_me(temp, clr2, clr2_map, offset)
            
            # Add grid
            plt.xticks(np.arange(0, 900, 100))
            plt.yticks(np.arange(0, 700, 100))
            plt.grid(True)
        
            #
            plt.title(tracks_names[id_pair[0]]+ " --> "+tracks_names[id_pair[1]]+"\n"+fnames[k] + ":  " + str(times[k][0])+ " to " + str(times[k][1]),fontsize=6)

            # Save each figure with a unique name
            plt.savefig(f"/mnt/b3a68699-495d-4ebb-9ab1-ac74f11c68c5/simba/chasing2/plots/plot_{fnames[k]}_{id_pair[0]}_{id_pair[1]}_{str(times[k][0])}_{str(times[k][1])}.png") 
            #plt.show()

        plt.close()  # Close the figure after saving to free up memory
