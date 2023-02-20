import numpy as np
import os
from tqdm import trange
import parmap
import glob

from tqdm import tqdm
import sleap
import parmap
from itertools import combinations

#
import sys
sys.path.append("/home/cat/code/gerbil/utils") # go to parent dir

from track import track
from convert import convert
from ethogram import ethogram

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd

#
class CohortProcessor():

    #
    def __init__(self):

        self.cohort_start_date = None
        self.cohort_end_date = None

        self.current_session = None

        #self.list_methods()

    #
    def process_feature_track(self, fname_slp):

        if os.path.exists(fname_slp):

            fname_spine_out = fname_slp.replace('.slp',"_spine.npy")
            if os.path.exists(fname_spine_out):
                return

            t = track.Track(fname_slp)
            t.track_type = 'features'

            ###### parameters for computing body centroid #######
            t.use_dynamic_centroid = True   # True: alg. serches for the first non-nan value in this body order [2,3,1,0,4,5]
                                                 # - advantage: much more robust to lost features
                                                 # False: we fix the centroid to a specific body part
                                                 # - advantage less jitter for some applications
            t.centroid_body_id = [2]         # if centroid flag is False; we use this body part instead

            ##### run track fixer #######
            t.fix_all_tracks()

            ##### join spatially close but temporally distant chunks #####
            if False:
                #
                t.memory_interpolate_tracks_spine()

            ##### save the fixed spines will overwrite the previous/defatul spine values####
            t.save_centroid()


    def process_huddle_track(self, fname_slp,
                             fix_track_flag,
                             interpolate_flag):


        #fname = '/media/cat/256GB/dan/huddles/p26_huddles.npy'

        #
        t = track.Track(fname_slp)
        t.fix_track_flag = fix_track_flag
        t.interpolate_flag = interpolate_flag

        #############################################
        ############# RUN TRACK FIXER ###############
        #############################################
        max_jump_allowed = 50,              # maximum distance that a gerbil can travel in 1 frame
        max_dist_to_join = 50,              # maximum distnace between 2 chunks that can safely be merged
        min_chunk_len = 25                  # shortest duration of

        t.fix_huddles(max_jump_allowed,
                          max_dist_to_join,
                          min_chunk_len)

        ##################################################
        ################# RUN HUDDLE FIXER ###############
        ##################################################

        #
        fps = 24
        t.max_distance_huddle = 100                   # how far away we can merge huddles together (pixels)
        t.max_time_to_join_huddle = fps*120            # how far in time can we merge huddle chunks (seconds x frames)
        t.min_huddle_time = 120*fps                     # minimum huddle duration in seconds
        t.memory_interpolate_huddle()

        ##################################################
        ############## SAVE FIXED TRACKS #################
        ##################################################

        ##### save the fixed spines will overwrite the previous/defatul spine values####
        t.save_centroid()

        #
        t.save_updated_huddle_tracks()

            #
    def preprocess_huddle_tracks(self):

        #
        self.root_dir_features = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                              'huddles')

        #
        fnames_slp = []
        for k in range(self.fnames_slp.shape[0]):
            fname = os.path.join(self.root_dir_features,self.fnames_slp[k][0]).replace(
                                    '.mp4','_'+self.NN_type[k][0])+"_huddle.slp"
            if os.path.exists(fname):
                fnames_slp.append(fname)


        #
        if self.parallel:
            parmap.map(self.process_huddle_track,
                       fnames_slp,
                       self.fix_track_flag,
                       self.interpolate_flag,
                       pm_processes=self.n_cores,
                       pm_pbar = True)
        else:
            for fname_slp in fnames_slp:
                self.process_huddle_track(fname_slp)

        #


    #
    def preprocess_feature_tracks(self):

        #
        self.root_dir_features = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                              'features')

        #
        fnames_slp = []
        for k in range(self.fnames_slp.shape[0]):
            fname = os.path.join(self.root_dir_features,self.fnames_slp[k][0]).replace(
                                    '.mp4','_'+self.NN_type[k][0])+".slp"
            if os.path.exists(fname):
                fnames_slp.append(fname)


        #
        if self.parallel:
            parmap.map(self.process_feature_track,
                   fnames_slp,
                   pm_processes=self.n_cores,
                   pm_pbar = True)
        else:
            for fname_slp in fnames_slp:
                self.process_track(fname_slp)

        #


    #
    def load_database(self):


        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns',504)
        pd.set_option('display.width',1000)

        df = pd.read_excel(self.fname_spreadsheet, engine='openpyxl')
        df.style.applymap(lambda x:'white-space:nowrap')
        print ("DF: ", df.head() )

        ###################################################################
        ########## SAVE FILENAMES WITH 6 ANIMALS AND NN TYPES #############
        ###################################################################
        #
        print ("Loading only recordings with 6 animals...")
        self.n_gerbils = df.loc[:,'# of Gerbils']
        #print ("# of gerbils: ", self.n_gerbils)

        #
        self.PDays = df.loc[:,'Dev Day']

        #
        self.Start_times = df.loc[:,'Start time']

        #
        idx = np.where(self.n_gerbils==6)[0]
        print (" ... total # : ", idx.shape[0], " / ", self.n_gerbils.shape[0])

        fnames = df.loc[:,'Filename']
        self.fnames_slp = np.vstack(fnames.iloc[idx].tolist())

        #
        self.NN_type = np.vstack(df.loc[:,'NN Type'].iloc[idx].tolist())


    #
    def list_methods(self):
        method_list = [func for func in dir(self)
                       if callable(getattr(self, func)) and
                       not func.startswith("__")]

        print ("Available methods: ",
               *method_list,
               sep='\n  ')



    def get_pairwise_interaction_time(self, a1, a2):

        res=[]
        for k in trange(0,991,1):
            self.track_id = k
            track = self.load_single_feature_spines()

            # if track is missing, skip it
            if track is None:
                res.append(np.zeros((6,6))[a1,a2])
                continue

            #
            self.symmetric_matrices=False
            self.plotting=False
            temp = self.compute_pairwise_interactions(track)
            try:
                temp = temp[a1,a2]
            except:
                print ("Missing animal track...")
                temp = np.zeros((6,6))[a1,a2]

            res.append(temp)

        res = np.array(res)
        print ("res: ", res.shape)

        self.res = res


    def format_behavior(self):

        #
        self.data = []
        for k in range(self.PDays.shape[0]):

            PDay = self.PDays[k]
            time = self.Start_times[k]
            self.data.append([int(PDay[1:]), time.hour, self.res[k]])

        #
        self.data = np.vstack(self.data)
        print (self.data)

        # compute average per hour
        self.data_ave = []
        s = []
        s.append(self.data[0,2])
        for k in range(0,self.data.shape[0]-1,1):
            if self.data[k,1]==self.data[k+1,1]:
                s.append(self.data[k+1,2])
            else:
                temp = self.data[k]
                temp[2] = np.mean(s)
                self.data_ave.append(temp)
                s=[]
        self.data = np.vstack(self.data_ave)

    def list_recordings(self):



        pass

    def compress_video(self):
        pass


    def load_single_feature_spines(self):

        fname = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                     'features',
                                     self.fnames_slp[self.track_id][0].replace('.mp4','_'+self.NN_type[self.track_id][0]+".slp"))
        #
        t = track.Track(fname)
        t.fname=fname
        #
        if os.path.exists(fname)==False:
            return None

        #
        t.get_track_spine_centers()

        return t

    def process_time(self):

        ''' Function that uses filename to generate metadata

            - generates Universal Dev Time from timestamp
            - generates Day vs. Nighttime label from timestamp
            - identifies the correct NN for the file

        '''

        #
        print ("current session: ", self.current_session)



    def track_video(self):
        ''' Function that takes as input filename

        :return:
        '''

        pass



    def detect_audio(self):
        pass

    #
    def compute_pairwise_interactions(self,track):

        #
        x_ticks=['female','male','pup1','pup2','pup3','pup4']


        self.distance_threshold = 250 # # of pixels away assume 1 pixel ~= 0.5mm -> 20cm
        time_window = 1*25 # no of seconds to consider
        self.smoothing_window = 3
        min_distance = 25 # number of frames window
        fps=24

        locs = track.tracks_spine.transpose(1,0,2)
        traces_23hrs = locs

        # COMPUTE PAIRWISE INTERACTIONS
        animals=np.arange(locs.shape[0])
        interactions = np.zeros((animals.shape[0],animals.shape[0]),'int32') + np.nan
        durations_matrix = np.zeros((animals.shape[0], animals.shape[0]),'int32') + np.nan

        ########################################################
        ########################################################
        ########################################################
        # loop over all pairwise combinations
        pair_interaction_times = []
        pairs1 = list(combinations(animals,2))
        #for pair in tqdm(pairs1, desc='pairwise computation'):
        for pair in pairs1:
            traces = []

            # smooth out traces;
            for k in pair:
                traces1=traces_23hrs[k].copy()
                traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((self.smoothing_window,))/self.smoothing_window, mode='same')
                traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((self.smoothing_window,))/self.smoothing_window, mode='same')
                traces1 = traces1
                traces.append(traces1)

            # COMPUTE PAIRWISE DISTANCES AND NEARBY TIMES POINTS
            idx_array = []
            diffs = np.sqrt((traces[0][:,0]-traces[1][:,0])**2+
                            (traces[0][:,1]-traces[1][:,1])**2)
            idx = np.where(diffs<self.distance_threshold)[0]

            # COMPUTE TOTAL TIME TOGETHER
            #print ("Pairwise: ", pair, idx.shape)
            durations_matrix[pair[0],pair[1]]=idx.shape[0]/fps

            # COMPUTE # OF INTERACTIONS;
            diffs_idx = idx[1:]-idx[:-1]
            idx2 = np.where(diffs_idx>5)[0]
            interactions[pair[0],pair[1]]=idx2.shape[0]

            # SAVE TIMES OF INTERACTION
            pair_interaction_times.append(idx)

        # SYMMETRIZE MATRICES
        if self.symmetric_matrices:
            for k in range(durations_matrix.shape[0]):
                for j in range(durations_matrix.shape[1]):
                    if np.isnan(durations_matrix[k,j])==False:
                        durations_matrix[j,k]=durations_matrix[k,j]
                        interactions[j,k]=interactions[k,j]


        # #################################################
        # ######### PLOT INTERACTIONS PAIRWISE ############
        # #################################################
        dur_matrix_percentage = durations_matrix/(locs.shape[1]/fps)*100

        if self.plotting:
            plt.figure()
            labelsize=14
            ax2=plt.subplot(1,1,1)
            im = plt.imshow(durations_matrix, cmap='viridis')

            #x_ticks=['female','male','pup1','pup2']
            plt.xticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=15)
            plt.yticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=75)
            plt.tick_params(labelsize=labelsize)

            cbar = plt.colorbar()
            cbar.set_label("time together (sec)", fontsize=labelsize)

            ##############################################
            ############ PLOT PAIRWISE DURATIONS ########
            #################################################
            plt.figure()
            ax2=plt.subplot(1,1,1)

            dur_matrix_percentage = durations_matrix/(locs.shape[1]/fps)*100
            plt.imshow(dur_matrix_percentage, cmap='viridis')

            #x_ticks=['female','male','pup1','pup2']
            plt.xticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=15)
            plt.yticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=75)
            plt.tick_params(labelsize=labelsize)

            cbar = plt.colorbar()
            cbar.set_label("time together (% of total recording)", fontsize=labelsize)
            #

            #
            plt.suptitle(os.path.split(track.fname)[1])


            plt.show()

        return dur_matrix_percentage
