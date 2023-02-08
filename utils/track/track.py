
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import glob

import scipy
import scipy.spatial
from scipy import signal
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import parmap
import time
import numpy as np
import os
import cv2
from tqdm.auto import tqdm, trange

import sleap
import h5py
from scipy import signal
import pandas as pd
from sleap.skeleton import Node, Skeleton
from typing import Callable, Dict, Iterator, List, Optional, Type, Tuple

#
names=  ['female','male','pup1','pup2','pup3','pup4']

############################################
class Track():

    #
    def __init__(self, fname_slp):

        #
        self.verbose=False

        #
        self.recompute_spine_centres = False

        #
        self.fname_slp = fname_slp

        #
        self.slp = None

    #
    def load_slp(self):

        self.slp = sleap.load_file(self.fname_slp)

    #
    def slp_to_h5(self):

        fname_h5 = self.fname_slp[:-4] + ".h5"
        if self.slp is None:
            #print("... slp file not loaded, loading now...")
            self.load_slp()
            if self.verbose:
                print("... done loading slp")

        self.slp.export(fname_h5)

    def slp_to_npy(self):

        fname_h5 = self.fname_slp[:-4] + ".h5"
        if os.path.exists(fname_h5) == False:
            if self.verbose:
                print("... h5 file missing, converting now...")
            self.slp_to_h5()
            if self.verbose:
                print("... done loading h5")

        #
        hf = h5py.File(fname_h5, 'r')

        keys = hf.keys()
        group2 = hf.get('tracks')
        #print ("group2: ", group2)
        tracks = []
        for k in range(len(group2)):
            tracks.append(group2[k])

        tracks = np.array(tracks).transpose(3, 0, 2, 1)

        #
        fname_npy = self.fname_slp[:-4] + ".npy"
        np.save(fname_npy, tracks)

    #
    def load_tracks(self):

        #
        self.verbose=False
        fname_npy = self.fname_slp[:-4] + '.npy'
        if os.path.exists(fname_npy) == False:
            if self.verbose:
                print("... npy missing, converting...")
            self.slp_to_npy()
            if self.verbose:
                print("... done loading npy")

        # full feature tracks
        self.tracks = np.load(fname_npy)

        # copy of full feature tracks to be corrected
        self.tracks_fixed = self.tracks.copy()

        #
        self.tracks_centers = np.nanmean(
                                self.tracks,
                                axis=2)

        #
        self.get_track_spine_centers()

    #
    def bin_track_spines(self):
        #
        self.tracks_spine = (self.tracks_spine//self.bining_bin_size)*self.bining_bin_size+ self.bining_bin_size//2

    #
    def save_binned_centroid(self):

        fname_spine = self.fname_slp[:-4]+"_spine_binned.npy"
        np.save(fname_spine, self.tracks_spine)

    #
    def save_centroid(self):

        #print ("saving centroid")
        text = ''
        if self.fix_track_flag:
            text = text + "_fixed"
        if self.interpolate_flag:
            text = text + "_interpolated"

        self.fname_spine_saved = self.fname_slp[:-4]+"_spine"+text+".npy"
        np.save(self.fname_spine_saved, self.tracks_spine)

    #
    def get_track_spine_centers(self):
        '''  This function returns single locations per animal
            with a focus on spine2, spine3, spine1 etc...
        '''
        #
        fname_spine = self.fname_slp[:-4]+"_spine.npy"
        if os.path.exists(fname_spine)==False or self.recompute_spine_centres==True:

            ############### CHECK IF WORKING ON HUDDLE ###########
            # check if huddle type
            if self.track_type == 'huddle':
                self.tracks_spine = self.tracks.squeeze()[:,None]
                np.save(fname_spine, self.tracks_spine)
                return

            ################## COMPUTE SPINES FOR ANIMAL TRACKS ###############
            # initialize spine to nans
            self.tracks_spine = np.zeros((self.tracks.shape[0],
                                          self.tracks.shape[1],
                                          self.tracks.shape[3]))*0 + np.nan

            #if self.n_animals==4:
            #    ids = [6,7,5,8,4,3,2,1,0]  # centred on spine2

            if self.use_dynamic_centroid == True:
                ids = [2,3,1,0,4,5]
            else:
                ids = self.centroid_body_id

            # search over all times
            for n in range(self.tracks.shape[0]):

                # search over all animals
                for a in range(self.tracks.shape[1]):

                    # search over all body ids for the first non-nan
                    for id_ in ids:
                        if np.isnan(self.tracks[n,a,id_,0])==False:
                            # this overwrites the "spine" with the first non-zero value
                            self.tracks_spine[n,a]=self.tracks[n,a,id_]
                            break

            np.save(fname_spine, self.tracks_spine)
        else:
            self.tracks_spine = np.load(fname_spine)

    # cleanup tracks
    def clean_tracks_spine(self,
                    max_jump_allowed = 50,
                    max_dist_to_join = 50,
                    min_chunk_len=5):

        ''' Method to fix the large jumps, short orphaned segments,
            and interpolate across short distances

            Input:
            - track array for a single animal: [n_time_steps, 2], where the 2 is for x-y locations
            - max_jump_allowed: maximum number of pixels (in euclidean distance) a track is allowed to jump before being split
            - max_dist_to_join: when joining orphaned tracks, the largest distacne allowed between 2 track bouts
            - min_chunk_len = shortest chunk allowed (this is applied at the end after joining all the chunks back

            Output: fixed track

        '''

        #
        for a in range(self.tracks_spine.shape[1]):
            track_xy1 = self.tracks_spine[:,a].copy()

            ########################################
            ###########  Delete big jumps ##########
            ########################################
            #for k in trange(1,track_xy1.shape[0]-1,1, desc='setting big jumps to nans'):
            for k in range(1,track_xy1.shape[0]-1,1):
                if np.linalg.norm(track_xy1[k]-track_xy1[k-1])>max_jump_allowed:
                    track_xy1[k]=np.nan

            ########################################
            ##### Join segments that are close #####
            ########################################
            #
            last_chunk_xy = None

            # check if we start outside chunk or inside
            if np.isnan(track_xy1[0,0])==False:
                inside = True
            else:
                inside = False

            # interpolate between small bits
            # for k in trange(1,track_xy1.shape[0]-1,1,
            #                 position=0, leave=True,
            #                 desc='join segements that are close'):
            for k in range(1,track_xy1.shape[0]-1,1):
                if np.isnan(track_xy1[k,0]):
                    if inside:
                        inside=False
                        last_chunk_xy = track_xy1[k]
                        last_chunk_idx = k
                else:
                    if inside==False:
                        inside=True
                        new_chunk_xy = track_xy1[k]
                        new_chunk_idx = k
                        if last_chunk_xy is not None:
                            dist = np.linalg.norm(track_xy1[k]-track_xy1[k-1])
                            if dist<= max_dist_to_join:
                                track_xy1[last_chunk_idx:new_chunk_idx] = new_chunk_xy

            ########################################
            ##  Delete short segments left behind ##
            ########################################
            #
            chunk_start_xy = None

            # check if we start outside chunk or inside
            if np.isnan(track_xy1[0,0])==False:
                chunk_start_idx = 0
                inside = True
            else:
                inside = False

            # interpolate between small bits
            # for k in trange(1,track_xy1.shape[0]-1,1,
            #                 position=0, leave=True,
            #                 desc='interpolate betweensmall bits'):
            for k in range(1,track_xy1.shape[0]-1,1):
                if np.isnan(track_xy1[k,0]):
                    if inside:
                        inside=False
                        chunk_end_idx = k
                        if (chunk_end_idx - chunk_start_idx)<min_chunk_len:
                            track_xy1[chunk_start_idx:chunk_end_idx]= np.nan
                else:
                    if inside==False:
                        inside=True
                        chunk_start_idx = k

            #
            self.tracks_spine[:,a] = track_xy1.copy()

    #
    def fix_huddles(self,
                    max_jump_allowed = 50,              # maximum distance that a gerbil can travel in 1 frame
                    max_dist_to_join = 50,              # maximum distnace between 2 chunks that can safely be merged
                    min_chunk_len = 5):

        #
        self.animal_ids = [0]
        self.track_type = 'huddle'
        self.tracks_names = ['huddle']
        self.recompute_spine_centres = True
        self.verbose = True                         # gives additional printouts
        self.n_animals = 1     # number of animals
        self.filter_width = 10                      # this is the median filter width in frames; e.g. 10 ~=0.4 seconds
                                                     # higher values provide more stability, but less temporally precise locations
        # load the tracks
        self.use_dynamic_centroid = False
        self.load_tracks()

        #print ("spine tracks: ", self.tracks_spine.shape)
        ####################################################
        ### OPTIONAL - MEDIAN FILTER ALL TRACKS ############
        ####################################################
        #
        #self.filter_tracks_spines()

        ####################################################
        #### CLEANUP: REMOVE BIG JUMPS, SHORT SEGS #########
        ####################################################
        #
        # max_jump_allowed = 50              # maximum distance that a gerbil can travel in 1 frame
        # max_dist_to_join = 50              # maximum distnace between 2 chunks that can safely be merged
        # min_chunk_len = 5                  # minimum number of frames that a chunk has to survive for in order to be saved
        #print (" cleaning tracks spine")
        self.clean_tracks_spine(max_jump_allowed,
                                max_dist_to_join,
                                min_chunk_len)

        ####################################################
        ########### BREAK UP TACKS INTO CHUNKS #############
        ####################################################
        # makes scores based on .slp output? (to check)
        # TODO: is this even used!?
        self.get_scores()

        # uses track_spines to break up all the data into continuous chunks
        self.max_jump_single_frame = 30  # max distance in pixels (?) that an animal can move in a single frame
        self.make_tracks_chunks_huddles()

        #############################################
        ########## RERUN TRACK CLEANUP ##############
        #############################################
        #
        #print (" cleaning tracks spine")
        self.clean_tracks_spine(max_jump_allowed,
                                max_dist_to_join,
                                min_chunk_len)

        #
        self.tracks_spine_fixed = self.tracks_spine

    #
    def fix_all_tracks(self):

        #
        self.animal_ids = [0,1,2,3,4,5]
        self.tracks_names = ['female','male','pup1','pup2','pup3','pup4']
        self.recompute_spine_centres=True
        self.verbose = True                         # gives additional printouts
        self.n_animals = len(self.animal_ids)      # number of animals
        self.filter_width = 10                      # this is the median filter width in frames; e.g. 10 ~=0.4 seconds
                                                     # higher values provide more stability, but less temporally precise locations

        # load the tracks
        self.load_tracks()

        ####################################################
        ### OPTIONAL - MEDIAN FILTER ALL TRACKS ############
        ####################################################
        #
        #self.filter_tracks_spines()

        ####################################################
        #### CLEANUP: REMOVE BIG JUMPS, SHORT SEGS #########
        ####################################################
        #
        max_jump_allowed = 50              # maximum distance that a gerbil can travel in 1 frame
        max_dist_to_join = 50              # maximum distnace between 2 chunks that can safely be merged
        min_chunk_len = 5                  # minimum number of frames that a chunk has to survive for in order to be saved
        self.clean_tracks_spine(max_jump_allowed,
                                max_dist_to_join,
                                min_chunk_len)

        ####################################################
        ########### BREAK UP TACKS INTO CHUNKS #############
        ####################################################
        # makes scores based on .slp output? (to check)
        # TODO: is this even used!?
        self.get_scores()

        # uses track_spines to break up all the data into continuous chunks
        self.max_jump_single_frame = 30  # max distance in pixels (?) that an animal can move in a single frame
        self.make_tracks_chunks()

        ##############################################
        ############## FIX TRACKS PARAMS #############
        ##############################################
        self.time_threshold = 25       # window to search for nearest chunks, about 1sec seems fair...
        self.safe_chunk_length = 25    # chunks this long will not change id
        self.min_chunk_len = 4         # min length of chukn to be used for anchoring/correcting
        self.max_distance_merge = 75   # max pix diff allowed for merging when using model;
                                        # - not just for neighbouring frames
        self.memory_length = 25      # how many frames back is it ok to remember a prev animal
        self.verbose = False
        self.update_tracks = True

        # parameters for fixing track chunking
        self.max_time_automerge = 25      # time to automerget chunks from same animal ???
        self.max_dist_automerge = 25     # distance to auto merge chunks from same animal separated by single time skip

        #
        self.reallocate_track_segments()

        #############################################
        ########## RERUN TRACK CLEANUP ##############
        #############################################
        #
        self.clean_tracks_spine(max_jump_allowed,
                                max_dist_to_join,
                                min_chunk_len)



#
    def make_tracks_chunks_huddles(self):
        ''' Function finds temporally continuous tracks
            Time-continuous-tracks
             Function breaks up continuous tracks that are too far apart;
             important for when ids jump around way too much
            Loop over the tcrs and check if jumps are too high to re-break track
        '''

        #print ("... Making tracks chunks...")

        # break distances that are very large over single jumps
        # join by time
        self.tracks_chunks = []

        track = self.tracks_spine[:,0]

        in_segment = False
        if np.isnan(track[0,0])==False:
            in_segment = True
            start = 0
            end = None

        for k in range(1,track.shape[0],1):
            if np.isnan(track[k,0])==True:
                if in_segment==True:
                    self.tracks_chunks.append([start, k-1])
                    in_segment = False
            else:
                if in_segment==False:
                    start = k
                    end = None
                    in_segment=True
        if in_segment==True:
            self.tracks_chunks.append([start, k-1])

            #
    def make_tracks_chunks(self):
        ''' Function finds temporally continuous tracks
            Time-continuous-tracks
             Function breaks up continuous tracks that are too far apart;
             important for when ids jump around way too much
            Loop over the tcrs and check if jumps are too high to re-break track
        '''

        #print ("... Making tracks chunks...")

        # break distances that are very large over single jumps
        # join by time
        self.time_cont_tracks = []

        # loop over animals
        for a in range(self.tracks_spine.shape[1]):
            track = self.tracks_spine[:,a]

            # check temporal distance
            idx = np.where(np.isnan(track)==False)[0]
            diff = idx[1:]-idx[:-1]
            idx2 = np.where(diff>1)[0]


            # make track list
            self.time_cont_tracks.append([])

            # append first track
            if idx2.shape[0]>0:
                self.time_cont_tracks[a].append([0,idx[idx2[0]]])
            else:
                # only a single track found throughout the entire dataset;
                #print ("Single track animal: ", a, "idx: ", idx.shape, "  idx2: ", idx2.shape)
                self.time_cont_tracks[a].append([0,idx[-1]])

            # append all other tracks
            for i in range(1,idx2.shape[0],1):
                self.time_cont_tracks[a].append([idx[idx2[i-1]+1],
                                idx[idx2[i]]])

        # break by space
        self.tracks_chunks = []
        for a in range(len(self.time_cont_tracks)):
            self.tracks_chunks.append([])
            while len(self.time_cont_tracks[a])>0:  #for k in range(len(self.tcrs[a])):
                times = self.time_cont_tracks[a][0]
                locs = self.tracks_spine[times[0]:times[1]+1, a]  # be careful to add +1

                dists = np.sqrt((locs[1:,0]-locs[:-1,0])**2+
                                (locs[1:,1]-locs[:-1,1])**2)
                idx = np.where(dists>=self.max_jump_single_frame)[0]
                t = np.arange(times[0],times[1],1)

                #
                if idx.shape[0]>0:
                    self.tracks_chunks[a].append([t[0],t[idx[0]]])
                    for i in range(1,idx.shape[0],1):
                        #if (t[idx[i]]-t[idx[i-1]+1])>1:
                        self.tracks_chunks[a].append([t[idx[i-1]+1],
                                                              t[idx[i]]])

                    # add residual snippet at the end
                    if (t[idx[-1]]+1)<=times[1]: # and (t[idx[-1]]+1-times[1])>1:
                        self.tracks_chunks[a].append([t[idx[-1]]+1,
                                                      times[1]])

                else:
                    self.tracks_chunks[a].append(times.copy())

                # del
                del self.time_cont_tracks[a][0]

        # Huddle tracks don't seem ot have proper score name files
        #  - and we don't use them for the correcting of the track errors
        if self.track_type == 'huddle':
            return

        # also make a similar array to tracks_spine that contains the mean confidence
        self.tracks_scores_mean = np.zeros((self.tracks_spine.shape[0],
                                                 self.tracks_spine.shape[1]),
                                                'float32')+np.nan

        #
        for animal_id in range(len(self.tracks_chunks)):
            for c in range(len(self.tracks_chunks[animal_id])):
                chunk = self.tracks_chunks[animal_id][c]
                mean = self.scores[chunk[0]:chunk[1]+1,animal_id].mean(0)
                self.tracks_scores_mean[chunk[0]:chunk[1]+1,animal_id]= mean



    #
    def memory_interpolate_huddle(self):
        ''' Idea is to merge all huddle pieces that are less than some distance apart over time
            - regardless of the time between
            - regardless of intervening chunks in between!
        '''

        #self.clean_tracks_spine()
        #print ("running memory interpolation on huddles")
        #
        self.tracks_spine_fixed = self.tracks_spine_fixed.squeeze()

        #
        all_segs = self.tracks_chunks

        #
        final_merged_times = []
        final_merged_locs = []

        # loop over all segs
        #print (" Total # of starting huddle segments: ", len(all_segs))
        #with tqdm(total=len(all_segs),
        #          position=0, leave=True,
        #          desc='Remaining inner segments to analyze') as pbar:

        if True:

                    #
            while len(all_segs)>0:

                #
                seg_current = all_segs[0]

                # remove the latest segment out
                del all_segs[0]

                #
                #pbar.set_description("Remaining segs to process "+ str(len(all_segs)))

                #
                all_segs_inner = all_segs.copy()

                # grab the first segment and check against all other segments for distance
                merged_times = []
                merged_locs = []
                merged_times.append(seg_current)
                merged_locs.append(self.tracks_spine_fixed[seg_current[0]:seg_current[1]])

                #
                non_merged_chunks = []

                #
                while len(all_segs_inner)>0:

                    # pick the next segement in line
                    seg_next = all_segs_inner[0]

                    #
                    #if len(all_segs_inner)==0:
                    #    non_merged_chunks.append(seg_next)
                    #    break

                    #
                    del all_segs_inner[0]

                    # this deals with a bug of the last segment being the same frame 2x [last_frame, last_frame]
                    if seg_next[0]==seg_next[1]:
                        if len(all_segs_inner)==0:
                            break
                        else:
                            continue

                    # compute distance between last location of the first chunk
                    #   and first location of the following chunk
                    merged_locs_temp = np.vstack(merged_locs)
                    loc_current = self.tracks_spine_fixed[seg_next[0]]

                    #
                    if False:
                        coords = (merged_locs_temp-loc_current).squeeze()
                        #print ("Coords: ", coords.shape, " seg_next: ", seg_next)
                        dist = np.min(np.linalg.norm(coords,axis=1))
                    # This just matches to centre of the previous track - not any random part
                    else:
                        mean_previous = np.mean(merged_locs_temp,axis=0)
                        # here we try to find shortest distance between existing huddle and any point
                        #  in hte new huddle seg;
                        # - alternative option is to take mean of new huddle seg
                        try:
                            diff = mean_previous-self.tracks_spine_fixed[seg_next[0]:seg_next[1]]
                            dist = np.min(np.linalg.norm(diff, axis=1))
                        except:
                            print ("seg_next: ", seg_next)
                            print ("merged_locs_temp: ", merged_locs_temp)
                            print ("mean_previous: ", mean_previous)
                            print ("dist: ", dist)
                            print ("diff : ", diff.shape)

                    # compute time between last chunk and current chunk
                    time_diff =  seg_next[0] - seg_current[1]

                    # if close enough
                    if dist<=self.max_distance_huddle and time_diff <= self.max_time_to_join_huddle:

                        ############# MERGE IN SPACE ##########
                        # fill in missing space
                        try:
                            temp_locs = np.zeros((seg_next[0] - seg_current[1],2))+loc_current
                        except:
                            print ("broke here: seg_next: ", seg_next, " seg_current: ", seg_current)


                        # add distance in between
                        merged_locs.append(temp_locs)

                        # add new segement locations
                        merged_locs.append(self.tracks_spine_fixed[seg_next[0]:seg_next[1]].squeeze())

                        ############### MERGE IN TIME #################
                        # make missing time
                        temp_times = [seg_current[1],seg_next[0]]

                        # add time in between
                        merged_times.append(temp_times)

                        # add new segment
                        merged_times.append(seg_next)

                        seg_current = [seg_current[0], seg_next[1]]

                    #
                    else:
                        non_merged_chunks.append(seg_next)
                        if False:
                            print (" segment too far: ", seg_next[0], seg_next[1],
                                   self.tracks_spine_fixed[seg_next[0]:seg_next[1]].squeeze()[0],
                                   self.tracks_spine_fixed[seg_next[0]:seg_next[1]].squeeze()[1])
                            print (" min distance to previous segment: ", dist)
                            print (" time to previous segment: ", time_diff)

                #
                huddle_duration = 0
                for k in range(len(merged_times)):
                    huddle_duration+= merged_times[k][1] - merged_times[k][0]

                if huddle_duration >= self.min_huddle_time:
                    final_merged_times.append(merged_times)
                    final_merged_locs.append(merged_locs)

                # on exit from while loop delete the first entry in the
                if False:
                    print ('')
                    print ("seg_current: ", seg_current)
                    print ("length leftovers: ", len(non_merged_chunks),
                           " non merged chunks: ", non_merged_chunks)
                    print ("#################")
                all_segs = non_merged_chunks.copy()

                #break

        #self.tracks_chunks[0] = final_merged_tracks
        self.final_merged_times = final_merged_times
        self.final_merged_locs = final_merged_locs

    #
    def save_updated_huddle_tracks(self):


        fname_out = self.fname_slp[:-4]+'_multi_track_huddles.npy'

        #
        #print (" # of detected tracks/huddles: ", len(self.final_merged_times))

        # make multi-huddle track
        self.tracks_huddles = np.zeros((self.tracks_spine.shape[0],
                                        len(self.final_merged_times),
                                        2))*np.nan
        # loop over all huddles
        for k in range(len(self.final_merged_times)):
            times = self.final_merged_times[k]
            segs = self.final_merged_locs[k]

            for time_chunk, seg_chunk in zip(times,segs):
                t = np.arange(time_chunk[0], time_chunk[1],1)
                self.tracks_huddles[t,k] = seg_chunk

        #print ("fname out: ", fname_out)
        np.save(fname_out, self.tracks_huddles)


    #
    def memory_interpolate_tracks_spine(self):

        ''' Idea is to check for neighbouring bouts of movement if there has not been a lot of
            space movement, we can freeze the centroid at the last location (or middle point)

            - middle point could help actulaly to visualize ugly errors
        '''

        #
        animal_ids = np.arange(self.tracks_spine.shape[1])

        #
        for animal_id in animal_ids:

            #
            max_dist = 50
            seg_previous = self.tracks_chunks[animal_id][0]
            for chunk_id in range(1,len(self.tracks_chunks[animal_id]),1):

                #
                seg_current = self.tracks_chunks[animal_id][chunk_id]

                # compute distance
                loc_previous = self.tracks_spine_fixed[seg_previous[1],animal_id]
                loc_current = self.tracks_spine_fixed[seg_current[0],animal_id]

                #
                dist = np.linalg.norm(loc_previous-loc_current)

                # merge
                if dist<=max_dist:
                   # print (seg_previous[1],  seg_current[0],
                    #       loc_previous+loc_current, np.mean((loc_previous, loc_current),axis=0))
                    self.tracks_spine[seg_previous[1]: seg_current[0], animal_id] = np.mean((loc_previous, loc_current),axis=0)

                #
                seg_previous = seg_current

    #
    def filter_tracks(self):
    
        #
        for k in self.animal_ids:
            for f in range(self.tracks.shape[3]):
                temp = self.tracks[:,k,:,f] #  
                self.tracks[:,k,:,f] = scipy.ndimage.median_filter(temp, 
                                                                   size=self.filter_width)


    def filter_tracks_spines_butterworth(self):

        def butter_filter(data, butterCutoffFreq, order=2):
            b, a = signal.butter(order, butterCutoffFreq, 'low', analog=True)
            y = signal.filtfilt(b, a, data)
            return y

        #
        for k in range(self.tracks_spine.shape[1]):
            for f in range(self.tracks_spine.shape[2]):
                temp = self.tracks_spine[:,k,f] #
                idx = np.where(np.isnan(temp))

                #
                butterCutoffFreq = 5
                temp = butter_filter(temp, butterCutoffFreq)
                #temp = scipy.ndimage.median_filter(temp,
                #                                   size=self.filter_width)

                #
                temp[idx] = np.nan
                self.tracks_spine[:,k,f] = temp

    #
    def filter_tracks_spines(self):
    
        # 
        for k in range(self.tracks_spine.shape[1]):
            for f in range(self.tracks_spine.shape[2]):
                temp = self.tracks_spine[:,k,f] #
                idx = np.where(np.isnan(temp))

                #
                temp = scipy.ndimage.median_filter(temp,
                                                   size=self.filter_width)

                #
                temp[idx] = np.nan
                self.tracks_spine[:,k,f] = temp
 
    #
    def del_short_chunks(self, min_chunk_len=2):
        print ("Deleting chunks < ", min_chunk_len)
        for a in range(len(self.tracks_chunks)):
            chunks = np.array(self.tracks_chunks[a])

            #
            #print (chunks.shape)
            lens = chunks[:,1]-chunks[:,0]

            #
            idx = np.where(lens<min_chunk_len)[0]
            # delete singles from the xy locs as well
            for id_ in range(idx.shape[0]):
                time = self.tracks_chunks[a][idx[id_]]
                self.tracks_spine[time[0]:time[1]+1, a]=np.nan

            #
            idx = np.where(lens>=min_chunk_len)[0]
            chunks = chunks[idx]
            self.tracks_chunks[a] = chunks
            #print (len(self.tracks_chunks[a]))

    def merge_single_jumps(self):

        #
        for a in range(len(self.tracks_chunks)):
            chunks = np.array(self.tracks_chunks[a])
            chunks_fixed = []

            for i in range(0,chunks.shape[0]-1,1):

                times1 = chunks[i]
                times2 = chunks[i+1]

                if times2[0]-times1[1]<=(self.max_time_automerge+1):
                    # check if distances are very small
                    locs1 = self.tracks_spine[times1[1],a]
                    locs2 = self.tracks_spine[times2[0], a]

                    dist = np.linalg.norm(locs1-locs2)

                    if dist<=self.max_dist_automerge:
                        chunks_fixed.append([times1[0],times2[1]])
                        if self.verbose:
                            print ("merged: ", a, i, times1, times2, locs1, locs2, dist)

                        # also replace the spine location
                        self.tracks_spine[times1[1]:times1[1]+
                                          self.max_time_automerge+1,a]=self.tracks_spine[times1[1],a]

                        continue

                chunks_fixed.append(times1)

            self.tracks_chunks[a] = chunks_fixed

    def get_scores(self):


        #
        if self.track_type=='huddle':
            return

        #
        fname_scores = self.fname_slp[:-4] + "_scores.npy"
        if os.path.exists(fname_scores) == False:
            if self.verbose:
                print("... slp file loading...")
            self.load_slp()

            #
            self.scores = np.zeros((len(self.slp), self.n_animals), 'float32') + np.nan
            for n in range(len(self.slp)):
                for a in range(len(self.slp[n])):
                    name = self.slp[n][a].track.name
                    idx = self.tracks_names.index(name)
                    self.scores[n, idx] = self.slp[n][a].score

            np.save(fname_scores, self.scores)

        else:
            self.scores = np.load(fname_scores)

    def plot_scores_distribution(self):
        width = 0.01
        for k in range(4):
            ax = plt.subplot(2, 2, k + 1)
            y = np.histogram(self.scores[:, k],
                             bins=np.arange(0, 1.02, width))
            plt.bar(y[1][:-1], y[0], width * .9)
            plt.title("animal " + str(k))
            plt.semilogy()
        plt.show()


    def find_nearest_forward(self, val, array2, window=1000.0):
        diff_temp = array2-val
        idx = np.where(diff_temp>0)[0]

        if idx.shape[0]==0 or diff_temp[idx[0]]>window:
            return 1E5, np.nan

        loc = idx[0]
        diff = diff_temp[idx[0]]
        return diff, loc

    #
    def find_nearest_backward(self, val, array2, window=1000.0):
        diff_temp = val-array2
        idx = np.where(diff_temp>0)[0]

        if idx.shape[0]==0 or diff_temp[idx[0]]>window:
            return 1E5, np.nan

        loc = idx[0]
        diff = diff_temp[idx[0]]
        return diff, loc

    #
    def get_chunk_info(self,
                       animal_id1,
                       t):


        ################### CURRENT ######################
        #track_local = np.array(self.tracks_chunks[animal_id1])
        track_local = np.array(self.tracks_chunks_fixed[animal_id1])
        chunk_current = []
        while len(chunk_current)==0:
            chunk_current = np.where(np.logical_and(t>=track_local[:,0], t<=track_local[:,1]))[0]

            # advance time until get to the first segment for this animal
            if len(chunk_current)==0:
                t+=1

        times_current = track_local[chunk_current][0]
        locs_current = self.tracks_spine_fixed[times_current, animal_id1]

        if self.verbose:
            print ("Active animal:", animal_id1)
            print ("locs active; ", locs_current)
            print ("times active animal: ", times_current)

        return times_current, locs_current, chunk_current, t


    def check_chunk_overlap(self,
                            times_active_animal,
                            animal_id1,
                            track_local):

        # if comparing current track with same animal tracks
        if self.animal_current == animal_id1:
            return False

        #
        # check if partial overlapping chunk the need to skip entirely;
        # we do not want to swap parts of tracks as we assume continous tracks have single identities

        # check if any tracks start before start of current track but end during/after
        idx = np.where(np.logical_and(track_local[:,0]<times_active_animal[0],
                                      track_local[:,1]>=times_active_animal[0]))[0]
        if idx.shape[0]>0:
            return True

        # start at same time but end after
        idx = np.where(np.logical_and(track_local[:,0]==times_active_animal[0],
                                      track_local[:,1]>times_active_animal[1]))[0]
        if idx.shape[0]>0:
            return True

        # start during but end after
        idx = np.where(np.logical_and(track_local[:,0]<=times_active_animal[1],
                                      track_local[:,1]>times_active_animal[1]))[0]
        if idx.shape[0]>0:
            return True

        return False

    #
    def get_cost(self, t,
                chunk_current,
                times_active_animal,
                locs_active_animal,
                min_chunk_len,
                verbose=False):

        ''' Find distances between active animal chunk and other animals
            Notes: look for minimum distance to nearest
                - do not change very long tracks, sleap likely has them right

        '''
        cost = np.zeros(self.n_animals, 'float32')+1E5
        chunk_ids_compare = np.zeros(self.n_animals, 'float32')+np.nan
        for animal_id1 in range(self.n_animals):

            # grab chunk info
            #track_local = np.array(self.tracks_chunks[animal_id1])
            track_local = np.array(self.tracks_chunks_fixed[animal_id1])

            # check if chunks overlap
            flag = self.check_chunk_overlap(times_active_animal,
                                            animal_id1,
                                            track_local)
            if flag:
                if self.verbose:
                    print ("Skipping overlapping chunk/animal, ", animal_id1)
                cost[animal_id1] = 1E5
                continue

            ################### CURRENT ######################
            # if current track very long, make sure you don't change it
            chunk_id = np.where(np.logical_and(t>=track_local[:,0], t<=track_local[:,1]))[0]
            if self.verbose:
                print ("chunk id: ", chunk_id)
            if chunk_id.shape[0]>0:
                times_current = track_local[chunk_id][0]

                if times_current[1]-times_current[0]>self.time_threshold:
                    cost[animal_id1] = 1E5
                    continue

            ################# PREVIOUS #################
            chunk_ids = np.where(track_local[:,1]<t)[0]
            if self.verbose:
                print ("ANIMLA ID: ", animal_id1,
                        "prev hcunk id: ",
                       chunk_ids[-1],
                       " times: ", track_local[chunk_ids[-1]],
                       "locs : ", self.tracks_spine_fixed[track_local[chunk_ids[-1]],
                                                          animal_id1])
            # ensure we only look at sufficiently long track,s do not compare to singles/ or few
            z=-1
            while True:
                try:
                    chunk_id_prev = chunk_ids[z]
                    times_prev = track_local[chunk_id_prev]
                    if self.verbose:
                        print ("times prev: ", times_prev)

                    # anchor only ot chunks that are above a certain size
                    if (times_prev[1]-times_prev[0]+1)>=min_chunk_len:
                        locs_prev = self.tracks_spine_fixed[times_prev, animal_id1]
                        break
                except:
                    if self.verbose:
                        print ("broken: chunk_ids", chunk_ids)
                    chunk_id_prev = None
                    times_prev = np.array([-1E6,-1E6])
                    locs_prev = np.array([1E5,1E5],'float32')
                    break
                z-=1

            ############ CHECK AND DISSALOW HUGE JUMPS IN SINGLE TIME STEPS #########
            # cannot connect to such animals
            if (times_prev[1]+1)==times_active_animal[0]:

                nearest_prev_dist = np.linalg.norm(locs_active_animal[0]-locs_prev[1])
                if self.verbose:
                    print ("!!!!!!!!!!!!! NEARBY ANIMAL, CHECKING DISTANCE JUMP")
                    print ("JUMP: ", nearest_prev_dist)
                if nearest_prev_dist>self.max_jump_single_frame:
                    cost[animal_id1] = 1E5
                    continue

            ################### NEXT ###################
            chunk_ids = np.where(track_local[:,0]>t)[0]
            if self.verbose:
                print ("ANIMLA ID: ", animal_id1,
                       "next chunk ids: ",
                       chunk_ids[0],
                       " times: ", track_local[chunk_ids[0]],
                          "locs : ", self.tracks_spine_fixed[track_local[chunk_ids[0]],
                          animal_id1])
            z=0
            while True:
                try:
                    chunk_id_next = chunk_ids[z]
                    times_next = track_local[chunk_id_next]
                    if (times_next[1]-times_next[0]+1)>=min_chunk_len:
                        locs_next = self.tracks_spine_fixed[times_next, animal_id1]
                        break
                except:
                    chunk_id_next = None
                    times_next = np.array([1E6,1E6])
                    locs_next = np.array([1E5,1E5],'float32')
                    break
                z+=1

            ############ CHECK AND DISSALOW HUGE JUMPS IN SINGLE TIME STEPS #########
            if (times_next[0])==times_active_animal[1]:
                nearest_prev_dist = np.linalg.norm(locs_active_animal[1]-locs_next[0])
                if nearest_prev_dist>self.max_jump_single_frame:
                    cost[animal_id1] = 1E5
                    continue

            ####################################################
            # make cost matrix
            # find distance between active chunk start and prev chunk:
            if self.verbose:
                print ("DIFF to prev animal: times_prev[1] ",
                       times_prev[1])
            if times_active_animal[0]-times_prev[1]<self.time_threshold:
                c_prev = np.linalg.norm(locs_active_animal[0]-locs_prev[1])
            else:
                c_prev = 1E5

            #
            if self.verbose:
                print ("DIFF to next animal times_next[0] ",
                       times_next[0])
            if times_next[0]-times_active_animal[1]<self.time_threshold:
                c_next = np.linalg.norm(locs_active_animal[1]-locs_next[0])
            else:
                c_next= 1E5

            #
            if self.verbose:
                print ("time: ", t)
                print ("animal id: ", animal_id1)
                #print ("  current: ", chunk_id, "times: ", times_current, " locs: ", locs_current)
                print ("  prev: ", chunk_id_prev, "times : ", times_prev, " locs: ", locs_prev)
                print ("  next: ", chunk_id_next, "times : ", times_next, " locs: ", locs_next)

                print ("times prev: ", times_prev)
                print ("cprev: ", c_prev, "  cnext: ", c_next)

            #
            if self.verbose:
                print ("animal ", animal_id1, " costs:  c_next, cprev ", c_next, c_prev)
            tot_cost = np.array([c_next,c_prev])
            cost[int(animal_id1)] = np.nanmin(tot_cost)

            if np.isnan(cost[int(animal_id1)]):
                if self.verbose:
                    print ("animal ", animal_id1, " has no self connection, replacing with 1E5")
                cost[int(animal_id1)]=1E5

        return cost

    #
    def swap_chunks(self,
                    correct_id,    # the correct assignemnt
                    times_current,
                    chunk_current,
                    ):


        if self.verbose:
            print ("swapping: ", correct_id, " with ", self.animal_current)
        ##########################################
        ###### SWAP SPINES (SINGLE FEATURES) #####
        ##########################################
        # hold memory
        temp_track = self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                                        self.animal_current].copy()
        #
        temp2_track = self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                                         correct_id].copy()

        #
        self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                           self.animal_current]= temp2_track

        # replace correct id with temp
        self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                           correct_id]= temp_track

        ##########################################
        ############ SWAP FULL FEATURES ##########
        ##########################################
        # hold memory
        temp_track = self.tracks_fixed[times_current[0]:times_current[1]+1,
                                        self.animal_current].copy()
        #
        temp2_track = self.tracks_fixed[times_current[0]:times_current[1]+1,
                                         correct_id].copy()

        #
        self.tracks_fixed[times_current[0]:times_current[1]+1,
                           self.animal_current]= temp2_track

        # replace correct id with temp
        self.tracks_fixed[times_current[0]:times_current[1]+1,
                           correct_id]= temp_track

        ##########################################
        ############# SWAP CHUNKS ################
        ##########################################
        #
        temp_chunk = times_current #self.tracks_chunks_fixed[animal_current][chunk_current].copy()
        if self.verbose:
            print ("***********************************************************************")
            print ("***********************SWAPPING ", self.animal_current ,  " WITH ", correct_id)
            print ("***********************************************************************")
            print ('track.tracks_chunks_fixed ', self.tracks_chunks_fixed[self.animal_current])
            print ("animal_current: ", self.animal_current)
            print ("chunk_current: ", chunk_current)

        # del chunk from active animal
        self.tracks_chunks_fixed[self.animal_current] = np.delete(
                                                self.tracks_chunks_fixed[self.animal_current],
                                                chunk_current,
                                                axis=0)

        # check to see if any smaller chunk inside current animal and swap it as well
        idx2 = np.where(np.logical_and(
                       self.tracks_chunks_fixed[correct_id][:,0]>=temp_chunk[0],
                       self.tracks_chunks_fixed[correct_id][:,1]<=temp_chunk[1]))[0]

        if idx2.shape[0]==1:
            # need to add swapped chunk from correct_id to animal_current

            swapped_chunk = self.tracks_chunks_fixed[correct_id][idx2][0]
            self.tracks_chunks_fixed[self.animal_current] = np.vstack((
                                                self.tracks_chunks_fixed[self.animal_current],
                                                swapped_chunk))
            # reorder by time
            idx = np.argsort(self.tracks_chunks_fixed[self.animal_current][:,0])
            self.tracks_chunks_fixed[self.animal_current] = self.tracks_chunks_fixed[
                                                self.animal_current][idx]

            # also delete the swapped chunk fro
            self.tracks_chunks_fixed[correct_id] = np.delete(
                                                        self.tracks_chunks_fixed[correct_id],
                                                        idx2[0],
                                                        axis=0)


        # add new chunk to correct_animal
        self.tracks_chunks_fixed[correct_id] = np.vstack((
                                                self.tracks_chunks_fixed[correct_id],
                                                temp_chunk))
        # reorder by time
        idx = np.argsort(self.tracks_chunks_fixed[correct_id][:,0])
        self.tracks_chunks_fixed[correct_id]= self.tracks_chunks_fixed[correct_id][idx]


    #
    def reallocate_track_segments(self,
                   t=None,
                   t_end=None):

        #
        #print ("... Fixing tracks...")

        #
        if t==None or t_end==None:
            t = 0
            t_end = self.tracks.shape[0]

        #
        if self.verbose:
            pbar = tqdm(total=(t_end-t))

        #
        self.animal_current = 0

        # make copies of fixed arrays
        self.tracks_chunks_fixed=[]
        for k in range(len(self.tracks_chunks)):
            self.tracks_chunks_fixed.append(np.array(self.tracks_chunks[k].copy()))

        #
        #if self.use_dynamic_centroid==True:
        self.tracks_spine_fixed = self.tracks_spine.copy()
        #else:
        #    self.tracks_spine_fixed = self.tracks[:,:,self.centroid_body_id]

        # loop while unfixed chunks remain
        # This function goes over each chunk of slp track starting at beginning
        #   - it assigns it to the correct animal based on history
        #   - then grabs the next chunk of slp track that begins most immediatley after
        while True:  # <-------- need to advance in time slowly by indexing through each animal chunk
            if self.verbose:
                 pbar.update(t)

            # grab location for current chunk and animal being analyzed:
            times_current, locs_current, chunk_current, t = self.get_chunk_info(
                                                                        self.animal_current,
                                                                        t)
            if self.verbose:
                print ("###: t: ", t,
                       " animal current ", self.animal_current,
                       "  chunk current",  chunk_current,
                       " times current: ", times_current)

            # some tracks will not change id if they are above a specific size:
            if (times_current[1]-times_current[0])<self.safe_chunk_length:

                # get cost:
                cost = self.get_cost(t,
                                     chunk_current,
                                     times_current,
                                     locs_current,
                                     self.min_chunk_len)

                if self.verbose:
                    print ("COST: ", cost)

                # closest swap attempt one animal at a time
                if np.min(cost)<self.max_distance_merge:

                    #
                    correct_id = np.argmin(cost)

                    # check if any data needs to be swapped
                    if self.verbose:
                        print ("INPUT TIMES CURRENT: ", times_current)

                    if correct_id!= self.animal_current:
                        self.swap_chunks(
                                        correct_id,     # the correct assignemnt
                                        times_current,
                                        chunk_current)  # the times to swap

            ##################################################
            ######### FIND NEXT SECTION OF TRACK #############
            ##################################################
            # find next track start and ends
            temp = []
            for i in range(self.n_animals):
                # grab all tracks for each animal
                temp2 = np.array(self.tracks_chunks_fixed[i])

                # append all track starts that occur after current time;
                # TODO: this may skip tracks that start at the same time...
                try:
                    next_chunk_time = np.where(temp2[:,0]>t)[0]
                    temp.append(temp2[next_chunk_time,0][0])
                except:

                    if self.update_tracks:
                       # print ("UPDATING TRACKS")
                        self.tracks_chunks = self.tracks_chunks_fixed
                        self.tracks_spine = self.tracks_spine_fixed

                    return

            # find the nearest start in time;
            t = np.min(temp)

            # select which animal the chunk belongs to
            self.animal_current = np.argmin(temp)

            #
            if self.verbose:
                print ('')
                print ('')
                print ("##########################################################################")
                print ("TIME: ", t, " active_animal: ", self.animal_current, " (temp: )", temp)

            if t>t_end:
                break

        pbar.close()

        if self.update_tracks:
            print ("UPDATING TRACKS")
            self.tracks_chunks = self.tracks_chunks_fixed
            self.tracks_spine = self.tracks_spine_fixed

#
def filter_trace(data,
                fc=1):

    from scipy import signal

    fs = 25  # Sampling frequency

    # Generate the time vector properly
    #fc = 1  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(2, w, 'low')
    data = signal.filtfilt(b, a, data)

    return data



#
def detect_id_switch3(tracks,
                     animal_id,
                     max_n_nans=0):
    #
    other_animal_ids = np.int32(np.delete(np.arange(6), animal_id))

    # select animal_id track
    track = tracks[:, animal_id]

    # calculate the feature-wise distance between current frame and next frame
    ctr = 0
    frame_ids = []
    corner = 300
    for k in trange(tracks.shape[0] - 1, desc='finding id switches in movie'):
        feats0 = track[k]

        for a in other_animal_ids:
            feats1 = tracks[k + 1, a]

            #
            dists = np.linalg.norm(feats1 - feats0, axis=1)

            #
            med = np.nanmedian(dists)

            # if the median distance is low
            if med < 15:
                #
                n = np.count_nonzero(np.isnan(dists))
                # if n<=(6-min_n_features):
                if n == max_n_nans:
                    # also skip switches in the huddle
                    if np.nanmedian(feats0[:, 0]) > corner:
                        if np.nanmedian(feats0[:, 1]) > corner:
                            frame_ids.append([k, k + 1, animal_id, a])

                        ctr += 1
                # else:
                #    print ("n toolow: ", n)
    print("found: ", len(frame_ids), " switches")
    return frame_ids


def find_id_switches(fnames_slp_list):
    fnames_slp = np.loadtxt(fnames_slp_list,
                            dtype='str')

    #
    overwrite_flag = True

    #
    for fname_slp in fnames_slp:

        #
        animal_ids = np.arange(6)

        #
        fname_out = fname_slp[:-4] + "_all_frames_with_switches.npy"

        #
        if overwrite_flag == True or os.path.exists(fname_out) == False:

            #
            track = Track(fname_slp)
            track.track_type = 'features'
            track.use_dynamic_centroid = True  # True: alg. serches for the first non-nan value in this body order [2,3,1,0,4,5]
            track.load_tracks()

            #### FIND ALL ID SWITCHES IN MOVIE ####
            all_frames = []

            if False:
                for animal_id in animal_ids:
                    # for animal_id in [2]:
                    frame_ids = detect_id_switch3(track.tracks,
                                                  animal_id)
                    if len(frame_ids) > 0:
                        all_frames.append(frame_ids)
            else:

                res = parmap.map(detect_id_switch_parallel,
                                 animal_ids,
                                 track.tracks)
                all_frames = []
                for r in res:
                    if len(r) > 0:
                        all_frames.append(r)

                        #
            all_frames = np.vstack(all_frames)
            # print ("all frames: ", all_frames.shape)

            idx = np.argsort(all_frames[:, 0])
            all_frames = all_frames[idx]

            #
            print("filename completed: ", fname_slp)
            print("all frames: ", all_frames.shape)

            #
            np.save(fname_out, all_frames)

def find_id_switches_parallel(fnames_slp,
                              fnames_vids,
                              n_cores):

    if True:
        in_ = list(zip(fnames_slp, fnames_vids))

        parmap.map(find_id_switches_single_file,
                   in_,
                   pm_pbar = True,
                   pm_processes = n_cores)






def find_id_switches_single_file(input,
                                 overwrite_flag = False):

    #
    fname_slp, fname_vid = input

    #
    animal_ids = np.arange(6)

    #
    fname_out = fname_vid[:-4] + "_all_frames_with_switches.npy"

    #
    if overwrite_flag == True or os.path.exists(fname_out) == False:

        #
        try:
            track = Track(fname_slp)
            track.track_type = 'features'
            track.use_dynamic_centroid = True  # True: alg. serches for the first non-nan value in this body order [2,3,1,0,4,5]
            track.load_tracks()
        except:
            print ("Missing : ", fname_slp)
            return
        #### FIND ALL ID SWITCHES IN MOVIE ####
        all_frames = []

        if False:
            for animal_id in animal_ids:
                # for animal_id in [2]:
                frame_ids = detect_id_switch3(track.tracks,
                                              animal_id)
                if len(frame_ids) > 0:
                    all_frames.append(frame_ids)
        else:

            res = parmap.map(detect_id_switch_parallel,
                             animal_ids,
                             track.tracks)
            all_frames = []
            for r in res:
                if len(r) > 0:
                    all_frames.append(r)

                    #
        if len(all_frames)>0:
            all_frames = np.vstack(all_frames)
            # print ("all frames: ", all_frames.shape)

            idx = np.argsort(all_frames[:, 0])
            all_frames = all_frames[idx]
        else:
            all_frames = np.zeros(0)

        #
        #print("filename completed: ", fname_slp)
        #print("all frames: ", all_frames.shape)

        #
        np.save(fname_out, all_frames)



def make_deleted_slp_files_human_labels(fname_slp):

    #
    names = ['female','male','pup1','pup2','pup3','pup4','none']

    #fname = "/media/cat/256GB/dan/testing_track_cleanup_code/cohort2_night_1000/videos/cohort2_night.1000.slp"
    labels = sleap.load_file(fname_slp)

    labels.videos  # Use in ipython to print list of videos in the labels object
    n_videos = len(labels.videos)

    #
    merged_video = sleap.load_video('/home/cat/data/dan/merge_hybrid_slp/movie_hybrid.mp4')


    n_all = 0
    for video_idx in range(n_videos):

        # Change this select the video of interest from the labels.video list
        video = labels.videos[video_idx]
        labeled_frames_from_video = labels.get(video)

        print ("# of labeled frames: ", len(labeled_frames_from_video))

        # Find indices of labeled frames to keep, note these are the indices of the `LabeledFrame`s in the labels.labeled_frames list
        # lf_indices = search_frames[:,0] #[0, 2, 7, 101]  # Replace with list of labeled frame indices (NOT the same as video indices)
       # lf_indices = all_  # [0, 2, 7, 101]  # Replace with list of labeled frame indices (NOT the same as video indices)
        lf_indices = np.arange(len(labeled_frames_from_video))

        # Create a new `Labels` object containing only the matched `LabeledFrame`s
        labels_keep = labels.extract(lf_indices, copy=True)

        # Loop through matched `LabeledFrame`s
        ctr = 0
        new_frames = []
        for i, lf in enumerate(labels_keep.labeled_frames):

            #
            lf.video = merged_video
            ctr_del = 0
            for inst_idx, inst in enumerate(lf.user_instances):


                new_frames = []
                for i, lf in enumerate(labels):

                    # Update reference to merged video.
                    lf.video = merged_video

                    # Update frame index to the frame number within the merged video.
                    lf.frame_idx = n_frames + i

                    # Update the track reference to use the reference tracks to prevent duplication.
                    for instance in lf:
                        if instance.track is not None:
                            if instance.track.name in reference_tracks:
                                instance.track = reference_tracks[instance.track.name]
                            else:
                                reference_tracks[instance.track.name] = instance.track

                    # Append the labeled frame to the list of frames we're keeping from these labels.
                    new_frames.append(lf)

                all_frames.extend(new_frames)
                n_frames += len(new_frames)

            merged_labels = sleap.Labels(all_frames)

            # save stuff
            merged_labels.save(fname_hybrid_slp)




            # get the predicted name
            # print ("instd idx: ", inst_idx)
            #name_predicted = lf.predicted_instances[inst_idx + ctr_del].track.name
            name_labeled = lf.user_instances[inst_idx + ctr_del].track.name
            print ("name labeled: ", name_labeled)

            #  see if it matches the track to keep name
            if name_predicted != track_to_keep:
                lf.instances.remove(inst)

                ctr_del -= 1
        ctr += 1

    # Save new slp
    ##new_slp_path = fname[:-4] + "_deleted.slp"
    #labels_keep.save_file(labels_keep, new_slp_path)
    # sleap.Labels.save_file(labels_keep, new_slp_path[:-4]+"_2.slp")

    print ("Done...")


def make_deleted_slp_files_parallel(fnames_slp, fnames_vids,
                                    n_cores):

    in_ = list(zip(fnames_slp, fnames_vids))

    parmap.map(make_deleted_slp_files, in_,
               pm_pbar=True,
               pm_processes=n_cores)


#
def make_deleted_slp_files(input):

    fname_slp, fname_vid = input

    names = ['female','male','pup1','pup2','pup3','pup4','none']

    fname_out = fname_slp[:-4]+"_deleted.slp"

    if os.path.exists(fname_out):
        return

    # load all frames:
    search_frames = np.load(fname_vid[:-4] + '_all_frames_with_switches.npy')

    # merge the frame pairs into each other so they all appear as a single sequence
    all_ = []
    names_idx = []
    for k in range(len(search_frames)):
        all_.append(search_frames[k,0])
        all_.append(search_frames[k,1])
        names_idx.append(search_frames[k,2])
        names_idx.append(search_frames[k,3])

    #
    tracks_to_keep = []
    for idx in names_idx:
        #print (idx)
        tracks_to_keep.append(names[idx])

    #
    labels = sleap.load_file(fname_slp)
    n_videos = len(labels.videos)

    #
    n_all=0
    for video_idx in range(n_videos):

        # Change this select the video of interest from the labels.video list
        video = labels.videos[video_idx]
        labeled_frames_from_video = labels.get(video)

        # Find indices of labeled frames to keep, note these are the indices of the `LabeledFrame`s in the labels.labeled_frames list
        #lf_indices = search_frames[:,0] #[0, 2, 7, 101]  # Replace with list of labeled frame indices (NOT the same as video indices)
        lf_indices = all_ #[0, 2, 7, 101]  # Replace with list of labeled frame indices (NOT the same as video indices)

        # Create a new `Labels` object containing only the matched `LabeledFrame`s
        labels_keep = labels.extract(lf_indices, copy=True)

        # Loop through matched `LabeledFrame`s
        ctr=0
        for lf, track_to_keep in zip(labels_keep.labeled_frames, tracks_to_keep):

            #
            ctr_del = 0
            for inst_idx, inst in enumerate(lf.predicted_instances):

                # get the predicted name
                #print ("instd idx: ", inst_idx)
                name_predicted = lf.predicted_instances[inst_idx+ctr_del].track.name

                #  see if it matches the track to keep name
                if name_predicted!=track_to_keep:
                    lf.instances.remove(inst)

                    ctr_del-=1
            ctr+=1

    # Save new slp
    labels_keep.save_file(labels_keep, fname_out)
    #sleap.Labels.save_file(labels_keep, new_slp_path[:-4]+"_2.slp")

#
def make_hybrid_slp_from_list(fnames_slp_list,
                              fname_hybrid_video):
    #
    merged_video = sleap.load_video(fname_hybrid_video)

    #
    fname_hybrid_slp = fname_hybrid_video[:-4] + ".slp"

    if os.path.exists(fname_hybrid_slp==True):
        return

    #
    slp_files = np.loadtxt(fnames_slp_list, dtype='str')

    n_frames = 0
    all_frames = []
    reference_tracks = {}

    first_labels = None
    for slp_file0 in slp_files:

        # replace slp_file with the deleted version
        slp_file = slp_file0[:-4]+"_deleted.slp"

        # Load saved labels.
        labels = sleap.load_file(slp_file, match_to=first_labels)
        if first_labels is None:
            first_labels = labels

        new_frames = []
        for i, lf in enumerate(labels):

            # Update reference to merged video.
            lf.video = merged_video

            # Update frame index to the frame number within the merged video.
            lf.frame_idx = n_frames + i

            # Update the track reference to use the reference tracks to prevent duplication.
            for instance in lf:
                if instance.track is not None:
                    if instance.track.name in reference_tracks:
                        instance.track = reference_tracks[instance.track.name]
                    else:
                        reference_tracks[instance.track.name] = instance.track

            # Append the labeled frame to the list of frames we're keeping from these labels.
            new_frames.append(lf)

        all_frames.extend(new_frames)
        n_frames += len(new_frames)

    merged_labels = sleap.Labels(all_frames)

    # save stuff
    merged_labels.save(fname_hybrid_slp)

#
def make_hybrid_slp(fnames_slp_list,
                    fname_hybrid_video):
    #
    merged_video = sleap.load_video(fname_hybrid_video)
    #print(merged_video)

    #
    fname_hybrid_slp = fname_hybrid_video[:-4] + ".slp"

    if os.path.exists(fname_hybrid_slp)==True:
        print ('    hybrid file already exists...exiting')
        return

    #
    slp_files = np.loadtxt(fnames_slp_list, dtype='str')

    n_frames = 0
    all_frames = []
    reference_tracks = {}

    first_labels = None
    for slp_file0 in slp_files:

        # replace slp_file with the deleted version
        slp_file = slp_file0[:-4]+"_deleted.slp"

        # Load saved labels.
        labels = sleap.load_file(slp_file, match_to=first_labels)
        if first_labels is None:
            first_labels = labels

        new_frames = []
        for i, lf in enumerate(labels):

            # Update reference to merged video.
            lf.video = merged_video

            # Update frame index to the frame number within the merged video.
            lf.frame_idx = n_frames + i

            # Update the track reference to use the reference tracks to prevent duplication.
            for instance in lf:
                if instance.track is not None:
                    if instance.track.name in reference_tracks:
                        instance.track = reference_tracks[instance.track.name]
                    else:
                        reference_tracks[instance.track.name] = instance.track

            # Append the labeled frame to the list of frames we're keeping from these labels.
            new_frames.append(lf)

        all_frames.extend(new_frames)
        n_frames += len(new_frames)

    merged_labels = sleap.Labels(all_frames)

    # save stuff
    merged_labels.save(fname_hybrid_slp)

    print("DONE...")


#
def make_hybrid_video(fnames_slp_list):

    # video settings
    colors = [
        (0, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
        (0, 128, 0),
        (128, 128, 128),
        (128, 0, 128),
    ]

    #
    animal_ids = np.arange(6)

    # make new video video settings
    size_vid = np.int32(np.array([900, 700]))
    fps_out = 1
    dot_size = 4
    thickness = -1
    window = 100
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # #fourcc = cv2.VideoWriter_fourcc(*'X264')
    # #fourcc = cv2.VideoWriter_fourcc(*args["codec"])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', '')

    # load videos
    fname_out = os.path.split(fnames_slp_list)[0] + '/hybrid.mp4'
    fname_out_cropped = fname_out[:-4] + "_cropped.mp4"
    fname_out_annotated = fname_out[:-4] + "_annotated.mp4"
    fname_out_annotated_cropped = fname_out[:-4] + "_annotated_cropped.mp4"

    #
    print("Fname hybrid movie: ", fname_out_cropped)

    # make new video video settings
    size_vid = np.int32(np.array([900, 700]))
    fps_out = 1
    dot_size = int(6)
    thickness = -1

    if False:
        video_out = cv2.VideoWriter(fname_out,
                                    fourcc,
                                    fps_out,
                                    (size_vid[0], size_vid[1]),
                                    True)

    video_out_cropped = cv2.VideoWriter(fname_out_cropped,
                                        fourcc,
                                        fps_out,
                                        (size_vid[0], size_vid[1]),
                                        True)

    #
    if False:
        video_out_annotated = cv2.VideoWriter(fname_out_annotated,
                                              fourcc,
                                              fps_out,
                                              (size_vid[0], size_vid[1]),
                                              True)

    #
    if False:
        video_out_annotated_cropped = cv2.VideoWriter(fname_out_annotated_cropped,
                                                      fourcc,
                                                      fps_out,
                                                      (size_vid[0], size_vid[1]),
                                                      True)

    # load slp files
    fnames_slp = np.loadtxt(fnames_slp_list, dtype='str')

    # loop over each slp file
    for fname_slp in fnames_slp:

        # load all frames:
        all_frames = np.load(fname_slp[:-4] + '_all_frames_with_switches.npy')

        # load movie name
        idx = fname_slp.find("compressed")
        # print ("idx: ", idx)

        print (fname_slp[:idx] + "*cropped*.mp4")

        fname_movie = glob.glob(fname_slp[:idx] + "*cropped*.mp4")[0]
        print("fname found: ", fname_movie)

        ############################################
        ############################################
        ############################################
        track = Track(fname_slp)
        track.track_type = 'features'
        track.use_dynamic_centroid = True  # True: alg. serches for the first non-nan value in this body order [2,3,1,0,4,5]
        track.load_tracks()

        # load current vid
        original_vid = cv2.VideoCapture(fname_movie)
        width = original_vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # set frames to new ones
        ctr = 0
        for frames in tqdm(all_frames, desc='adding frames to movies'):
            # print ("frame #s: ", frames)
            if len(frames) == 0:
                continue

            #
            animal_id = frames[2]

            #
            original_vid.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

            #
            for z in range(2):

                # find centre of animal black out surrounding area
                feats = track.tracks[frames[0] + z]

                #
                if z == 0:
                    # print ("feats: ", feats)
                    # print ("feats: animal id: ", feats[animal_id])
                    feats_centre = np.nanmedian(feats[animal_id], 0)
                    # print ("feats centre: ", feats_centre)
                    #
                    min_x = int(max(feats_centre[0] - window, 0))
                    max_x = int(min(feats_centre[0] + window, 900))
                    min_y = int(max(feats_centre[1] - window, 0))
                    max_y = int(min(feats_centre[1] + window, 700))

                ret, img_out = original_vid.read()

                ###################################
                ########## REGULAR VIDS ###########
                ###################################
                # save non-annotated video
                if False:
                    video_out.write(img_out)

                # crop the image
                img_out_cropped = img_out.copy()
                img_out_cropped[:, :min_x] = 0
                img_out_cropped[:, max_x:] = 0
                img_out_cropped[:min_y] = 0
                img_out_cropped[max_y:] = 0

                video_out_cropped.write(img_out_cropped)

                ###################################
                ######### ANNOTATED VIDS ##########
                ###################################
                #
                if False:
                    #
                    img_out = plot_feats(
                        feats,
                        colors,
                        img_out,
                        dot_size)
                    #
                    if True:
                        #
                        video_out_annotated.write(img_out)

                        img_out_cropped = img_out.copy()
                        img_out_cropped[:, :min_x] = 0
                        img_out_cropped[:, max_x:] = 0
                        img_out_cropped[:min_y] = 0
                        img_out_cropped[max_y:] = 0

                        #
                        video_out_annotated_cropped.write(img_out_cropped)

            # video_out_annotated.write(img_out*0)

        # break

    video_out_cropped.release()
    if False:
        video_out.release()
        video_out_annotated.relea
        se()
        video_out_annotated_cropped.release()

    #
    print("DONE...")

#



#
def make_hybrid_video_from_list(fnames_slp, fnames_vids, fname_out_cropped):

    # make new video video settings
    window = 100
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # #fourcc = cv2.VideoWriter_fourcc(*'X264')
    # #fourcc = cv2.VideoWriter_fourcc(*args["codec"])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', '')

    # make output video
    #fname_out_cropped = os.path.split(fnames_slp[0])[0] + '/id_switches_cropped.mp4'
    if os.path.exists(fname_out_cropped):
        print ("    movie already generated... exiting")
        return fname_out_cropped
    #
    #print("   fname output hybrid movie: ", fname_out_cropped)

    # make new video video settings
    size_vid = np.int32(np.array([900, 700]))
    fps_out = 1

    #
    video_out_cropped = cv2.VideoWriter(fname_out_cropped,
                                        fourcc,
                                        fps_out,
                                        (size_vid[0], size_vid[1]),
                                        True)

    # loop over each slp file
    for fname_slp, fname_vid in tqdm(list(zip(fnames_slp, fnames_vids)), desc='making hybrid movie'):

        # load all frames:
        all_frames = np.load(fname_vid[:-4] + '_all_frames_with_switches.npy')

        # load movie name
        idx = fname_slp.find("compressed")
        #print (fname_slp[:idx] + "*cropped*.mp4")

        fname_movie = glob.glob(fname_slp[:idx] + "*cropped*.mp4")[0]
       # print("fname found: ", fname_movie)

        ############################################
        ############################################
        ############################################
        track = Track(fname_slp)
        track.track_type = 'features'
        track.use_dynamic_centroid = True  # True: alg. serches for the first non-nan value in this body order [2,3,1,0,4,5]
        track.load_tracks()

        # load current vid
        original_vid = cv2.VideoCapture(fname_movie)

        # set frames to new ones
        #for frames in tqdm(all_frames, desc='adding frames to movies'):
        for frames in all_frames:

            if len(frames) == 0:
                continue

            #
            animal_id = frames[2]

            #
            original_vid.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

            #
            for z in range(2):

                # find centre of animal black out surrounding area
                feats = track.tracks[frames[0] + z]

                #
                if z == 0:
                    feats_centre = np.nanmedian(feats[animal_id], 0)

                    #
                    min_x = int(max(feats_centre[0] - window, 0))
                    max_x = int(min(feats_centre[0] + window, 900))
                    min_y = int(max(feats_centre[1] - window, 0))
                    max_y = int(min(feats_centre[1] + window, 700))

                ret, img_out = original_vid.read()

                ###################################
                ########## REGULAR VIDS ###########
                ###################################
                # crop the image
                img_out_cropped = img_out.copy()
                img_out_cropped[:, :min_x] = 0
                img_out_cropped[:, max_x:] = 0
                img_out_cropped[:min_y] = 0
                img_out_cropped[max_y:] = 0

                video_out_cropped.write(img_out_cropped)

                ###################################
                ######### ANNOTATED VIDS ##########
                ###################################

    video_out_cropped.release()

    return fname_out_cropped

#
def make_human_label_only_video(idxs,
                                fnames_videos,
                                fname_out = None):
    ''' Function makes a hybrid video from a list of video names and frame indexes.
        This video accompanies the compressed .slp file made that contains only human labeled
        frames.

    '''

    if os.path.exists(fname_out):
        print ("   video already exists ... exiting")
        return

    #
    #root_dir = os.path.split(fnames_videos[0])[0] + '/'

    # make new video video settings
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'X264')
    # #fourcc = cv2.VideoWriter_fourcc(*args["codec"])
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', '')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    #
    print ("Fname hybrid movie: ", fname_out)

    # make new video video settings
    size_vid = np.int32(np.array([900, 700]))
    fps_out = 1

    #
    video_out = cv2.VideoWriter(fname_out,
                                fourcc,
                                fps_out,
                                (size_vid[0], size_vid[1]),
                                True)

    # loop over each slp file
    ctr = 0
    for idx, fname_video in tqdm(zip(idxs, fnames_videos), desc='total frames: '+str(len(idxs))):
        # load current vid
        original_vid = cv2.VideoCapture(fname_video)
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, idx)

        #
        ret, img_out = original_vid.read()

        #
        video_out.write(img_out)

        # fname_video_old = fname_video
        original_vid.release()

        ctr += 1

    #
    print ("# of images saved ", ctr)
    video_out.release()



#
def make_human_label_only_slp(fname_merged_video,
                              fname_slp,
                              human_slp_dir):

    ''' This function reduces an .slp file containing human labels and other frames
        to a much shroter file containing only human labeled frames.
        It also saves the frame indexes and the video names so that the required hybrid.avi file
        can also be made to match the shorter .slp file

    '''

    #
    merged_video = sleap.load_video(fname_merged_video)
    #print (merged_video)

    # Load saved labels.
    first_labels = None

    labels = sleap.load_file(fname_slp)

    # loop over
    new_frames = []
    frames_idx = []
    fnames = []
    ctr1 = 0
    for lf in labels:

        # Catalin added
        if lf.has_user_instances == False:
            continue

        # save indexes in order to make the hybrid videos
        frames_idx.append(lf.frame_idx)

        #
        temp_fname_video = lf.video.backend.filename
        #print (" original video: ", temp_fname_video)
        _, fname = os.path.split(temp_fname_video)
        root_dir = os.path.split(fname_merged_video)[0]

        vid_fname = os.path.join(root_dir,
                                 human_slp_dir,
                                 fname)

        #print (" corrected video: ", vid_fname)
        #
        fnames.append(vid_fname)

        # Update reference to merged video.
        lf.video = merged_video
        lf.video.backend.filename = fname_merged_video

        # Update frame index to the frame number within the merged video.
        lf.frame_idx = ctr1

        # Append the labeled frame to the list of frames we're keeping from these labels.
        new_frames.append(lf)

        #
        ctr1 += 1

    # format labels and save them
    merged_labels = sleap.Labels(new_frames)
    fname_slp_out = os.path.split(fname_merged_video)[0] + '/human_labels.slp'
    merged_labels.save(fname_slp_out)

    # save indexes and video names to make hybrid video
    np.save(os.path.split(fname_merged_video)[0] + '/human_frames_idx.npy', frames_idx)
    np.save(os.path.split(fname_merged_video)[0] + '/human_fnames.npy', fnames)


    return fname_slp_out
#
def detect_id_switch_parallel(animal_id,
                              tracks,
                              max_n_nans=0):
    #
    other_animal_ids = np.int32(np.delete(np.arange(6), animal_id))

    # select animal_id track
    track = tracks[:, animal_id]

    # calculate the feature-wise distance between current frame and next frame
    ctr = 0
    frame_ids = []
    corner = 300
    for k in range(tracks.shape[0] - 1):
        feats0 = track[k]

        for a in other_animal_ids:
            try:
                feats1 = tracks[k + 1, a]
            except:
                continue
            #
            dists = np.linalg.norm(feats1 - feats0, axis=1)

            #
            med = np.nanmedian(dists)

            # if the median distance is low
            if med < 15:
                #
                n = np.count_nonzero(np.isnan(dists))
                # if n<=(6-min_n_features):
                if n == max_n_nans:
                    # also skip switches in the huddle
                    if np.nanmedian(feats0[:, 0]) > corner:
                        if np.nanmedian(feats0[:, 1]) > corner:
                            frame_ids.append([k, k + 1, animal_id, a])

                        ctr += 1
                # else:
                #    print ("n toolow: ", n)
    #print("found: ", len(frame_ids), " switches")
    return frame_ids


def make_slp(fname_slp):

    labels = sleap.load_file(fname_slp)

    labels.videos  # Use in ipython to print list of videos in the labels object
    n_videos = len(labels.videos)

    #
    for video_idx in trange(n_videos):
        #video_idx =   # Change this select the video of interest from the labels.video list
        video = labels.videos[video_idx]
        labeled_frames_from_video = labels.get(video)

        # Loop through all labeled frames (from a specific video)
        n_frames = 0
        for ctr1, lf in enumerate(labeled_frames_from_video):

            # Loop through all user instances in the current labeled frame
            for ctr2, inst in enumerate(lf.user_instances):

                #
                remove_inst = False
                points_array = inst.points_array  # Returns a (num_nodes, 2) np.recarray

                # SQUARE EXCLUDE this checks for any features at all in the exclusion radius
                if True:
                    idx = np.where(np.isnan(points_array.sum(1))==False)[0]
                    points_array = points_array[idx]

                    # check the x and y locatoins
                    idx1 = np.where(points_array[:,0]<huddle_block[0])[0]
                    idx2 = np.where(points_array[:,1]<huddle_block[1])[0]

                    # if any point is in both lists then it's in the huddle zone
                    if set(idx1) & set(idx2):
                        lf.instances.remove(inst)
                        #
                    #dist = np.linalg.norm(points_array-huddle_location, axis=1)
                    #dist = np.min(dist)


                #print ('')

    # Save the modified slp
    new_filename = fname[:-4]+"_deleted.slp"  # Use a different path from the current slp to be safe
    labels.save_file(labels, new_filename)


def plot_flip2(n,
               frame0,
               frame1,
               track0,
               track1):
    clrs = ['blue', 'red', 'pink', 'cyan', 'magenta', 'lightgreen',
            ]

    # print ("track0: ", track0)

    plt.figure(figsize=(15, 15))

    ax = plt.subplot(121)
    ax.imshow(frame0)

    for k in range(6):
        feats0 = track0[k]
        # print("Feats0: ", feats0, feats0.shape)
        if len(feats0.shape) > 1:
            plt.scatter(feats0[:, 0],
                        feats0[:, 1],
                        c=clrs[k])

    #
    ax = plt.subplot(122)
    ax.imshow(frame1)
    for k in range(6):
        feats1 = track1[k]
        # print("Feats1: ", feats1)

        if len(feats1.shape) > 1:
            plt.scatter(feats1[:, 0],
                        feats1[:, 1],
                        c=clrs[k])

    plt.savefig('/home/cat/example_' + str(n) + '.png')
    plt.close()


#
def plot_feats(feats,
               colors,
               img_out,
               dot_size):
    for a in range(feats.shape[0]):
        color = colors[a]  # (255, 0, 0)
        # print ("animal id: ", a, ", color: ", color)
        for f in range(feats.shape[1]):
            x = feats[a, f, 0]
            y = feats[a, f, 1]

            if np.isnan(x) or np.isnan(y):
                continue

            x = int(x)
            y = int(y)

            # plot cicrl
            center_coordinates = (x, y)
            img_out = cv2.circle(img_out,
                                 center_coordinates,
                                 dot_size,
                                 color,
                                 -1)
    # print ('')
    return img_out


#
def detect_id_switch_no_plot(fname_movie,
                             tracks,
                             animal_id,
                             min_n_features=4):
    #
    other_animal_ids = np.int32(np.delete(np.arange(6), animal_id))

    # select animal_id track
    track = tracks[:, animal_id]

    # calculate the feature-wise distance between current frame and next frame
    ctr = 0
    frame_ids = []
    corner = 300
    for k in trange(tracks.shape[0] - 1, desc='finding id switches in movie'):
        feats0 = track[k]

        for a in other_animal_ids:
            feats1 = tracks[k + 1, a]

            #
            dists = np.linalg.norm(feats1 - feats0, axis=1)

            #
            med = np.nanmedian(dists)
            # n = np.count_nonzero(np.isnan(dists))
            # if (6-n)>0:
            #     print ("n: ", n, ", median distances: ", med, ", dists: ", dists)

            #
            if med < 15:
                #
                n = np.count_nonzero(np.isnan(dists))
                # if n<=(6-min_n_features):
                if n == 0:
                    # also skip switches in the huddle
                    if np.nanmedian(feats0[:, 0]) > corner:
                        if np.nanmedian(feats0[:, 1]) > corner:
                            frame_ids.append([k, k + 1])

                        ctr += 1
                # else:
                #    print ("n toolow: ", n)
    print("found: ", len(frame_ids), " switches")
    return frame_ids

def compare_skeletons(skeleton: Skeleton,
                      new_skeleton: Skeleton) -> Tuple[List[str], List[str]]:

    delete_nodes = []
    add_nodes = []
    if skeleton.node_names != new_skeleton.node_names:
        # Compare skeletons
        base_nodes = skeleton.node_names
        new_nodes = new_skeleton.node_names
        delete_nodes = [node for node in base_nodes if node not in new_nodes]
        add_nodes = [node for node in new_nodes if node not in base_nodes]

    return delete_nodes, add_nodes


class DatabaseLoader():

    #
    def __init__(self, fname, network_type):
	    
        print ("reading excel spreasheet...")
        df = pd.read_excel(fname, engine='openpyxl')
        # print ("DF: ", df)

        self.df = df
        self.network_type = network_type

        #
        self.make_deleted_slp_files_parallel = make_deleted_slp_files_parallel
        self.make_hybrid_video_from_list = make_hybrid_video_from_list
        self.find_id_switches_parallel = find_id_switches_parallel
        self.make_hybrid_slp = make_hybrid_slp
        self.make_human_label_only_slp = make_human_label_only_slp
        self.make_human_label_only_video = make_human_label_only_video

#    #
    def make_id_switch_files(self):

        # parse
        fnames_slp = []
        fnames_vids = []
        print (" parsing dataframe of size: ", len(self.df))
        ctr=0
        for dd in self.df.iterrows():

            #
            if dd[1]['NN Type']==self.network_type:

                fname_vid = dd[1]['Filename']
                fname_slp = dd[1]['Slp filename']
                fnames_slp.append(os.path.join(self.root_dir,
                                               self.input_dir,
                                               fname_slp))
                fnames_vids.append(os.path.join(self.root_dir,
                                                self.input_dir,
                                                fname_vid))
                                                
            ctr+=1
            if ctr>5000:
                break
        #
        print ("\nrunning .slp preprocessing and id_switch detection ...")
        self.find_id_switches_parallel(fnames_slp,
                                       fnames_vids,
                                       self.n_cores)

        # make the id-switch only deleted slp files
        print ("\nmaking shortened id_switch .slp files...")
        self.make_deleted_slp_files_parallel(fnames_slp,
                                             fnames_vids,
                                             self.n_cores)

        # make the hybrid video containing all the id-switch frames
        print ("\nmaking shortened id_switch video file...")
        self.fname_id_switch_vid = os.path.join(self.root_dir,
                                               'id_switches_cropped.mp4')
        self.make_hybrid_video_from_list(fnames_slp,
                                         fnames_vids,
                                         self.fname_id_switch_vid)

        # combine all deleted .slp files into a single slp
        #  and link to the hybrid video
        self.make_hybrid_slp(fnames_slp,
                             self.fname_id_switch_vid)

        #

  #
    def clean_human_slp(self):

        self.fname_slp = os.path.join(self.root_dir,
                                      self.human_slp_dir,
                                      self.fname_slp_human_labels)


        # remove non-human labels from .slp file
        #  and save movie indexes etc.
        self.fname_human_slp = os.path.join(self.root_dir,
                                            'human_labels.slp')
        self.fname_human_vid = os.path.join(self.root_dir,
                                            'human_labels.avi')
        self.fname_human_slp = self.make_human_label_only_slp(self.fname_human_vid,
                                                              os.path.join(self.root_dir,
                                                                           self.human_slp_dir,
                                                                           self.fname_slp_human_labels),
                                                              self.human_slp_dir)

        #
        self.idx = np.load(self.fname_human_slp.replace("human_labels.slp",
                                                         "human_frames_idx.npy"))
        self.fnames = np.load(self.fname_human_slp.replace("human_labels.slp",
                                                           "human_fnames.npy"))

        #

        self.make_human_label_only_video(self.idx,
                                         self.fnames,
                                         self.fname_human_vid)

    def merge_slp_files(self):
              
        # human slp file
        fname_slp1 = self.fname_human_slp

        # id switch slp file
        fname_slp2 = self.fname_id_switch_vid.replace("mp4","slp")

        #
        fname_hybrid_video = os.path.split(fname_slp1)[0]+"/merged.avi"

        #
        merged_video = sleap.load_video(fname_hybrid_video)

        #
        fname_hybrid_slp = fname_hybrid_video[:-4] + ".slp"

        # Load labels
        slp_files = [fname_slp1,
                     fname_slp2
                    ]
        first_labels = sleap.load_file(slp_files[0])
        #slp_labels = [first_labels, sleap.load_file(slp_files[1])]#match_to=first_labels)]
        slp_labels = [first_labels, sleap.load_file(slp_files[1], match_to=first_labels)]

        # Compare skeletons
        assert len(slp_labels[0].skeletons) <= 1  # Ensure each project only has a single skeleton
        assert len(slp_labels[1].skeletons) <= 1
        delete_nodes, add_nodes = compare_skeletons(slp_labels[0].skeleton, slp_labels[1].skeleton)
        assert len(delete_nodes) == 0  # Ensure project skeletons are identical
        assert len(add_nodes) == 0
        the_skeleton = slp_labels[0].skeleton or slp_labels[1].skeleton

        n_frames = 0
        all_frames = []
        reference_tracks = {}

        #
        ctrl=0
        for ctrv, labels in enumerate(slp_labels):

            # start counter for id switch .slp
            if ctrv==1:
                ctrl=-1

            #
            new_frames = []
            ctri = 0
            for i, lf in enumerate(labels):
                ctrl += 1

                # increase counter during id switch editing and skip if required
                if ctrv==1:
                    if (ctrl%self.subsample==0) or (ctrl%self.subsample==1):
                        pass
                    else:
                        continue

                # Update reference to merged video.
                lf.video = merged_video

                # Update frame index to the frame number within the merged video.
                lf.frame_idx = n_frames + ctri

                # # Update the track reference to use the reference tracks to prevent duplication.
                for instance in lf:
                    if instance.track is not None:
                        if instance.track.name in reference_tracks:
                            instance.track = reference_tracks[instance.track.name]
                        else:
                            reference_tracks[instance.track.name] = instance.track

                    # Set the instance to reference the correct instance of `Skeleton`
                    instance.skeleton = the_skeleton

                # Append the labeled frame to the list of frames we're keeping from these labels.
                new_frames.append(lf)
                ctri+=1

            all_frames.extend(new_frames)
            n_frames += len(new_frames)

        #merged_labels = sleap.Labels(all_frames)

        merged_labels = sleap.Labels(all_frames)
        merged_labels.skeletons = [the_skeleton]

        assert all([inst.skeleton == the_skeleton for inst in merged_labels.instances()])

        # save stuff
        merged_labels.save(fname_hybrid_slp)

    #
    def merge_human_and_id_switches(self):

        #
        self.merge_video_files()

        #
        self.merge_slp_files()
        
    #
    def merge_video_files(self):

        fnames = [
            self.fname_human_slp.replace('slp','avi'),
            self.fname_id_switch_vid
        ]

        # make new video video settings
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        # #fourcc = cv2.VideoWriter_fourcc(*args["codec"])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', '')

        # load videos
        fname_out = os.path.join(os.path.split(self.fname_id_switch_vid)[0]+"/merged.avi")

        # make new video video settings
        size_vid = np.int32(np.array([900, 700]))
        fps_out = 1

        #
        video_out = cv2.VideoWriter(fname_out,
                                    fourcc,
                                    fps_out,
                                    (size_vid[0], size_vid[1]),
                                    True)

        # loop over each video file
        ctrl = 0
        #for idx, fname_video in tqdm(zip(idxs, fnames_videos)):
        for ctrv, fname_video in enumerate(fnames):
            print ("processing: ", fname_video)

            # load either original labels or id switch videos
            original_vid = cv2.VideoCapture(fname_video)

            # start the counter for the id switch video
            if ctrv==1:
                ctrl=-1

            while True:
                ctrl+=1
                ret, img_out = original_vid.read()
                # exit on end of video
                if ret==False:
                    break

                # skip every required frame
                if ctrv==1:
                    if (ctrl%self.subsample==0) or (ctrl%self.subsample==1):
                        pass
                    else:
                        continue


                #
                video_out.write(img_out)

            # fname_video_old = fname_video
            original_vid.release()

            #ctr += 1

        video_out.release()

        print("done merging video files...")

