
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import scipy
import scipy.spatial
from scipy import signal
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

import numpy as np
import os
import cv2
from tqdm import trange

import sleap

import h5py


# Simplified motion predictor without variance/bayes rule updates
# as our measurements are NOT noisy (a simplified Kalman filter)


class Correct():

    def __init__(self, tracks, verbose):

        #
        self.tracks = tracks
        print ("input into Predict: ", self.tracks.shape)

        #
        self.tracks_fixed = np.zeros(self.tracks.shape,'float32')

        #
        self.verbose=verbose

    def get_positions(self):

        '''     ###############################################
                ###### GET ANIMAL POSITIONS AT TIME T #########
                ###############################################
                # get x,y positions of each animal
                # for most recent 3 values
                # eventually, add uncertainty based on how far back we go; BUT COMPLICATED
        '''

        # Problem: we don't always have data in sequence so need to work on interpolating also
        # make array [n_animals, n_time_points, 2]
        n_time_points = 3
        self.pos= np.zeros((self.tracks.shape[1], n_time_points, 2), 'float32')
        for a in range(self.tracks.shape[1]):

            # TO IMPLEMENT LATER
            # loop back in time to find nearest non zero vals;
            #for t1 in np.arange(t,0,-1):
            #    if np.isnan(self.tracks[t1,a,0])==False:
            #        self.pos[a]= self.tracks[t1-3:t1,a]
            #        break

            self.pos[a]=self.tracks[self.time-3:self.time,a]

            # temporarily replace read positions that are nans with most recent non-nan val
            idx = np.where(np.isnan(self.pos[a]))[0]
            for id_ in idx:
                idx2 = np.where(np.isnan(self.tracks_fixed[:self.time,a])==False)[0]
                self.pos[a] = self.tracks_fixed[idx2[-1],a]
#             #
#             idx = np.where(np.isnan(self.pos[a]))[0]
#             if idx.shape[0]>0:
#                 print ('found nan data', a, self.pos[a])
#                 self.pos[a,idx]==self.pos[a,2]

        if self.verbose:
            print ("self.positions; ", self.pos)


    def compute_vel_acceleration(self):
        '''###############################################
        ###### COMPUTE VELOCITY AND ACCELERATION ######
        ###############################################
        # compute velocity from 2 values
        # compute acceleration from 3 values
        '''

        self.vel = np.zeros((self.tracks.shape[1], 2), 'float32')
        self.acc = np.zeros((self.tracks.shape[1], 2), 'float32')

        # turn off predictive dynamics, seems to be mostly problematic
        if self.dynamics_off_flag==False:

            for a in range(self.tracks.shape[1]):
                vel = self.pos[a,2]-self.pos[a,1]
                acc = (self.pos[a,2]-self.pos[a,1])- \
                      (self.pos[a,1]-self.pos[a,0])
                self.vel[a]=vel
                self.acc[a]=acc

            # set all vel and acc nans to zeros
            idx = np.where(np.isnan(self.vel))
            self.vel[idx]=0
            idx = np.where(np.isnan(self.acc))
            self.acc[idx]=0

        # use dampening on acceleration vector - animals can't reach inifite velocities
        #self.acc[:]=0

        if self.verbose:
            print ("vel; ", self.vel)
            print ("accel; ", self.acc)


    def compute_position(self,
                         p0,
                         vel0,
                         acc0):

        # for now assume dt==1 always
        # return p0 + vel0*dt + 0.5*acc0*(dt**2)

        return p0 + vel0 + 0.5*acc0


    def predict_location(self):

        '''###############################################
        ###### PREDICT LOCATION OF EACH ANIMAL ########
        ###############################################
        # predict location at the next time step
        # may also wish to run prediction backwards in time as well
        '''

        self.pos_pred = np.zeros((self.tracks.shape[1], 2), 'float32')

        for a in range(self.tracks.shape[1]):
            self.pos_pred[a]= self.compute_position(self.pos[a,2],
                                                self.vel[a],
                                                self.acc[a])
        if self.verbose:
            print ("self.pos_predicted: ", self.pos_pred)


    def compute_distances(self):

        '''#######################################################
        ### DISTANCE MATRIX BETWEEN OBSERVATIONS/PRECITIONS ###
        #######################################################
        # compute distance matrix
        '''

        # read locations at time t+1
        self.pos_read = self.tracks[self.time+1]

        if self.verbose:
            print ("self.pos_read at t+1:", self.pos_read)

        # compute pairwise dist between predicted and read
        cost = scipy.spatial.distance.cdist(self.pos_pred, self.pos_read)
        if self.verbose:
            print ("Cost: ", cost)

        # NEED TO SPEED THIS UP
        idx = np.where(np.isnan(cost))
        cost[idx]=1E4

        #
        _, assignment = linear_sum_assignment(cost)
        self.assignment = assignment

        #
        if self.verbose:
            print ("assignemnt: ", assignment)
            print ("costs: ", cost[_,assignment])


    def update_positions(self):

        ''' ###############################################
            ########### HUNGARIAN ASSIGNMENT ##############
            ###############################################
        #    assign observed data to each animal

            if nan, just use previous
        '''

        #
        for a in range(self.assignment.shape[0]):

            # check if updated position is a nan and replace it with most recent non-nan
            #print (self.pos_read[self.assignment[a]])
            temp1 = self.pos_read[self.assignment[a]]
            if np.isnan(temp1[0]):
                idx2 = np.where(np.isnan(self.tracks_fixed[:self.time,a])==False)[0]
                temp1 = self.tracks_fixed[idx2[-1],a]

            # shift 1 cell over and grab new val
            #print (self.pos[a,1:3].shape, temp1.shape)
            self.pos[a] = np.vstack((self.pos[a,1:3], temp1))

            # save data
            self.tracks_fixed[self.time,a]=self.pos_read[self.assignment[a]]

        if self.verbose:
            print ("new hugnarian positoins: ", self.pos)

#
# def find_nearest_forward(val, array2, window=1000.0):
#     diff_temp = array2-val
#     idx = np.where(diff_temp>0)[0]
#
#     if idx.shape[0]==0 or diff_temp[idx[0]]>window:
#         return 1E5, np.nan
#
#     loc = idx[0]
#     diff = diff_temp[idx[0]]
#     return diff, loc
#
# #
# def find_nearest_backward(val, array2, window=1000.0):
#     diff_temp = val-array2
#     idx = np.where(diff_temp>0)[0]
#
#     if idx.shape[0]==0 or diff_temp[idx[0]]>window:
#         return 1E5, np.nan
#
#     loc = idx[0]
#     diff = diff_temp[idx[0]]
#     return diff, loc


#

    def get_chunk_info(track,
                       animal_id1,
                       t,
                       verbose=False):


        ################### CURRENT ######################
        track_local = np.array(track.tracks_chunks[animal_id1])
        chunk_current = np.where(np.logical_and(t>=track_local[:,0], t<=track_local[:,1]))[0]
        times_current = track_local[chunk_current][0]
        locs_current = track.tracks_spine_fixed[times_current, animal_id1]

        if verbose:
            print ("Active animal:", animal_id1)
            print ("locs active; ", locs_current)
            print ("times active animal: ", times_current)

        return times_current, locs_current, chunk_current

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
        cost = np.zeros(self.n_animals, 'float32')  #
        in_chunk = np.bool(4)       # this variable checks if other animal chunks partially overlap
                                    # with current chunk; if so, need to skip them;
        for animal_id1 in range(self.n_animals):

            # grab chunk info
            track_local = np.array(self.tracks_chunks[animal_id1])

            ################### CURRENT ######################
            # if current track very long, make sure you don't change it
            chunk_id = np.where(np.logical_and(t>=track_local[:,0], t<=track_local[:,1]))[0]
            if chunk_id.shape[0]>0:
                times_current = track_local[chunk_id][0]

                if times_current[1]-times_current[0]>self.safe_chunk_length:
                    cost[animal_id1] = 1E5
                    continue


            ################# PREVIOUS #################
            chunk_ids = np.where(track_local[:,1]<t)[0]

            # ensure we only look at sufficiently long track,s do not compare to singles/ or few
            z=-1
            while True:
                try:
                    chunk_id_prev = chunk_ids[z]
                    times_prev = track_local[chunk_id_prev]
                    if times_prev[1]-times_prev[0]>min_chunk_len:
                        locs_prev = self.tracks_spine_fixed[times_prev, animal_id1]
                        break
                except:
                    print ("broken: chunk_ids", chunk_ids)
                    chunk_id_prev = None
                    times_prev = np.array([-1E6,-1E6])
                    locs_prev = np.array([1E5,1E5],'float32')
                    break
                z-=1

            ################### NEXT ###################
            chunk_ids = np.where(track_local[:,0]>t)[0]
            z=0
            while True:
                try:
                    chunk_id_next = chunk_ids[z]
                    times_next = track_local[chunk_id_next]
                    if times_next[1]-times_next[0]>min_chunk_len:
                        locs_next = self.tracks_spine_fixed[times_next, animal_id1]
                        break
                except:
                    chunk_id_next = None
                    times_next = np.array([1E6,1E6])
                    locs_next = np.array([1E5,1E5],'float32')
                    break
                z+=1


            # check if partial overlapping chunk the need to skip entirely;
            # we do not want to swap parts of tracks as we assume continous tracks have single identities
            print ("333333333333333333333333333333333333333Overlapping chunk/animal, skip it")
            print ("times_prev: ", times_prev, "  chunk_current: ", chunk_current)
            if (times_prev[1] >= t) or (times_next[0]<=chunk_current[1]):
                if verbose:
                    print ("Overlapping chunk/animal, skip it")
                    print ("times_prev: ", times_prev, "  chunk_current: ", chunk_current)
                    cost[animal_id1] = 1E5
                    continue


            # make cost matrix
            # find distance between active chunk start and prev chunk:
            if times_active_animal[0]-times_prev[1]<self.time_threshold:
                c_prev = np.linalg.norm(locs_active_animal[0]-locs_prev[1])
            else:
                c_prev = 1E5

            #
            if times_next[0]-times_active_animal[1]<self.time_threshold:
                c_next = np.linalg.norm(locs_active_animal[1]-locs_next[0])
            else:
                c_next= 1E5

            #
            if verbose:
                print ("time: ", t)
                print ("animal id: ", animal_id1)
                #print ("  current: ", chunk_id, "times: ", times_current, " locs: ", locs_current)
                print ("  prev: ", chunk_id_prev, "times : ", times_prev, " locs: ", locs_prev)
                print ("  next: ", chunk_id_next, "times : ", times_next, " locs: ", locs_next)

                print ("times prev: ", times_prev)
                print ("cprev: ", c_prev, "  cnext: ", c_next)

            #cost[animal_id1] = max(c_next, c_prev)
            #c_next*=1.5
            cost[animal_id1] = min(c_next, c_prev)
            #cost[animal_id1] = c_prev

#
    def swap_chunks(self,
                    correct_id,    # the correct assignemnt
                    animal_current, # the current assignemnt
                    times_current,
                    chunk_current,
                    verbose=False):

        if verbose:
            print ("swapping: ", correct_id, " with ", animal_current)
        ##########################################
        ##########################################
        ##########################################
        # hold memory
        temp_track = self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                                        animal_current].copy()
        #
        temp2_track = self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                                         correct_id].copy()

        #
        self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                           animal_current]= temp2_track

        # replace correct id with temp
        self.tracks_spine_fixed[times_current[0]:times_current[1]+1,
                           correct_id]= temp_track


        ##########################################
        ##########################################
        ##########################################
        #
        try:
            temp_chunk = self.tracks_chunks_fixed[animal_current][chunk_current].copy()
        except:
            print ('track.tracks_chunks_fixed[: , ', self.tracks_chunks_fixed[animal_current].shape)
            print ("animal_current: ", animal_current)
            print ("chunk_current: ", chunk_current)
            #return track

        #
        self.tracks_chunks_fixed[correct_id] = np.vstack((
                    self.tracks_chunks_fixed[correct_id], temp_chunk))
        idx = np.sort(self.tracks_chunks_fixed[correct_id][:,0])[0]
        self.tracks_chunks_fixed[correct_id]= self.tracks_chunks_fixed[correct_id][idx]

        #
        self.tracks_chunks_fixed[animal_current] = np.delete(
                                self.tracks_chunks_fixed[animal_current],
                                chunk_current,axis=0)

    #
    def fix_tracks(self,
                   t, t_end, time_threshold,
                  safe_chunk_length,
                  min_chunk_len,
                  animal_current,
                  track,
                  n_animals,
                  verbose=True):

        from tqdm import tqdm
        pbar = tqdm(total=self.tracks_spine.shape[0])

        #
        while True:  # <-------- need to advance in time slowly by indexing through each animal chunk
            pbar.update(t)

            # grab location for current chunk and animal being aanlyzed:
            times_current, locs_current, chunk_current = self.get_chunk_info(track,
                                                                        animal_current,
                                                                        t,
                                                                        verbose)
            print (t, animal_current, chunk_current)
            # check to not change tracks that are very long:
            if times_current[1]-times_current[0]<safe_chunk_length:

                # get cost:
                cost = self.get_cost(track, t,
                                     chunk_current,
                                     times_current,
                                     locs_current,
                                    min_chunk_len,
                                    verbose)

                if verbose:
                    print ("COST: ", cost)

                # closest swap attempt one animal at a time
                if np.min(cost)!=1E5:

                    #
                    correct_id = np.argmin(cost)

                    # check if any data needs to be swapped
                    if correct_id!= animal_current:
                        track = self.swap_chunks(track,
                                            correct_id,    # the correct assignemnt
                                            animal_current, # the current assignemnt
                                            times_current,
                                            chunk_current,
                                            verbose) # the times to swap

            ##################################################
            ######### FIND NEXT SECTION OF TRACK #############
            ##################################################
            # find next track start and ends
            temp = []
            for i in range(n_animals):
                # grab all tracks for each animal
                try:
                    temp2 = np.array(track.tracks_chunks[i])
                except:
                    print ("I: ", i)
                    print ("track.tracks_chunks[i]: ", track.tracks_chunks[i])

                # append all track starts that occur after current time;
                # TODO: this may skip tracks that start at the same time...
                next_chunk_time = np.where(temp2[:,0]>t)[0]

                #
                temp.append(temp2[next_chunk_time,0][0])

            # find the nearest start in time;
            t = np.min(temp)

            # select which animal the chunk belongs to
            animal_current = np.argmin(temp)

            #
            if verbose:
                print ('')
                print ('')
                print ("##########################################################################")
                print ("TIME: ", t, " active_animal: ", animal_current, " (temp: )", temp)


            if t>t_end:
                break

        pbar.close()
