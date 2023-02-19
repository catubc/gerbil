
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math

#

from numpy import arccos, array
from numpy.linalg import norm

import numpy as np
import os
from tqdm import trange
import parmap
import glob
from sklearn.decomposition import PCA
#import umap
# import seaborn as sns
# import pandas as pd

# import pickle
# #
from tqdm import tqdm

# import sleap


import sklearn.experimental
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#
from scipy.io import loadmat
import scipy
import scipy.ndimage

class Ethogram():

    def __init__(self, fname_slp):

        #
        self.fname_slp = fname_slp

        #
        self.root_dir = os.path.split(self.fname_slp)[0]

        #
        self.fps = 24

        #
        self.rad_to_degree= 57.2958

        #
        #self.feature_ids = np.array([0,5,6,7,8,9])

        self.feature_ids = np.array([1,2,3,0,5,6])

    #
    def get_durations_single_frame(self, data):

        # find starts and ends from the non-nan Indexes
        starts = []
        ends = []
        starts.append(data[0])
        for d in range(1,data.shape[0],1):
            if (data[d]-data[d-1])>1:
                ends.append(data[d-1]+1)
                starts.append(data[d])

        #
        starts = np.array(starts)
        ends = np.array(ends)
        if starts.shape[0]==ends.shape[0]+1:
            starts=starts[:-1]

        durations = (ends - starts)

        # loop over the
        vectors = []
        for k in range(durations.shape[0]):
            temp = durations[k]
            if temp>=self.min_duration:

                vectors.append([starts[k], ends[k]])

        vectors = np.vstack(vectors)

        return vectors

    #
    def get_lengths(self, animal_id):

        # nose + spine1-5 data
        feature_ids = self.feature_ids #

        #
        # root_dir = self.root_dir # '/media/cat/1TB/dan/cohort1/'

        #
        if self.fixed_tracks_flag:
            fname_in = os.path.join(self.fname_slp.replace('.slp','_fixed.npy'))
        else:
            fname_in = os.path.join(self.fname_slp.replace('.slp','.npy'))

        d = np.load(fname_in)

        #
        print ("feature ids: ", feature_ids)
        d = d[:,animal_id, feature_ids]
        print ("ANIMAL Specific data: ", d.shape)

        #
        two = []
        six = []
        headnose = []
        headspine = []
        for k in trange(d.shape[0]):
            # grab frame
            temp = d[k]

            # find nans
            idx1 = np.where(np.isnan(temp[:,0])==False)[0]
            idx2 = np.where(np.isnan(temp[:,1])==False)[0]

            # make sure # of nans in x is same as y (Sleap error)
            if idx1.shape[0]!=idx2.shape[0]:
                continue

            #
            if idx1.shape[0]>=2:
                two.append(k)
                if idx1[0]==0 and idx1[1]==1:
                #if idx1[0]==self.feature_ids[0] and idx1[1]==self.feature_ids[1]:
                    headnose.append(k)
                if 1 in idx1 and 2 in idx2:
                    headspine.append(k)

            # if all 6 features found
            if idx1.shape[0]==6:
                six.append(k)

        #
        two = np.array(two)
        six = np.array(six)
        headnose = np.array(headnose)
        headspine = np.array(headspine)
        print ("two: ", two.shape, " six: ", six.shape)

        print ("copy headspine to headnose ")
        headnose = headspine
        return two, six, headnose

    #
    def centre_and_align2_pairwise(self, data, centre_pt = 0, rotation_pt = 1):

        # centre the data on the nose
        # data[:,0] -= data[centre_pt,0]
        # data[:,1] -= data[centre_pt,1]
        #print ("data; ", data.shape)

        # align all data in window to first time point
        translation_pt = data[0,centre_pt]
        data -= translation_pt

        # get angle nose and head for first frame
        angle = -np.arctan2(*data[0,rotation_pt].T[::-1])-np.pi/2

        # get rotation for nose and head for first frame
        rotmat = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])

        # Apply rotation to all data in window
        data_rot = []
        for p in range(data.shape[0]):
            data_rot.append((rotmat @ data[p].T).T)

        data_rot = np.array(data_rot)

        idx = np.where(np.isnan(data_rot))
        if idx[0].shape[0]>0:
            print (data_rot)

            print ("rotation has NANS !!!!!!! : ", data)
            return None

        #
        return data_rot

    #
    def get_vectors(self, animal_id, vectors_idx, feature_ids):

        # root_dir = self.root_dir #'/media/cat/1TB/dan/cohort1/slp/'

        if self.fixed_tracks_flag:
            fname_in = os.path.join(self.fname_slp.replace('.slp','_fixed.npy'))
        else:
            fname_in = os.path.join(self.fname_slp.replace('.slp','.npy'))

        d = np.load(fname_in)
        d = d[:,animal_id]

        #
        vecs=[]
        good_idx=[]
        for k in trange(vectors_idx.shape[0], desc='computing vecs'):
            temp = d[vectors_idx[k][0]:vectors_idx[k][1]][:,feature_ids]
            idx = np.where(np.isnan(temp))

            # if nose or head contain zeros skip
            if idx[0].shape[0]>0:
                continue

            vecs.append(temp)
            good_idx.append(k)

        vecs = np.array(vecs)
        good_idx = np.array(good_idx)

        return vecs, good_idx

    #
    def vectors_to_egocentric(self, vecs, animal_id,min_duration):

        #
        vecs_ego = np.zeros(vecs.shape, 'float32')+np.nan

        #
        for s in trange(vecs.shape[0],  desc='Getting egocentric vectors', leave=True):

            vec = vecs[s]
            #print ("vec: ", vec.shape)
            # centre and align data
            vecs_ego[s] = centre_and_align2_pairwise(vec)

        #     np.save(fname_out, vecs_ego)
        # else:
        #     vecs_ego = np.load(fname_out)

        return vecs_ego

    #
    def find_movements(self, vecs_ego):

        print ("vecs: ", vecs_ego.shape)


        min_quiet_n_frames = 2      # number of frames of static
        min_quiet_movement = 1

        #
        min_velocity = 1             # minimum pixels to move to indicate movement intiaitins


        #
        ctr_q = 0
        idx_movement = []
        s = 0
        while s<vecs_ego.shape[0]:
    #    for s in trange(vecs_ego.shape[0]):

            # check if there is at least 10 frames quiet followed by high movement

            # get location of noses
            noses = vecs_ego[s,:,0]
            #print ("noses: ", noses.shape)

            # get velocities
            temp = noses[1:]-noses[:-1]
            vel = np.linalg.norm(temp,axis=1)
            #print ("vel: ", vel, vel.shape)

            # find periods with at least 10 frames of low or no movement
            if np.max(vel[:min_quiet_n_frames])<=min_quiet_movement:
                ctr_q+=1

                # require minimum movement in next frame
                if True:
                    if vel[min_quiet_n_frames]>=min_velocity:
                        idx_movement.append(s)
                        #vecs_movement.append(vecs_ego[s])
                else:
                    #vecs_movement.append(vecs_ego[s])
                    idx_movement.apppend(s)
            s+=1


        print ("ctrq: ", ctr_q)
        idx_movement = np.array(idx_movement)

        return idx_movement


    def dim_red(self, X_pca):
        from sklearn import decomposition
        import sklearn

        #
        print ("X_pca: ", X_pca.shape)

        #
        if False:
            pca = decomposition.PCA(n_components=3)

            X_pca = pca.fit_transform(X_pca)
            print (X_pca.shape)

        else:
            import umap
            umap = umap.UMAP(n_components=2,
                            init='random',
                            random_state=0)

            print ("fitting umap")
            #umap_ = umap.fit(vecs_pca[::10])
            umap_ = umap.fit(X_pca)

            print ("transforming alldata")
            X_pca = umap_.transform(X_pca)


        print ("plotting: ", X_pca.shape)

        return X_pca


    #
    def make_pca_data(self, vecs_movement, angles):

        vecs_nose = vecs_movement[:,:,0]
        print ("vecs_nose: ", vecs_nose.shape)

        X_pca = np.zeros((vecs_movement.shape[0], vecs_movement.shape[1],vecs_movement.shape[2]+1))

        for k in range(X_pca.shape[0]):
            X_pca[k,:,:2] = vecs_nose[k]
            X_pca[k,:,2] = angles[k]


        print (X_pca.shape)

        X_pca = X_pca.reshape(X_pca.shape[0],-1)

        return X_pca


    # Note: returns angle in radians
    def theta(self, v, w):
        return arccos(v.dot(w)/(norm(v)*norm(w)))


    def get_acceleration_persec_continuous_all_pairs(self):

        #
        self.acc = np.zeros((self.tracks.shape[0],
                        self.tracks.shape[1]), dtype=np.float32)+np.nan
        self.vel = np.zeros((self.tracks.shape[0],
                        self.tracks.shape[1]), dtype=np.float32)+np.nan

        # make all pair-wise combinations to run through
        import itertools
        all_pairs = np.int32(list(itertools.combinations(self.feature_ids, 2)))

        # loop over animals
        for a in range(self.tracks.shape[1]):

            acc_list = []
            for i in self.feature_ids:
                # use the first feature to compute velocity
                locs1 = self.tracks[:,a,i]

                # velocity
                vel_ap = locs1[1:,0] - locs1[:-1,0]
                vel_ml = locs1[1:,1] - locs1[:-1,1]
                self.vel[1:,a] = np.sqrt(vel_ap**2+vel_ml**2)*self.fps

                # acceleration
                acc_ap = vel_ap[1:]-vel_ap[:-1]
                acc_ml = vel_ml[1:]-vel_ml[:-1]

                acc_list.append(np.sqrt(acc_ap**2+acc_ml**2)*self.fps)

            self.acc[2:,a] = np.nanmedian(acc_list,axis=0)


    #
    def get_acceleration_persec_continuous(self):

        #
        self.acc = np.zeros((self.tracks.shape[0],
                        self.tracks.shape[1]), dtype=np.float32)
        self.vel = np.zeros((self.tracks.shape[0],
                        self.tracks.shape[1]), dtype=np.float32)

        #
        print (" .... >>>> Try computing vel and acc from the median of all locs <<<")

        # loop over animals
        for a in range(self.tracks.shape[1]):

            # use the first feature to compute velocity
            locs1 = self.tracks[:,a,self.features_anchor[0]]

            # velocity
            vel_ap = locs1[1:,0] - locs1[:-1,0]
            vel_ml = locs1[1:,1] - locs1[:-1,1]
            self.vel[1:,a] = np.sqrt(vel_ap**2+vel_ml**2)*self.fps

            # acceleration
            acc_ap = vel_ap[1:]-vel_ap[:-1]
            acc_ml = vel_ml[1:]-vel_ml[:-1]
            self.acc[2:,a] = np.sqrt(acc_ap**2+acc_ml**2)*self.fps

    #
    def get_acceleration_persec_single_frame(self, vecs, animal_id):
        root_dir = '/media/cat/1TB/dan/cohort1/slp/'

        fps = 25

        acc = []
        vel = []
        #if os.path.exists(fname_out)==False:
        for f in trange(vecs.shape[0], desc='Getting velocity and acceleration'):

            # use head position to compute velocity, NOT nose as the head is more stable
            #vecs_nose = vecs[f][:,0]
            vecs_nose = vecs[f][:,1]

            # velocity
            vel_ap = vecs_nose[1:,0] - vecs_nose[:-1,0]
            vel_ml = vecs_nose[1:,1] - vecs_nose[:-1,1]
            vel.append(np.sqrt(vel_ap**2+vel_ml**2)*fps)

            # acceleration
            acc_ap = vel_ap[1:]-vel_ap[:-1]
            acc_ml = vel_ml[1:]-vel_ml[:-1]
            acc.append(np.sqrt(acc_ap**2+acc_ml**2)*fps)

        acc = np.array(acc)
        vel = np.array(vel)

        return acc_ap, acc_ml, acc, vel

    #
    def load_vecs_single_frame(self, animal_id):

        ##################################
        ##################################
        ##################################
        two, six, headnose = self.get_lengths(animal_id)
        print ("# of headnose locations: ", headnose.shape)
        print ("HEADNOSE: ", headnose)

        #
        ##################################
        ##################################
        ##################################
        vectors_idx = self.get_durations_single_frame(headnose)

        print ("# of segments with min duration: ", self.min_duration,
                " for single frame analysis ", vectors_idx.shape)

        ##################################
        ##################################
        ##################################
        feature_ids = self.feature_ids[:2]  # grab the nose + head
        vecs, good_idx = self.get_vectors(animal_id, vectors_idx, feature_ids)

        ##################################
        ##################################
        ##################################
        times = vectors_idx[good_idx]

        return vecs, times

    def plot_angle_acceleration_distributions(self, animal_ids, min_duration):
        #
        fps = 25

        #
        root_dir = '/media/cat/1TB/dan/cohort1/slp/'

        #
        vecs_mov_array = []
        for animal_id in animal_ids:
            fname = os.path.join(root_dir + '/all_continuous_'+
                                 str(animal_id)+'_min_duration'+
                                 str(min_duration)+'.npz')
            d = np.load(fname)
            angles = d['angles']
            acc = d['acc']

            ##################################
            ######### PLOT ANGLES ############
            ##################################
            plt.subplot(4,2,animal_id*2+1)

            print ("angles: ", angles.shape)
            width = 1
            # rad_to_degree= 57.2958
            lims = 360
            bins = np.arange(-lims,lims+width, width)
            temp = angles.flatten()# *fps*rad_to_degree

            y = np.histogram(temp, bins=bins)
            plt.plot(y[1][1:]-width/2, y[0],c='black')
            plt.semilogy()
            plt.ylim(bottom=1)
            plt.plot([0,0],[0,np.max(y[0])],'--')
            plt.plot([45,45],[0,np.max(y[0])],'--')
            plt.plot([-45,-45],[0,np.max(y[0])],'--')
            if animal_id==0:
                plt.title("angles (deg/sec) pdf")
            plt.ylabel("Animal "+str(animal_id))

            ##################################
            ###### PLOT ACC-OVERALL ##########
            ##################################
            plt.subplot(4,2,animal_id*2+2)
            width = 5
            temp3 = acc #*25

            #
            y = np.histogram(temp3, bins=bins)
            plt.plot(y[1][1:]-width/2, y[0],c='magenta')
            plt.semilogy()
            #plt.semilogx()
            plt.plot([0,0],[0,np.max(y[0])],'--')
            plt.plot([40,40],[0,np.max(y[0])],'--')
            # plt.plot([150,150],[0,np.max(y[0])],'--')
            # plt.plot([245,245],[0,np.max(y[0])],'--')
            plt.ylim(bottom=1)
            plt.xlim(left=-1)
            if animal_id==0:
                plt.title("abs acceleration (pix/sec) pdf")

            print ('')

    def discretize_data_continuous(self):

        #
        print ("DISCRETIZING ")

        ##########################################
        ################# ANGLES #################
        ##########################################
        self.angles_discretized = np.zeros(self.angles.shape, dtype=np.float32)+np.nan
        for a in trange(self.angles.shape[1]):
            temp = self.angles[:,a] # *self.fps*self.rad_to_degree

            for c in range(len(self.angles_thresh)):
                idx = np.where(np.logical_and(
                                    temp>=self.angles_thresh[c][0],
                                    temp<self.angles_thresh[c][1],
                               ))[0]
                self.angles_discretized[idx, a] = c

        ##########################################
        ############## ACCELERATION ##############
        ##########################################
        #
        self.acc_discretized = np.zeros(self.acc.shape, dtype=np.float32)+np.nan
        for a in trange(self.acc.shape[1]):
            temp = self.acc[:,a] # *self.fps*self.rad_to_degree

            #
            for c in range(len(self.acc_thresh)):
                idx = np.where(np.logical_and(
                                    temp>=self.acc_thresh[c][0],
                                    temp<self.acc_thresh[c][1],
                               ))[0]

                self.acc_discretized[idx, a]=c


    #
    def discretize_data_single_frame(self, animal_id,
                                    angles,
                                    acc,
                                    fps,
                                    rad_to_degree,
                                    angles_thresh,
                                    acc_thresh):

        # #
        # root_dir = '/media/cat/1TB/dan/cohort1/slp/'
        #
        # #
        # fname_out = os.path.join(root_dir, 'all_discretized_'+str(animal_id)+
        #             '_single_frame.npz')

        #
        #if os.path.exists(fname_out)==False:
        print ("DISCRETIZING ")

        ##########################################
        ##########################################
        ##########################################
        angles_discretized = [] # np.zeros(angles.shape, 'float32')+np.nan
        for k in trange(angles.shape[0]):
            temp = angles[k]*fps*rad_to_degree

            ang_discretized_seg = np.zeros(temp.shape[0])
            for a in range(len(angles_thresh)):
                idx = np.where(np.logical_and(
                                    temp>=angles_thresh[a][0],
                                    temp<angles_thresh[a][1],
                               ))[0]
                ang_discretized_seg[idx] = a

            angles_discretized.append(ang_discretized_seg)
        angles_discretized = np.array(angles_discretized)

        ##########################################
        ##########################################
        ##########################################
        # discretize accelaration
        acc_discretized = []
        for k in trange(acc.shape[0]):
            temp = acc[k]
            acc_discretized_seg = np.zeros(temp.shape[0])
            for a in range(len(acc_thresh)):
                idx = np.where(np.logical_and(
                                    temp>=acc_thresh[a][0],
                                    temp<acc_thresh[a][1],
                               ))[0]

                acc_discretized_seg[idx]=a
            acc_discretized.append(acc_discretized_seg)
        acc_discretized = np.array(acc_discretized)

        ##########################################
        ##########################################
        #########################################
        all_discretized = []
        for k in range(angles_discretized.shape[0]):
            all_discretized.append(np.hstack((angles_discretized[k][2:], acc_discretized[k])))

        all_discretized=np.array(all_discretized)

        ##########################################
        ##########################################
        #########################################
        # np.savez(fname_out,
        #         all_discretized = all_discretized,
        #         angles_discretized = angles_discretized,
        #         acc_discretized = acc_discretized)
        #
        #
        # else:
        #     data = np.load(fname_out, allow_pickle=True)
        #
        #     all_discretized=data['all_discretized']
        #     angles_discretized=data['angles_discretized']
        #     acc_discretized=data['acc_discretized']


        return all_discretized, angles_discretized, acc_discretized


    def median_filter(self, data, filter_width):
        #print ("smoothing vector: ", data.shape)
        for k in range(data.shape[0]):
            data[k] = scipy.ndimage.median_filter(data[k], size=filter_width)

        return data

    #
    def smooth_vecs_ego_single_frame(self, vecs_ego, window=5):

        #
        vecs_smooth = []
        for k in trange(vecs_ego.shape[0], desc='smoothing vecs'):
            temp = np.zeros(vecs_ego[k].shape,'float32')
            for p in range(vecs_ego[k].shape[1]):
                for r in range(vecs_ego[k].shape[2]):
                    temp[:,p,r] = self.median_filter(vecs_ego[k][:,p,r], window)
            vecs_smooth.append(temp)
        vecs_smooth = np.array(vecs_smooth)

        return vecs_smooth

    #
    def smooth_vecs_ego(self, vecs_ego, window = 5):

        for k in trange(vecs_ego.shape[2], desc='smoothing vecs_ego'):
            for p in range(vecs_ego.shape[3]):
                vecs_ego[:,:,k,p] = median_filter(vecs_ego[:,:,k,p], window)

        return vecs_ego

    #
    def smooth_angles(self, angles, window=5):
        print ("...smoothing angles...")
        angles = self.median_filter(angles, window)

        return angles

    #
    def smooth_tracks(self, window=5):

        #
        # loop over animals
        for a in trange(self.tracks.shape[1], desc='smoothing'):
            # loop over features
            for f in range(self.tracks.shape[2]):
                # loop over x,y
                for l in range(self.tracks.shape[3]):
                    #temp[:,p,r] = self.median_filter(vecs_ego[k][:,p,r], window)
                    self.tracks[:,a,f,l] = self.median_filter(self.tracks[:,a,f,l],
                                                              window)


    def replace_nans(self):

        print ("self. tracks: ", self.tracks.shape)

        # loop over each animal
        for a in trange(self.tracks.shape[1], desc='Replacing nans'):
            # loop over each feature
            for f in range(self.tracks.shape[2]):
                # loop over x and y coords
                for c in range(self.tracks.shape[3]):

                    temp = self.tracks[:,a,f,c]

                    # search forward
                    idx = np.where(np.isnan(temp))[0]
                    for id_ in idx:
                        temp[id_] = temp[id_-1]

                    # search backward
                    temp = temp[::-1]
                    idx = np.where(np.isnan(temp))[0]
                    for id_ in idx:
                        temp[id_] = temp[id_-1]

                    self.tracks[:,a,f,c] = temp[::-1]

    #
    def compute_discretized_and_histograms_continuous_all_pairs(self):

        #
        fname_out = os.path.join(self.root_dir,
                                 os.path.split(self.fname_slp)[1].replace('.slp','')+
                                 '_continuous_allData_allPairs.npz')

        ##################################
        if os.path.exists(fname_out)==False:

            #
            print ("angles thresholds: ", self.angles_thresh)

            #
            print ("accelaration thresholds: ", self.acc_thresh)

            #
            ##################################
            ##################################
            ##################################
            #
            if self.fixed_tracks_flag:
                fname_in = os.path.join(self.fname_slp.replace('.slp','_fixed.npy'))
            else:
                fname_in = os.path.join(self.fname_slp.replace('.slp','.npy'))

            self.tracks = np.load(fname_in)
            print ("Loaded tracks; ", self.tracks.shape)

            ##################################
            ########## SMOOTH TRACKS #########
            ##################################
            if self.replace_nans_with_last_loc:
                self.replace_nans()

            ##################################
            ########## SMOOTH TRACKS #########
            ##################################
            if self.smooth_tracks_flag:
                fname_1 = self.fname_slp.replace('.slp','_tracks_smooth.npy')
                if os.path.exists(fname_1)==False:
                    self.smooth_tracks()
                    np.save(fname_1, self.tracks)
                else:
                    self.tracks = np.load(fname_1)
                    print ("smoothed tracks; ", self.tracks.shape)

            ##################################
            ############# GET ANGLES #########
            ##################################
            self.get_angles_continuous_all_pairs()
            print ("angles: ", self.angles.shape, " eg.g. ", self.angles[0])

            if self.smooth_angles_flag:
                self.angles = self.smooth_angles(self.angles)  # same function for single frame

            ##################################
            ##### ACCUMULATE ANGLES ##########
            ##################################
            if self.cumulative_sum:
                self.compute_angles_cumulative_continuous()
                print ("angles cumsum: ", self.angles_cumsum.shape, " eg.g. ", self.angles_cumsum[0])

                # assign cumulative angles to main arrays
                self.angles = self.angles_cumsum

            ##################################
            ##### DISCRETIZE ACCELERATION ####
            ##################################
            self.get_acceleration_persec_continuous_all_pairs()

            ##################################
            #### DISCRETIZE ANGLES AND ACC ###
            ##################################
            self.discretize_data_continuous()

            #
            np.savez(fname_out,
                     locs=self.tracks,
                     angles_thresh=self.angles_thresh,
                     angles=self.angles,
                     angles_discretized=self.angles_discretized,
                     acc_thresh=self.acc_thresh,
                     acc=self.acc,
                     acc_discretized=self.acc_discretized,
                     vel=self.vel
                     )

        print ("... DONE...")

    #
    def compute_discretized_and_histograms_continuous(self):

        #
        fps = self.fps
        rad_to_degree= self.rad_to_degree

        # discretized thresholds for angles
        # self.angles_thresh = [[-1E8, -45],
        #                  [-45,+45],
        #                  [+45,1E8]
        #                 ]
        # #
        # self.acc_thresh = [[0,40],
        #               #[20,40],
        #               [40,1E8]]

        #
        fname_out = os.path.join(self.root_dir,
                                 os.path.split(self.fname_slp)[1].replace('.slp','')+
                                 '_continuous_allData.npz')

        #
        if os.path.exists(fname_out)==False:

            #
            print ("agnels thresholds: ", self.angles_thresh)

            #
            print ("accelaration thresholds: ", self.acc_thresh)

            #
            ##################################
            ##################################
            ##################################
            #
            if self.fixed_tracks_flag:
                fname_in = os.path.join(self.fname_slp.replace('.slp','_fixed.npy'))
            else:
                fname_in = os.path.join(self.fname_slp.replace('.slp','.npy'))

            self.tracks = np.load(fname_in)
            print ("Loaded tracks; ", self.tracks.shape)

            #
            ##################################
            ########## SMOOTH TRACKS #########
            ##################################
            if self.smooth_tracks_flag:
                fname_1 = self.fname_slp.replace('.slp','_tracks_smooth.npy')
                if os.path.exists(fname_1)==False:
                    self.smooth_tracks()
                    np.save(fname_1, self.tracks)
                else:
                    self.tracks = np.load(fname_1)
                    print ("smoothed tracks; ", self.tracks.shape)

            ##################################
            ############# GET ANGLES #########
            ##################################
            self.get_angles_continuous()
            print ("angles: ", self.angles.shape, " eg.g. ", self.angles[0])

            if self.smooth_angles_flag:
                self.angles = self.smooth_angles(self.angles)  # same function for single frame

            ##################################
            ##### ACCUMULATE ANGLES ##########
            ##################################

            if self.cumulative_sum:
                self.compute_angles_cumulative_continuous()
                print ("angles cumsum: ", self.angles_cumsum.shape, " eg.g. ", self.angles_cumsum[0])

                # assign cumulative angles to main arrays
                self.angles = self.angles_cumsum

            ##################################
            ##### DISCRETIZE ACCELERATION ####
            ##################################
            self.get_acceleration_persec_continuous()

            ##################################
            #### DISCRETIZE ANGLES AND ACC ###
            ##################################
            self.discretize_data_continuous()

            #
            np.savez(fname_out,
                     locs=self.tracks,
                     angles_thresh=self.angles_thresh,
                     angles=self.angles,
                     angles_discretized=self.angles_discretized,
                     acc_thresh=self.acc_thresh,
                     acc=self.acc,
                     acc_discretized=self.acc_discretized,
                     vel=self.vel
                     )

        print ("... DONE...")


    #
    def compute_discretized_and_histograms_single_frame(self):

        animal_id = self.animal_id

        #
        fps = self.fps
        rad_to_degree= self.rad_to_degree

        # discretized thresholds for angles
        angles_thresh = [[-1E8, -45],
                         [-45,+45],
                         [+45,1E8]
                        ]

        # discretized thresholds for acceleration
    #     acc_thresh = [[0,30],
    #                   [30,75],
    #                   [75,150],
    #                   [150,1E8]]

        acc_thresh = [[0,40],
                      #[20,40],
                      [40,1E8]]

        # acc_thresh = [[0,30],
        #               [30,75],
        #               [75,1E8]]
        #

        #
        fname_out = os.path.join(self.root_dir,
                                 os.path.split(self.fname_slp)[1].replace('.slp','')+
                                 '_animalID_'+str(animal_id)+
                                 '_single_frame_allData.npz')

        #
        if os.path.exists(fname_out)==False:

            #
            print ("agnels thresholds: ", angles_thresh)

            #
            print ("accelaration thresholds: ", acc_thresh)

            #
            ##################################
            ##################################
            ##################################
            if False:
                vecs, times = self.load_vecs_single_frame(animal_id)
            else:
                vecs, times = self.load_vecs_single_frame(animal_id)

            locs = vecs.copy()

            ##################################
            ############ GET VECTORS #########
            ##################################
            if False:
                vecs_smooth = self.smooth_vecs_ego_single_frame(vecs)
                print ("vecs smooth: ", vecs_smooth.shape, " e.g.: ", vecs_smooth[0].shape)
            else:
                vecs_smooth = vecs

            ##################################
            ############# GET ANGLES #########
            ##################################
            angles = self.get_angles_single_frame(vecs_smooth, animal_id)
            print ("angles: ", angles.shape, " eg.g. ", angles[0])

            if False:
                angles = self.smooth_angles(angles)  # same function for single frame

            ##################################
            ##### ACCUMULATE ANGLES ##########
            ##################################

            if True:
                angles = self.compute_angles_cumulative(angles)
                print ("angles: ", angles.shape, " eg.g. ", angles[0])

            ##################################
            ##### DISCRETIZE ACCELERATION ####
            ##################################
            acc_ap, acc_ml, acc, vel = self.get_acceleration_persec_single_frame(
                                                                        vecs_smooth,
                                                                        animal_id)

            ##################################
            ###### MAKE CONTINOUS DATA #######
            ##################################
            # all_continuous = np.hstack((angles[:,2:], acc))

            ##################################
            #### DISCRETIZE ANGLES AND ACC ###
            ##################################

            (all_discretized,
             angles_discretized,
             acc_discretized) = self.discretize_data_single_frame(
                                                    animal_id,
                                                    angles,
                                                    acc,
                                                    fps,
                                                    rad_to_degree,
                                                    angles_thresh,
                                                    acc_thresh)

            #
            np.savez(fname_out,
                     locs=locs,
                     locs_smooth=vecs_smooth,
                     times=times,
                     all_discretized=all_discretized,
                     angles_thresh=angles_thresh,
                     angles=angles,
                     angles_discretized=angles_discretized,
                     acc_thresh=acc_thresh,
                     acc=acc,
                     acc_ap=acc_ap,
                     acc_ml=acc_ml,
                     acc_discretized=acc_discretized,
                     vel=vel
                     )

        print ("... DONE...")

    def compute_angles_cumulative_continuous(self):

        #
        # min_rot = self.min_rot
        #rad_to_degree= 57.2958
        self.angles_cumsum = np.zeros(self.angles.shape,
                                      dtype=np.float32)+np.nan

        # this is complicated: leaving raw angles is most accurate, but might miss movements
        for a in range(self.angles.shape[1]):

            cumsum = 0
            for n in trange(self.angles.shape[0], desc='Getting cumulative angles', leave=True):
                # check for nan, if so just leave it
                if np.isnan(self.angles[n,a])==False:
                    cumsum += self.angles[n,a]
                    if np.abs(cumsum) >= self.min_rot:
                        self.angles_cumsum[n,a]=cumsum
                        # print ("large angelf ound: ", cumsum)
                        cumsum=0
                    else:
                        self.angles_cumsum[n,a]=0
                # if hit nan, reset the cumulative sum
                else:
                    cumsum = 0


    def compute_angles_cumulative(self, angles):

        min_rot = 20
        fps = 25
        rad_to_degree= 57.2958

        #
        for k in trange(angles.shape[0]):

            temp = np.cumsum(angles[k]*rad_to_degree*fps)
            m = np.where(np.abs(temp)>min_rot)[0]

            #
            temp_cleaned = np.zeros(temp.shape[0])
            while m.shape[0]>0:
                # print (k, "angular accelaration / sec: ", temp, " first loc: ", m[0], " angle: ", temp[m[0]])
                #arg = np.argmax(np.abs(temp))
                #print ("max angle reached: ", temp[arg])

                temp_cleaned[m[0]] = temp[m[0]]

                # zero out all entries up to this point;
                temp[m[0]+1:] = temp[m[0]+1:]-temp[m[0]]
                temp[:m[0]+1]=0

                #
                m = np.where(np.abs(temp)>min_rot)[0]

            #
            angles[k] = temp_cleaned

        return angles



    def get_angles_single_frame(self, vecs_ego, animal_id):

        #
        angles = []
        #
        for f in trange(vecs_ego.shape[0],  desc='Getting angles', leave=True):

            # grab the f chunk and t=0 (ie. first frame) xy location
            temp1 = vecs_ego[f][0]

            # grab xy differences between head and nose at t=0
            temp1 = temp1[1] - temp1[0]
            temp_prev = temp1.copy()

            angles_seg = np.zeros(vecs_ego[f].shape[0], dtype=np.float32)
            # loop over all times in a chunk and find angle relative the first frame
            for m in range(0,vecs_ego[f].shape[0],1):

                # grab frame
                temp2 = vecs_ego[f][m]

                # compute xy diff between nose and head
                temp2 = temp2[1] - temp2[0]

                # compute angle between t=0 frame and current frame
                angle = math.atan2(temp_prev[0]*temp2[1] - temp_prev[1]*temp2[0],
                                   temp_prev[0]*temp2[0] + temp_prev[1]*temp2[1])
                angles_seg[m]=angle

                temp_prev = temp2.copy()

            angles.append(angles_seg)


        #     np.save(fname_out, angles)
        # else:
        #     angles = np.load(fname_out)

        angles = np.array(angles)

        return angles

    #
    def get_angles_continuous(self):

        #
        print ("features_anchor: ", self.features_anchor)

        #
        self.angles = np.zeros((self.tracks.shape[0],
                           self.tracks.shape[1]),dtype=np.float32)+np.nan

        # loop over animals
        for a in range(self.tracks.shape[1]):

            # loop over frames starting at the second frame:
            for n in trange(0,self.tracks.shape[0]-1,1):

                # grab the current frame locations
                temp1 = self.tracks[n,a,self.features_anchor]
                if np.any(np.isnan(temp1)):
                    continue

                # grab next frame location
                temp2 = self.tracks[n+1,a,self.features_anchor]
                if np.any(np.isnan(temp2)):
                    continue

                # grab xy differences between head and nose at t=0
                temp1 = temp1[1] - temp1[0]

                # compute xy diff between nose and head
                temp2 = temp2[1] - temp2[0]

                # compute angle between t=0 frame and current frame
                self.angles[n,a] = math.atan2(temp1[0]*temp2[1] - temp1[1]*temp2[0],
                                              temp1[0]*temp2[0] + temp1[1]*temp2[1])*self.rad_to_degree*self.fps

    #
    def get_angles_continuous_all_pairs(self):

        #
        print ("features_anchor: ", self.features_anchor)

        #
        self.angles = np.zeros((self.tracks.shape[0],
                           self.tracks.shape[1]),dtype=np.float32)+np.nan

        # make all pair-wise combinations to run through
        import itertools
        all_pairs = np.int32(list(itertools.combinations(self.feature_ids, 2)))

        # loop over animals
        for a in range(self.tracks.shape[1]):

            # loop over frames
            for n in trange(0,self.tracks.shape[0]-1,1, desc='Getting all pairwise angles', leave=True):

                # loop over all pairs:
                frame_angles = []
                for pair in all_pairs:

                    # grab the current frame locations
                    temp1 = self.tracks[n,a,pair]
                    if np.any(np.isnan(temp1)):
                        continue

                    # grab next frame location
                    temp2 = self.tracks[n+1,a,pair]
                    if np.any(np.isnan(temp2)):
                        continue

                    # grab xy differences between head and nose at t=0
                    temp1 = temp1[1] - temp1[0]

                    # compute xy diff between nose and head
                    temp2 = temp2[1] - temp2[0]

                    # compute angle between t=0 frame and current frame
                    frame_angles.append(math.atan2(temp1[0]*temp2[1] - temp1[1]*temp2[0],
                                  temp1[0]*temp2[0] + temp1[1]*temp2[1])*self.rad_to_degree*self.fps)

                self.angles[n,a] = np.median(frame_angles)


# 
def make_random_data(n_days,
                     n_partitions):
    #
    data = np.random.randint(0,200, size=(n_days*n_partitions,3)).astype('float32')
    
    # offset of data
    data[0:10,2]=np.nan
        
    # 
    data[-10:,2]=np.nan

    # generate random set days and hours
    for k in range(n_days):
        for p in range(n_partitions):
            data[k*n_partitions+p,0] = k+15
            data[k*n_partitions+p,1] = p
    
    return data

#
def plot_ethogram_hourly(n_days, 
                         n_partitions, 
                         data):
    
    #
    day = data[0][0]
    start_day = day.copy()
    print ("start day: ", day)

    #
    time = data[0][1]
    print ("start time: ", time)
    
    #
    temp = np.zeros(n_partitions)
    
    #
    img=[]
    while data.shape[0]>0:
        
        # grab the time
        time = data[0][1]
        
        # grab the next value
        temp[int(time)]=data[0][2]
        #print ("day: ", day, "time: ", time, " value: ", data[0,2])
    
        # pop the stack
        data = np.delete(data,0,axis=0)
        
        #
        if data.shape[0]==0:
            img.append(temp)
            break
    
        # check to see if day changed
        if data[0,0] != day:
            
            # apend the day to the image stack
            img.append(temp)

            # reset day number
            day = data[0][0]
            
            # reset partition number
            time = data[0][1]
            
            # reset the 
            temp = temp*0
            
    img = np.array(img)[::-1]
    
    #
    end_day = day
    
    #
    plt.figure()
    plt.imshow(img,
              #aspect='auto',
              interpolation='none',
              extent=[0+0.5,n_partitions+0.5, start_day-0.5,end_day-0.5])
    
    #
    plt.colorbar()
    plt.ylabel("Post natal day")
    plt.xlabel("Time of day")
    plt.title("Behavior ethogram")
    plt.show()
