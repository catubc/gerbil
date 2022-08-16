import matplotlib
#
import matplotlib.pyplot as plt
#
import matplotlib.cm as cm
import sleap

import numba
from numba import jit

import h5py
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor


#
import numpy as np
import os
from tqdm import trange
import parmap
import glob
from sklearn.decomposition import PCA
#import umap
import seaborn as sns
import pandas as pd

import pickle
#
from tqdm import tqdm

import sklearn.experimental
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#
from scipy.io import loadmat
import scipy
import scipy.ndimage

class CentreBody():

    def __init__(self):


        self.March16_file_order = [
                '2020-3-16_11-56-56-704655',  # day time starts correct day
                '2020-3-16_12-57-12-418305',
                '2020-3-16_01-57-27-327194',
                '2020-3-16_02-57-41-995158',
                '2020-3-16_03-57-56-902379',
                '2020-3-16_04-58-11-998956',
                '2020-3-16_05-58-27-193818',
                '2020-3-16_06-58-43-678014',
                '2020-3-16_07-59-00-362242',
                '2020-3-16_08-59-17-534732',
                '2020-3-16_09-59-34-731308',
                '2020-3-16_10-59-50-448686',

                '2020-3-16_12-54-07-193951',  # night time of previous day though
                '2020-3-16_01-54-23-358257',
                '2020-3-16_02-54-39-170978',
                '2020-3-16_03-54-54-231226',
                '2020-3-16_04-55-09-841582',
                '2020-3-16_05-55-25-305681',
                '2020-3-16_06-55-40-714236',
                '2020-3-16_07-55-55-775234',
                '2020-3-16_08-56-11-096689',
                '2020-3-16_09-56-26-362091',
                '2020-3-16_10-56-41-406701',
                ]


        #
        self.node_names = ['nose',          # 0
                          'lefteye',       # 1
                          'righteye',      # 2
                          'leftear',       # 3
                          'rightear',      # 4
                          'spine1',        # 5
                          'spine2',        # 6
                          'spine3',        # 7
                          'spine4',        # 8
                          'spine5',        # 9
                          'tail1',         # 10
                          'tail2',         # 11
                          'tail3',         # 12
                          'tail4']         # 13

        #
        self.feature_ids = np.array([0,5,6,7,8,9])

        #
        self.animal_ids = np.arange(4)


    def process_slp(self):

        self.fnames_slp = glob.glob(self.root_dir+"/*.slp")

        if self.parallel:
            parmap.map(convert_slp, self.fnames_slp,
                       pm_processes=16,
                       pm_pbar=True)

        else:

            for fname in self.fnames_slp:
                self.fname_slp = fname
                self.convert_slp()



    def get_fnames(self):

        self.fnames = glob.glob(self.root_dir+"/*_compressed.npy")


    def filter_data(self):

        print ("  ... median filtering ...")


        if self.parallel:
            parmap.map(filter_data1, self.fnames,
                      pm_processes=8,
                      pm_pbar=True)
        else:
            for fname in self.fnames:
                filter_data1(fname)



    def reject_outliers(self, max_dist=40):

        print ("  ... rejecting outliers....")

        self.fnames = glob.glob(self.root_dir+"/*_compressed.npy")

        if self.parallel:
            parmap.map(reject_outliers1, self.fnames,
                       self.feature_ids,
                       max_dist,
                       pm_processes=8,
                       pm_pbar=True)
        else:
            for fname in tqdm(self.fnames):
                reject_outliers1(fname,
                                      self.feature_ids,
                                      max_dist)


    def centre_and_align1(self, fname,
                         feature_ids):

        fname2 = fname.replace('.npy','_median_filtered_outliers.npy')

        fname_out = fname2.replace('.npy','_centre_aligned.npy')
        #fname_out_good_only = fname2.replace('.npy','_centre_aligned.npy')

        if os.path.exists(fname_out)==False:

            data = np.load(fname2)

            #
            centre_pt = 0

            features_full = np.zeros((data.shape[0],data.shape[1],feature_ids.shape[0],2),
                                          'float32')+np.nan

            features_array = []
            for k in range(4):
                features_array.append([])

            for f in range(0,data.shape[0],1):

                # loop over each animal
                for k in range(data.shape[1]):

                    x = data[f,k,feature_ids,0]
                    y = data[f,k,feature_ids,1]

                    idx = np.where(np.isnan(x))[0]
                    if idx.shape[0]==0:
                        #print (f, k, x.shape)

                        locs = np.vstack((x,y)).T

                        # centre and align data
                        locs_pca = self.centre_and_align2(locs,f,centre_pt)

                        if locs_pca is not None:
                            idx = np.where(np.isnan(locs_pca))[0]

                            if idx.shape[0]>0:
                                continue

                            features_full[f,k] = locs_pca
                            features_array[k].append(locs_pca)

            np.save(fname_out, features_full)


    def centre_and_align2(self, data, frame, centre_pt=0):

        if True:
            # centre the data on the nose
            data[:,0] -= data[centre_pt,0]
            data[:,1] -= data[centre_pt,1]

            # get angle between +x axis and head location (i.e. 2nd position)
            # 2ND POSITION ALIGNMENT
            t = -np.arctan2(*data[1].T[::-1])-np.pi/2

            # get rotation
            rotmat = np.array([[np.cos(t), -np.sin(t)],
                               [np.sin(t),  np.cos(t)]])

            # Apply rotation to each row of m
            m2 = (rotmat @ data.T).T

            return m2

    def centre_and_align_all_pairs(self):

        self.fnames = glob.glob(self.root_dir+"/*_compressed.npy")

        feature_ids = np.arange(6)
        for f1 in trange(feature_ids.shape[0]):
            for f2 in range(f1+1, feature_ids.shape[0],1):

                #
                if self.parallel:
                    parmap.map(centre_and_align1_pairwise, self.fnames,
                               self.feature_ids,
                               f1,
                               f2,
                               pm_processes=8,
                               # pm_pbar=True
                               )
                else:
                    for fname in self.fnames:

                        centre_and_align1_pairwise(fname,
                                                   self.feature_ids,
                                                   f1,
                                                   f2)
        print (" DONE Generating 2-point ground truth datasets for imputation")


    def centre_and_align(self):

        print ("  ... center and aligning ...")

        self.fnames = glob.glob(self.root_dir+"/*_compressed.npy")

        if self.parallel:
            parmap.map(self.centre_and_align1, self.fnames,
                       self.feature_ids,
                       pm_processes=8,
                       pm_pbar=True)
        else:
            for fname in fnames:
                pass


    def load_processed_data(self, f1, f2, data_type=1, remove_nans=True):

        ''' NEW DATA LOADER
             data_type can be 3 types:
            0: filtered, outlier cleaned, centred and rotated to specifci 2 features
            1: not used
            2: not used

            remove_nans - cleans up the data so only full body axis data is used
            True: loading clean data for model training
            False: loading all data for prediction

        '''
        print ("   note: pipeline currently applied only to Cohort1 March 16th datasets")

        # stack the postures for each animal for the day
        features_array = []
        for k in range(4):
            features_array.append([])

        for file in self.March16_file_order:

            # this data is filtered, outlier triaged + centred to nose and aligned to face up
            if data_type==0:
                fname = glob.glob(os.path.join(self.root_dir,file+
                                               "*_median_filtered_outliers_centre_aligned_"+
                                               str(f1)+"_"+str(f2)+".npz").replace("-","_"))[0]
                # load data + angles + translations
                temp = np.load(fname)
                d3 = temp['features_full']
                #d3 = d3[:,:,self.feature_ids]  Data already limited to features

            # this data is filtered and outlier triaged
            if data_type==1:
                print (" DATA NOT AVAILABLE ...")
                self.features_array = None
                return

            # this data is not outlier traiged; - very poor results/bad to work with
            if data_type==2:
                print (" DATA NOT AVAILABLE ...")
                self.features_array = None
                return

            # loop over animals and keep only complete data (i.e. 6 pts)
            for k in range(self.animal_ids.shape[0]):

                # find nans and delete any frame that is missing even a single value for that animal
                if remove_nans:

                    idx = np.where(np.isnan(d3[:,k]))
                    ids, counts = np.unique(idx[0], return_counts=True)
#                     print (k, "Coutns: ", counts.shape, counts)
#                     print (k, "ids: ", ids.shape, " idx: ", idx[0].shape, idx[0])

                    idx_all = np.arange(d3.shape[0])
                    idx_good = np.delete(idx_all, ids)

                    temp = d3[idx_good,k]

                    features_array[k].append(temp)
                else:
                    features_array[k].append(d3[:,k])


        # return stacked feature array
        self.features_array = features_array

        print (" DATA SIZES: ")
        print (" female [n_samples, n_featres, xy]: ", np.vstack(self.features_array[0]).shape)
        print (" male:                              ", np.vstack(self.features_array[1]).shape)
        print (" pup1:                              ", np.vstack(self.features_array[2]).shape)
        print (" pup2:                              ", np.vstack(self.features_array[3]).shape)


#
def impute_real_data_stand_alone1(pair,
                                  root_dir,
                                  animal_ids,
                                  feature_ids,
                                  model_type_name,
                                  March16_file_order,
                                  ):

    # get pairs
    f1 = pair[0]
    f2 = pair[1]

    # load specific egocentric + animal _ id data
    feats = load_processed_data_stand_alone(f1, f2,
                                            March16_file_order,
                                            root_dir,
                                            animal_ids,
                                            data_type=0,
                                            remove_nans=False)

    #
    feats = np.array(feats)

    # loop over all animals and impute
    for animal_id in animal_ids:

        # Select animal and specific feature data
        #feats_animal = np.vstack(feats[animal_id]) #[:,feature_ids]

        # load frames ids which have f1/f2 anchor points
        fname_frame_ids = os.path.join(root_dir, "centre_rotated_animalID_"+
                                       str(animal_id)+"_"+str(f1)+"_"+str(f2)+".npz")
        d = np.load(fname_frame_ids)

        feats_animal = d['feats_rotated']
        frame_ids = d['frame_ids_rotated']
        angles = d['angles']
        translations = d['translations']

        print (f1, f2, " frame_ids with anchor features in animal: ", animal_id, " is: ", frame_ids.shape)

        fname_out = fname_frame_ids.replace('.npz', '_imputed_original_locations.npz')

        #
        if os.path.exists(fname_out)==False:
            test = feats_animal[frame_ids]
            angles = angles[frame_ids]
            translations = translations[frame_ids]

            # reshape data
            X_test = test.reshape(test.shape[0],-1)

            ##############################################
            # load f1 - f2 model
            fname_in = os.path.join(root_dir,
                        model_type_name+"_"+str(f1)+"_"+str(f2)+
                        "_animalID_"+str(animal_id)+".pkl")

            #
            with open(fname_in, "rb") as f:
                model = pickle.load(f)

            ##############################################
            # impute missing data in egocentric coordinates
            # print ("  imputing missing data...")
            res = model.transform(X_test).reshape(-1,6,2)


            ##############################################
            res_untranslated = untranslate_stand_alone(res,
                                                       angles,
                                                       translations)

            #
            np.savez(fname_out,
                    imputed_translated = res_untranslated,
                    imputed = res,
                    imputed_frames = frame_ids)

            print ('')
        print ('')


def untranslate_stand_alone(data, angles, translations):

    # translate data to original locations
    data_untranslated = np.zeros(data.shape,'float32')
    for k in trange(data_untranslated.shape[0]):
        angle = -angles[k]
        rotmat = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])
        data_untranslated[k] = (rotmat @ data[k].T).T

    #
    data_untranslated[:,:,0] = (data_untranslated[:,:,0].T + translations[:,0]).T
    data_untranslated[:,:,1] = (data_untranslated[:,:,1].T + translations[:,1]).T

    # return rotated body, angle and translation pt
    return data_untranslated

#
def impute_real_data_stand_alone(I):

    #
    pairs = []
    for f1 in range(I.feature_ids.shape[0]):
        for f2 in range(f1+1, I.feature_ids.shape[0],1):
            pairs.append([f1,f2])

    if I.parallel:
        parmap.map(impute_real_data_stand_alone1,
                   pairs,
                   I.root_dir,
                   I.animal_ids,
                   I.feature_ids,
                   I.model_type_names[I.model_type],
                   I.March16_file_order
                   )
    else:
        for pair in pairs:
            impute_real_data_stand_alone1(
                pair,
                I.root_dir,
                I.animal_ids,
                I.feature_ids,
                I.model_type_names[I.model_type],
                I.March16_file_order
            )


def centre_rotate_features_pairs(I):

    #
    pairs = []
    for f1 in range(I.feature_ids.shape[0]):
        for f2 in range(f1+1, I.feature_ids.shape[0],1):
            pairs.append([f1,f2])

    if I.parallel:
        parmap.map(centre_rotate_standalone_f1f2,
                   pairs,
                   I.root_dir,
                   I.animal_id,
                   I.feature_ids)
    else:
        for pair in pairs:
            centre_rotate_standalone_f1f2(
                pair,
                I.root_dir,
                I.animal_id,
                I.feature_ids
            )



def centre_rotate_standalone_f1f2(pair,
                                  root_dir,
                                  animal_id,
                                  feature_ids):
    #
    f1 = pair[0]
    f2 = pair[1]

    #
    fname_frame_ids = os.path.join(root_dir, "anchor_frames_animalID_"+
                             str(animal_id)+"_"+str(f1)+"_"+str(f2)+".npy")
    frame_ids = np.load(fname_frame_ids)

    fname_feats = os.path.join(root_dir, "animalID_" +
                                str(animal_id) +
                                "_alldata.npy")
    feats = np.load(fname_feats)

    # index into body spine axis
    feats = feats[:,feature_ids]

    fname_out = fname_frame_ids.replace('anchor_frames','centre_rotated').replace('npy','npz')

    if os.path.exists(fname_out)==False:

        #
        # centre_pt = f1        # centre-point fix the data
        # rotation_pt = f2      # rotate it to 2nd feature;  e.g. nose centred (1st feature)
                              # and rotated to head (i.e. 2nd feature)

        # feats_rotated = np.zeros((feats.shape[0],feats.shape[1],feature_ids.shape[0],2),
        #                               'float32')+np.nan
        feats_rotated = feats.copy()

        angles = np.zeros((feats.shape[0]),'float32')+np.nan
        translations = np.zeros((feats.shape[0], 2),'float32')+np.nan

        # loop over frames that contain f1-f2 anchor points;
        # for f in range(0,feats.shape[0],1):
        for f in tqdm(frame_ids):

            x = feats_rotated[f,:,0]
            y = feats_rotated[f,:,1]

            idx = np.where(np.isnan(x)==False)[0]

            # must rotate all data regardless of how many nans
            x = x[idx]
            y = y[idx]

            f1 = np.where(idx == f1)[0]#.squeeze()
            f2 = np.where(idx == f2)[0]#.squeeze()

            # there are still some f1/f2 misses that snuck through...so just skip fixing those frames;
            if len(f1)==0 or len(f2)==0:
                continue
            f1 = f1.squeeze()
            f2 = f2.squeeze()

            # make 2D vector stack from non
            locs = np.vstack((x,y)).T
            #print ("locs:" , locs)

            # centre and align data
            locs_rot, angle, translation_pt = centre_and_align2_pairwise(locs,
                                                                         f1,
                                                                         f2)

            # if locs_pca is not None:
            # idx = np.where(np.isnan(locs_pca))[0]
            #
            # if idx.shape[0]>0:
            #     continue

            feats_rotated[f,idx] = locs_rot
            angles[f] = angle
            translations[f] = translation_pt

        #
        np.savez(fname_out,
                 feats_rotated = feats_rotated,
                 frame_ids_rotated = frame_ids,
                 angles = angles,
                 translations = translations,
                 )


#
def break_features_pairs(f1, f2, I):

    # load features
    fname_out = os.path.join(I.root_dir, "anchor_frames_animalID_"+
                             str(I.animal_id)+"_"+str(f1)+"_"+str(f2)+".npy")

    #
    if os.path.exists(fname_out)==False:

        feats = np.load(os.path.join(I.root_dir, "animalID_"+str(I.animal_id)+"_alldata.npy"))
        print (feats.shape)

        #
        feats = feats[:,I.feature_ids]

        #
        I.frame_ids = np.hstack(parse_data_for_anchors(f1, f2, feats))

        print (I.frame_ids.shape)
        np.save(fname_out, I.frame_ids)


@numba.jit()
def parse_data_for_anchors(f1, f2, feats):

    frame_ids = []
    for k in range(feats.shape[0]):
        # count # of non nan entries
        idx = np.where(np.isnan(feats[k,:,0])==False)[0]

        # if # > 3 and both our features are included then save the entry
        #if (idx.shape[0]>=3) and (f1 in idx) and (f2 in idx):
        if (f1 in idx) and (f2 in idx):
            frame_ids.append(k)

    return frame_ids


def load_processed_data_stand_alone(f1, f2,
                                    March16_file_order,
                                    root_dir,
                                    animal_ids,
                                    data_type=0,
                                    remove_nans=True):

    ''' NEW DATA LOADER
         data_type can be 3 types:
        0: filtered, outlier cleaned, centred and rotated to specifci 2 features
        1: not used
        2: not used

        remove_nans - cleans up the data so only full body axis data is used
        True: loading clean data for model training
        False: loading all data for prediction

    '''

    print ("   note: pipeline currently applied only to Cohort1 March 16th datasets")

    # stack the postures for each animal for the day
    features_array = []
    for k in range(4):
        features_array.append([])

    for file in March16_file_order:

        # this data is filtered, outlier triaged + centred to nose and aligned to face up
        if data_type==0:
            temp_temp = os.path.join(root_dir,file+
                                           "*_median_filtered_outliers_centre_aligned_"+
                                           str(f1)+"_"+str(f2)+".npz").replace("-","_")
            fname = glob.glob(temp_temp)[0]

            # load data + angles + translations
            temp = np.load(fname)
            d3 = temp['features_full']
            #d3 = d3[:,:,self.feature_ids]  Data already limited to features
            if remove_nans==True:
                print ("   Nans-loading in for f1 / f2 centred data is irrelevant as all incomplete frames are auto set to nans ")
                #return

        # this data is filtered and outlier triaged
        if data_type==1:
            temp_temp = os.path.join(root_dir,file+
                                           "*_median_filtered_outliers.npy").replace("-","_")
            fname = glob.glob(temp_temp)[0]

            # load data + angles + translations
            d3 = np.load(fname)

        # this data is not outlier traiged; - very poor results/bad to work with
        if data_type==2:
            return

        # loop over animals and keep only complete data (i.e. 6 pts)
        for k in animal_ids:

            # find nans and delete any frame that is missing even a single value for that animal
            if remove_nans:

                idx = np.where(np.isnan(d3[:,k]))
                ids, counts = np.unique(idx[0], return_counts=True)
#                     print (k, "Coutns: ", counts.shape, counts)
#                     print (k, "ids: ", ids.shape, " idx: ", idx[0].shape, idx[0])

                idx_all = np.arange(d3.shape[0])
                idx_good = np.delete(idx_all, ids)

                temp = d3[idx_good,k]

                features_array[k].append(temp)
            else:
                features_array[k].append(d3[:,k])


    # return stacked feature array
    #self.features_array = features_array

    print (" DATA SIZES: ")
    print (" female [n_samples, n_featres, xy]: ", np.vstack(features_array[0]).shape)
    print (" male:                              ", np.vstack(features_array[1]).shape)
    print (" pup1:                              ", np.vstack(features_array[2]).shape)
    print (" pup2:                              ", np.vstack(features_array[3]).shape)

    return features_array


#
def centre_and_align1_pairwise(fname,
                               feature_ids,
                               f1,
                               f2):

    ''' Function generates 2 point fixed body axis arrays;
        Input: median filtered and outlier triaged data
        Output: 2 point fixed body axis data (0-centred);
                angle of rotation
                0-centre translation

    '''
    fname2 = fname.replace('.npy','_median_filtered_outliers.npy')

    fname_out = fname2.replace('.npy','_centre_aligned_'+str(f1)+"_"+str(f2)+'.npz')

    if os.path.exists(fname_out)==False:

        data = np.load(fname2)

        #
        centre_pt = f1        # centre-point fix the data
        rotation_pt = f2      # rotate it to 2nd feature;  e.g. nose centred (1st feature)
                              # and rotated to head (i.e. 2nd feature)

        features_full = np.zeros((data.shape[0],data.shape[1],feature_ids.shape[0],2),
                                      'float32')+np.nan
        angles = np.zeros((data.shape[0], data.shape[1]),'float32')+np.nan
        translations = np.zeros((data.shape[0], data.shape[1], 2),'float32')+np.nan

        for f in range(0,data.shape[0],1):

            # loop over each animal
            for k in range(data.shape[1]):

                x = data[f,k,feature_ids,0]
                y = data[f,k,feature_ids,1]

                idx = np.where(np.isnan(x))[0]
                if idx.shape[0]==0:

                    # make 2D vector stack from non
                    locs = np.vstack((x,y)).T

                    # centre and align data
                    locs_pca, angle, translation_pt = centre_and_align2_pairwise(locs,
                                                                              centre_pt,
                                                                              rotation_pt)

                    #if locs_pca is not None:
                    idx = np.where(np.isnan(locs_pca))[0]

                    if idx.shape[0]>0:
                        continue

                    features_full[f,k] = locs_pca
                    angles[f,k] = angle
                    translations[f,k] = translation_pt

        #
        np.savez(fname_out,
                 features_full = features_full,
                 angles = angles,
                 translations = translations,
                 )


def centre_and_align2_pairwise(data, centre_pt, rotation_pt):

    # centre the data on the nose
    # data[:,0] -= data[centre_pt,0]
    # data[:,1] -= data[centre_pt,1]

    translation_pt = data[centre_pt]
    data -= translation_pt


    # get angle between +x axis and head location (i.e. 2nd position)
    # 2ND POSITION ALIGNMENT
    angle = -np.arctan2(*data[rotation_pt].T[::-1])-np.pi/2

    # get rotation
    rotmat = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)]])

    # Apply rotation to each row of m
    try:
        m2 = (rotmat @ data.T).T
    except:
        print ("centre_pt:", centre_pt)
        print ("rotation pt: ", rotation_pt)
        print ("angle; ", angle)
        print ("rotmat: ", rotmat)
        print ("data: ", data)

    # return rotated body, angle and translation pt
    return m2, angle, translation_pt


def convert_slp(fname):

    load_slp_convert_to_h5(fname)

    slp_to_npy(fname)

def load_slp_convert_to_h5(fname):


    fname_h5 = fname.replace('.slp',".h5")
    if os.path.exists(fname_h5)==False:
        #
        slp = sleap.load_file(fname)

        #
        slp.export(fname_h5)

def slp_to_npy(fname):

    fname_h5 = fname.replace('.slp',".h5")
    fname_npy = fname.replace('.slp','.npy')

    if os.path.exists(fname_npy)==False:

        #
        hf = h5py.File(fname_h5, 'r')

        keys = hf.keys()
        group2 = hf.get('tracks')
        print ("group2: ", group2)
        tracks = []
        for k in range(len(group2)):
            tracks.append(group2[k])

        tracks = np.array(tracks).transpose(3, 0, 2, 1)

        #
        #fname_npy = fname_slp[:-4] + ".npy"
        np.save(fname_npy, tracks)



def reject_outliers1(fname, feature_ids, max_dist):

    fname2 = fname.replace('.npy','_median_filtered.npy')

    fname_out = fname2.replace('.npy','_outliers.npy')

    if os.path.exists(fname_out)==False:

        data = np.load(fname2)
        for f in range(0,data.shape[0],1):

            for k in range(data.shape[1]):
                #
                x = data[f,k,feature_ids,0]
                y = data[f,k,feature_ids,1]

                x, y = reject_outliers2(x,y, max_dist)

                data[f,k,feature_ids,0] = x
                data[f,k,feature_ids,1] = y

        np.save(fname_out, data)

def reject_outliers2(x,y,
                    max_dist_pairwise,
                    max_dist_all=100):  # number of deviations away

    ''' Function returns indexes for which [x,y] array points are close to at least 2 other points

        Goal 1: to generate very clean data which has 6 body features well connected for downstream analysis and imputation

        Goal 2: to remove outliers and leave small clusters of feautres only

    '''

    # method 2: explicitly reject points that are > max distance from nearest 2 points
    temp = np.vstack((x,y))
    dists = scipy.spatial.distance.cdist(temp.T, temp.T)

    # first check points inside the array to ensure they have 2 close neighbours
    # if they don't, remove them so other points can't be connected to them.
    idx_far = []
    for k in range(1,temp.shape[1]-1,1):
        #idx = np.where(dists[k]<=max_dist_pairwise)[0]
        temp = dists[k]
        if np.abs(temp[k]-temp[k-1])>max_dist_pairwise or np.abs(temp[k]-temp[k+1])>max_dist_pairwise:
            idx_far.append(k)
            dists[:,k]= 1E3

    # check start and end points to ensure they have nearby val
    if np.abs(dists[0,1])>max_dist_pairwise:
        idx_far.append(0)
        #print (dists[0], 'excluded ', 0)

    if np.abs(dists[dists.shape[1]-1,dists.shape[1]-2])>max_dist_pairwise:
        idx_far.append(dists.shape[1]-1)
        #print (dists[0], 'excluded ', dists.shape[1]-1)


    x[idx_far] = np.nan
    y[idx_far] = np.nan

    return x, y



#
class Impute():

    def __init__(self, cb):

        #
        self.n_cores = 8

        #
        self.root_dir = cb.root_dir

        #
        self.cb = cb

        #
        self.model_type_names = ["BayesianRidge",
                                 "DecisionTreeRegressor",
                                 "ExtraTreesRegressor",
                                 "KNeighborsRegressor"]

        self.estimators = [BayesianRidge(),
                DecisionTreeRegressor(max_features='sqrt', random_state=0),
                ExtraTreesRegressor(n_estimators=10, random_state=0),
                KNeighborsRegressor(n_neighbors=15)]

        #
        self.feature_ids = cb.feature_ids

        #
        self.March16_file_order = cb.March16_file_order

        #
        self.animal_ids = cb.animal_ids

    #
    def evaluate_imputation_error(self,features_array, animal_id, res, idx_train, idx_test):

        temp = np.vstack(features_array[animal_id])
        temp = temp-temp[:,0][:,None]

        fig=plt.figure()
        ax=plt.subplot()

        diff = np.abs(temp[idx_test]-res)
        #print ("diff: ", diff.shape)

        errors = []
        for k in range(diff.shape[1]):
            errors.append([])

        for k in range(diff.shape[0]):
            for p in range(diff.shape[1]):
                temp = diff[k,p]
                #print (temp)
                tdiff = np.linalg.norm(temp)
                if tdiff>0:
                    errors[p].append(tdiff)

        t =[]
        for k in range(len(errors)):
            temp = errors[k]
            pad = np.zeros(100000-len(errors[k]),'float32')+np.nan
            temp = np.concatenate((temp, pad))
            t.append(temp)

        data = np.array(t).T
        #print (data.shape)
        columns = ['nose','spine1','spine2', 'spine3', 'spine4', 'spine5']
        #columns = ['errors']
        df = pd.DataFrame(data, columns = columns)

        #print ("DF: ", df)

        # plot
        ax=plt.subplot(2,1,1)
        sns.violinplot(data=df) #x=df['spine2'])
        plt.ylim(bottom=0)
        plt.ylabel(" pixel error")

        ax=plt.subplot(2,1,2)
        plt.title("Zoom",fontsize=20)
        sns.violinplot(data=df) #x=df['spine2'])
        plt.ylabel(" pixel error")
        plt.ylim(0,50)

    #
    def predict_novel_data(self):

        #
        pairs = []
        for f1 in range(self.feature_ids.shape[0]):
            for f2 in range(f1+1, self.feature_ids.shape[0],1):
                pairs.append([f1,f2])

        #
        if self.parallel:
            parmap.map(predict_novel_data1, pairs,
                        self.March16_file_order,
                        self.root_dir,
                        self.model_type_names[self.model_type],
                        self.animal_ids,
                        self.animals_selected,
                        pm_processes=self.n_cores)
        #
        else:
            for pair in pairs:
                print (pair)
                predict_novel_data1(pair,
                                    self.March16_file_order,
                                    self.root_dir,
                                    self.model_type_names[self.model_type],
                                    self.animal_ids,
                                    self.animals_selected
                                    )
        print ("DONE")

    #
    def predict_imputation_ground_truth_all_pairs(self,
                                                 max_n_drops = 3):

        # load actual data and impute missing locations
        if self.generate_random_drops==False:
            print (" TO DO: load actual data")

            self.predict_novel_data()

        # load ground truth data and generate random drop outs
        else:
            # print ("LOADING head-egocentric clean data")
            fname_out = os.path.join(self.root_dir,'temp_f1f2.npy')
            if os.path.exists(fname_out)==False:
                self.cb.load_processed_data(f1=0, f2=1, data_type=0, remove_nans=True)

                np.save(fname_out, self.cb.features_array)

                # select some random drops for each
                temp = np.vstack(self.cb.features_array[self.animal_id])
                drops = np.zeros((temp.shape[0], 3), 'int32')
                for k in range(drops.shape[0]):
                    drops[k] = np.random.choice(np.arange(6),3, replace=False)

                # drops = np.vstack(drops)
                #print (drops.shape, drops)

                np.save(fname_out.replace('.npy','_drops.npy'),
                        drops)
            else:
                #
                self.cb.features_array = np.load(fname_out, allow_pickle=True)
                drops = np.load(fname_out.replace('.npy','_drops.npy'), allow_pickle=True)


        if self.parallel:

            pairs = []
            for f1 in range(self.feature_ids.shape[0]):
                for f2 in range(f1+1, self.feature_ids.shape[0],1):
                    pairs.append([f1,f2])

            #
            parmap.map(evaluate_errors_stand_alone, pairs,
                        drops,
                        self.root_dir,
                        self.model_type_names[self.model_type],
                        self.March16_file_order,
                        self.animal_ids,
                        self.animals_selected,
                        pm_processes=8)
        else:
            for f1 in range(6):
                for f2 in range(f1+1, 6, 1):
                    pair = np.array([f1,f2])
                    evaluate_errors_stand_alone(pair,
                                                drops,
                                                self.root_dir,
                                                self.model_type_names[self.model_type],
                                                self.March16_file_order,
                                                self.animal_ids,
                                                self.animals_selected)


    def evaluate_res_error(self):
        # imputed = self.res #np.loadtxt('/home/cat/data_imputed.tsv')
        # gt = np.loadtxt('/home/cat/feats_ground_truth.tsv')

        print ("Self res: ", self.res.shape)
        print ("gt: ", self.gt.shape)

        # comptue distances
        diff = np.float32(np.abs(self.gt-self.res))

        # initialize error lists to track missing data
        errors = []
        for k in range(diff.shape[1]):
            errors.append([])

        # loop over frames
        for k in range(diff.shape[0]):
            # loop over features
            for p in range(diff.shape[1]):
                temp = diff[k,p]
                tdiff = np.linalg.norm(temp)
                if tdiff>0:
                    errors[p].append(tdiff)

        return errors


    #
    def predict_imputation_ground_truth(self,
                                        drops=None,
                                        idx_test=None,
                                        n_drops = 3):

        ''' OLD VERSION NOT USED ANYMORE

            Function used to test the prediction between the ground truth complete body axis and dropout data

            Need to set the 'remove_nans' flag about to True to get clean model data

            Need to set the idx_test to ~90% of data and run validation on the 10% holdout.

        '''

        #


        if self.fname_dropout is not None:
            X_test = np.loadtxt(self.fname_dropout)
            print (X_test.shape)

            idx_drop = []

        # GENERATE RANDOM DROPS
        else:
            #
            print ("LOADING head-egocentric clean data")
            self.cb.load_processed_data(data_type=0, remove_nans=True)
            temp = np.vstack(self.cb.features_array[self.animal_id])
            print (temp.shape)

            print ("Centering to head only <----------- NEEDS UPDATE FOR OTHER EGOCENTRIC DATA")
            temp = temp-temp[:,0][:,None]

            X_all = temp.reshape(temp.shape[0],-1)

            # select frames to predict
            if idx_test is not None:
                X_test = X_all[idx_test]
            else:
                X_test = X_all

            # do drop outs in the test set:
            X_test = X_test.reshape(-1,6,2)
            idx_drop = np.zeros((n_drops, X_test.shape[0]),'int32')
            for k in range(idx_drop.shape[1]):#n_drops):
                if drops is None:
                    idx_drop[:,k] = np.random.choice(np.arange(6),n_drops,replace=False)
                else:
                    idx_drop[:,k] = drops #np.random.choice(np.arange(6),n_drops,replace=False)


            print ("idx drop: ", idx_drop[0].shape)
            for k in range(len(idx_drop)):
                for p in range(idx_drop[k].shape[0]):
                    X_test[p,idx_drop[k][p]]=np.nan

            #
            X_test = X_test.reshape(X_test.shape[0],-1)


        # TEST STEP
        #res = imp.transform(X_test).reshape(-1,6,2)
        print ("... predting data ...")
        #res = self.models[self.animal_id][0].transform(X_test).reshape(-1,6,2)
        res = self.model.transform(X_test).reshape(-1,6,2)
        #print ("Res: ", res.shape)

        self.res = res
        self.drop = idx_drop

        return res, idx_drop

    #
    def load_models(self):

        body_centre = 0
        #

        #fname_in = self.root_dir+"imputation_animal_id"+str(self.animal_id)+"_body_centre"+str(p)+".pckl"
        fname_in = os.path.join(self.root_dir,
                                "model_type"+str(self.model_type)+
                                "_imputation_animal_id"+str(self.animal_id)+
                                "_body_centre"+str(body_centre)+".pckl")

        with open(fname_in, "rb") as f:
            model = pickle.load(f)

        self.model = model

    #
    def calculate_missing_features(self):

        ''' Function used to grab statistisc of missing data

        '''

        animal_ids = np.arange(4)

        idxc_array = []
        for animal_id in tqdm(animal_ids):
            self.animal_id = animal_id
            idxc_array.append([])
            for max_drops in range(7):

                #
                temp = np.vstack(self.cb.features_array[self.animal_id])

                # find which frames have that feature intact
                idx = np.where(np.isnan(temp))
                ids, counts = np.unique(idx[0], return_counts=True)

                # these are the frames where at least max_drops worth of frames are present
                idxc = np.where(counts<=max_drops*2)[0]

                # these are frames where all 6 features are available.
                idx_all = np.arange(temp.shape[0])
                idx_good = np.delete(idx_all, ids)

                idxc_array[animal_id].append(idxc.shape[0] + idx_good.shape[0])

        self.idxc_array = idxc_array
        print (self.idxc_array)
        ax=plt.subplot()

        t = np.arange(6,-1,-1)
        for k in range(len(self.idxc_array)):
            plt.plot(t, np.array(self.idxc_array[k])/2069710., label=" animal: "+str(k))
            #plt.plot(t, I.idxc_array[k], label=" animal: "+str(k))

        plt.legend()
        plt.ylim(0,1)
        plt.xlabel("min # of present features ", fontsize=20)
        ax.ticklabel_format(useOffset=False)
        ax.ticklabel_format(style='plain')
        ax.set_ylabel("% of total # of frames in 23hr period",
                     fontsize=20)
        plt.plot([3,3],[0,1],'--')
        #plt.xlim(-0.1,6.1)
        plt.show()

    #



    def plot_imputation_results(self, features_array, animal_id, idx_test, res, idx_drop):

        #
        labels = ['n','s1','s2','s3','s4','s5']

        # grab the selected data:
        temp = np.vstack(features_array[animal_id])
        temp = temp-temp[:,0][:,None]

        #print ("temp: ", temp.shape)
        fig = plt.figure()
        ax = plt.axes()
        shift = 0
        for k in range(10):
            plt.subplot(2,5,k+1)

            ############ PLOT GROUND TRUTH ##############
            id2 = np.random.choice(idx_test,1)[0]
            #print (id2)

            #print (temp[id2].shape)
            print (temp[id2,:,0])
            plt.scatter(temp[id2,:,0],
                        temp[id2,:,1],
                        c='blue',
                        s=np.arange(1,7,1)[::-1]*20,
                        alpha=.7,
                        edgecolor='black', label='truth')

            #
            id3 = np.where(idx_test==id2)[0]
            #print (id3)

            ############ PLOT IMPUTED LOCS ##############
            plt.scatter(res[id3,:,0]+shift,
                        res[id3,:,1],
                        c='red',
                        s=np.arange(1,7,1)[::-1]*20,
                        alpha=.7,
                        edgecolor='black', label='imputed')

            if True: #k==0:
                for p in range(6):
    #                 plt.text(res[id3,p,0],
    #                          res[id3,p,1],labels[p])
                    plt.text(temp[id2,p,0],
                             temp[id2,p,1],labels[p])

            # draw lines
            for p in range(len(idx_drop)):
                #print ("connectgin: ", p, idx_drop[p][id3])
                #print ("idx_drop full: ", np.array(idx_drop).shape)
                plt.plot([temp[id2,idx_drop[p][id3],0], res[id3,idx_drop[p][id3],0]+shift],
                         [temp[id2,idx_drop[p][id3],1], res[id3,idx_drop[p][id3],1]],
                         '--',c='black')

            if k==0:
                plt.legend(fontsize=8)

            plt.title("frame id: "+str(id2) + "\ndrops: "+str(np.array(idx_drop)[:,id3]),fontsize=8)

            x1 = (np.max(np.abs(temp[id2,:,0])),
                                 np.max(np.abs(res[id3,:,0])),
                                 np.max(np.abs(temp[id2,:,1])),
                                 np.max(np.abs(res[id3,:,1])))
            #print ("x1: ", x1)

            max_ = np.max(x1)*1.2
            #plt.xlim(-max_, max_+shift)
            plt.xlim(-200, 200)
            plt.ylim(-200,200)
            #plt.ylim(-max_,10)
            ax.set_aspect('equal', 'datalim')
            #print ('')

        plt.show()


    def predict_imputation_new_data(self, max_drops=3):

        ''' Function used to impute missing data

        '''

        # NED

        print ("NEED TO LOAD NOnNAN DATA FIRST")
        self.cb.load_processed_data(data_type=0, remove_nans=False)

        #
        all_locs = np.vstack(self.cb.features_array[self.animal_id])

        # find which frames have that feature intact
        idx = np.where(np.isnan(all_locs))
        ids, counts = np.unique(idx[0], return_counts=True)

        #idx_all = np.arange(all_locs.shape[0])
        #idx_good = np.delete(idx_all, ids)

        # find frame Ids where at most max_drops of features are missing
        idxc = np.where(counts<=max_drops*2)[0]

        #
        X_all = all_locs[idxc]
        print ("X_all: ", X_all.shape)

        # loop over each feature
        res = np.zeros((6,X_all.shape[0],
                              X_all.shape[1],
                              X_all.shape[2]),'float32')+np.nan
        for k in range(6):

            # find frames where current feature is present so we can apply model to it
            idx_feature = np.where(np.isnan(X_all[:,k,0])==False)[0]

            X_test = X_all[idx_feature]

            # load model
            temp_model = self.models[self.animal_id][k]
            #print (k, temp_model)

            # centre the data on the selected feature
            offsets = X_test[:,k]
            X_test = X_test-offsets[:,None]

            #print ("XTES: ", X_test.shape)

            # format the data
            X_test = X_test.reshape(X_test.shape[0],-1)

            # run mprediction
            r = temp_model.transform(X_test).reshape(-1,6,2)
            print ("Result: ", r.shape)

            # add offset back in
            r = r + offsets[:,None]
            #
            res[k,idx_feature] = r


        # average over all models
        print (" res: ", res.shape)
        #res_ave = np.nanmean(res,axis=0)

        # use only front body prediction averages
        #res_ave = np.nanmedian(res[:3],axis=0)
        #print (" res_ave; ", res_ave.shape)

        res_ave = res[0]

        return res_ave, all_locs, idxc

    def gen_imputation_multi_models(pairs,
                                    root_dir,
                                    model_type_name,
                                    animal_ids,

                                    ):
        f1 = pairs[0]
        f2 = pairs[1]

        #
        fname_model = os.path.join(root_dir, model_type_name+
                                           "_"+str(f1)+"_"+str(f2))

        # load filtered + outliner triaged data with Nans removed
        self.cb.load_processed_data(f1, f2,
                                    data_type=0,
                                    remove_nans=True)

        #
        for a in range(self.animal_ids.shape[0]):

            # fname_out = fname_features.replace('.npz','_animalID'+str(a)+"_"+
            #                                   self.model_type_names[self.model_type]+".pkl")
            fname_out = fname_model+'_animalID_'+str(a)+".pkl"

            if os.path.exists(fname_out)==False:
                temp = np.vstack(self.cb.features_array[a])
                print ("Raw Data: ", temp.shape)

                X_train = temp.reshape(temp.shape[0],-1)
                print ("X_train: ", X_train.shape)

                print ("fitting...animal id: ", a, "  body centre: ", f1, "  rotation pt: ", f2)
                imp = IterativeImputer(max_iter=10,
                                       #n_nearest_features = 2,
                                       #sample_posterior = True,
                                       estimator=self.estimators[self.model_type],
                                       random_state=0)
                imp.fit(X_train)
                print ("done")

                #
                with open(fname_out, "wb") as f:
                    pickle.dump(imp, f)
        #



    def generate_imputation_models_all_pairs(self):

        # make a model for every animal and every pair-wise
        #feature_ids = np.arange(6)

        pairs = []
        for f1 in range(self.feature_ids.shape[0]):
            for f2 in range(f1+1, self.feature_ids.shape[0],1):
                pairs.append([f1,f2])

        if self.parallel:
            parmap.map(models_parallel,
                       pairs,
                       self.March16_file_order,
                       self.model_type_names[self.model_type],
                       self.root_dir,
                       self.animal_ids,
                       self.feature_ids,
                       self.estimators[self.model_type],
                       data_type=0,
                       pm_processes= 8)
        else:
            for pair in pairs:
                models_parallel(
                               pair,
                               self.March16_file_order,
                               self.model_type_names[self.model_type],
                               self.root_dir,
                               self.animal_ids,
                               self.feature_ids,
                               self.estimators[self.model_type],
                               data_type=0)


    def make_vae_data(self):
        import csv

        print (np.vstack(cb.features_array[0]).shape)

        X = np.vstack(cb.features_array[0])
        #X = X.reshape(X.shape[0],-1)

        max_drop_outs = 3
        n_frames = 50000
        n_frames = X.shape[0]

        #
        with open('/home/cat/feats_ground_truth.tsv', 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            #for record in SeqIO.parse("/home/fil/Desktop/420_2_03_074.fastq", "fastq"):
            #for k in trange(X.shape[0]):
            for k in trange(n_frames):

                temp = X[k]
                if False:
                    idx = np.random.choice(np.arange(6), max_drop_outs, replace=False)

                    if idx.shape[0]>0:
                        temp[idx]=np.nan
                temp = temp.reshape(-1)
                writer.writerow(temp)

        with open('/home/cat/feats_dropout.tsv', 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            #for record in SeqIO.parse("/home/fil/Desktop/420_2_03_074.fastq", "fastq"):
            #for k in trange(X.shape[0]):
            for k in trange(n_frames):

                temp = X[k]
                if True:
                    idx = np.random.choice(np.arange(6),max_drop_outs, replace=False)

                    if idx.shape[0]>0:
                        temp[idx]=np.nan
                temp = temp.reshape(-1)
                writer.writerow(temp)


    def plot_multiple_imputation_results(self):

        #
        all_locs = np.array(np.loadtxt('/home/cat/feats_ground_truth.tsv'))
        all_locs = all_locs.reshape(all_locs.shape[0],6,2)
        #print (all_locs.shape)

        #
        drops = np.arange(3,6,1)
        leftin = np.arange(0,3,1)

        frame_ids = self.frame_ids
        for k in range(len(self.frame_ids)):

            id_ = frame_ids[k]


            ############## PLOT IMPUTED #################
            self.axes[k].scatter(self.res[id_,self.dropped_features,0],
                       self.res[id_,self.dropped_features,1],
                        s=np.arange(1,7,1)[::-1][self.dropped_features]*50,
                        hatch='***',
                        c='green'
                       )
            ymin, ymax = self.axes[k].get_ylim()
            xmin, xmax = self.axes[k].get_xlim()

            self.axes[k].set_xlim(min(xmin, -50), max(xmax,50))
            self.axes[k].set_ylim(min(ymin, -50), max(ymax,50))


        #plt.suptitle(str(I.model_type_names[I.model_type]))
        #plt.suptitle(self.model_type_names[self.model_type])

    def plot_vae_scatter(self, dropped_features = None, fig=None):

        self.dropped_features = np.array(dropped_features)
        #
        if fig is None:
            fig=plt.subplot()

        #
        try:
            print (self.gt.shape)
            print (self.imputed.shape)
            print (self.gt_dropout.shape)
        except:

            self.gt = np.loadtxt('/home/cat/feats_ground_truth.tsv')
            self.gt_dropout = np.loadtxt('/home/cat/feats_dropout.tsv')
            self.imputed = np.loadtxt('/home/cat/data_imputed.tsv')

        #
        print (self.gt.shape)
        print (self.imputed.shape)
        print (self.gt_dropout.shape)

        frame_ids = []
        self.axes = []
        for k in range(10):
            #
            while True:
                id_ = np.random.choice(np.arange(self.gt.shape[0]),1)

                temp = self.gt_dropout[id_]
                idx = np.where(np.isnan(temp))[1]

                # look only for the bottom 3 feature
                if dropped_features is not None:
                    self.kept_features = np.int32(np.delete(np.arange(6),dropped_features))
                    match_drop = []
                    for d in range(len(dropped_features)):
                        match_drop.append ([dropped_features[d]*2,dropped_features[d]*2+1])

                    match_drop = np.hstack(match_drop)
                    if np.array_equal(idx,match_drop)==False:
                        continue
                    self.match_drop = match_drop
                    print ("match_drop:", match_drop)
                else:
                    self.match_drop = []


                #
                temp = self.gt[id_]
                temp = temp.reshape(6,2)

                self.axes.append(plt.subplot(2,5,k+1))

                if k==0:
                    import matplotlib.patches as mpatches

                    # where some data has already been plotted to ax
                    #handles, labels = ax.get_legend_handles_labels()

                    # manually define a new patch
                    labels = []
                    labels.append(mpatches.Patch(color='black', label='fixed - gt'))
                    labels.append(mpatches.Patch(color='blue', label='dropout - gt'))
                    labels.append(mpatches.Patch(color='red', label='vae - imputed'))
                    labels.append(mpatches.Patch(color='green', label=self.model_type_names[self.model_type]+' - imputed'))

                    # plot the legend
                    plt.legend(handles=labels,fontsize=8)

                plt.scatter(temp[self.dropped_features,0], #-100,
                           temp[self.dropped_features,1],
                           s=np.arange(1,7,1)[::-1][self.dropped_features]*50,
                           alpha=1,
                           c='blue')

                temp = self.gt_dropout[id_]
                temp = temp.reshape(6,2)
                plt.scatter(temp[self.kept_features ,0], #-100,
                           temp[self.kept_features ,1],
                           s=np.arange(1,7,1)[::-1][self.kept_features]*50,
                           alpha=1,
                           c='black')

                # PLOT IMPUTATION
                temp5 = self.imputed[int(id_*5):int(id_*5+5)]
                print (temp5.shape)
                for t in range(temp5.shape[0]):
                    temp = temp5[t]
                    temp = temp.reshape(6,2)
                    plt.scatter(temp[self.dropped_features,0],
                           temp[self.dropped_features,1],
                           s=np.arange(1,7,1)[::-1][self.dropped_features]*50,
                           c='red',
                           alpha=.1)

                temp = np.mean(temp5,axis=0)
                temp = temp.reshape(6,2)
                plt.scatter(temp[self.dropped_features,0],
                       temp[self.dropped_features,1],
                       s=np.arange(1,7,1)[::-1][self.dropped_features]*50,
                       c='red',
                            hatch='***',
                       alpha=1)

                frame_ids.append(id_)

                ymin, ymax = self.axes[k].get_ylim()
                xmin, xmax = self.axes[k].get_xlim()

                plt.xlim(min(xmin, -50), max(xmax,50))
                plt.ylim(min(ymin, -50), max(ymax,50))


                break


        plt.suptitle("Variational Autoencoder with Arbitrary Conditioning (https://github.com/tigvarts/vaeac)")
        plt.show()

        self.fig = fig
        self.frame_ids = frame_ids

        #return frame_ids

    #
    def evaluate_imputation_vae(self):

        # VAE ERROR
        imputed = np.loadtxt('/home/cat/data_imputed.tsv')
        gt = np.float32(np.loadtxt('/home/cat/feats_ground_truth.tsv'))

        gt = np.float32(gt.reshape(gt.shape[0],6,2))


        imputed = np.array(np.array_split(imputed, 5))
        imputed = np.mean(imputed,axis=0)

        imputed = imputed.reshape(imputed.shape[0],6,2)


        diff = np.float32(np.abs(gt-imputed))
        #print ("diff: ", diff.shape)

        errors = []
        for k in range(diff.shape[1]):
            errors.append([])

        for k in range(diff.shape[0]):
            for p in range(diff.shape[1]):
                temp = diff[k,p]
                #print (temp)
                tdiff = np.linalg.norm(temp)
                if tdiff>0:
                    errors[p].append(tdiff)

        t =[]
        for k in range(len(errors)):
            temp = errors[k]
            pad = np.zeros(350000-len(errors[k]),'float32')+np.nan
            temp = np.concatenate((temp, pad))

            idx = np.where(temp<1E-8)[0]
            temp[idx]=np.nan
            t.append(temp)

        errors = np.float64(t).T

        columns = ['nose','spine1','spine2', 'spine3', 'spine4', 'spine5']
        df = pd.DataFrame(errors, columns = columns)

        print (df)

        fig=plt.figure()
        ax = sns.violinplot(data=df) #x=df['spine2'])
        plt.ylim(0,50)
        plt.title("VAE IMPUTATION")
        #plt.ylabel(" pixel error")

        return df

    def evaluate_imputation_multivariate(self):

        # VAE ERROR
        imputed = self.res #np.loadtxt('/home/cat/data_imputed.tsv')
        gt = np.loadtxt('/home/cat/feats_ground_truth.tsv')

        gt = gt.reshape(gt.shape[0],6,2)

        # comptue distances
        diff = np.float32(np.abs(gt-imputed))
        errors = []
        for k in range(diff.shape[1]):
            errors.append([])

        for k in range(diff.shape[0]):
            for p in range(diff.shape[1]):
                temp = diff[k,p]
                tdiff = np.linalg.norm(temp)
                if tdiff>0:
                    errors[p].append(tdiff)

        t =[]
        for k in range(len(errors)):
            temp = errors[k]
            pad = np.zeros(350000-len(errors[k]),'float32')+np.nan
            temp = np.concatenate((temp, pad))
            t.append(temp)

        data = np.array(t).T
        columns = ['nose','spine1','spine2', 'spine3', 'spine4', 'spine5']
        df = pd.DataFrame(data, columns = columns)

        print (df)

        fig=plt.figure()

        ax = sns.violinplot(data=df) #x=df['spine2'])
        plt.ylim(0,50)
        plt.title(self.model_type_names[self.model_type])

        return df


def filter_data1(fname):

    fname_out = fname.replace('.npy','_median_filtered.npy')
    if os.path.exists(fname_out)==False:
        data = np.load(fname)

        data_filtered = data.copy()
        # loop over animals
        for a in range(data.shape[1]):
            # loop over feature
            for f in range(data.shape[2]):
                # loop over each x,y
                for l in range(data.shape[3]):
                    data_filtered[:,a,f,l] = filter_data2(data[:,a,f,l])

        np.save(fname_out, data_filtered)


def filter_data2(x, width=25):

    # replace data with previous value until all nans are replacesd
    for k in range(1000):
        idx = np.where(np.isnan(x))[0]
        if idx.shape[0]==0:
            break

        if idx[0]==0:
            idx=idx[1:]
        x[idx] = x[idx-1]


    x = scipy.ndimage.median_filter(x, size=width)

    return x


#
def models_parallel(pairs,
                    March16_file_order,
                    model_name,
                    root_dir,
                    animal_ids,
                    feature_ids,
                    estimator,
                    data_type=0):

    #
    f1 = pairs[0]
    f2 = pairs[1]

    #
    fname_model = os.path.join(root_dir, model_name+"_"+str(f1)+"_"+str(f2))

    # load filtered + outliner triaged data with Nans removed
    features_array = load_processed_data_stand_alone(f1, f2, March16_file_order,
                                                    root_dir,
                                                    animal_ids,
                                                    data_type=data_type,
                                                    remove_nans=True)

    # loop over animals
    for a in range(animal_ids.shape[0]):

        # fname_out = fname_features.replace('.npz','_animalID'+str(a)+"_"+
        #                                   self.model_type_names[self.model_type]+".pkl")
        fname_out = fname_model+'_animalID_'+str(a)+".pkl"

        if os.path.exists(fname_out)==False:
            temp = np.vstack(features_array[a])
            print ("Raw Data: ", temp.shape)

            X_train = temp.reshape(temp.shape[0],-1)
            print ("X_train: ", X_train.shape)

            print ("fitting...animal id: ", a, "  body centre: ", f1, "  rotation pt: ", f2)
            imp = IterativeImputer(max_iter=10,
                                   #n_nearest_features = 2,
                                   # sample_posterior = True,
                                   estimator=estimator,
                                   random_state=0)
            imp.fit(X_train)
            print ("done")

            #
            with open(fname_out, "wb") as f:
                pickle.dump(imp, f)


def predict_novel_data1(pairs,
                        March16_file_order,
                        root_dir,
                        model_type_name,
                        animal_ids,
                        animals_selected
                        ):
    #
    f1 = pairs[0]
    f2 = pairs[1]

    # load specific egocentric data
    features_array = load_processed_data_stand_alone(f1, f2,
                                                     March16_file_order,
                                                     root_dir,
                                                     animal_ids,
                                                     data_type=0,
                                                     remove_nans=False)

    # the original, non rotated data needs to be rotated
    # must first find out which frames contain the features we need


    # drop out specific data
    for animal_id in animals_selected:
        fname_out = os.path.join(root_dir,"prediction_"+model_type_name+"_animalID_"+str(animal_id)+
                         "_"+str(f1)+"_"+str(f2)+'.npy')

        #
        if os.path.exists(fname_out)==False:
            gt = np.vstack(features_array[animal_id])

            # predict on all data
            X_test = gt.reshape(gt.shape[0],-1)

            print ("X_test: ", X_test.shape)

            # load f1 - f2 model
            fname_in = os.path.join(root_dir,
                        model_type_name+"_"+str(f1)+"_"+str(f2)+
                        "_animalID_"+str(animal_id)+".pkl")

            # print ("  loading model...")
            with open(fname_in, "rb") as f:
                model = pickle.load(f)

            # impute missing data in egocentric coordinates
            # print ("  imputing missing data...")
            res = model.transform(X_test).reshape(-1,6,2)

            #
            np.save(fname_out,
                    res)



def evaluate_errors_stand_alone(pair,
                                drops,
                                root_dir,
                                model_type_name,
                                March16_file_order,
                                animal_ids,
                                animals_selected,
                                ):

    f1 = pair[0]
    f2 = pair[1]

    # find which frame drops contain f1 and f2
    idx1 = np.where(drops==f1)[0]
    idx2 = np.where(drops==f2)[0]
    idx3 = np.intersect1d(idx1, idx2)
    # print (idx1.shape, idx2.shape, " intersection f1 f2: ", idx3.shape)

    # load specific egocentric data
    features_array = load_processed_data_stand_alone(f1, f2,
                                                     March16_file_order,
                                                     root_dir,
                                                     animal_ids,
                                                     data_type=0,
                                                     remove_nans=True)

    # drop out specific data
    for animal_id in animals_selected:
        fname_out = os.path.join(root_dir,"errors_"+model_type_name+"_animalID_"+str(animal_id)+
                         "_"+str(f1)+"_"+str(f2)+'.npy')

        #
        if os.path.exists(fname_out)==False:
            gt = np.vstack(features_array[animal_id])
            #print ("self.gt: ", self.gt.shape)

            # initialize test arrays
            test = np.zeros((gt.shape),'float32')+np.nan
            for k in range(idx3.shape[0]):
                # print (self.gt[k].shape, self.gt[k][drops[idx3[k]]].shape)
                test[k,drops[idx3[k]]] = gt[k,drops[idx3[k]]]

            #
            # print ("test = ", test.shape)

            # reshape data
            X_test = test.reshape(test.shape[0],-1)

            # load f1 - f2 model
            fname_in = os.path.join(root_dir,
                        model_type_name+"_"+str(f1)+"_"+str(f2)+
                        "_animalID_"+str(animal_id)+".pkl")

            # print ("  loading model...")
            with open(fname_in, "rb") as f:
                model = pickle.load(f)

            # impute missing data in egocentric coordinates
            # print ("  imputing missing data...")
            res = model.transform(X_test).reshape(-1,6,2)

            # compute error
            # print ("  computing error ...")
            error = evaluate_res_error_stand_alone(gt, res)

            # print ("Saving errrors....")


            np.save(fname_out,
                    error)


def evaluate_res_error_stand_alone(res, gt):
    # imputed = self.res #np.loadtxt('/home/cat/data_imputed.tsv')
    # gt = np.loadtxt('/home/cat/feats_ground_truth.tsv')

    print ("res: ", res.shape)
    print ("gt: ", gt.shape)

    # comptue distances
    diff = np.float32(np.abs(gt-res))

    # initialize error lists to track missing data
    errors = []
    for k in range(diff.shape[1]):
        errors.append([])

    # loop over frames
    for k in range(diff.shape[0]):
        # loop over features
        for p in range(diff.shape[1]):
            temp = diff[k,p]
            tdiff = np.linalg.norm(temp)
            if tdiff>0:
                errors[p].append(tdiff)

    return errors




def plot_errors(I):
    ctr=0
    sns.reset_defaults()

    fig=plt.figure()
    for f1 in range(6):
        for f2 in range(f1+1, 6, 1):
            print (" PLOTTING: ", f1, f2)
            fname = os.path.join(I.root_dir,'errors_'+I.model_type_names[I.model_type]+
                                 "_animalID_"+str(I.animals_selected[0])+"_"+str(f1)+"_"+str(f2)+".npy")
            error = np.load(fname,allow_pickle=True)
            error_f1f2 = np.array(error)

            #
            ax = plt.subplot(5,6,f1*6+f2)
            t =[]
            for k in range(len(error_f1f2)):
                temp = error_f1f2[k]
                pad = np.zeros(250000-len(error_f1f2[k]),'float32')+np.nan
                temp = np.concatenate((temp, pad))

                idx = np.where(temp<1E-8)[0]
                temp[idx]=np.nan
                t.append(temp)

            errors = np.float64(t).T

            columns = ['nose','spine1','spine2', 'spine3', 'spine4', 'spine5']
            df = pd.DataFrame(errors, columns = columns)


            p = sns.violinplot(data=df) #x=df['spine2'])
            _, ylabels = plt.yticks()
            _, xlabels = plt.xticks()

            p.set_xticklabels(xlabels, size=10, rotation=45)
            #p.set_yticklabels(ylabels, size=13)


            plt.ylim(0,50)
            plt.title(str(f1)+" "+str(f2),pad=.8)
            if f1==0 and f2==1:
                pass
            else:
                plt.xticks([])
                plt.yticks([])

            ctr+=1
            print (ctr,)
            print ('')

            #break
        #break
    plt.show()

