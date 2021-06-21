import matplotlib
import matplotlib.pyplot as plt
import scipy

import numpy as np
import os
import cv2
import warnings

from scipy.io import loadmat, savemat
from tqdm import trange, tqdm



###########################################################
###### CONVERT DATA TO JAABA trx.mat FORMAT ###############
###########################################################

class Convert():

    #
    def __init__(self, fname):

        #
        self.fname = fname

        #
        #self.fname_movie = self.fname.replace("_fixed.npy",".avi")
        self.fname_movie = self.fname.replace("_fixed.npy",".mp4")

        # load full feature tracks
        self.tracks = np.load(self.fname)
        print (" full features track data: ", self.tracks.shape)

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
        self.get_track_spine_centers()

    #
    def get_track_spine_centers(self):
        '''  This function returns single locations per animal
            with a focus on spine2, spine3, spine1 etc...
        '''
        #
        fname_out = self.fname[:-4]+"_spine.npy"
        if os.path.exists(fname_out)==False:
            self.tracks_spine = np.zeros((self.tracks.shape[0],
                                          self.tracks.shape[1],
                                          self.tracks.shape[3]),
                                         'float32')
            ids = [6,7,5,8,4,3,2,1,0,9,10,11,12]  # centred on spine2
            #points=[nose, lefteye, righteye, leftear,rightear, spine1, spine2, spine3,spine4]
            #         0       1        2        3         4      5        6        7      8

            # loop over time
            for n in trange(self.tracks.shape[0]):
                # loop over animals
                for a in range(self.tracks.shape[1]):
                    # loop over features to find nearest to spine2

                    idx = np.where(np.isnan(self.tracks[n,a,:,0])==False)[0]
                    if idx.shape[0]>=3:
                        #print ("idx: ", idx)
                        x_idx = idx[self.reject_outliers(self.tracks[n,a,idx,0])]
                        y_idx = idx[self.reject_outliers(self.tracks[n,a,idx,1])]
                        z_idx = np.intersect1d(x_idx, y_idx)
                        #print (" zidx: ", z_idx)

                        #print ("self.tracks[n,a,idx[z_idx]]: ", self.tracks[n,a,z_idx])
                        for id_ in ids:
                            if id_ in z_idx:
                                self.tracks_spine[n,a]=self.tracks[n,a,id_]

                                break
            np.save(fname_out,
                    self.tracks_spine)
        else:
            self.tracks_spine = np.load(fname_out)

    def median_filter(self, x):

        for k in range(1000):
            idx = np.where(np.isnan(x))[0]
            if idx.shape[0]==0:
                break

            if idx[0]==0:
                idx=idx[1:]
            x[idx] = x[idx-1]

        x = scipy.ndimage.median_filter(x, size=25)

        return x


    def get_angle_and_axes_section2(self):

        from math import atan2, pi

        def angle_trunc(a):
            while a < 0.0:
                a += pi * 2
            return a

        def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
            deltaY = y_landmark - y_orig
            deltaX = x_landmark - x_orig
            return angle_trunc(atan2(deltaY, deltaX)) #* 180 / PI

  #
        if self.end is None:
            self.end = self.tracks.shape[0]

        self.angles = np.zeros((self.end-self.start,
                                self.tracks.shape[1]),
                                'float32')+np.nan
        self.axes = np.zeros((self.end-self.start,
                                  self.tracks.shape[1],
                                  2),
                                    'float32')+np.nan
        #
        # grab nose which is ID = 0 and ids 5:9 for body

        for a in range(self.tracks.shape[1]):
            for k in tqdm(range(self.start, self.end,1)):
                x = self.tracks[k,a,5:10,0]
                y = self.tracks[k,a,5:10,1]

                # # try adding the nose location also
                try:
                    x = np.concatenate((self.tracks[k,a,0,0], x))
                    y = np.concatenate((self.tracks[k,a,0,1], y))
                except:
                    pass

                idx = np.where(np.isnan(x)==False)[0]
                if idx.shape[0]>0:
                    x=x[idx]
                    y=y[idx]

                    if x.shape[0]<=3:
                        continue

                    angle = getAngleBetweenPoints(np.mean(x[2:4]), np.mean(y[2:4]),
                                                  np.mean(x[0:2]), np.mean(y[0:2]))

                    self.angles[k,a] = angle


                    if False:
                        # find 2 consecutive values and compute the axes as a function of those values NOT
                        #  standard deviatoin and other measures that average over
                        #  BUT DOESN"T WORK, MUCH TOO NOISY
                        x_idx = self.reject_outliers(x)
                        y_idx = self.reject_outliers(y)
                        z_idx = np.intersect1d(x_idx, y_idx)
                        x=x[z_idx]
                        y=y[z_idx]

                        # convert back to original animal sequential ids
                        z_idx_original = idx[z_idx]
                        # find 2 consecutive points and use them as measure of length
                        for i in range(z_idx_original.shape[0]-1):
                            if z_idx_original[i+1] - z_idx_original[i] == 1:
                                self.axes[k,a,1] = x[i-1]-x[i]
                                self.axes[k,a,0] = y[i-1]-y[i]

                    else:
                        #
                        locs = np.vstack((x,y)).T

                        # rotate
                        theta = np.radians(angle)
                        c, s = np.cos(theta), np.sin(theta)
                        R = np.array(((c, -s), (s, c)))
                        locs_r = locs@R


                        # Reject outliers that are substantially outside of data
                        x_idx = self.reject_outliers(locs_r[:,0])
                        y_idx = self.reject_outliers(locs_r[:,1])

                        z_idx = np.intersect1d(x_idx, y_idx)
                        x=locs_r[z_idx,0]
                        y=locs_r[z_idx,1]

                        #
                        #self.axes[k,a,1] = np.max(x)-np.min(x)
                        #self.axes[k,a,0] = np.max(y)-np.min(y)

                        self.axes[k,a,1] = np.std(x)
                        self.axes[k,a,0] = np.std(y)

            # apply 1D angle filter
            if self.apply_median_filter:

                self.angles[:,a] = self.median_filter((self.angles[:,a]))
                self.axes[:,a,0] = self.median_filter(self.axes[:,a,0])
                self.axes[:,a,1] = self.median_filter(self.axes[:,a,1])

    # # remove outliars
    # def reject_outliers(self, data, m = 4.):
    #     d = np.abs(data - np.median(data))
    #     mdev = np.median(d)
    #     s = d/mdev if mdev else 0.
    #
    #     idx = np.where(s<m)[0]
    #
    #     return data[s<m]

    # remove outliars
    def reject_outliers(self, data, m = 4.):  # number of deviations away
        d = np.abs(data - np.median(data))
        mdev = np.median(d)

        if mdev:
            s = d/mdev
        else:
            s = 0.

        idx = np.where(s<m)[0]
        return idx

    #
    def convert_npy_to_jaaba(self):

        #
        if self.end is None:
            end = self.tracks_spine.shape[0]

        # get rotations
        self.get_angle_and_axes_section2()

        # get body shrinking/stretching
        self.axes = self.axes/self.scale

        # shirnk axes a bit
        #self.axes = self.axes/2.5
        self.axes[:,:,0] = self.axes[:,:,0]#/5
        self.axes[:,:,1] = self.axes[:,:,1]#/2.5

        #
        trx_array=[]
        for k in self.animal_ids:

            # x and y locations of the animals
            x = np.array(self.tracks_spine[self.start:self.end,k,0], self.dtype) # x-coordinate of the animal in pixels (1 x nframes).
            y = np.array(self.tracks_spine[self.start:self.end,k,1], self.dtype) # y-coordinate of the animal in pixels (1 x nframes).

            # remove zeros
            idx = np.where(x==0)[0]
            x[idx]=np.nan
            y[idx]=np.nan

            if self.apply_median_filter:
                x = self.median_filter(x)
                y = self.median_filter(y)

            # convert values according to scale:
            if self.scale !=1:
                x = x/self.scale
                y = y/self.scale

            # meta data for the video and animal
            nframes = np.array(self.end-self.start,self.dtype)    # Number of frames in the trajectory of the current animal (scalar).
            firstframe = np.array(1,self.dtype)             # First frame of the animal's trajectory (scalar). USE MATLAB INDEXES
            endframe = np.array(nframes,self.dtype)   # Last frame of the animal's trajectory (scalar).
            off = 0                    # Offset for computing index into x, y, etc. Always equal to 1 - firstframe (scalar).
            id_ = k+1                  # Identity number of the trajectory (scalar). USE MATLAB VALUES

            # convert from pixels to distances
            pixels_per_mm = 2                       # number of pixels per mm # to figure out
            x_mm = x//pixels_per_mm                 # x-coordinate of the animal in mm (1 x nframes).
            y_mm = y//pixels_per_mm                 # y-coordinate of the animal in mm (1 x nframes).

            #
            theta_mm = self.angles                     # Orientation of the animal in real coordinates. This is often the same as theta, if no transformation other than translation and scaling is performed between pixels and real coordinates (1 x nframes).

            #
            a_mm = self.axes[:,k,0]//pixels_per_mm  # 1/4 of the major-axis length in mm (1 x nframes).
            b_mm = self.axes[:,k,1]//pixels_per_mm  # 1/4 of the major-axis length in mm (1 x nframes).

            #
            sex = self.sexes[k]  # Sex of the animal. Can be just one value ('M' or 'F' or '?') or a cell array of 'M' and 'F' giving the sex for each frame. The size of the cell array should be nframes.

            # frame-rate based info
            dt = np.ones(int(nframes)-1, self.dtype)   # Difference in timestamps of the current frame and next frame, in seconds (1 x nframes-1).
            fps = 25                               # Average frames-per-second (scalar).

            days_per_timestamp = 1./(24*60*60*self.fps)
            timestamps = np.array(np.arange(nframes)*days_per_timestamp,
                                  self.dtype) # Timestamp of each frame (optional), in days (1 x nframes).

            #
            trx = {
             'x': x,
             'y':y,
             'theta': self.angles[:,k],
             'a': self.axes[:,k,0],
             'b': self.axes[:,k,1],
             'nframes':nframes,
             'firstframe':firstframe,
             'endframe':endframe,
             'off':off,
             'id':id_,
             'x_mm':x_mm,
             'y_mm':y_mm,
             'theta_mm':theta_mm[:,k],
             'a_mm':a_mm,
             'b_mm':b_mm,
             'sex':sex,
             'dt':dt,
             'fps':self.fps,
             'timestamps': timestamps
            }

            trx_array.append(trx)


        trx_dict = {'a':trx_array[0],
                    'b':trx_array[1],
                    'c':trx_array[2],
                    'd':trx_array[3]
                   }

        #
        savemat(self.fname[:-4]+'_'+str(self.start)+"_"+str(self.end)+'_trx.mat',
                {
                 'trx':trx_dict
                }
               )

        if self.make_movie:
            self.split_movie()

    #
    def split_movie(self):
        from tqdm import trange

        #fname_out = self.fname_movie[:-4]+"_"+str(self.start)+"_"+str(self.end)+".mp4"
        fname_out = self.fname_movie[:-4]+"_"+str(self.start)+"_"+str(self.end)+self.extension
        #if os.path.exists(fname_out):
        #    return

        #
        if os.path.exists(self.fname_movie)==False:
            print ("Video file missing: ", self.fname_movie)
            print ("... exiting...")
            return
        original_vid = cv2.VideoCapture(self.fname_movie)

        # video sizes
        size_vid = np.array([1280,1024])//self.scale
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, self.start)

        # Initialize video out
        fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
        video_out = cv2.VideoWriter(fname_out, fourcc, 25, (size_vid[0],size_vid[1]), True)

        for n in trange(0, self.end-self.start, 1):
            ret, frame = original_vid.read()

            # dsize
            if self.scale!=1:
                #print ("scaling")
                dsize = (size_vid[0], size_vid[1])

                # resize image
                frame = cv2.resize(frame, dsize)

            video_out.write(frame)
        video_out.release()
        original_vid.release()






