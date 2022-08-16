import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
import cv2

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
        self.fname_movie = self.fname.replace("_fixed.npy",".avi")

        # load full feature tracks
        self.tracks = np.load(self.fname)
        print (" full features track data: ", self.tracks.shape)



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
            ids = [6,7,5,8,4,3,2,1,0]  # centred on spine2
            #points=[nose, lefteye, righteye, leftear,rightear, spine1, spine2, spine3,spine4]
            #         0       1        2        3         4      5        6        7      8

            # loop over time
            for n in trange(self.tracks.shape[0]):
                # loop over animals
                for a in range(self.tracks.shape[1]):
                    # loop over features to find nearest to spine2
                    for id_ in ids:
                        if np.isnan(self.tracks[n,a,id_,0])==False:
                            self.tracks_spine[n,a]=self.tracks[n,a,id_]

                            break
            np.save(fname_out,
                    self.tracks_spine)
        else:
            self.tracks_spine = np.load(fname_out)

    # def get_angle(self):
    #
    #     #
    #     fname_out = self.fname[:-4]+"_angles.npy"
    #
    #     if os.path.exists(fname_out)==False:
    #
    #         #
    #         self.angles = np.zeros((self.tracks.shape[0],
    #                                 self.tracks.shape[1]),
    #                                 'float32')+np.nan
    #         #
    #         deg_scale = 180/3.1415926
    #         deg_scale = 1
    #         #
    #         for k in trange(self.tracks.shape[0]):
    #             for a in range(self.tracks.shape[1]):
    #                 #x = self.tracks[k,a,:,0]
    #                 #y = self.tracks[k,a,:,1]
    #                 x = self.tracks[k,a,5:10,0]
    #                 y = self.tracks[k,a,5:10,1]
    #                 idx = np.where(np.isnan(x)==False)[0]
    #                 if idx.shape[0]>0:
    #                     x=x[idx]
    #                     y=y[idx]
    #                     m,b = np.polyfit(y, x, 1)
    #                     self.angles[k,a] = np.arctan(m)*deg_scale
    #                     if x[0]<x[-1]:
    #                         self.angles[k,a]+=180
    #                     #self.angles[k,a] = np.arctan(m)*deg_scale
    #
    #         np.save(fname_out,
    #                 self.angles)
    #     else:
    #         self.angles = np.load(fname_out)
    #
    #     self.angles = self.angles[self.start:self.end]

    def get_angle_and_axes_section(self):

       #
        self.angles = np.zeros((self.end-self.start,
                                self.tracks.shape[1]),
                                'float32')+np.nan
        self.axes = np.zeros((self.end-self.start,
                                  self.tracks.shape[1],
                                  2),
                                    'float32')+np.nan


        #
        deg_scale = 180/3.1415926
        deg_scale = 1
        #
        for k in tqdm(range(self.start, self.end,1)):
            for a in range(self.tracks.shape[1]):
                #x = self.tracks[k,a,:,0]
                #y = self.tracks[k,a,:,1]
                x = self.tracks[k,a,5:10,0]
                y = self.tracks[k,a,5:10,1]
                idx = np.where(np.isnan(x)==False)[0]
                if idx.shape[0]>0:
                    x=x[idx]
                    y=y[idx]
                    m,b = np.polyfit(x, y, 1)
                    angle = np.arctan(m)*deg_scale
                    self.angles[k,a] = angle
                    if x[0]<x[-1]:
                        self.angles[k,a]+=180

                    # stack locations
                    locs = np.vstack((x,y)).T

                    # rotate
                    theta = np.radians(angle)
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array(((c, -s), (s, c)))
                    locs_r = locs@R

                    # Reject outliers that are substantially outside of data
                    x = self.reject_outliers(locs_r[:,0])
                    y = self.reject_outliers(locs_r[:,1])

                    self.axes[k,a,0] = np.max(x)-np.min(x)
                    self.axes[k,a,1] = np.max(y)-np.min(y)


                    #self.angles[k,a] = np.arctan(m)*deg_scale

    # remove outliars
    def reject_outliers(self, data, m = 4.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    def get_axes(self):

        #
        fname_out = self.fname[:-4]+"_major_minor.npy"

        if os.path.exists(fname_out)==False:

            self.axes = np.zeros((self.tracks.shape[0],
                                  self.tracks.shape[1],
                                  2),
                                    'float32')+np.nan

            #
            for k in trange(self.tracks.shape[0]):
                for a in range(self.tracks.shape[1]):

                    x = self.tracks[k,a,:,0]
                    y = self.tracks[k,a,:,1]
                    idx = np.where(np.isnan(x)==False)[0]

                    #
                    if idx.shape[0]>0:

                        # load data
                        x=x[idx]
                        y=y[idx]
                        angle = self.angles[k,a]

                        # stack locations
                        locs = np.vstack((x,y)).T

                        # rotate
                        theta = np.radians(angle)
                        c, s = np.cos(theta), np.sin(theta)
                        R = np.array(((c, -s), (s, c)))
                        locs_r = locs@R



                        # Reject outliers that are substantially outside of data
                        x = self.reject_outliers(locs_r[:,0])
                        y = self.reject_outliers(locs_r[:,1])

                        self.axes[k,a,0] = np.max(x)-np.min(x)
                        self.axes[k,a,1] = np.max(y)-np.min(y)

            np.save(fname_out, self.axes)
        else:
            self.axes=np.load(fname_out)

        self.axes = self.axes[self.start:self.end]/self.scale

    #
    def convert_npy_to_jaaba(self):

        #
        if self.end is None:
            end = self.tracks_spine.shape[0]

        # get rotations
        self.get_angle_and_axes_section()

        # get body shrinking/stretching
        #self.get_axes()
        self.axes = self.axes/self.scale

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

        fname_out = self.fname_movie[:-4]+"_"+str(self.start)+"_"+str(self.end)+".mp4"
        if os.path.exists(fname_out):
            return

        #
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
                dsize = (size_vid[0], size_vid[1])

                # resize image
                frame = cv2.resize(frame, dsize)

            video_out.write(frame)
        video_out.release()
        original_vid.release()






