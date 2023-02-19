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

import h5py

import sleap


#
class Visualize():

    def __init__(self,
                 tracker=None):

        #
        self.tracker = tracker

    def make_video_skeleton(self,
                            fps=25,
                            start=None,
                            end=None):

        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 128, 0)
        ]

        #
        names = ['female', 'male', 'pup1', 'pup2']
        clrs = ['blue', 'red', 'cyan', 'green']

        if start is None:
            start = 0
        if end is None:
            end = self.tracks.shape[0]

        # load and videos
        fname_out = (self.fname_video[:-4] + '_' + str(start) + "_" + str(end) + '.mp4')

        # video settings
        size_vid = np.array([1280, 1024])
        dot_size = 4
        thickness = -1

        # load original vid
        original_vid = cv2.VideoCapture(self.fname_video)
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        video_out = cv2.VideoWriter(fname_out,
                                    fourcc,
                                    fps,
                                    (size_vid[0], size_vid[1]),
                                    True)

        original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

        font = cv2.FONT_HERSHEY_PLAIN

        # loop over frames
        for n in trange(start, end, 1):
            # for n in trange(start, 100, 1):
            ret, frame = original_vid.read()

            cv2.putText(frame, str(n), (50, 50), font, 5, (255, 255, 0), 5)

            for i in range(self.tracks.shape[1]):
                color = colors[i]  # (255, 0, 0)

                for f in range(self.tracks.shape[2]):
                    x = self.tracks[n, i, f, 0]
                    y = self.tracks[n, i, f, 1]

                    if np.isnan(x) or np.isnan(y):
                        continue

                    x = int(x)
                    y = int(y)

                    # plot cicrl
                    center_coordinates = (x, y)
                    radius = dot_size
                    frame = cv2.circle(frame, center_coordinates, radius, color, thickness)

                    # plot line
                    if f < (self.tracks.shape[2] - 1):
                        if np.isnan(self.tracks[n, i, f + 1, 0]) == False:
                            start_point = (int(self.tracks[n, i, f + 1, 0]),
                                           int(self.tracks[n, i, f + 1, 1]))

                            # End coordinate, here (250, 250)
                            # represents the bottom right corner of image
                            end_point = center_coordinates

                            # Line thickness of 9 px
                            thickness = 3

                            # Using cv2.line() method
                            # Draw a diagonal green line with thickness of 9 px
                            frame = cv2.line(frame,
                                             start_point,
                                             end_point,
                                             color,
                                             thickness)

            #                 frame[y-dot_size:y+dot_size,x-dot_size:x+dot_size]= (np.float32(
            #                     matplotlib.colors.to_rgb(clrs[i]))*255.).astype('uint8')

            video_out.write(frame)

        video_out.release()
        original_vid.release()

    def make_video_centroid(self,
                            tracks,
                            fname_video,
                            start,
                            end,
                            fps):

        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 128, 0)
        ]

        #
        names = ['female', 'male', 'pup1', 'pup2']
        clrs = ['blue', 'red', 'cyan', 'green']

        if start is None:
            start = 0
        if end is None:
            end = tracks.shape[0]

        # load and videos
        video_name = fname_video
        fname_out = (video_name[:-4] + '_' + str(start) + "_" + str(end) + '_centroid.mp4')
        if False:  # os.path.exists(fname_out):
            print("fname: exists", fname_out)
            return

        # video settings
        size_vid = np.array([1280, 1024])
        dot_size = 12
        thickness = -1

        # load original vid
        original_vid = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')

        video_out = cv2.VideoWriter(fname_out, fourcc, fps, (size_vid[0], size_vid[1]), True)
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

        font = cv2.FONT_HERSHEY_PLAIN

        # loop over frames
        histories = np.zeros((tracks.shape[1],
                              5, 2), 'float32') + np.nan
        print("Histories: ", histories.shape)
        for n in trange(start, end, 1):
            # for n in trange(start, 100, 1):
            ret, frame = original_vid.read()

            cv2.putText(frame, str(n), (50, 50), font, 5, (255, 255, 0), 5)

            #
            for i in range(tracks.shape[1]):
                color = colors[i]  # (255, 0, 0)

                # for f in range(tracks.shape[2]):
                x = tracks[n, i, 0]
                y = tracks[n, i, 1]

                try:
                    histories[i, 1:] = histories[i, :4]
                    histories[i, 0, 0] = x
                    histories[i, 0, 1] = y
                except:
                    histories[i, 1:] = histories[i, :4]
                    histories[i, 0, 0] = np.nan
                    histories[i, 0, 1] = np.nan

                # plot centroids and history lines
                for h in range(5):
                    center_coordinates = (histories[i, h, 0], histories[i, h, 1])
                    # if i==0:

                    # check centre points
                    idx = np.where(np.isnan(center_coordinates) == True)[0]
                    if idx.shape[0] == 0:
                        radius = int(dot_size * (1 - h / 5 / 2))
                        frame = cv2.circle(frame,
                                           center_coordinates,
                                           radius,
                                           color,
                                           -1)
                    else:
                        continue

                    # draw lines
                    if h < 4:
                        start_point = center_coordinates
                        end_point = (histories[i, h + 1, 0],
                                     histories[i, h + 1, 1])

                        idx = np.where(np.isnan(end_point) == True)[0]
                        if idx.shape[0] == 0:
                            # Line thickness of 9 px
                            thickness = 3

                            # Using cv2.line() method
                            # Draw a diagonal green line with thickness of 9 px
                            frame = cv2.line(frame,
                                             start_point,
                                             end_point,
                                             color,
                                             thickness)

            video_out.write(frame)

        video_out.release()
        original_vid.release()

    def make_video_centroid_scores(self,
                                   tracks,
                                   scores_aggregate,
                                   scores_frame,
                                   fname_video,
                                   start,
                                   end,
                                   fps):

        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 128, 0),
            (255,255,255)
        ]

        #
        #print(scores.shape)

        #
        names = ['female', 'male', 'pup1', 'pup2']
        clrs = ['blue', 'red', 'cyan', 'green']

        if start is None:
            start = 0
        if end is None:
            end = tracks.shape[0]

        #
        n_history = 5

        # load and videos
        video_name = fname_video
        fname_out = (video_name[:-4] + '_' + str(start) + "_" + str(end) + '_centroid.mp4')

        # video settings
        size_vid = np.array([1280, 1024])
        dot_sizes = np.int32(np.linspace(14, 2, n_history))
        thickness = -1  # fills circles

        # load original vid
        original_vid = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')

        video_out = cv2.VideoWriter(fname_out, fourcc, fps, (size_vid[0], size_vid[1]), True)
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

        font = cv2.FONT_HERSHEY_PLAIN

        # loop over frames
        histories = np.zeros((tracks.shape[1],
                              n_history, 2), 'float32') + np.nan
        # loop over frames
        for n in trange(start, end, 1):
            ret, frame = original_vid.read()

            #
            cv2.putText(frame, str(n), (50, 50), font, 5, (255, 255, 0), 5)

            # loop over animals
            for i in range(tracks.shape[1]):

                color = colors[i]

                #
                x = tracks[n, i, 0]
                y = tracks[n, i, 1]

                try:
                    histories[i, 1:] = histories[i, :n_history - 1]
                    histories[i, 0, 0] = x
                    histories[i, 0, 1] = y
                except:
                    histories[i, 1:] = histories[i, :n_history - 1]
                    histories[i, 0, 0] = np.nan
                    histories[i, 0, 1] = np.nan

                # loop over history
                for h in range(n_history):
                    center_coordinates = (histories[i, h, 0], histories[i, h, 1])

                    # check if center exists (i.e. is not nan)
                    idx = np.where(np.isnan(center_coordinates) == True)[0]
                    if idx.shape[0] == 0:
                        center_coordinates_shifted = (histories[i, h, 0], int(histories[i, h, 1]-40))

                        # radius = int(dot_size*(1-h/n_history/2))
                        radius = dot_sizes[h]
                        frame = cv2.circle(frame,
                                           center_coordinates,
                                           radius,
                                           color,
                                           -1)

                        # also plot score value if animal ided in present frame
                        if h == 0:
                            cv2.putText(frame, str(round(scores_aggregate[n, i], 4)),
                                        center_coordinates,
                                        font, 3,
                                        colors[4], 5)
                            #print (center_coordinates)
                            cv2.putText(frame, str(round(scores_frame[n, i], 4)),
                                        center_coordinates_shifted,
                                         font, 3,
                                         color, 5)
                    else:
                        continue

                    # draw lines
                    if h < (n_history - 1):
                        start_point = center_coordinates
                        end_point = (histories[i, h + 1, 0],
                                     histories[i, h + 1, 1])

                        idx = np.where(np.isnan(end_point) == True)[0]
                        if idx.shape[0] == 0:
                            # Line thickness of 9 px
                            thickness = 3

                            # Using cv2.line() method
                            # Draw a diagonal green line with thickness of 9 px
                            frame = cv2.line(frame,
                                             start_point,
                                             end_point,
                                             color,
                                             thickness)

            video_out.write(frame)

        video_out.release()
        original_vid.release()

    def show_track_centers(self,
                           tracks,
                           start,
                           end):

        clrs = ['red', 'blue', 'cyan', 'green']
        names = ['female', 'male', 'pup1', 'pup2']

        t = np.arange(start, end, 1)

        for k in range(tracks.shape[1]):
            temp = tracks[start:end, k].sum(1)

            # plt.plot(t,temp,c=clrs[k],label=names[k])
            plt.scatter(t, temp, c=clrs[k], label=names[k])

        plt.legend(fontsize=10)
        plt.xlabel("Frames", fontsize=20)
        plt.show()

    def show_track_centers_x(self,
                           tracks,
                           start,
                           end):

        clrs = ['red', 'blue', 'cyan', 'green']
        names = ['female', 'male', 'pup1', 'pup2']

        t = np.arange(start, end, 1)

        for k in range(tracks.shape[1]):
            temp = tracks[start:end, k,0]

            # plt.plot(t,temp,c=clrs[k],label=names[k])
            plt.scatter(t, temp, c=clrs[k], label=names[k])

        plt.legend(fontsize=10)
        plt.xlabel("Frames", fontsize=20)
        plt.show()

    #
    # def show_track_distances(self,
    #                        tracks,
    #                        start,
    #                        end):
    #
    #     clrs = ['red', 'blue', 'cyan', 'green']
    #     names = ['female', 'male', 'pup1', 'pup2']
    #
    #     t = np.arange(start, end, 1)
    #
    #     idx = np.where(tracks==0)
    #     tracks[idx]=np.nan
    #
    #     for k in range(tracks.shape[1]):
    #         locs = tracks[start: end, k]
    #         dist = []
    #         for p in range(tracks.shape[1]):
    #             if p==k:
    #                 continue
    #             temp = np.sqrt((tracks[start:end, k,0]-tracks[start:end, p,0])**2+
    #                            (tracks[start:end, k,1]-tracks[start:end, p,1])**2)
    #             idx = np.where(temp==0)[0]
    #
    #             dist.append(temp)
    #
    #         dist = np.vstack(dist)
    #         print (dist.shape)
    #         dist = np.nansum(dist,axis=0)
    #         print (dist.shape)
    #
    #         idx = np.where(dist==0)[0]
    #         dist[idx]=np.nan
    #
    #         locs = np.linalg.norm(tracks[start:end, k])
    #         print ("LOCS: ", locs.shape)
    #         dist+=locs
    #
    #         # plt.plot(t,temp,c=clrs[k],label=names[k])
    #         plt.scatter(t, dist+0.1*k, c=clrs[k], label=names[k])
    #
    #     plt.legend(fontsize=10)
    #     plt.xlabel("Frames", fontsize=20)
    #     plt.show()



#
class Track():

    #
    def __init__(self, fname_slp):

        #
        self.fname_slp = fname_slp

        #
        self.slp = None

    def load_slp(self):

        self.slp = sleap.load_file(self.fname_slp)

    def slp_to_h5(self):

        fname_h5 = self.fname_slp[:-4] + ".h5"
        if self.slp is None:
            print("... slp file not loaded, loading now...")
            self.load_slp()
            print("... done loading slp")

        self.slp.export(fname_h5)

    def slp_to_npy(self):

        fname_h5 = self.fname_slp[:-4] + ".h5"
        if os.path.exists(fname_h5) == False:
            print("... h5 file missing, converting now...")
            self.slp_to_h5()
            print("... done loading h5")

        #
        hf = h5py.File(fname_h5, 'r')

        keys = hf.keys()
        group2 = hf.get('tracks')
        tracks = []
        for k in range(len(group2)):
            tracks.append(group2[k])

        tracks = np.array(tracks).transpose(3, 0, 2, 1)

        #
        fname_npy = self.fname_slp[:-4] + ".npy"
        np.save(fname_npy, tracks)

    def load_tracks(self):

        #
        fname_npy = self.fname_slp[:-4] + '.npy'
        if os.path.exists(fname_npy) == False:
            print("... npy missing, converting...")
            self.slp_to_npy()
            print("... done loading npy")

        #
        self.tracks = np.load(fname_npy)

        #
        self.tracks_centers = np.nanmean(
                                self.tracks,
                                axis=2)
        #
        self.get_track_spine_centers()

    def get_track_spine_centers(self):
        '''  This function returns single locations per animal
            with a focus on spine2, spine3, spine1 etc...
        '''
        #
        fname_spine = self.fname_slp[:-4]+"_spine.npy"
        if os.path.exists(fname_spine)==False:

            self.tracks_spine = self.tracks_centers.copy()*0 + np.nan
            ids = [6,7,5,8,4,3,2,1,0]  # centred on spine2
            #points=[nose, lefteye, righteye, leftear,rightear, spine1, spine2, spine3,spine4]
            #         0       1        2        3         4      5        6        7      8

            #
            for n in trange(self.tracks.shape[0]):
                for a in range(self.tracks.shape[1]):
                    for id_ in ids:
                        if np.isnan(self.tracks[n,a,id_,0])==False:
                            self.tracks_spine[n,a]=self.tracks[n,a,id_]

                            break
            np.save(fname_spine, self.tracks_spine)
        else:
            self.tracks_spine = np.load(fname_spine)

    def make_tracks_chunks(self):
        ''' Function finds temporally continuous tracks
            Time-continuous-tracks
             Function breaks up continuous tracks that are too far apart;
             important for when ids jump around way too much
            Loop over the tcrs and check if jumps are too high to re-break track
        '''

        # break distances that are very large over single jumps
        # join by time
        self.time_cont_tracks = []
        for a in range(self.tracks_spine.shape[1]):
            track = self.tracks_spine[:,a]
            idx = np.where(np.isnan(track)==False)[0]
            diff = idx[1:]-idx[:-1]
            idx2 = np.where(diff>1)[0]

            # make track list
            self.time_cont_tracks.append([])

            # append first track
            self.time_cont_tracks[a].append([0,idx[idx2[0]]])

            # append all other tracks
            for i in range(1,idx2.shape[0],1):
                #if (idx[idx2[i]] - idx[idx2[i-1]+1])>0:
                self.time_cont_tracks[a].append([idx[idx2[i-1]+1],
                                idx[idx2[i]]])
                #else:
                #    print ("short: ", idx[idx2[i]] - idx[idx2[i-1]+1])

        # break by space
        thresh_dist = 100
        self.tracks_chunks = []
        for a in range(len(self.time_cont_tracks)):
            self.tracks_chunks.append([])
            while len(self.time_cont_tracks[a])>0:  #for k in range(len(self.tcrs[a])):
                times = self.time_cont_tracks[a][0]
                locs = self.tracks_spine[times[0]:times[1]+1, a]  # be careful to add +1
                dists = np.sqrt((locs[1:,0]-locs[:-1,0])**2+
                                (locs[1:,1]-locs[:-1,1])**2)
                idx = np.where(dists>=thresh_dist)[0]
                t = np.arange(times[0],times[1],1)
                if idx.shape[0]>0:
                    #if t[idx[0]]- t[0]>1:
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


        # also make a similar array to tracks_spine that contains the mean confidence
        self.tracks_scores_mean = np.zeros((self.tracks_spine.shape[0],
                                                 self.tracks_spine.shape[1]),
                                                'float32')+np.nan
        for animal_id in range(len(self.tracks_chunks)):
            for c in range(len(self.tracks_chunks[animal_id])):
                chunk = self.tracks_chunks[animal_id][c]
                mean = self.scores[chunk[0]:chunk[1]+1,animal_id].mean(0)
                self.tracks_scores_mean[chunk[0]:chunk[1]+1,animal_id]= mean

    def del_single_chunks(self, min_chunk_len=2):

        for a in range(len(self.tracks_chunks)):
            chunks = np.array(self.tracks_chunks[a])

            #
            print (chunks.shape)
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
            print (len(self.tracks_chunks[a]))




    def get_scores(self):

        #
        fname_scores = self.fname_slp[:-4] + "_scores.npy"

        if os.path.exists(fname_scores) == False:
            print("... slp file loading...")
            self.load_slp()

            tracks = ['female', 'male', 'pup shaved', 'pup unshaved']
            self.scores = np.zeros((len(self.slp), 4), 'float32') + np.nan
            for n in trange(len(self.slp)):
                for a in range(len(self.slp[n])):
                    name = self.slp[n][a].track.name
                    idx = tracks.index(name)
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





# Simplified motion predictor without variance/bayes rule updates
# as our measurements are NOT noisy (a simplified Kalman filter)

class Predict():

    def __init__(self, tracks, verbose):

        #
        self.tracks = tracks
        print("input into Predict: ", self.tracks.shape)

        #
        self.tracks_fixed = np.zeros(self.tracks.shape, 'float32')

        #
        self.verbose = verbose

    def get_positions(self):

        '''     ###############################################
                ###### GET ANIMAL POSITIONS AT TIME T #########
                ###############################################
                # get x,y positions of each animal
                # for most recent 3 values
                # eventually, add uncertainty based on how far back we go; BUT COMPLICATED
        '''

        # make array to hold history [n_animals, n_time_points, 2]
        # Problem: we don't always have data in previous time steps
        # note, the 3 time points were required for the ballistic model containing velocity
        # and acceleration; memory only based model does not require this information
        n_time_points = 3
        self.pos = np.zeros((self.tracks.shape[1], n_time_points, 2), 'float32')
        for a in range(self.tracks.shape[1]):

            # assign historical values
            self.pos[a] = self.tracks[self.time - 3:self.time, a]

            # temporarily replace read positions that are nans with most recent non-nan val
            idx = np.where(np.isnan(self.pos[a]))[0]
            for id_ in idx:
                idx2 = np.where(np.isnan(self.tracks_fixed[:self.time, a]) == False)[0]
                self.pos[a] = self.tracks_fixed[idx2[-1], a]
        #             #
        #             idx = np.where(np.isnan(self.pos[a]))[0]
        #             if idx.shape[0]>0:
        #                 print ('found nan data', a, self.pos[a])
        #                 self.pos[a,idx]==self.pos[a,2]

        if self.verbose:
            print("self.positions; ", self.pos)

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
        if self.dynamics_off_flag == False:

            for a in range(self.tracks.shape[1]):
                vel = self.pos[a, 2] - self.pos[a, 1]
                acc = (self.pos[a, 2] - self.pos[a, 1]) - \
                      (self.pos[a, 1] - self.pos[a, 0])
                self.vel[a] = vel
                self.acc[a] = acc

            # set all vel and acc nans to zeros
            idx = np.where(np.isnan(self.vel))
            self.vel[idx] = 0
            idx = np.where(np.isnan(self.acc))
            self.acc[idx] = 0

        # use dampening on acceleration vector - animals can't reach inifite velocities
        # self.acc[:]=0

        if self.verbose:
            print("vel; ", self.vel)
            print("accel; ", self.acc)

    def compute_position(self,
                         p0,
                         vel0,
                         acc0):

        # for now assume dt==1 always
        # return p0 + vel0*dt + 0.5*acc0*(dt**2)

        return p0 + vel0 + 0.5 * acc0

    def predict_location_single_frame(self):

        '''################################################################
        ###### PREDICT LOCATION OF EACH ANIMAL USING SINGLE FRAMES ########
        ###################################################################
        # predict location at the next time step
        # may also wish to run prediction backwards in time as well
        '''

        self.pos_pred = np.zeros((self.tracks.shape[1], 2), 'float32')

        for a in range(self.tracks.shape[1]):
            self.pos_pred[a] = self.compute_position(self.pos[a, 2],
                                                     self.vel[a],
                                                     self.acc[a])
        if self.verbose:
            print("self.pos_predicted: ", self.pos_pred)

    def compute_distances(self):

        '''#######################################################
        ### DISTANCE MATRIX BETWEEN OBSERVATIONS/PREDICTIONS ###
        #######################################################
        # compute distance matrix
        '''

        # read locations at time t+1
        self.pos_read = self.tracks[self.time + 1]

        if self.verbose:
            print("self.pos_read at t+1:", self.pos_read)

        # compute pairwise dist between predicted and read
        cost = scipy.spatial.distance.cdist(self.pos_pred, self.pos_read)
        if self.verbose:
            print("Cost: ", cost)

        # NEED TO SPEED THIS UP
        idx = np.where(np.isnan(cost))
        cost[idx] = 1E4

        #
        _, assignment = linear_sum_assignment(cost)
        self.assignment = assignment

        #
        if self.verbose:
            print("assignemnt: ", assignment)
            print("costs: ", cost[_, assignment])

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
            # print (self.pos_read[self.assignment[a]])
            temp1 = self.pos_read[self.assignment[a]]
            if np.isnan(temp1[0]):
                idx2 = np.where(np.isnan(self.tracks_fixed[:self.time, a]) == False)[0]
                temp1 = self.tracks_fixed[idx2[-1], a]

            # shift 1 cell over and grab new val
            # print (self.pos[a,1:3].shape, temp1.shape)
            self.pos[a] = np.vstack((self.pos[a, 1:3], temp1))

            # save data
            self.tracks_fixed[self.time, a] = self.pos_read[self.assignment[a]]

        if self.verbose:
            print("new hugnarian positoins: ", self.pos)

    def match_chunk_forward_backward(self):

        pass

