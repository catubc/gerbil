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


#
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
            (0, 128, 0),
            (128,128, 128),
            (128, 0, 128),
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
                            fname_video_out,
                            start,
                            end,
                            fps,
                            shrink):
        #
        print ("input tracks size: ", tracks.shape)
        
        
        #
        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 255),

            (0, 255, 255),
            (0, 128, 255),
            (0, 128, 128),
            (128, 0, 128),
            (128, 128, 0),


        ]

        #
        names = ['female', 'male', 'pup1', 'pup2', 'pup3','pup4']
        clrs = ['blue', 'red', 'cyan', 'green', 'pink','brown']

        if start is None:
            start = 0
        if end is None:
            end = tracks.shape[0]

        # load and videos
        video_name = fname_video
        fname_out = (video_name[:-4] + '_' + str(start) + "_" + str(end) + '_'+ fname_video_out+'.mp4')
        if False:  # os.path.exists(fname_out):
            print("fname: exists", fname_out)
            return
            
        # rescale tracks
        tracks= tracks/shrink

        # load original vid
        original_vid = cv2.VideoCapture(video_name)
        
        width = original_vid.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT )
        fps_out =  fps

        print ("width, heigh: ", width,height)
        # video settings
        size_vid = np.int32(np.array([width, height])/shrink)
        dot_size = int(12/shrink)
        thickness = -1
        #print ("size vid: ", size_vid)
        
        # make new video
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        video_out = cv2.VideoWriter(fname_out, fourcc, fps_out, (size_vid[0], size_vid[1]), True)
        
        # set frames to new ones
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

        font = cv2.FONT_HERSHEY_PLAIN

        # loop over frames
        history_len = self.history_len
        histories = np.zeros((tracks.shape[1],
                              history_len, 2), 'float32') + np.nan
        print("Histories: ", histories.shape)

        #
        for n in trange(start, end, 1):

            # for n in trange(start, 100, 1):
            ret, frame = original_vid.read()

            # scale frame
            #print ("Frame: ", frame.shape, type(frame[0][0][0]))
            #frame = frame[::shrink, ::shrink]
            #print ("Frame: ", frame.shape, type(frame[0][0][0]))
            shrink_ratio = 1/shrink
            frame = cv2.resize(frame, # original image
                       (0,0), # set fx and fy, not the final size
                       fx=shrink_ratio,
                       fy=shrink_ratio,
                       interpolation=cv2.INTER_NEAREST)

            # print frame #
            cv2.putText(frame,
                        str(n),
                        (int(50/shrink), int(150/shrink)),
                        font,
                        int(10/shrink),
                        (255, 255, 0),
                        5)
            #
            # cv2.putText(frame, fname_video_out,
            #             (int(500/shrink), int(50/shrink)),
            #             font,
            #             int(5/shrink),
            #             (255, 255, 0),
            #             5)

            #
            for i in range(tracks.shape[1]):
                color = colors[i]  # (255, 0, 0)

                # for f in range(tracks.shape[2]):
                x = tracks[n, i, 0]
                y = tracks[n, i, 1]

                try:
                    histories[i, 1:] = histories[i, :history_len-1]
                    histories[i, 0, 0] = x
                    histories[i, 0, 1] = y
                except:
                    #print ("history error...")
                    histories[i, 1:] = histories[i, :history_len-1]
                    histories[i, 0, 0] = np.nan
                    histories[i, 0, 1] = np.nan

                # plot centroids and history lines
                for h in range(history_len):
                    
                    # plot centroids
                    try:
                        center_coordinates = (int(histories[i, h, 0]), int(histories[i, h, 1]))
                    except:
                        continue
                    # if i==0:

                    # check centre points
                    idx = np.where(np.isnan(center_coordinates) == True)[0]
                    if idx.shape[0] == 0:
                        radius = int(dot_size * (1 - h / history_len / 2))
                        frame = cv2.circle(frame,
                                           center_coordinates,
                                           radius,
                                           color,
                                           thickness)
                    else:
                        continue

                    # plot lines connecting centroids
                    if h < (history_len-1):
                        start_point = center_coordinates
                        try:
                            end_point = (int(histories[i, h + 1, 0]),
                                     int(histories[i, h + 1, 1]))
                        except:
                            continue
                            
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
        
    def make_video_centroid_velocity(self,
                            tracks,
                            fname_video,
                            fname_video_out,
                            start,
                            end,
                            fps,
                            shrink):
        #
        print ("input tracks size: ", tracks.shape)
        
        # calculate the velocities for each animal in each frame
        diffs = np.diff(tracks, axis=0)
        squared_diffs = diffs ** 2
        velocities = np.sqrt(np.sum(squared_diffs, axis=-1))  # velocities is (n_frames-1, n_animals)
        
        # apply box filter
        def apply_box_filter(velocity, box_size):
            box_filter = np.ones(box_size)/box_size
            filtered_velocity = np.convolve(velocity, box_filter, mode='same')
            return filtered_velocity
        
        # width of the filter (number of points for the moving average)
        filter_width = 100

        # apply the filter to the signal
        filtered_velocities = np.zeros_like(velocities)
        for animal in range(2):
            for dim in range(2):
                filtered_velocities[:, animal] = apply_box_filter(velocities[:, animal], filter_width)
                
        from bottleneck import nanmedian  # This function can ignore np.nan values

        # Assuming velocities is your array
        N, num_animals = velocities.shape  # This gives you 28802, 2
        window_size = 10

        # Initialize an output array of the same size
        filtered_velocities = np.zeros_like(velocities)

        # Iterate over the second dimension (i.e., number of animals)
        for i in range(num_animals):
            # Apply the moving median to this slice of the velocities
            filtered_velocities[:, i] = np.array([nanmedian(velocities[max(0, k-window_size):min(N, k+window_size), i]) for k in range(N)])


        #
        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 255),

            (0, 255, 255),
            (0, 128, 255),
            (0, 128, 128),
            (128, 0, 128),
            (128, 128, 0),


        ]

        #
        names = ['female', 'male', 'pup1', 'pup2', 'pup3','pup4']
        clrs = ['blue', 'red', 'cyan', 'green', 'pink','brown']

        if start is None:
            start = 0
        if end is None:
            end = tracks.shape[0]

        # load and videos
        video_name = fname_video
        fname_out = (video_name[:-4] + '_' + str(start) + "_" + str(end) + '_'+ fname_video_out+'.mp4')
        if False:  # os.path.exists(fname_out):
            print("fname: exists", fname_out)
            return
            
        # rescale tracks
        tracks= tracks/shrink

        # load original vid
        original_vid = cv2.VideoCapture(video_name)
        
        width = original_vid.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT )
        fps_out =  fps

        print ("width, heigh: ", width,height)
        # video settings
        size_vid = np.int32(np.array([width, height])/shrink)
        dot_size = int(12/shrink)
        thickness = -1
        #print ("size vid: ", size_vid)
        
        # make new video
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        video_out = cv2.VideoWriter(fname_out, fourcc, fps_out, (size_vid[0], size_vid[1]), True)
        
        # set frames to new ones
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

        font = cv2.FONT_HERSHEY_PLAIN

        # loop over frames
        history_len = self.history_len
        histories = np.zeros((tracks.shape[1],
                              history_len, 2), 'float32') + np.nan
        print("Histories: ", histories.shape)

        #
        for n in trange(start, end-1, 1):

            # for n in trange(start, 100, 1):
            ret, frame = original_vid.read()

            # scale frame
            #print ("Frame: ", frame.shape, type(frame[0][0][0]))
            #frame = frame[::shrink, ::shrink]
            #print ("Frame: ", frame.shape, type(frame[0][0][0]))
            shrink_ratio = 1/shrink
            frame = cv2.resize(frame, # original image
                       (0,0), # set fx and fy, not the final size
                       fx=shrink_ratio,
                       fy=shrink_ratio,
                       interpolation=cv2.INTER_NEAREST)

            # print frame #
            cv2.putText(frame,
                        str(n),
                        (int(50/shrink), int(150/shrink)),
                        font,
                        int(10/shrink),
                        (255, 255, 0),
                        5)

            # print velocities for each animal
            if n > start:  # can't calculate velocity for the first frame
                for i in range(tracks.shape[1]):
                    #velocity_text = "{:.2f}".format(velocities[n-1, i])

                    # adjust the y-coordinate for each animal to avoid overlap
                    #y_coordinate = int((50 * (i+1))/shrink)  # adjust vertical position for each animal

                    # adjust x-coordinate to move text to the top right corner
                    #x_coordinate = int(width/shrink - 200) 


                    #color = (0, 0, 255) if i == 0 else (255, 0, 0)
                    #cv2.putText(frame,
                                #velocity_text,
                                #(x_coordinate, y_coordinate),
                                #font,
                                #int(2/shrink),  
                               # color,
                                #2)
                                
                    # get the FILTERED velocities of each animal
                    vel_animal1 = filtered_velocities[n, 0]
                    vel_animal2 = filtered_velocities[n, 1]
                    
                    # get the NON-FILTERED velocities of each animal
                    # vel_animal1 = velocities[n, 0]
                    # vel_animal2 = velocities[n, 1]
                    
                    # Prepare the texts
                    vel1_text = str(round(vel_animal1, 2)) if vel_animal1 > 5 else ""
                    vel2_text = str(round(vel_animal2, 2)) if vel_animal2 > 5 else ""
                    # vel1_text = str(round(vel_animal1, 2))
                    # vel2_text = str(round(vel_animal2, 2))

                    # Display the velocities at the bottom left corner
                    cv2.putText(frame, vel1_text, (int(50/shrink), int(height-110/shrink)), font, int(5/shrink), (0, 0, 255), 3)
                    cv2.putText(frame, vel2_text, (int(50/shrink), int(height-50/shrink)), font, int(5/shrink), (255, 0, 0), 3)

            #
            for i in range(tracks.shape[1]):
                color = colors[i]  # (255, 0, 0)

                # for f in range(tracks.shape[2]):
                x = tracks[n, i, 0]
                y = tracks[n, i, 1]

                try:
                    histories[i, 1:] = histories[i, :history_len-1]
                    histories[i, 0, 0] = x
                    histories[i, 0, 1] = y
                except:
                    #print ("history error...")
                    histories[i, 1:] = histories[i, :history_len-1]
                    histories[i, 0, 0] = np.nan
                    histories[i, 0, 1] = np.nan

                for ii in range(history_len):
                    try:
                        cv2.circle(frame,
                                   (np.int(histories[i, ii, 0]),
                                    np.int(histories[i, ii, 1])),
                                   dot_size,
                                   color,
                                   thickness)
                    except:
                        pass

            # write
            video_out.write(np.uint8(frame))

        #
        video_out.release()


    #
    def make_video_pre_and_post(self,
                                tracks1,
                                tracks2,
                                fname_video,
                                fname_video_out,
                                start,
                                end,
                                fps):
        #
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
            end = tracks1.shape[0]

        # load and videos
        video_name = fname_video
        fname_out = (video_name[:-4] + '_' + str(start) + "_" + str(end) + '_'+ fname_video_out+'.mp4')
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

        #
        video_out = cv2.VideoWriter(fname_out,
                                    fourcc,
                                    fps,
                                    (size_vid[0]*2, size_vid[1]),
                                    True)

        #
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

        #
        font = cv2.FONT_HERSHEY_PLAIN

        # loop over frames
        histories1 = np.zeros((tracks1.shape[1],
                              5, 2), 'float32') + np.nan
        histories2 = np.zeros((tracks1.shape[1],
                              5, 2), 'float32') + np.nan

        #
        for n in trange(start, end, 1):

            # for n in trange(start, 100, 1):
            ret, frame_original = original_vid.read()

            frame_out = np.hstack((frame_original,frame_original))#.transpose(1,0,2)

            #
            cv2.putText(frame_out, str(n), (50, 50), font, 5, (255, 255, 0), 5)
            cv2.putText(frame_out, 'original', (500, 50), font, 5, (255, 255, 0), 5)
            cv2.putText(frame_out, 'post-fix', (500+1280, 50), font, 5, (255, 255, 0), 5)

            # make first vid
            offset = 0
            frame_out, histories1 = self.draw_locs(n,
                                       frame_out,
                                       dot_size,
                                       tracks1,
                                       colors,
                                       histories1,
                                       offset,
                                       )
            offset = 1280
            frame_out, histories2 = self.draw_locs(n,
                                       frame_out,
                                       dot_size,
                                       tracks2,
                                       colors,
                                       histories2,
                                       offset,
                                       )

            video_out.write(frame_out)

        video_out.release()
        original_vid.release()



    #
    def make_annotated_video_single_track(self,
                                        tracks1,
                                        fname_video,
                                        fname_video_out,
                                        tag_loc,
                                        start=None,
                                        end=None,
                                        scale = 1,
                                        fps=25):

        #
        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 128, 0)
        ]

        # select start/end of track
        if start is None:
            start = 0
        if end is None:
            end = tracks1.shape[0]

        # load and videos
        video_name = fname_video
        fname_out = (video_name[:-4] + '_' + str(start) + "_" + str(end) + '_'+ fname_video_out+'.mp4')

        # load original vid
        original_vid = cv2.VideoCapture(video_name)

        # video out settings
        width = int(original_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print ("width, height: ", width, height)
        size_vid = np.array([width, height])

        #
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        video_out = cv2.VideoWriter(fname_out,
                                    fourcc,
                                    fps,
                                    (width//scale, height//scale),
                                    True)

        #
        original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

        #
        font = cv2.FONT_HERSHEY_PLAIN

        # loop over frames
        for n in trange(start, end, 1):

            # for n in trange(start, 100, 1):
            ret, frame_original = original_vid.read()

            #
            if np.isnan(tracks1[n])==False:
                cv2.putText(frame_original,
                            "000000000",
                            #u'\u9632',
                            (tag_loc[0], tag_loc[1]),
                            font,
                            5,
                            (170, 255, 0),
                            5)

            frame_original = frame_original[::scale,::scale,:]

            video_out.write(frame_original)

        video_out.release()
        original_vid.release()



    #
    def draw_locs(self,
                  n,
                  frame_out,
                  dot_size,
                  tracks,
                  colors,
                  histories,
                  offset,
                  ):

        for i in range(tracks.shape[1]):
            color = colors[i]  # (255, 0, 0)

            # for f in range(tracks.shape[2]):
            x = tracks[n, i, 0]+offset
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
                center_coordinates = (histories[i, h, 0],
                                      histories[i, h, 1])

                # check centre points
                idx = np.where(np.isnan(center_coordinates) == True)[0]
                if idx.shape[0] == 0:
                    radius = int(dot_size * (1 - h / 5 / 2))
                    frame_out = cv2.circle(frame_out,
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
                        frame_out = cv2.line(frame_out,
                                             start_point,
                                             end_point,
                                             color,
                                             thickness)

        return frame_out, histories

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
            cv2.putText(frame, str(n), (50, 50), font, 3, (255, 255, 0), 5)

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
