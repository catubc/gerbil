import numpy as np
from cv2 import imread, remap, imwrite, matchTemplate, circle, line, putText, resize, imshow, waitKey, INTER_CUBIC, IMREAD_GRAYSCALE, TM_CCOEFF_NORMED, FONT_HERSHEY_DUPLEX
from math import sqrt, atan2, pi
from scipy.optimize import curve_fit
from glob import glob
from os.path import split, join
from tqdm import tqdm, trange
import os
import cv2

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    
    for k in trange(1E6):
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            break
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#06d.jpg" % (count+1), frame)
        count = count + 1


    cap.release()
     


def make_dirs(video_name):

    dir_frames = video_name.replace('.mp4','/')
    dir_fixed_frames = video_name.replace('.mp4','_corrected/')

    try:
        os.mkdir(dir_frames)
    except:
        pass

    # 
    try:
        os.mkdir(dir_frames)
    except:
        pass

    return dir_frames, dir_fixed_frames
