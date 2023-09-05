import numpy as np
import os
from tqdm import trange
import parmap
import glob

from tqdm import tqdm
import sleap
import parmap

#
import sys
sys.path.append("/home/cat/code/gerbil/") # go to parent dir

from utils.track import track
from utils.convert import convert
from utils.ethogram import ethogram

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#
class CohortProcessor():

    def __init__(self):

        self.cohort_start_date = None
        self.cohort_end_date = None

        self.current_session = None

        self.list_methods()

    def list_methods(self):
        method_list = [func for func in dir(self)
                       if callable(getattr(self, func)) and
                       not func.startswith("__")]

        print ("Available methods: ",
               *method_list,
               sep='\n  ')


    def list_recordings(self):

        fnames = os.path
        pass

    def compress_video(self):
        pass

    def load_recording(self):

        pass

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


    #def load
