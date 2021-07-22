import numpy as np
from cv2 import imread, remap, imwrite, matchTemplate, circle, line, putText, resize, imshow, waitKey, INTER_CUBIC, IMREAD_GRAYSCALE, TM_CCOEFF_NORMED, FONT_HERSHEY_DUPLEX
from math import sqrt, atan2, pi
from scipy.optimize import curve_fit
from glob import glob
from os.path import split, join
from tqdm import tqdm, trange

def defish(img_names,
           map_x, 
           map_y,
           destination_folder):

    # 
    for fn in img_names:
        
        orig = imread(fn)
        corrected = remap(orig, 
                          map_x, 
                          map_y, 
                          INTER_CUBIC)
                
        imwrite(join(destination_folder, 
                     split(fn)[-1]), 
                 corrected)  
    
        
