import deeplabcut
import sys

args = str(sys.argv)

config = args[0] #'/media/cat/4TBSSD/yuki_lever-ariadna-2020-07-21/IJ2/config.yaml'

videos_list = args[1] 
videos = np.loadtxt(videos_list,dtype='str')
print ("Processing videos: ", videos)


#####################################################
#####################################################
#####################################################
deeplabcut.analyze_videos(config,
                          videos,
                          videotype="avi")
