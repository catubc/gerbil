import numpy as np
import os
from tqdm import trange
import parmap
import glob

from tqdm import tqdm
import scipy
import parmap
from itertools import combinations
import statistics
#
# TODO: bad solution
import sys
sys.path.append("/home/cat/code/gerbil/utils") # go to parent dir

from track import track
from convert import convert
from ethogram import ethogram

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd

#
class CohortProcessor():

    #
    def __init__(self, fname_spreadsheet):

        self.cohort_start_date = None
        self.cohort_end_date = None

        self.current_session = None

        self.fname_spreadsheet = fname_spreadsheet

        self.root_dir = os.path.split(self.fname_spreadsheet)[0]

        #self.list_methods()

    #
   

    #
    def remove_huddles_from_feature_tracks(self):

        #
        #self.root_dir_features = os.path.split(self.fname_spreadsheet)[0],
                                              #          'huddles')

        #
        fnames_slp_huddle = []
        fnames_slp_features = []
        found = 0
        for k in range(self.fnames_slp.shape[0]):
            fname_huddle = os.path.join(self.root_dir,
                                 'huddles',
							     self.fnames_slp[k][0]).replace('.mp4','_'+self.NN_type[k][0])+"_huddle_spine_fixed_interpolated.npy"

            fname_features = os.path.join(self.root_dir,
                                 'features',
							     self.fnames_slp[k][0]).replace('.mp4','_'+self.NN_type[k][0])+"_spine.npy"

            #
            if os.path.exists(fname_features) and os.path.exists(fname_features):
                fnames_slp_huddle.append(fname_huddle)
                fnames_slp_features.append(fname_features)
                found+=1
            else:
                pass
                #print ("missing file: ", self.fnames_slp[k][0])

        print ("# file pairs found: ", found, " (if less  than above, please check missing)")

        #
        fnames_all = list(zip(fnames_slp_features,fnames_slp_huddle))

        if self.parallel:
            parmap.map(self.remove_huddles,
                       fnames_all,
                       pm_processes=self.n_cores,
                       pm_pbar = True)
        else:
            for fnames in tqdm(fnames_all):
                self.remove_huddles(fnames)

    #
    def remove_huddles(self, fnames):

        fname_features, fname_huddles = fnames[0], fnames[1]

        #
        huddles = np.load(fname_huddles)
        features = np.load(fname_features)

        #
        for k in range(huddles.shape[0]):

            h_locs = huddles[k]
            
            for h_loc in h_locs:
                
                if np.isnan(h_loc[0]):
                    continue
                #print (h_loc)

                f_loc = features[k]
                #print ("h_loc: ", h_loc.shape, "  f_loc: ", f_loc.shape)

                dists = np.linalg.norm(f_loc-h_loc,axis=1)
                #print ("dists: ", dists)

                # set
                idx = np.where(dists<=self.huddle_min_distance)
                #print ("idx: ", idx)
                features[k,idx]=np.nan

        fname_out = fname_features.replace('.npy','_nohuddle.npy')
        np.save(fname_out, features)

    #
    def preprocess_huddle_tracks(self):

        #
        self.root_dir_features = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                              'huddles')

        #
        fnames_slp = []
        for k in range(self.fnames_slp.shape[0]):
            fname = os.path.join(self.root_dir_features,
							     self.fnames_slp[k][0]).replace('.mp4','_'+self.NN_type[k][0])+"_huddle.slp"
            #
            if os.path.exists(fname):
                fnames_slp.append(fname)

        #
        print ("# of files: ", len(fname))
        if self.parallel:
            parmap.map(process_huddle_track,
                       fnames_slp,
                       self.fix_track_flag,
                       self.interpolate_flag,
                       pm_processes=self.n_cores,
                       pm_pbar = True)
        else:
            for fname_slp in tqdm(fnames_slp):
                process_huddle_track(fname_slp,
											self.fix_track_flag,
											self.interpolate_flag,)

    #
    def load_huddle_tracks(self):

        #
        self.root_dir_features = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                              'huddles')

        # add the name extensions to the file names
        text = '_spine'
        if self.fix_track_flag:
           text = text + "_fixed"
        if self.interpolate_flag:
           text = text + "_interpolated"


        #
        fnames_slp = []
        self.tracks_huddles = []
        for k in range(self.fnames_slp.shape[0]):
            fname = os.path.join(self.root_dir_features,
							     self.fnames_slp[k][0]).replace('.mp4','_'+self.NN_type[k][0])+"_huddle.slp"
            #
            if os.path.exists(fname):
                #
                temp = np.load(os.path.join(fname[:-4]+text+".npy"))
                self.tracks_huddles.append(temp)
        #


        #
        print (" # of files: ", len(self.tracks_huddles))






    #
    def compute_huddle_composition(self):

        #
        if True:
            self.huddle_comps_binned = parmap.map(compute_huddle_parallel,
											   self.tracks_features,
											   #self.tracks_features[:50],
											   self.median_filter_width,
											   self.n_frames_per_bin,
											   pm_processes = self.n_cores,
											   pm_pbar = True
											   )
        else:
            self.huddle_comps_binned = []
            for s in trange(len(self.tracks_features)):
                session = self.tracks_features[s]

                huddle_comp_binned = compute_huddle_parallel(session,
                                                        self.median_filter_width,
                                                        self.n_frames_per_bin,
                                                        )

                self.huddle_comps_binned.append(huddle_comp_binned)


    #
    def load_feature_tracks(self):
        #
        self.root_dir_features = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                              'features')

        #
        self.tracks_features = []
        self.tracks_features_pdays = []
        self.tracks_features_start_times_absolute_mins = []
        self.tracks_features_start_times_absolute_sec = []
        for k in range(self.fnames_slp.shape[0]):
            fname = os.path.join(self.root_dir_features,self.fnames_slp[k][0]).replace(
                                    '.mp4','_'+self.NN_type[k][0])+".slp"
            if os.path.exists(fname):
                #
                if self.use_nohuddle:
                    self.fname_spine_saved = fname[:-4]+"_spine_nohuddle.npy"
                else:
                    self.fname_spine_saved = fname[:-4]+"_spine.npy"

                #
                self.tracks_features.append(np.load(self.fname_spine_saved))
                self.tracks_features_pdays.append(self.PDays[k])
                self.tracks_features_start_times_absolute_mins.append(self.start_times_absolute_minute[k])
                self.tracks_features_start_times_absolute_sec.append(self.start_times_absolute_sec[k])
        #
        print ("# of feature tracks: ", len(self.tracks_features))

    #
    def preprocess_feature_tracks(self):

        #
        self.root_dir_features = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                              'features')

        #
        fnames_slp = []
        for k in range(self.fnames_slp.shape[0]):
            fname = os.path.join(self.root_dir_features,self.fnames_slp[k][0]).replace(
                                    '.mp4','_'+self.NN_type[k][0])+".slp"
            if os.path.exists(fname):
                fnames_slp.append(fname)


        #
        if self.parallel:
            parmap.map(process_feature_track,
                       fnames_slp,
                       self.exclude_huddles,
                       pm_processes=self.n_cores,
                       pm_pbar = True)
        else:
            for fname_slp in tqdm(fnames_slp):
                process_feature_track(fname_slp, self.exclude_huddles)

    #
    def show_3D_plots(self):

        locs = []
        
        # find the time ranges required
        pday_starts = []
        for pd in self.pdays:
            temp = (int(pd[1:])-15)*24*60
            pday_starts.append(temp)
            
        print ("pday starts: ", pday_starts)

        #
        day_in_mins = 24*60
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((1,1,2))  # aspect ratio is 1:1:1 in data space
        for k in trange(0,1500,1):
            self.track_id = k
            track = self.load_single_feature_spines()
            # if track is missing, skip it
            if track is None:
                continue
            
            try:
                temp = track.tracks_spine[:,self.animal_id]
            except:
                print ("Error loading track: ", k)
                continue


            start_time = self.start_times_absolute_minute[k]

	        #chek to see if if the time is within in day (in mins) of the 
            good_time = False
            for pday_start in pday_starts:
                dd = start_time - pday_start
                if dd < day_in_mins and dd>=0:
                    good_time=True
                    break
                    
            # see if we're within
            if good_time==False:
                continue
            x = temp[:,0]
            y = temp[:,1]
            z = np.arange(0,x.shape[0],1)#*10
            z = start_time + z/24/60
            #print (x.shape, y.shape, z.shape)
            #if self.subsample:

            idx_split = np.arange(0,x.shape[0],self.subsample_rate)[1:]

            x = np.nanmean(np.split(x, idx_split)[:-1], axis=1)
            y = np.nanmean(np.split(y, idx_split)[:-1], axis=1)
            z = np.nanmean(np.split(z, idx_split)[:-1], axis=1)

            # remove big jumps
            min_dist = 150
            locs = np.vstack((x,y)).T
            dists = locs[1:]-locs[:-1]
            diffs = np.linalg.norm(dists, axis=1)
            idx = np.where(diffs>min_dist)[0]
            x[idx] = np.nan
            y[idx] = np.nan
            z[idx] = np.nan
            
            # remove short chunks
            inside = False
            if np.isnan(x[0])==False:
                inside=True
                start = 0
            idxs = []
            for k in range(1,x.shape[0],1):
                if np.isnan(x[k]):
                    if inside:
                        end = k
                        if (k-start)<self.min_chunk_len:
                            idxs.append(np.arange(start,end,1))
                        inside=False
                else:
                    if inside==False:
                        start = k
                        inside=True
            if len(idxs)>0:
                idxs = np.hstack(idxs)
                x[idxs] = np.nan
                y[idxs] = np.nan
                z[idxs] = np.nan
                        
            #print (x.shape)
            ax.plot(x,y,z, c= 'blue',
                    linewidth=1,
                    alpha=.2)
            #plt.plot()

            #return
        #pday_labels = 
        #p21 = 6*24*60
        #p28 = 13*24*60
        #print (p21, p28)
        xticks = pday_starts
        xticks_new = self.pdays

        ax.set_zticks(xticks)
        ax.set_zticklabels(xticks_new)
        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")
        ax.set_zlabel("Development day")
        ax.set_zlim(bottom=0)
        plt.suptitle("Animal #: " + str(self.animal_id))
        plt.show()

    def load_database(self):


        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns',504)
        pd.set_option('display.width',1000)

        df = pd.read_excel(self.fname_spreadsheet, engine='openpyxl')
        df.style.applymap(lambda x:'white-space:nowrap')
        print ("DF: ", df.head() )

        ###################################################################
        ########## SAVE FILENAMES WITH 6 ANIMALS AND NN TYPES #############
        ###################################################################
        #
        print ("Loading only recordings with 6 animals...")
        self.n_gerbils = df.loc[:,'# of Gerbils']
        #print ("# of gerbils: ", self.n_gerbils)

        #
        self.PDays = df.loc[:,'Dev Day']

        #
        self.Start_times = df.loc[:,'Start time']

        #
        idx = np.where(self.n_gerbils==6)[0]
        print (" ... total # : ", idx.shape[0], " / ", self.n_gerbils.shape[0])

        fnames = df.loc[:,'Filename']
        self.fnames_slp = np.vstack(fnames.iloc[idx].tolist())

        #
        self.NN_type = np.vstack(df.loc[:,'NN Type'].iloc[idx].tolist())

        #
        self.start_times_military = np.array(df.loc[:,'Start time'])

        # generate absolute start times for the data:
        fps = 24
        self.start_times_absolute_minute = []
        self.start_times_absolute_frame = []
        self.start_times_absolute_sec = []
        for k in range(len(self.start_times_military)):
            time = self.start_times_military[k]
            #time = [time.hour, time.minute, time.second]
            pday = int(self.PDays[k][1:])

            # get minutes
            pday_abs = pday - 15
            
            # get time in mins
            time_in_mins = time.hour*60 + time.minute
            abs_time_in_mins = pday_abs*24*60 + time_in_mins  # convert from day to minute;
            self.start_times_absolute_minute.append(abs_time_in_mins)
            
            # get time in sec
            time_in_sec = time.hour*60*60 + time.minute*60
            abs_time_in_sec = pday_abs*24*60*60 + time_in_sec  # convert from day to minute;
            self.start_times_absolute_sec.append(abs_time_in_sec)
            
            #
            time_in_frames= time.hour*60*60*fps+time.minute*60*fps +time.second*fps
            self.start_times_absolute_frame.append(time_in_frames)

    #
    def list_methods(self):
        method_list = [func for func in dir(self)
                       if callable(getattr(self, func)) and
                       not func.startswith("__")]

        print ("Available methods: ",
               *method_list,
               sep='\n  ')


    #
    def compute_rectangle_occupancy(self,
                                 track_local,
                                 lower_left,
                                 upper_right):
        #
        idx = np.where(np.isnan(track_local.sum(1))==False)[0]
        track_local2 = track_local[idx]
        idx2 = np.where(np.all(np.logical_and(track_local2>=lower_left,
                                     track_local2 <= upper_right), axis=1))[0]
        #print (track_local2.shape, idx2.shape)

        return idx2.shape[0]/track_local.shape[0]*100

    #
    def get_rectangle_occupancy(self, a1):

        #
        res=[]
        lower_left = self.rect_coords[0]
        upper_right = self.rect_coords[1]

        #
        for k in trange(0,1500,1):
            self.track_id = k
            track = self.load_single_feature_spines()
            # if track is missing, skip it
            if track is None:
                res.append(0)
                continue

            #
            try:
                #print (track.tracks_spine.shape)
                temp = self.compute_rectangle_occupancy(track.tracks_spine[:,a1],
                                                        lower_left,
                                                    upper_right)

                res.append(temp)
            except:
                print("anima;", a1, "error loading track: ", k)
                res.append(0)

        res = np.array(res)
        print ("res: ", res.shape)

        self.res = res


    #
    def compute_circle_occupancy(self,
                                 track_local,
                                 centre,
                                 radius):
        #
        xx = track_local-centre
        idx = np.where(np.isnan(xx.sum(1))==False)[0]
        dists = np.linalg.norm(xx[idx],axis=1)
        idx = np.where(dists<=radius)[0]

        return idx.shape[0]/xx.shape[0]*100

    #
    def get_circle_occupancy(self, a1):

        res=[]
        centre = self.circle_coords[0]
        radius = np.linalg.norm(self.circle_coords[0]-self.circle_coords[1])

        #
        for k in trange(0,1500,1):
            self.track_id = k
            track = self.load_single_feature_spines()
            # if track is missing, skip it
            if track is None:
                res.append(0)
                continue

            #
            temp = self.compute_circle_occupancy(track.tracks_spine[:,a1],
                                                 centre,
                                                 radius)


            res.append(temp)

        res = np.array(res)
        #print ("res: ", res.shape)

        self.res = res

    def load_ethograms(self):

        #
        print(self.animal_ids)

        #
        d = []
        for animal_id in self.animal_ids:

            #
            # fname_in = os.path.join(self.root_dir,
            #                        self.behavior_name+"_"+str(animal_id)+'.npy').replace('(','[').replace(')',']')
            fname_in = os.path.join(self.root_dir,
                                    self.behavior_name + "_" + str(animal_id) + "_excludehuddles_"
                                    + str(self.exclude_huddles) + '.npy').replace('(', '[').replace(')', ']')

            #
            try:
                temp = np.load(fname_in)
                d.append(temp)
            except:
                print("file missing: ", fname_in)

        #d = np.vstack(d)
        self.ethograms = d

    def show_developmental_trajectories(self):

        #
        len_ = self.ethograms[0].shape[0]
        self.ethograms = np.vstack(self.ethograms)

        #
        print (self.ethograms.shape)
        print ("sums: ", np.nansum(self.ethograms))

        #
        import sklearn

        idx = np.where(np.isnan(self.ethograms))
        self.ethograms[idx]=0

        #
        if self.dim_red_method == 'pca':
            pca = sklearn.decomposition.PCA(n_components=3)
            X_out = pca.fit_transform(self.ethograms)
        elif self.dim_red_method == 'tsne':
            from sklearn.manifold import TSNE
            #>> > X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
            X_out = sklearn.manifold.TSNE(n_components=2, learning_rate='auto',
            init = 'random', perplexity = 3).fit_transform(self.ethograms)

        #
        print ("X_out: ", X_out.shape)
        clrs = ['black','blue','red','green','brown','pink','magenta','lightgreen','lightblue',
                'yellow','lightseagreen','orange','grey','cyan','teal','lawngreen']

        # removes days which have zero entries
        if self.remove_zeros:
            idx = np.where(self.ethograms.sum(1)==0)
            print ("removing zeros: ", idx[0].shape)
            X_out[idx]=np.nan

        #
        plt.figure()
        #len_ = self.ethograms[0].shape[0]
        for k in range(0,X_out.shape[0],len_):

            x = X_out[k:k+len_,0]
            y = X_out[k:k+len_,1]
            sizes = np.arange(1,10+len_)*10

            #
            idx = np.where(np.isnan(x)==False)
            x = x[idx]
            y = y[idx]
            sizes = sizes[idx]
            #print (idx)

            plt.scatter(x,
                        y,
                        label=str(self.animal_ids[k//len_]),
                        s=sizes,
                        c=clrs[k//len_])

            # connect lines
            plt.plot(x,
                     y,
                        c=clrs[k//len_])

        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(self.behavior_name)
        plt.legend()
        plt.suptitle(self.dim_red_method)
        plt.show()


    #
    def get_pairwise_interaction_time(self, a1, a2):

        names = ['female','male','pup1','pup2','pup3','pup4']
        self.animals_interacting = ''+names[self.animal_ids[0]] + " " + names[self.animal_ids[1]]


        res=[]
        for k in trange(0,1500,1):
            self.track_id = k
            track = self.load_single_feature_spines()

            # if track is missing, skip it
            if track is None:
                res.append(np.zeros((6,6))[a1,a2])
                continue

            #
            self.symmetric_matrices=False
            self.plotting=False
            temp = self.compute_pairwise_interactions(track)
            try:
                temp = temp[a1,a2]
            except:
                print ("Missing animal track...")
                temp = np.zeros((6,6))[a1,a2]

            res.append(temp)

        res = np.array(res)

        self.res = res

    #
    def format_behavior(self):

        #
        self.data = []
        for k in range(self.PDays.shape[0]):

            PDay = self.PDays[k]
            time = self.Start_times[k]
            self.data.append([int(PDay[1:]), time.hour, self.res[k]])

        #
        self.data = np.vstack(self.data)
        #print (self.data)

        # compute average per hour
        self.data_ave = []
        s = []
        s.append(self.data[0,2])
        for k in range(0,self.data.shape[0]-1,1):
            if self.data[k,1]==self.data[k+1,1]:
                s.append(self.data[k+1,2])
            else:
                temp = self.data[k]
                temp[2] = np.mean(s)
                self.data_ave.append(temp)
                s=[]

        self.data = np.vstack(self.data_ave)

    #
    def list_recordings(self):



        pass

    def compress_video(self):
        pass

    #
    def load_single_feature_spines(self):


        try:
            fname = os.path.join(os.path.split(self.fname_spreadsheet)[0],
                                     'features',
                                     self.fnames_slp[self.track_id][0].replace('.mp4','_'+self.NN_type[self.track_id][0]+".slp"))
        except:
            return None


        #
        t = track.Track(fname)
        t.fname=fname
        #
        if os.path.exists(fname)==False:
            return None

        #
        t.exclude_huddles = self.exclude_huddles
        t.get_track_spine_centers()

        return t

    def process_time(self):

        ''' Function that uses filename to generate metadata

            - generates Universal Dev Time from timestamp
            - generates Day vs. Nighttime label from timestamp
            - identifies the correct NN for the file

        '''

        #
        print ("current session: ", self.current_session)

    def set_roi(self):

        #
        global circle_coords, rect_coords
        global ax1, fig

        from matplotlib import pyplot as plt, patches

        fig, ax1 = plt.subplots(figsize=(13,10))
       # #line, = ax1.plot(x, y, 'o', picker=10)
        plt.ylim(0,700)

        #
        track_local = np.load(self.fname_slp_npy)
        print (track_local.shape)

        plt.imshow(self.video_frame,
                   aspect='auto')
        plt.plot(track_local[:,0,0],
                 track_local[:,0,1])
#
        plt.title("Left button: 1- centre; 2-radius\n "+
                   "Right button: 1- bottom left; 2-top right\n " +
                   "Centre button: exit")


        circle_coords=np.zeros((2,2))
        rect_coords=np.zeros((2,2))

        #
        def click_handler(event):
            global circle_coords, rect_coords
            global ax1, fig

            if event.button == 3:
                if rect_coords[0].sum()==0:
                    rect_coords[0] = [event.xdata, event.ydata]
                else:
                    rect_coords[1] = [event.xdata, event.ydata]
                #print("left-click!", circle_coords)
                # do something

            if event.button == 2:
                plt.close(fig)
                return

            #
            if event.button == 1:
                if circle_coords[0].sum()==0:
                    circle_coords[0] = [event.xdata, event.ydata]
                else:
                    circle_coords[1] = [event.xdata, event.ydata]

            if circle_coords[1].sum()>0:
                diff =  circle_coords[0]-circle_coords[1]

                dist = np.linalg.norm(diff)
                circle1 = patches.Circle(circle_coords[0],
                                         dist,
                                         color='r',
                                         alpha=.5)
                ax1.add_patch(circle1)
                fig.canvas.draw()
                fig.canvas.flush_events()
                #

            #
            if rect_coords[1].sum()>0:
                rect_coords = np.vstack(rect_coords)

                for k in range(2):
                    plt.plot([rect_coords[k][0], rect_coords[k][0]],
                             [rect_coords[0][1], rect_coords[1][1]],
                            '--',
                            c='red')

                for k in range(2):
                    plt.plot([rect_coords[0][0], rect_coords[1][0]],
                             [rect_coords[k][1], rect_coords[k][1]],
                            '--',
                            c='red')

                fig.canvas.draw()
                fig.canvas.flush_events()

                #


        #
        fig.canvas.mpl_connect('button_press_event', click_handler)

        #
        plt.show(block=True)

        #
        self.circle_coords=circle_coords
        self.rect_coords=rect_coords


    def load_video2(self):
        
        import av
        print ("ASSUMING VIDEO IS 700 x 900... this only works for cohort2...")

        container = av.open(self.fname_video)

        frames = container.decode()
        print (frames)
        ctr=0
        for frame in frames:
              
            ctr+=1
            if ctr>1000:
                break
                
                

        self.video_frame = frame.to_ndarray()[:700]#[::-1]
      
        
    #
    def load_video(self):
        ''' Function that takes as input filename

        :return:
        '''

        import cv2

        #
        cap = cv2.VideoCapture(self.fname_video)
        
        print (self.fname_video)

        while(cap.isOpened()):
            ret, frame = cap.read()

            break

        cap.release()
        cv2.destroyAllWindows()

        self.video_frame = frame

    #
    def detect_audio(self):
        pass

    def generate_huddle_composition_ethograms(self):
	    
        #fps = 24
        # img width in binned values  hrs x mins x sec x fps
        img_width = int(24*60*60*self.video_frame_rate/self.n_frames_per_bin)
        img = np.zeros((img_width,16*6))
        img_flattened = np.zeros((img_width*16,6))
        print ("size of img: ", img.shape, " size of flatten image: ", img_flattened.shape)


        #
        # loop over every recording
        for ctr,start in enumerate(tqdm(self.tracks_features_start_times_absolute_sec)):
            
            #
            start_frames = start*self.video_frame_rate//self.n_frames_per_bin
            #start_row = start_frames//(img_width)#+15
            #start_row_flatten = start_frames #//(img_width)#+15
            #start_col = start_frames%(img_width)
            start_col_flatten = start_frames #%(img_width)
            #print (self.fnames_slp[ctr], "start frames: ", start_frames, "start row: ", start_row, " start _col: ", start_col)

            # loop over animals
            #print ("start-col: ", start_col_flatten)
            for k in range(len(self.huddle_comps_binned[0])):
                start_row_flatten = k
                #if k>2:
                #    break
                try:
                    temp = self.huddle_comps_binned[ctr][k].squeeze()
                except:
                    continue
                idx = np.where(temp==1)[0]
                temp[idx] = k+1
                len_ = min(img_flattened[start_col_flatten:start_col_flatten+temp.shape[0]].shape[0], temp.shape[0])
                #print ("len: ", len_)
                #img[start_col:start_col+len_,start_row*6+k] = temp[:len_]
                img_flattened[start_col_flatten:start_col_flatten+len_,start_row_flatten] = temp[:len_]
                
                # pad the next 10% of the data with the last 10% of the data
                if True:
                    pad_len = temp.shape[0]//3
                    #print ("pad len: ", pad_len, " len_: ", len_, "temp: ", temp.shape)
                    #try:
                    img_flattened[start_col_flatten+len_:start_col_flatten+ len_+pad_len, start_row_flatten] = temp[:len_][-pad_len:]
        # remake non-flattened images
        ctr=0
        for k in range(0, img_flattened.shape[0], img_width):
            temp = img_flattened[k:k+img_width]
            #print (ctr, int(ctr*6), int((ctr+1)*6), k, k+img_width)
            img[:,int(ctr*6):int((ctr+1)*6)] = temp
            ctr+=1
        img = np.array(img)
        #print ("img.sha: ", img.shape)

        #
        plt.figure(figsize=(10,5))
        plt.imshow(img.T[::-1],
                  aspect='auto',
                  interpolation = "None",
                  extent= [0,24,14.5,30.5],
                  cmap="Accent_r"
                  )

        #
        #yticks = np.arange(0,96,6)
        #yticks_new = yticks//6 + 15
        plt.ylabel("PDay")
        plt.xlabel("Time (hr)")
        plt.show()

    #
    def compute_pairwise_interactions(self,track):

        #
        x_ticks=['female','male','pup1','pup2','pup3','pup4']


        self.distance_threshold = 250 # # of pixels away assume 1 pixel ~= 0.5mm -> 20cm
        time_window = 1*25 # no of seconds to consider
        self.smoothing_window = 3
        min_distance = 25 # number of frames window
        fps=24

        locs = track.tracks_spine.transpose(1,0,2)
        traces_23hrs = locs

        # COMPUTE PAIRWISE INTERACTIONS
        animals=np.arange(locs.shape[0])
        interactions = np.zeros((animals.shape[0],animals.shape[0]),'int32') + np.nan
        durations_matrix = np.zeros((animals.shape[0], animals.shape[0]),'int32') + np.nan

        ########################################################
        ########################################################
        ########################################################
        # loop over all pairwise combinations
        pair_interaction_times = []
        pairs1 = list(combinations(animals,2))
        #for pair in tqdm(pairs1, desc='pairwise computation'):
        for pair in pairs1:
            traces = []

            # smooth out traces;
            for k in pair:
                traces1=traces_23hrs[k].copy()
                traces1[:,0]=np.convolve(traces_23hrs[k,:,0], np.ones((self.smoothing_window,))/self.smoothing_window, mode='same')
                traces1[:,1]=np.convolve(traces_23hrs[k,:,1], np.ones((self.smoothing_window,))/self.smoothing_window, mode='same')
                traces1 = traces1
                traces.append(traces1)

            # COMPUTE PAIRWISE DISTANCES AND NEARBY TIMES POINTS
            idx_array = []
            diffs = np.sqrt((traces[0][:,0]-traces[1][:,0])**2+
                            (traces[0][:,1]-traces[1][:,1])**2)
            idx = np.where(diffs<self.distance_threshold)[0]

            # COMPUTE TOTAL TIME TOGETHER
            #print ("Pairwise: ", pair, idx.shape)
            durations_matrix[pair[0],pair[1]]=idx.shape[0]/fps

            # COMPUTE # OF INTERACTIONS;
            diffs_idx = idx[1:]-idx[:-1]
            idx2 = np.where(diffs_idx>5)[0]
            interactions[pair[0],pair[1]]=idx2.shape[0]

            # SAVE TIMES OF INTERACTION
            pair_interaction_times.append(idx)

        # SYMMETRIZE MATRICES
        if self.symmetric_matrices:
            for k in range(durations_matrix.shape[0]):
                for j in range(durations_matrix.shape[1]):
                    if np.isnan(durations_matrix[k,j])==False:
                        durations_matrix[j,k]=durations_matrix[k,j]
                        interactions[j,k]=interactions[k,j]


        # #################################################
        # ######### PLOT INTERACTIONS PAIRWISE ############
        # #################################################
        dur_matrix_percentage = durations_matrix/(locs.shape[1]/fps)*100

        if self.plotting:
            plt.figure()
            labelsize=14
            ax2=plt.subplot(1,1,1)
            im = plt.imshow(durations_matrix, cmap='viridis')

            #x_ticks=['female','male','pup1','pup2']
            plt.xticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=15)
            plt.yticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=75)
            plt.tick_params(labelsize=labelsize)

            cbar = plt.colorbar()
            cbar.set_label("time together (sec)", fontsize=labelsize)

            ##############################################
            ############ PLOT PAIRWISE DURATIONS ########
            #################################################
            plt.figure()
            ax2=plt.subplot(1,1,1)

            dur_matrix_percentage = durations_matrix/(locs.shape[1]/fps)*100
            plt.imshow(dur_matrix_percentage, cmap='viridis')

            #x_ticks=['female','male','pup1','pup2']
            plt.xticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=15)
            plt.yticks(np.arange(locs.shape[0]),
                       x_ticks,rotation=75)
            plt.tick_params(labelsize=labelsize)

            cbar = plt.colorbar()
            cbar.set_label("time together (% of total recording)", fontsize=labelsize)
            #

            #
            plt.suptitle(os.path.split(track.fname)[1])


            plt.show()

        return dur_matrix_percentage

#

def compute_huddle_parallel(tracks_features,
                            median_filter_width,
                            n_frames_per_bin):

    #
    #fps = 24

    # # time points , # of animals
    huddle_comp = np.zeros((tracks_features.shape[0], tracks_features.shape[1]))
    idx = np.where(np.isnan(tracks_features[:,:,0]))  # set all nans to potential values in the huddles
    huddle_comp[idx] = 1  # if we can't find the track, animals probably in the huddle

    res = []
    for k in range(huddle_comp.shape[1]):

        #
        temp1 = scipy.signal.medfilt(huddle_comp[:,k], kernel_size=median_filter_width)
        huddle_comp[:,k] = temp1

        # split the data in # bins as per the variable
        idxs = np.arange(0,temp1.shape[0],n_frames_per_bin)[1:]
        temp2 = np.array(np.array_split(temp1, idxs))

        # find the mode (most common element) in each 1min bin
        temp3 = []
        for t_ in temp2:
            temp3.append(scipy.stats.mode(t_)[0])
        res.append(np.hstack(temp3))

    #
    huddle_comp_binned = np.vstack(res)
    return huddle_comp_binned


def process_feature_track(fname_slp, exclude_huddles):

    if os.path.exists(fname_slp):

        fname_spine_out = fname_slp.replace('.slp',"_spine.npy")
        if os.path.exists(fname_spine_out):
            return

        t = track.Track(fname_slp)
        t.exclude_huddles = exclude_huddles
        t.track_type = 'features'

        ###### parameters for computing body centroid #######
        t.use_dynamic_centroid = True   # True: alg. serches for the first non-nan value in this body order [2,3,1,0,4,5]
                                             # - advantage: much more robust to lost features
                                             # False: we fix the centroid to a specific body part
                                             # - advantage less jitter for some applications
        t.centroid_body_id = [2]         # if centroid flag is False; we use this body part instead

        ##### run track fixer #######
        t.fix_all_tracks()

        ##### join spatially close but temporally distant chunks #####
        if False:
            #
            t.memory_interpolate_tracks_spine()

        ##### save the fixed spines will overwrite the previous/defatul spine values####
        t.save_centroid()

#
def process_huddle_track(fname_slp,
						 fix_track_flag,
						 interpolate_flag):

	text = '_spine'
	if fix_track_flag:
	   text = text + "_fixed"
	if interpolate_flag:
	   text = text + "_interpolated"

	#
	fname_out = os.path.join(fname_slp[:-4]+text+".npy")

	#
	if os.path.exists(fname_out):
		return

	#
	t = track.Track(fname_slp)
	t.fix_track_flag = fix_track_flag
	t.interpolate_flag = interpolate_flag

	#############################################
	############# RUN TRACK FIXER ###############
	#############################################
	max_jump_allowed = 50,              # maximum distance that a gerbil can travel in 1 frame
	max_dist_to_join = 50,              # maximum distnace between 2 chunks that can safely be merged
	min_chunk_len = 25                  # shortest duration of

	t.fix_huddles(max_jump_allowed,
					  max_dist_to_join,
					  min_chunk_len)

	##################################################
	################# RUN HUDDLE FIXER ###############
	##################################################

	#
	fps = 24
	t.max_distance_huddle = 100                   # how far away we can merge huddles together (pixels)
	t.max_time_to_join_huddle = fps*120            # how far in time can we merge huddle chunks (seconds x frames)
	t.min_huddle_time = 120*fps                     # minimum huddle duration in seconds
	t.memory_interpolate_huddle()

	##################################################
	############## SAVE FIXED TRACKS #################
	##################################################

	##### save the fixed spines will overwrite the previous/defatul spine values####
	t.save_centroid()

	#
	t.save_updated_huddle_tracks(fname_out)


















