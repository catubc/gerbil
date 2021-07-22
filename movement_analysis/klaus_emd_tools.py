
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#
import numpy as np
import os
from tqdm import trange
import parmap
import glob
from sklearn.decomposition import PCA
#import umap
# import seaborn as sns
# import pandas as pd

# import pickle
# #
from tqdm import tqdm

# import sleap


import sklearn.experimental
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#
from scipy.io import loadmat
import scipy
import scipy.ndimage

#####################################################
#####################################################
#####################################################
def median_filter(data, max_gap, filter_width):

    for f in trange(data.shape[1]):
        for l in range(data.shape[2]):
            x = data[:,f,l]
            for k in range(1000):
                idx = np.where(np.isnan(x))[0]
                if idx.shape[0]==0:
                    break

                if idx[0]==0:
                    idx=idx[1:]
                x[idx] = x[idx-1]

            x = scipy.ndimage.median_filter(x, size=filter_width)
            data[:,f,l]= x

    return data

#
def get_durations(data, ctr, animal_id, min_duration = 25, plotting = False):
    clrs = ['black','green', 'magenta','brown']
    labels = ['two', 'all body', 'headnose', 'sliding']


    #
    root_dir = '/media/cat/1TB/dan/cohort1/slp/'
    d = np.load(root_dir + '/animalID_'+
                str(animal_id)+
                '_alldata_imputed.npy')

    #
    starts = []
    ends = []
    starts.append(data[0])
    for d in range(1,data.shape[0],1):
        if (data[d]-data[d-1])>1:
            ends.append(data[d-1])
            starts.append(data[d])

    #
    starts = np.array(starts)
    ends = np.array(ends)
    if starts.shape[0]==ends.shape[0]+1:
        starts=starts[:-1]

    durations = (ends - starts)+1

    dd = []
    vectors = []
    for k in range(durations.shape[0]):
        temp = durations[k]
        if temp>=min_duration:
            for p in range(0,temp-min_duration,1):
                dd.append(min_duration)

                #print (starts[k]+p, ends[k]+p)
                vectors.append([starts[k]+p, starts[k]+min_duration+p])

    vectors = np.vstack(vectors)


    #
    if plotting:
        width = 10
        bins = np.arange(0,250,width)
        y = np.histogram(durations, bins= bins)
        plt.plot(y[1][1:], y[0], label=labels[ctr], c=clrs[ctr])

        # also plot sliding window data
        y = np.histogram(dd, bins= bins)
        plt.plot(y[1][1:], y[0], label=labels[-4], c=clrs[-4])

            #
        plt.semilogy()
        plt.plot([25,25],[0,5000],'r--', label='1sec')
        plt.plot([12.5,12.5],[0,5000],'b--', label='0.5sec')
        plt.ylim(bottom=1)
        plt.title("Cohort 1, March 26, 23hrs;  animal_id " + str(animal_id))
        plt.legend()

    return durations, dd, vectors

#
def get_lengths(animal_id):

    # nose + spine1-5 data
    feature_ids = np.array([0,5,6,7,8,9])

    #
    root_dir = '/media/cat/1TB/dan/cohort1/'

    #
    try:
        d = np.load(os.path.join(root_dir, 'slp',
                             'alldata_fixed.npz'))
        d = d['frames']
        t = d['times']
    except:
        print (" FRAME IDS MISSING , UPGRADE DATAFILES ")
        d = np.load(os.path.join(root_dir, 'slp',
                     'alldata_fixed.npy'))

    #
    d = d[:,animal_id, feature_ids]
    print ("ANIMAL Specific data: ", d.shape)

    #
    fname_out = os.path.join(root_dir,"animalID_"+str(animal_id)+"lengths.npz")
    times_two = []
    times_six = []
    times_headnose = []

    #
    if os.path.exists(fname_out)==False:

        #
        two = []
        six = []

        headnose = []
        for k in trange(d.shape[0]):
            temp = d[k]
            idx1 = np.where(np.isnan(temp[:,0])==False)[0]
            idx2 = np.where(np.isnan(temp[:,1])==False)[0]

            # make sure sme missing
            if idx1.shape[0]!=idx2.shape[0]:
                continue

            #
            if idx1.shape[0]==6:
                six.append(k)
                continue

            #
            if idx1.shape[0]>=2:
                two.append(k)
                if idx1[0]==0 and idx1[1]==1:
                    headnose.append(k)

        #
        two = np.array(two)
        six = np.array(six)
        headnose = np.array(headnose)
        print ("two: ", two.shape, " six: ", six.shape)

        np.savez(fname_out,
                 two=two,
                 six=six,
                 headnose=headnose)
    else:
        d = np.load(fname_out, allow_pickle=True)
        two = d['two']
        six = d['six']
        headnose = d['headnose']

    return two, six, headnose

#
def visualize_durations():
    fig=plt.figure()
    animal_ids = np.arange(4)
    clrs = ['red','blue', 'cyan','green']
    labels = ['two', 'all body', 'headnose', 'sliding']

    two_feat_flag = False
    min_durations = np.array([25,50,125,250,500,750,1500,3000,4500,6000,7500])
    for animal_id in animal_ids:

        #
        # max_gap = 10
        # filter_width = 25
        # d = median_filter(d, max_gap, filter_width)

        #
        two, six, headnose = get_lengths(animal_id)

        res = []
        for min_duration in min_durations:
            #
            if two_feat_flag:
                _, dur_1sec, _ = get_durations(two, 0, animal_id, min_duration = min_duration, plotting = False)
            else:
                _, dur_1sec, _ = get_durations(six, 0, animal_id, min_duration = min_duration, plotting = False)

            print (animal_id, len(dur_1sec))
            res.append(len(dur_1sec))

        #
        plt.plot(min_durations/25., res,
                 color=clrs[animal_id],
                 linewidth=3, label = "animal: "+str(animal_id))

        #
        #break
    plt.ylim(bottom=1)
    #plt.xlim(0,7500/25)
    plt.xlim(0,1500/25)
    #plt.suptitle("# of segments of specific duration containing head + nose features",fontsize=20)
    plt.suptitle("# of segments of specific duration containing all six features",fontsize=20)
    #plt.semilogy()
    plt.xlabel("Duration of segment (Sec)", fontsize=20)
    plt.ylabel("# of segments", fontsize=20)
    plt.legend()
    plt.show()




#
def centre_and_align2_pairwise(data, centre_pt = 0, rotation_pt = 1):

    # centre the data on the nose
    # data[:,0] -= data[centre_pt,0]
    # data[:,1] -= data[centre_pt,1]
    #print ("data; ", data.shape)

    # align all data in window to first time point
    translation_pt = data[0,centre_pt]
    data -= translation_pt

    # get angle nose and head for first frame
    angle = -np.arctan2(*data[0,rotation_pt].T[::-1])-np.pi/2

    # get rotation for nose and head for first frame
    rotmat = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)]])

    # Apply rotation to all data in window
    data_rot = []
    for p in range(data.shape[0]):
        data_rot.append((rotmat @ data[p].T).T)

    data_rot = np.array(data_rot)

    idx = np.where(np.isnan(data_rot))
    if idx[0].shape[0]>0:
        print (data_rot)

        print ("rotation has NANS !!!!!!! : ", data)
        return None

    #
    return data_rot



#
def get_vectors(animal_id, vectors_idx, feature_ids, min_duration):

    root_dir = '/media/cat/1TB/dan/cohort1/slp/'
    # d = np.load(root_dir + '/animalID_'+
    #             str(animal_id)+
    #             '_alldata_imputed.npy')

    d = np.load(root_dir + '/animalID_'+str(animal_id)
                +'_alldata_fixed.npy')

    #
    vecs=[]
    for k in trange(vectors_idx.shape[0], desc='computing vecs'):
        temp = d[vectors_idx[k][0]:vectors_idx[k][1]][:,feature_ids]
        idx = np.where(np.isnan(temp))

        # if nose or head contain zeros skip
        if idx[0].shape[0]>0:
            continue
            # print (k, vectors_idx[k][0], vectors_idx[k][1])
            # print (temp)
            # break

        vecs.append(temp)

    vecs = np.array(vecs)

    np.save(root_dir+'/animalID_'+str(animal_id)+"_alldata_imputed_featureIDS"+str(feature_ids)+"_nonNan.npy",
            vecs)

    return vecs


#
def vectors_to_egocentric(vecs, animal_id,min_duration):

    # print ("vecs: ", vecs.shape)
    root_dir = '/media/cat/1TB/dan/cohort1/slp/'
    fname_out = root_dir + '/vecs_ego_animalID_'+str(animal_id)+"_duration_"+str(min_duration)+'.npy'
    if os.path.exists(fname_out)==False:
        vecs_ego = np.zeros(vecs.shape, 'float32')+np.nan

        #
        for s in trange(vecs.shape[0],  desc='Getting egocentric vectors', leave=True):

            vec = vecs[s]
            #print ("vec: ", vec.shape)
            # centre and align data
            vecs_ego[s] = centre_and_align2_pairwise(vec)

        np.save(fname_out, vecs_ego)
    else:
        vecs_ego = np.load(fname_out)

    return vecs_ego

def find_movements(vecs_ego):

    print ("vecs: ", vecs_ego.shape)


    min_quiet_n_frames = 2      # number of frames of static
    min_quiet_movement = 1

    #
    min_velocity = 1             # minimum pixels to move to indicate movement intiaitins


    #
    ctr_q = 0
    idx_movement = []
    s = 0
    while s<vecs_ego.shape[0]:
#    for s in trange(vecs_ego.shape[0]):

        # check if there is at least 10 frames quiet followed by high movement

        # get location of noses
        noses = vecs_ego[s,:,0]
        #print ("noses: ", noses.shape)

        # get velocities
        temp = noses[1:]-noses[:-1]
        vel = np.linalg.norm(temp,axis=1)
        #print ("vel: ", vel, vel.shape)

        # find periods with at least 10 frames of low or no movement
        if np.max(vel[:min_quiet_n_frames])<=min_quiet_movement:
            ctr_q+=1

            # require minimum movement in next frame
            if True:
                if vel[min_quiet_n_frames]>=min_velocity:
                    idx_movement.append(s)
                    #vecs_movement.append(vecs_ego[s])
            else:
                #vecs_movement.append(vecs_ego[s])
                idx_movement.apppend(s)
        s+=1


    print ("ctrq: ", ctr_q)
    idx_movement = np.array(idx_movement)

    return idx_movement


def dim_red(X_pca):
    from sklearn import decomposition
    import sklearn

    #
    print ("X_pca: ", X_pca.shape)

    #
    if False:
        pca = decomposition.PCA(n_components=3)

        X_pca = pca.fit_transform(X_pca)
        print (X_pca.shape)

    else:
        import umap
        umap = umap.UMAP(n_components=2,
                        init='random',
                        random_state=0)

        print ("fitting umap")
        #umap_ = umap.fit(vecs_pca[::10])
        umap_ = umap.fit(X_pca)

        print ("transforming alldata")
        X_pca = umap_.transform(X_pca)


    print ("plotting: ", X_pca.shape)

    return X_pca


#
def make_pca_data(vecs_movement, angles):

    vecs_nose = vecs_movement[:,:,0]
    print ("vecs_nose: ", vecs_nose.shape)

    X_pca = np.zeros((vecs_movement.shape[0], vecs_movement.shape[1],vecs_movement.shape[2]+1))

    for k in range(X_pca.shape[0]):
        X_pca[k,:,:2] = vecs_nose[k]
        X_pca[k,:,2] = angles[k]


    print (X_pca.shape)

    X_pca = X_pca.reshape(X_pca.shape[0],-1)

    return X_pca

from numpy import arccos, array
from numpy.linalg import norm

# Note: returns angle in radians
def theta(v, w):
    return arccos(v.dot(w)/(norm(v)*norm(w)))


def get_angles(vecs, animal_id, min_duration):

    root_dir = '/media/cat/1TB/dan/cohort1/slp/'
    fname_out = root_dir  + '/angles_ego_animalID_'+str(animal_id)+"_duration_"+str(min_duration)+'.npy'

    if os.path.exists(fname_out)==False:
        angles = np.zeros((vecs.shape[0],
                           vecs.shape[1]),
                           'float32')+np.nan
        #
        for f in trange(vecs.shape[0],  desc='Getting angles', leave=True):

            #
            temp1 = vecs[f,0]
            temp1 = temp1[1] - temp1[0]
            angle1 = np.angle(complex(*(temp1)))
            for m in range(0,vecs.shape[1],1):

                #
                temp2 = vecs[f,m]
                temp2 = temp2[1] - temp2[0]
                angle2 = np.angle(complex(*(temp2)), deg=True)

                angle = angle1-angle2
                angles[f,m]=angle

        np.save(fname_out, angles)
    else:
        angles = np.load(fname_out)

    return angles

# def get_angles2(vecs, vecs_ego, animal_id, min_duration):
#     import math
#     root_dir = '/media/cat/1TB/dan/cohort1/slp/'
#
#     fname_out = root_dir + '/angles_ego_animalID_'+str(animal_id)+"_duration_"+str(min_duration)+'.npy'
#
#     if os.path.exists(fname_out)==False:
#         angles = np.zeros((vecs.shape[0],
#                            vecs.shape[1]),
#                            'float32')+np.nan
#         #
#         for f in trange(vecs.shape[0],  desc='Getting angles', leave=True):
#
#             #
#             temp1 = vecs[f,0]
#             temp1 = temp1[1] - temp1[0]
#             angle1 = np.angle(complex(*(temp1)))
#             for m in range(0,vecs.shape[1],1):
#
#                 #
#                 #
#                 temp2 = vecs[f,m]
#                 temp2 = temp2[1] - temp2[0]
#
#                 angle = math.atan2(temp1[0]*temp2[1] - temp1[1]*temp2[0],
#                               temp1[0]*temp2[0] + temp1[1]*temp2[1])
#
#                 angles[f,m]=angle
#
#         np.save(fname_out, angles)
#     else:
#         angles = np.load(fname_out)
#
#
#     return angles
#


def get_acceleration_persec(vecs, animal_id, min_duration):
    root_dir = '/media/cat/1TB/dan/cohort1/slp/'

    fname_out = root_dir + '/acceleration_ego_animalID_'+str(animal_id)+"_duration_"+str(min_duration)+'.npz'

    acc_ap = np.zeros((vecs.shape[0],vecs.shape[1]-2),'float32')
    acc_ml = np.zeros((vecs.shape[0],vecs.shape[1]-2),'float32')

    if os.path.exists(fname_out)==False:
        for f in trange(vecs.shape[0], desc='Getting acceleration'):

            #
            vecs_nose = vecs[f,:,0]

            vel_ap = vecs_nose[1:,0] - vecs_nose[:-1,0]
            vel_ml = vecs_nose[1:,1] - vecs_nose[:-1,1]

            #
            aa_ap = vel_ap[1:]-vel_ap[:-1]
            aa_ml = vel_ml[1:]-vel_ml[:-1]

            #print ("aa_ap: ", aa_ap.shape)
            acc_ap[f]=aa_ap
            acc_ml[f]=aa_ml

        np.savez(fname_out,
                acc_ap=acc_ap,
                acc_ml=acc_ml)
    else:
        d = np.load(fname_out)
        acc_ap = d['acc_ap']
        acc_ml = d['acc_ml']

    acc = np.sqrt(acc_ml**2+acc_ap**2)*25

    return acc_ap, acc_ml, acc



def load_vecs_single_frame(animal_id):

    ##################################
    ##################################
    ##################################
    two, six, headnose = get_lengths(animal_id)
    print ("# of headnose locations: ", headnose.shape)
    #
    ##################################
    ##################################
    ##################################
    _,_,vectors_idx = get_durations(headnose, 0,
                                    animal_id,
                                    min_duration = min_duration,
                                    plotting = False)
    
    print ("# of segments for single frame analysis ", vectors_idx.shape)

    ##################################
    ##################################
    ##################################
    feature_ids = np.arange(2)
    vecs = get_vectors(animal_id, vectors_idx, feature_ids, min_duration)

    return vecs


def load_vecs(animal_id, min_duration):

    ##################################
    ##################################
    ##################################
    two, six, headnose = get_lengths(animal_id)
    print ("# of headnose locations: ", headnose.shape)
    #
    ##################################
    ##################################
    ##################################
    _,_,vectors_idx = get_durations(headnose, 0,
                                    animal_id,
                                    min_duration = min_duration,
                                    plotting = False)
    print ("# of durations of at least ", min_duration,
           vectors_idx.shape)
    ##################################
    ##################################
    ##################################
    feature_ids = np.arange(2)
    vecs = get_vectors(animal_id, vectors_idx, feature_ids, min_duration)

    return vecs

def generate_egocentric_and_angles_data(animal_ids, min_duration):

    vecs_mov_array = []
    #
    for animal_id in animal_ids:

        #
        ##################################
        ##################################
        ##################################
        two, six, headnose = get_lengths(animal_id)

        #
        ##################################
        ##################################
        ##################################
        _,_,vectors_idx = get_durations(headnose, 0,
                                        animal_id,
                                        min_duration = min_duration,
                                        plotting = False)

        ##################################
        ##################################
        ##################################
        feature_ids = np.arange(2)
        vecs = get_vectors(animal_id, vectors_idx, feature_ids, min_duration)

        ##################################
        ##################################
        ##################################
        vecs_ego = vectors_to_egocentric(vecs, animal_id, min_duration)

        ##################################
        ##################################
        ##################################
        angles = get_angles2(vecs, vecs_ego, animal_id, min_duration)

        ##################################
        ##################################
        ##################################
        _,_ = get_acceleration(vecs_ego,  animal_id, min_duration)


        print ("angles: ", angles.shape)

        print ('')

def plot_angle_acceleration_distributions(animal_ids, min_duration):
    fps = 25

    root_dir = '/media/cat/1TB/dan/cohort1/slp/'

    #
    vecs_mov_array = []
    for animal_id in animal_ids:
        fname = os.path.join(root_dir + '/all_continuous_'+
                             str(animal_id)+'_min_duration'+
                             str(min_duration)+'.npz')
        d = np.load(fname)
        angles = d['angles']
        acc = d['acc']

        ##################################
        ######### PLOT ANGLES ############
        ##################################
        plt.subplot(4,2,animal_id*2+1)

        print ("angles: ", angles.shape)
        width = 1
        # rad_to_degree= 57.2958
        lims = 360
        bins = np.arange(-lims,lims+width, width)
        temp = angles.flatten()# *fps*rad_to_degree
        
        y = np.histogram(temp, bins=bins)
        plt.plot(y[1][1:]-width/2, y[0],c='black')
        plt.semilogy()
        plt.ylim(bottom=1)
        plt.plot([0,0],[0,np.max(y[0])],'--')
        plt.plot([45,45],[0,np.max(y[0])],'--')
        plt.plot([-45,-45],[0,np.max(y[0])],'--')
        if animal_id==0:
            plt.title("angles (deg/sec) pdf")
        plt.ylabel("Animal "+str(animal_id))

        # ##################################
        # ######### PLOT ACC_AP ############
        # ##################################
        # plt.subplot(4,4,animal_id*4+2)
        # lims = 0.5*fps
        # width = .01*fps
        # bins = np.arange(-lims,lims+width, width)
        # temp = acc_ap.flatten()
        # y = np.histogram(temp, bins=bins)
        # plt.plot(y[1][1:]-width/2., y[0],c='blue')
        # plt.semilogy()
        # plt.ylim(bottom=1)
        # plt.plot([0,0],[0,np.max(y[0])],'--')
        # if animal_id==0:
        #     plt.title("acceleration ap (pix/sec) pdf")
        # 
        # ##################################
        # ######### PLOT ACC_ML ############
        # ##################################
        # plt.subplot(4,4,animal_id*4+3)
        # #width = 1
        # bins = np.arange(-lims,lims+width, width)
        # temp = acc_ml.flatten()
        # y = np.histogram(temp, bins=bins)
        # plt.plot(y[1][1:]-width/2, y[0],c='red')
        # plt.semilogy()
        # plt.plot([0,0],[0,np.max(y[0])],'--')
        # plt.ylim(bottom=1)
        # if animal_id==0:
        #     plt.title("acceleration ml (pix/sec) pdf")

        ##################################
        ###### PLOT ACC-OVERALL ##########
        ##################################
        plt.subplot(4,2,animal_id*2+2)
        width = 5
        temp3 = acc #*25

        #
        y = np.histogram(temp3, bins=bins)
        plt.plot(y[1][1:]-width/2, y[0],c='magenta')
        plt.semilogy()
        #plt.semilogx()
        plt.plot([0,0],[0,np.max(y[0])],'--')
        plt.plot([40,40],[0,np.max(y[0])],'--')
        # plt.plot([150,150],[0,np.max(y[0])],'--')
        # plt.plot([245,245],[0,np.max(y[0])],'--')
        plt.ylim(bottom=1)
        plt.xlim(left=-1)
        if animal_id==0:
            plt.title("abs acceleration (pix/sec) pdf")

        print ('')


def discretize_data(animal_id, min_duration,
                    angles, acc, fps,
                    rad_to_degree, angles_thresh, acc_thresh):

    root_dir = '/media/cat/1TB/dan/cohort1/slp/'

    fname_out = os.path.join(root_dir, 'all_discretized_'+str(animal_id)+
                '_min_duration'+str(min_duration)+'.npz')


    if os.path.exists(fname_out)==False:
        print ("DISCRETIZING ")
        angles_discretized = np.zeros(angles.shape, 'float32')+np.nan
        for k in trange(angles.shape[0]):
            temp = angles[k]*fps*rad_to_degree

            for a in range(len(angles_thresh)):
                idx = np.where(np.logical_and(
                                    temp>=angles_thresh[a][0],
                                    temp<angles_thresh[a][1],
                               ))[0]
                #temp[idx]=a

                angles_discretized[k,idx]= a


        # discretize accelaration
        acc_discretized = np.zeros(acc.shape, 'float32')+np.nan
        for k in trange(acc.shape[0]):
            temp = acc[k]

            for a in range(len(acc_thresh)):
                idx = np.where(np.logical_and(
                                    temp>=acc_thresh[a][0],
                                    temp<acc_thresh[a][1],
                               ))[0]

                acc_discretized[k,idx]= a


        #
        all_discretized = np.hstack((angles_discretized[:,2:], acc_discretized))

        #

        np.savez(fname_out,
                all_discretized = all_discretized,
                angles_discretized = angles_discretized,
                acc_discretized = acc_discretized)


    else:
        data = np.load(fname_out)

        all_discretized=data['all_discretized']
        angles_discretized=data['angles_discretized']
        acc_discretized=data['acc_discretized']


    return all_discretized, angles_discretized, acc_discretized

def median_filter(data, filter_width):

    for k in range(data.shape[0]):
        data[k] = scipy.ndimage.median_filter(data[k], size=filter_width)

    return data

def smooth_vecs_ego(vecs_ego, window = 5):

    for k in trange(vecs_ego.shape[2], desc='smoothing vecs_ego'):
        for p in range(vecs_ego.shape[3]):
            vecs_ego[:,:,k,p] = median_filter(vecs_ego[:,:,k,p], window)

    return vecs_ego

def smooth_angles(angles, window=5):
    print ("...smoothing angles...")
    angles = median_filter(angles, window)

    return angles


def compute_discretized_and_histograms_single_frame(animal_id):
    fps = 25
    rad_to_degree= 57.2958

    # discretized thresholds for angles
    angles_thresh = [[-1E8, -45],
                     [-45,+45],
                     [+45,1E8]
                    ]

    # discretized thresholds for acceleration
#     acc_thresh = [[0,30],
#                   [30,75],
#                   [75,150],
#                   [150,1E8]]

    acc_thresh = [[0,5],
                  [5,40],
                  [40,1E8]]

    # acc_thresh = [[0,30],
    #               [30,75],
    #               [75,1E8]]
    #
    root_dir = '/media/cat/1TB/dan/cohort1/slp/'

    fname_out = os.path.join(root_dir, 'all_continuous_'+
                             str(animal_id)+
                             '_single_frame.npz')


    if os.path.exists(fname_out)==False:

        #
        print ("agnels thresholds: ", angles_thresh)

        #
        print ("accelaration thresholds: ", acc_thresh)

        #
        ##################################
        ##################################
        ##################################
        vecs = load_vecs(animal_id, min_duration)

        ##################################
        ##################################
        ##################################
        # vecs = None                        # if this is already computed it is not required
        vecs_ego = vectors_to_egocentric(vecs, animal_id, min_duration)
        print ('vecs_ego: ', vecs_ego.shape)

        ##################################
        ##################################
        ##################################
        if False:
            vecs_ego = smooth_vecs_ego(vecs_ego, window=5)

        ##################################
        ##################################
        ##################################
        angles = get_angles3(vecs_ego, animal_id, min_duration)


        ##################################
        ##################################
        ##################################
        if True:
            angles = smooth_angles(angles)

        if True:
            angles = compute_angles_cumulative(angles)

        ##################################
        ##################################
        ##################################
        acc_ap, acc_ml, acc = get_acceleration_persec(vecs_ego,
                                                      animal_id,
                                                      min_duration)

        ##################################
        ###### MAKE CONTINOUS DATA #######
        ##################################
        all_continuous = np.hstack((angles[:,2:], acc))

        ##################################
        ######### DISCRETIZE #############
        ##################################

        (all_discretized,
         angles_discretized,
         acc_discretized) = discretize_data(animal_id,
                                            min_duration,
                                            angles,
                                            acc,
                                            fps,
                                            rad_to_degree,
                                            angles_thresh,
                                            acc_thresh)


        ##################################
        ######### MAKE HISTOGRAMS ########
        ##################################

        ang_hist, acc_hist, all_hist = get_normalized_histograms(
                                                    angles_discretized,
                                                    acc_discretized)
        #
        ang_unique, acc_unique, all_unique = get_unique_angles_accs(
                                                    ang_hist,
                                                    acc_hist,
                                                    all_hist)



        np.savez(fname_out,
                 all_continuous=all_continuous,
                 all_discretized=all_discretized,
                 all_hist=all_hist,
                 all_unique=all_unique,
                 angles_thresh=angles_thresh,
                 angles=angles,
                 angles_discretized=angles_discretized,
                 ang_unique=ang_unique,
                 ang_hist=ang_hist,
                 acc_thresh=acc_thresh,
                 acc=acc,
                 acc_ap=acc_ap,
                 acc_ml=acc_ml,
                 acc_hist=acc_hist,
                 acc_discretized=acc_discretized,
                 acc_unique=acc_unique,
                 )
    else:
        d = np.load(fname_out, allow_pickle=True)
        all_continuous=d['all_continuous']
        all_discretized=d['all_discretized']
        all_hist=d['all_hist']
        all_unique=d['all_unique']
        angles_thresh=d['angles_thresh']
        angles=d['angles']
        angles_discretized=d['angles_discretized']
        ang_unique=d['ang_unique']
        ang_hist=d['ang_hist']
        acc_thresh=d['acc_thresh']
        acc=d['acc']
        acc_hist=d['acc_hist']
        acc_discretized=d['acc_discretized']
        acc_unique=d['acc_unique']

    #
    print ("... done")
    return (all_continuous,
            all_discretized,
            all_hist,
            all_unique,
            angles_thresh,
            angles,
            angles_discretized,
            ang_unique,
            ang_hist,
            acc_thresh,
            acc,
            acc_hist,
            acc_discretized,
            acc_unique)


#
def compute_discretized_and_histograms(animal_id, min_duration):
    fps = 25
    rad_to_degree= 57.2958

    # discretized thresholds for angles
    angles_thresh = [[-1E8, -45],
                     [-45,+45],
                     [+45,1E8]
                    ]

    # discretized thresholds for acceleration
#     acc_thresh = [[0,30],
#                   [30,75],
#                   [75,150],
#                   [150,1E8]]

    acc_thresh = [[0,40],
                  [40,1E8]]

    # acc_thresh = [[0,30],
    #               [30,75],
    #               [75,1E8]]
    #
    root_dir = '/media/cat/1TB/dan/cohort1/slp/'

    fname_out = os.path.join(root_dir, 'all_continuous_'+
                             str(animal_id)+
                             '_min_duration'+str(min_duration)+
                             '.npz')


    if os.path.exists(fname_out)==False:

        #
        print ("agnels thresholds: ", angles_thresh)

        #
        print ("accelaration thresholds: ", acc_thresh)

        #
        ##################################
        ##################################
        ##################################
        vecs = load_vecs(animal_id, min_duration)

        ##################################
        ##################################
        ##################################
        # vecs = None                        # if this is already computed it is not required
        vecs_ego = vectors_to_egocentric(vecs, animal_id, min_duration)
        print ('vecs_ego: ', vecs_ego.shape)

        ##################################
        ##################################
        ##################################
        if False:
            vecs_ego = smooth_vecs_ego(vecs_ego, window=5)

        ##################################
        ##################################
        ##################################
        angles = get_angles3(vecs_ego, animal_id, min_duration)


        ##################################
        ##################################
        ##################################
        if True:
            angles = smooth_angles(angles)

        if True:
            angles = compute_angles_cumulative(angles)

        ##################################
        ##################################
        ##################################
        acc_ap, acc_ml, acc = get_acceleration_persec(vecs_ego,
                                                      animal_id,
                                                      min_duration)

        ##################################
        ###### MAKE CONTINOUS DATA #######
        ##################################
        all_continuous = np.hstack((angles[:,2:], acc))

        ##################################
        ######### DISCRETIZE #############
        ##################################

        (all_discretized,
         angles_discretized,
         acc_discretized) = discretize_data(animal_id,
                                            min_duration,
                                            angles,
                                            acc,
                                            fps,
                                            rad_to_degree,
                                            angles_thresh,
                                            acc_thresh)


        ##################################
        ######### MAKE HISTOGRAMS ########
        ##################################

        ang_hist, acc_hist, all_hist = get_normalized_histograms(
                                                    angles_discretized,
                                                    acc_discretized)
        #
        ang_unique, acc_unique, all_unique = get_unique_angles_accs(
                                                    ang_hist,
                                                    acc_hist,
                                                    all_hist)



        np.savez(fname_out,
                 all_continuous=all_continuous,
                 all_discretized=all_discretized,
                 all_hist=all_hist,
                 all_unique=all_unique,
                 angles_thresh=angles_thresh,
                 angles=angles,
                 angles_discretized=angles_discretized,
                 ang_unique=ang_unique,
                 ang_hist=ang_hist,
                 acc_thresh=acc_thresh,
                 acc=acc,
                 acc_ap=acc_ap,
                 acc_ml=acc_ml,
                 acc_hist=acc_hist,
                 acc_discretized=acc_discretized,
                 acc_unique=acc_unique,
                 )
    else:
        d = np.load(fname_out, allow_pickle=True)
        all_continuous=d['all_continuous']
        all_discretized=d['all_discretized']
        all_hist=d['all_hist']
        all_unique=d['all_unique']
        angles_thresh=d['angles_thresh']
        angles=d['angles']
        angles_discretized=d['angles_discretized']
        ang_unique=d['ang_unique']
        ang_hist=d['ang_hist']
        acc_thresh=d['acc_thresh']
        acc=d['acc']
        acc_hist=d['acc_hist']
        acc_discretized=d['acc_discretized']
        acc_unique=d['acc_unique']

    #
    print ("... done")
    return (all_continuous,
            all_discretized,
            all_hist,
            all_unique,
            angles_thresh,
            angles,
            angles_discretized,
            ang_unique,
            ang_hist,
            acc_thresh,
            acc,
            acc_hist,
            acc_discretized,
            acc_unique)

#
# def get_unique_angles_accs(angles_hist, acc_hist):
#     #
#     ang_unique = np.unique(angles_hist, axis=0)
#     print ("Unique angles: ", ang_unique.shape)
#
#     #
#     acc_unique = np.unique(acc_hist, axis=0)
#     print ("Unique accel: ", acc_unique.shape)
#
#     #
#     all_hist = np.hstack((angles_hist, acc_hist))
#     all_unique = np.unique(all_hist, axis=0)
#
#     print ("# possible combinations: ", acc_unique.shape[0]*ang_unique.shape[0])
#     print ("# of actual combinations: ", all_unique.shape)
#
#
#     print ("normalized angles histograms: ", angles_hist[50:52])
#     print ("ang unique: ", ang_unique[:2])
#     print ("acc unique: ", acc_unique[:2])
#
#     return ang_unique, acc_unique, all_unique


# CLUSTER STEP
def cluster_and_visualize(X_pca_fit,
                          X_pca,
                          n_clusters=2,
                          cluster_id = None):

    #clrs = ['black','blue','red','green', 'magenta','brown','cyan','yellow',
    #        'pink','olive']

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize


    # use GaussianMixture clusters
    if n_clusters is not None:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=n_clusters).fit(X_pca_fit)
        labels = gmm.predict(X_pca_fit)
        print(np.unique(labels))
          # PLOT STEP
        n_clusters = np.unique(labels).shape[0]
        print ("n_clusters: ", n_clusters, np.unique(labels))
        colors = plt.cm.tab20(np.linspace(0,1,20))

    # use metadata
    else:
        labels = []
#         for k in trange(X_pca.shape[0], desc='Making colors'):
#             #print (X_pca[k].shape)
#             temp = X_pca[k]
#             ave_angle = np.median(temp[:,0])
#             ave_acc = np.median(temp[:,1])

#             labels.append(ave_angle+ave_acc*3)


#         labels = np.int32(labels)
#         print ("labels: ", labels)

        #
        if True:
            medians = np.mean(X_pca, axis=1)
            print ("means: ", medians.shape)

            labels = np.log(medians[:,0]+medians[:,1]*3)
            print ("Labels log: ", labels)

            labels = (labels-np.min(labels))/(np.max(labels)-np.min(labels))
            print ("Labels norm: ", labels)

            labels = np.abs(np.log(labels+0.000001))
            print ("labels: ", labels)

            labels = (labels-np.min(labels))/(np.max(labels)-np.min(labels))
            print ("labels: ", labels)

            labels = np.around(labels, 1)
            labels = 1-labels
            print ("Final labels: ", labels)

            cmap = cm.viridis

            colors = plt.cm.viridis(np.linspace(0,1,100))

            ctr=0
            for k in np.unique(labels):
                plt.subplot(2,3,ctr+1)

                idx = np.where(labels==k)[0]
                print ("k: ", k, idx.shape)
                plt.scatter(X_pca_fit[idx,0],
                            X_pca_fit[idx,1],
                            alpha=.1,
                            edgecolor='black',
                            #color = colors[labels]
                            color = cmap(labels[idx])
                            #color = colors[labels[idx]]
                       )
                ctr+=1
                plt.xlim(-15,25)
                plt.ylim(-20,30)


            return idx, labels


        else:
            medians = np.median(X_pca, axis=1)
            labels = np.log(medians[:,0]+medians[:,1]*3)
            print ("labels: ", labels, np.unique(labels))
            labels = (labels-np.min(labels))/(np.max(labels)-np.min(labels))
            print ("labels: ", labels, np.unique(labels))
            labels = np.int32(labels*99)

            cmap = cm.tab20c
            colors = plt.cm.viridis(np.linspace(0,1,100))
            #norm = Normalize(vmin=np.min(labels), vmax=np.max(labels))

            fig=plt.figure()
            ctr=0
            for k in np.unique(labels):
                plt.subplot(2,3,ctr+1)

                idx = np.where(labels==k)[0]
                plt.scatter(X_pca_fit[idx,0],
                        X_pca_fit[idx,1],
                        alpha=.1,
                        edgecolor='black',
                        #color = colors[labels]
                        #color = cmap(labels)
                        color = colors[labels[idx]]
                       )
                ctr+=1
                plt.xlim(-15,25)
                plt.ylim(-20,30)

            return idx, labels

    #
    if cluster_id is None:
        fig=plt.figure()

        for id_ in np.unique(labels):
            idx = np.where(labels==id_)[0]
            print ("cluster: ", id_, " has # events: ", idx.shape[0])

            plt.scatter(X_pca_fit[idx,0],
                        X_pca_fit[idx,1],
                        alpha=.1,
                        edgecolor='black',
                        color = colors[labels[idx]]
                       )

    #
    print ("PLOTTING...")
    plt.show()

    return idx, labels



#


def get_normalized_histograms(angles_discretized,
                              acc_discretized):

    print ("COMPUTING Klaus normalized histograms")

    #
    bins = np.arange(np.unique(angles_discretized).shape[0]+1)-0.5
    print ("BINS angles: ", bins)
    angles_hist = np.zeros((angles_discretized.shape[0],
                             bins.shape[0]-1), 'float32')
    for k in trange(angles_discretized.shape[0]):
        angles_hist[k] = np.histogram(angles_discretized[k], bins = bins)[0]

    ## normalize
    angles_hist = angles_hist / np.sum(angles_hist[0])

    #
    bins = np.arange(np.unique(acc_discretized).shape[0]+1)-0.5
    print ("BINS accel: ", bins)
    acc_hist = np.zeros((acc_discretized.shape[0],
                             bins.shape[0]-1), 'float32')
    for k in trange(acc_discretized.shape[0]):
        acc_hist[k] = np.histogram(acc_discretized[k], bins = bins)[0]

    ## normalize
    acc_hist = acc_hist / np.sum(acc_hist[0])

    # make all hist
    all_hist = np.hstack((angles_hist, acc_hist))

    print ("Angles histograms: ", angles_hist.shape)
    print ("Accel histograms: ", acc_hist.shape)
    print ("all histograms: ", all_hist.shape)
    print ("")

    return  angles_hist, acc_hist, all_hist


def get_unique_angles_accs(angles_hist,
                           acc_hist,
                           all_hist):
    #
    ang_unique = np.unique(angles_hist, axis=0)
    print ("Unique angles: ", ang_unique.shape)

    acc_hist = np.around(acc_hist, 3)
    acc_unique = np.unique(acc_hist, axis=0)
    print ("Unique accel: ", acc_unique.shape)

    #
    all_unique = np.unique(all_hist, axis=0)

    print ("# possible combinations: ", acc_unique.shape[0]*ang_unique.shape[0])
    print ("# of actual combinations: ", all_unique.shape)

    print ("ang unique: ", ang_unique[:2])
    print ("acc unique: ", acc_unique[:2])

    return ang_unique, acc_unique, all_unique

def compute_angles_cumulative(angles):

    min_rot = 20
    fps = 25
    rad_to_degree= 57.2958

    #
    for k in trange(angles.shape[0]):

        temp = np.cumsum(angles[k]*rad_to_degree*fps)
        m = np.where(np.abs(temp)>min_rot)[0]

        #
        temp_cleaned = np.zeros(temp.shape[0])
        while m.shape[0]>0:
            # print (k, "angular accelaration / sec: ", temp, " first loc: ", m[0], " angle: ", temp[m[0]])
            #arg = np.argmax(np.abs(temp))
            #print ("max angle reached: ", temp[arg])

            temp_cleaned[m[0]] = temp[m[0]]

            # zero out all entries up to this point;
            temp[m[0]+1:] = temp[m[0]+1:]-temp[m[0]]
            temp[:m[0]+1]=0

            #
            m = np.where(np.abs(temp)>min_rot)[0]

        #
        angles[k] = temp_cleaned

    return angles

def get_angles3(vecs_ego, animal_id, min_duration):
    import math

    root_dir = '/media/cat/1TB/dan/cohort1/slp/'

    fname_out = root_dir + '/angles_ego_animalID_'+str(animal_id)+"_duration_"+str(min_duration)+'.npy'

    if os.path.exists(fname_out)==False:
        angles = np.zeros((vecs_ego.shape[0],
                           vecs_ego.shape[1]),
                           'float32') #+np.nan
        #
        for f in trange(vecs_ego.shape[0],  desc='Getting angles', leave=True):

            # grab the f chunk and t=0 (ie. first frame) xy location
            temp1 = vecs_ego[f,0]

            # grab xy differences between head and nose at t=0
            temp1 = temp1[1] - temp1[0]
            temp_prev = temp1.copy()

            # angle1 = np.angle(complex(*(temp1)))

            # loop over all times in a chunk and find angle relative the first frame
            for m in range(1,vecs_ego.shape[1],1):

                # grab frame
                temp2 = vecs_ego[f,m]

                # compute xy diff between nose and head
                temp2 = temp2[1] - temp2[0]

                # compute angle between t=0 frame and current frame
                # angle = math.atan2(temp1[0]*temp2[1] - temp1[1]*temp2[0],
                #                    temp1[0]*temp2[0] + temp1[1]*temp2[1])
                angle = math.atan2(temp_prev[0]*temp2[1] - temp_prev[1]*temp2[0],
                                   temp_prev[0]*temp2[0] + temp_prev[1]*temp2[1])
                angles[f,m]=angle

                temp_prev = temp2.copy()

        np.save(fname_out, angles)
    else:
        angles = np.load(fname_out)


    return angles


# #
# def compute_emd_normalized_unique_angle_acc(ang_unique, acc_unique):
#     emd_angle=np.zeros((ang_unique.shape[0],ang_unique.shape[0]))
#     for i in trange(ang_unique.shape[0]):
#         for p in range(i+1, ang_unique.shape[0],1):
#             emd_angle[i,p] = scipy.stats.wasserstein_distance(ang_unique[i],
#                                                               ang_unique[p])
#
#     # normalize by n-1
#     emd_angle = emd_angle / ang_unique.shape[1]
#
#     #
#     emd_acc=np.zeros((acc_unique.shape[0],acc_unique.shape[0]))
#     for i in trange(acc_unique.shape[0]):
#         for p in range(i+1, acc_unique.shape[0],1):
#             emd_acc[i,p] = scipy.stats.wasserstein_distance(acc_unique[i],
#                                                             acc_unique[p])
#
#     # normalize by n-1
#     emd_acc = emd_acc / acc_unique.shape[1]
#
#
#     return emd_angle, emd_acc


def get_index(array, feature_array):

    idx = np.sum(np.equal(feature_array, array), axis=1)
    idx = np.where(idx==len(array))[0]

    return idx


def compute_S_matrix(indexes, all_unique, ang_unique, acc_unique, n_split):

    root_dir = '/media/cat/1TB/dan/cohort1/slp/'

    fname_out = os.path.join(root_dir,"S_unique","S_unique_index"+str(indexes[0])+"_nsplit_"+str(n_split)+'.npz')

    if os.path.exists(fname_out)==False:

        S_ang = np.zeros((len(indexes),all_unique.shape[0]),'float32') #+np.nan
        S_acc = np.zeros((len(indexes),all_unique.shape[0]),'float32') #+np.nan

        #for p in trange(all_unique.shape[0]):
        for ctrp, p in enumerate(indexes):

            # find outer loop all feature match to ang unique features
            idx1_ang = get_index(all_unique[p,:3],
                                   ang_unique)
            # same for acceleration
            idx1_acc = get_index(all_unique[p,3:],
                                   acc_unique)

            # loop over inner dimensions
            for q in range(p+1, all_unique.shape[0],1):

                # get index of current frame for angle
                idx2_ang = get_index(all_unique[q,:3],
                                     ang_unique)
                #
                S_ang[ctrp,q] = emd_angle_unique_norm[idx1_ang,idx2_ang]

                # get index for acceleration
                idx2_acc = get_index(all_unique[q,3:],
                                     acc_unique)

               # print (idx1_ang,idx2_ang,idx1_acc,idx2_acc )
                #
                S_acc[ctrp,q] = emd_acc_unique_norm[idx1_acc,idx2_acc]

        # compute sum and normalize

        S = -((S_ang + S_acc)/2)**2
        #print (indexes[0], " S: ", S.shape)

        np.savez(fname_out,
                S = S,
                indexes = indexes)



def run_umap_params(X,
                    min_duration,
                    min_dist,
                    n_neighbors,
                    subsample_fit = 10,
                    subsample_predict = 1,  # do not subsample at prediction time
                    n_components = 2
                    ):

    print ("mindist: ", min_dist, "  n_neibhors: ", n_neighbors)
    #
    root_dir = '/media/cat/1TB/dan/cohort1/slp/'
    fname_out = (root_dir + '/umap_x_fit_min_duration_'+str(min_duration)+
                  '_dataType_'+data_type+
                  '_minDist_'+str(min_dist)+
                  '_nNeighbors_'+str(n_neighbors)+
                  '.npy')

    #
    if os.path.exists(fname_out)==False:

        X_unique, indices, counts = np.unique(X, axis=0,
                                              return_counts=True,
                                              return_index=True)
        #
        X_sampled = []
        for k in range(X_unique.shape[0]):
            ct = max(1, int(counts[k]/subsample_fit))
            #print (k, ct, counts[k])
            for c in range(ct):
                X_sampled.append(X_unique[k])

        X_sampled = np.array(X_sampled)
        print ("X_sampled (min 1 instance of each unique vector): ", X_sampled.shape)

        import umap
        umap = umap.UMAP(n_components=n_components,
                         n_neighbors=n_neighbors,
                         min_dist = min_dist,
                         init='random',
                         random_state=0)
        #

        #if data_type == 'continuous':
        print ("fitting / tranforming all", X.shape)
        X_fit = umap.fit_transform(X)


#         else:
#             X_subsampled = X_sampled
#             print ("fitting umap on" ,X_subsampled.shape)
#             umap_ = umap.fit(X_subsampled)

#             #
#             print ("transforming alldata ", X.shape)
#             X_fit = umap_.transform(X[::subsample_predict])


        #
        np.save(fname_out,
                X_fit)

        print ("X_fit: ", X_fit.shape)


    else:
        X_fit = np.load(fname_out)



    return X_fit



def get_random_sample_angles_accs(angles_hist, acc_hist, subsample):

    # round acc and ang to 3 decimals; Doesn't make a real difference in the end;
    angles_hist = np.around(angles_hist, 3)
    acc_hist = np.around(acc_hist, 3)

    # uniformly subsample the data:
    idx = np.arange(angles_hist.shape[0])[::subsample]


    all_histograms = np.hstack((angles_hist, acc_hist))[idx]
    print (idx.shape, angles_hist.shape, "all_histograms: ", all_histograms.shape)

    ang_unique = np.unique(all_histograms[:,:3], axis=0)
    acc_unique = np.unique(all_histograms[:,3:], axis=0)

    print ("Ang unique: ", ang_unique.shape, "acc unique: ", acc_unique.shape)

    return ang_unique, acc_unique, all_histograms



def dim_red3(X,
             method,
             subsample_fit=5,
             subsample_predict=1,
             n_components=2):

    ###################################
    ###################################
    ###################################
    if method==0:
        pca = decomposition.PCA(n_components=n_components)

        X_fit = pca.fit_transform(X)
        print (X_fit.shape)

    elif method==1:
        import umap
        umap = umap.UMAP(n_components=n_components,
                         #n_neighbors=100,
                         init='random',
                         random_state=0)

        X_subsampled = X[::subsample_fit]
        print ("fitting umap on" ,X_subsampled.shape)
        umap_ = umap.fit(X_subsampled)

        print ("transforming alldata ", X.shape)
        X_fit = umap_.transform(X[::subsample_predict])

    elif method==2:
        #X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        X_subsampled = X[::subsample_fit]
        print ("fitting tsne on" ,X_subsampled.shape)
        tsne_ = TSNE(n_components=2).fit(X)
        X_fit = tsne_.transform(X[::subsample_predict])

    elif method==3:

        # use principled subsampling to select at least 1 of each unique vector
        X_unique, indices, counts = np.unique(X, axis=0,
                                              return_counts=True,
                                              return_index=True)
#         print ("X_unique: ", X_unique.shape, X_unique[0])
#         print ("counts: ", counts.shape, counts[0])
#         print ("indices: ", indices.shape, indices[0])

        X_sampled = []
        for k in range(X_unique.shape[0]):
            ct = max(1, int(counts[k]/subsample_fit))
            #print (k, ct, counts[k])
            for c in range(ct):
                X_sampled.append(X_unique[k])

        X_sampled = np.array(X_sampled)
        print ("X_sampled (min 1 instance of each unique vector): ", X_sampled.shape)

        import umap
        umap = umap.UMAP(n_components=n_components,
                         #n_neighbors=100,
                         init='random',
                         random_state=0)
        #
        X_subsampled = X_sampled
        print ("fitting umap on" ,X_subsampled.shape)
        umap_ = umap.fit(X_subsampled)

        #
        print ("transforming alldata ", X.shape)
        X_fit = umap_.transform(X[::subsample_predict])

    return X_fit







def reconstruct_S_unique_from_files(all_unique, min_duration):

    root_dir = '/media/cat/1TB/dan/cohort1/slp/'
    path = os.path.join(root_dir, "S_unique",
                        "minDuration"+str(min_duration)+"_S_unique*.npz")
   # print ("path: ", path)
    fnames = glob.glob(path)
    #print ("Fnames: ", fnames)


    #S_unique = np.zeros((all_unique.shape[0],all_unique.shape[0]),'float32') #+np.nan
    S_unique = np.zeros((all_unique.shape[0],
                         all_unique.shape[0]),'float32') #+np.nan

    for fname in tqdm(fnames):
        #
        data = np.load(fname, allow_pickle=True)

        #
        S = data['S']
        indexes = data['indexes']

        #
        S_unique[indexes] = S

    #
    S_unique_symmetric = S_unique + S_unique.T - np.diag(np.diag(S_unique))

    return S_unique_symmetric

#
def compute_emd_normalized_unique_angle_acc(ang_unique, acc_unique):

    #
    emd_angle=np.zeros((ang_unique.shape[0],ang_unique.shape[0]))
    for i in range(ang_unique.shape[0]):
        for p in range(i+1, ang_unique.shape[0],1):
            emd_angle[i,p] = scipy.stats.wasserstein_distance(ang_unique[i],
                                                              ang_unique[p])

    # normalize by n-1
    emd_angle = emd_angle / (ang_unique.shape[1]-1)

    #
    emd_acc=np.zeros((acc_unique.shape[0],acc_unique.shape[0]))
    for i in range(acc_unique.shape[0]):
        for p in range(i+1, acc_unique.shape[0],1):
            emd_acc[i,p] = scipy.stats.wasserstein_distance(acc_unique[i],
                                                            acc_unique[p])

    # normalize by n-1
    emd_acc = emd_acc / (acc_unique.shape[1]-1)


    return emd_angle, emd_acc


def get_index(array, feature_array):

    idx = np.sum(np.equal(feature_array, array), axis=1)
    idx = np.where(idx==len(array))[0]

    return idx

