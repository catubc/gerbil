#
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from umap.umap_ import UMAP
import os
import numpy as np
from scipy.signal import savgol_filter
import scipy

from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vedo
from shapely.geometry import Polygon
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde, entropy
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm, trange
from scipy.stats import ks_2samp
from scipy.spatial import distance


# load data
class GerbilPCA():

    def __init__(self, 
                 root_dir,
                 behaviors):
        
        #
        self.root_dir = root_dir

        #
        self.behaviors = behaviors

        #
        self.clrs = ['black','blue','red','green','magenta','cyan','brown','navy',
                'orange','purple','pink','olive','teal','coral','lightblue']


    def get_kl_matrix(self, idx):

        #
        t1_window = np.arange(1,10,1)
        t1_matrix = np.zeros((14,14))+np.nan
        t2_matrix = np.zeros((14,14))+np.nan

        for t1 in t1_window:
            for t2 in range(t1+1, 13, 1):

                # grab chunk 1 of data
                temp1 = self.data[self.behavior_id][:,0:t1].copy()
                temp2 = self.data[self.behavior_id][:,t1:t2].copy()
                temp3 = self.data[self.behavior_id][:,t2:].copy()
                #print (temp1.shape, temp2.shape, temp3.shape)

                # flatten each chunk 
                temp1 = temp1.flatten()
                temp2 = temp2.flatten()
                temp3 = temp3.flatten()
                #print (temp1.shape, temp2.shape, temp3.shape)

                #
                if self.dist_method == 'kl_div':
                    # Perform kernel density estimation (KDE) 
                    samples1, samples2 = get_samples_from_kde(temp1, temp2, self.n_samples_kde)
                    kl_divergence1 = entropy(samples1, samples2)
                    t1_matrix[t1,t2] = kl_divergence1

                    #
                    samples1, samples2 = get_samples_from_kde(temp2, temp3, self.n_samples_kde)
                    kl_divergence2 = entropy(samples1, samples2)
                    t2_matrix[t1,t2] = kl_divergence2

                # jensen-shannon divergence
                elif self.dist_method == "js_div":
                    # Perform kernel density estimation (KDE) 
                    samples1, samples2 = get_samples_from_kde(temp1, temp2, self.n_samples_kde)
                    js_divergence1 = (entropy(samples1, samples2) + entropy(samples2, samples1))/2.0
                    t1_matrix[t1,t2] = js_divergence1

                    #
                    samples1, samples2 = get_samples_from_kde(temp2, temp3, self.n_samples_kde)
                    js_divergence2 = (entropy(samples1, samples2) + entropy(samples2, samples1))/2.0
                    t2_matrix[t1,t2] = js_divergence2

                elif self.dist_method == 'maha':
                    # calculate the mahalanobis distance between the two chunks
                    mean1 = np.mean(temp1, axis=0)
                    covariance_matrix1 = np.cov(temp1, rowvar=False)
                    mean2 = np.mean(temp2, axis=0)
                    covariance_matrix2 = np.cov(temp2, rowvar=False)

                    # Calculate the Mahalanobis distance
                    dists1 = []
                    for point in temp2:
                        mahalanobis_distance = mahalanobis(point, mean1, 
                                                    np.linalg.inv(covariance_matrix1))
                        dists1.append(mahalanobis_distance)

                    dists2 = []
                    for point in temp1:
                        mahalanobis_distance = mahalanobis(point, mean2, 
                                                    np.linalg.inv(covariance_matrix2))
                        dists2.append(mahalanobis_distance)

                    # compute the mean of the mahalanobis distances
                    t1_matrix[t1,t2] = np.mean(dists1)
                    t2_matrix[t1,t2] = np.mean(dists2)

                #
                elif self.dist_method == 'gausianity':

                    # fit a gaussian to the temp1 distribution        
                    mean1 = np.mean(temp1, axis=0)
                    covariance_matrix1 = np.cov(temp1, rowvar=False)
                    gaussian1 = multivariate_normal(mean=mean1, cov=covariance_matrix1)

                    # check the error of the gaussian fit
                    error1 = gaussian1.logpdf(temp1)

                    # fit a gaussian to the temp2 distribution
                    mean2 = np.mean(temp3, axis=0)
                    covariance_matrix2 = np.cov(temp3, rowvar=False)
                    gaussian2 = multivariate_normal(mean=mean2, cov=covariance_matrix2)

                    # check the error of the gaussian fit
                    error2 = gaussian2.logpdf(temp2)

                    #
                    #print (error1)
                    # compute the mean of the mahalanobis distances
                    t1_matrix[t1,t2] = np.mean(error1)
                    t2_matrix[t1,t2] = np.mean(error2)

                #
                elif self.dist_method == 'wasserstein_distance':
                    
                    
                    #
                    samples1, samples2 = get_samples_from_kde(temp1, temp2, self.n_samples_kde)
                    t1_matrix[t1,t2] = scipy.stats.wasserstein_distance(temp1, temp2)


                    # compute the wasserstein distance between the two chunks
                    samples1, samples2 = get_samples_from_kde(temp2, temp3, self.n_samples_kde)
                    t2_matrix[t1,t2] = scipy.stats.wasserstein_distance(temp2, temp3)
        

        #
        return t1_matrix, t2_matrix

    #
    def get_rapid_dev_plots(self):

        #
        dev_changes = np.zeros((len(self.behaviors),14))
       # dev_stable = np.zeros((len(self.behaviors),14))

        #
        for k in trange(len(self.behaviors)):

            # select a specific behavior
            self.behavior_id = k

            #
            t1_m = self.get_sliding_window_dists()

            #
            idx1 = np.where(t1_m<=self.pval_thresh)[0]

            #
            if self.smooth_pval:
                # 
                temp = np.zeros(t1_m.shape[0])
                temp[idx1] = 1
                
                #
               # print ("temp: ", temp)
                temp = replace_isolated_ones_with_zeros(temp, self.sliding_window_size)
                idx1 = np.where(temp==1)[0]
                #idx2 = np.where(temp==0)[0]
               # print ("temp: ", temp)
            dev_changes[k,idx1] = 1
            #dev_stable[k,idx2] = 1

        #         
        self.dev_changes = dev_changes
        #self.dev_stable = dev_stable


    #
    def get_area_under_curve_Fig2(self):

        # 
        auc_f = []
        for s in self.stack:
            print (s.shape)
            auc_f.append(np.trapz(s, dx=1))

        #
        auc_f = np.array(auc_f)
        print ("auc_f :", auc_f.shape)
        #auc_m = np.array(auc_m)

        #
        print ("auc_f mean: ", np.mean(auc_f,axis=1), " ,std: ", np.std(auc_f, axis=1))
        
        # # run anova test on data
        anova = scipy.stats.f_oneway(auc_f[0],
                                     auc_f[1],
                                     auc_f[2],
                                     auc_f[3],)
        self.anova = anova
        print ("anova: ", anova)

    #
    def get_pups_exploration_distance_food_water(self):
    # 
        print (len(self.stack))
        print (self.stack[0].shape)
        print (self.stack[1].shape)

        #
        c1_females = self.stack[0][:2]
        c2_females = self.stack[0][2:5]
        print ("c1_females: ", c1_females.shape)
        print ("c2_females: ", c2_females.shape)

        # make male average for each cohort to subtraact against
        c1_male_ave = np.mean(self.stack[1][:2], axis=0)
        print ("c1 male average over dev: ", c1_male_ave.shape)
        c2_male_ave = self.stack[1][2]
        print ("c2 male average over dev: ", c2_male_ave.shape)

        # compute diffs over each cohort manually
        diffs_c1 = []
        for c1_fem in c1_females:
            temp = c1_fem - c1_male_ave
            #print ("c1 diff: ", temp)
            diffs_c1.append(temp)
        diffs_c1 = np.array(diffs_c1).flatten()

        #
        diffs_c2 = []
        for c2_fem in c2_females:
            temp = c2_fem - c2_male_ave
            #print ("c2 diff: ", temp)
            diffs_c2.append(temp)

        diffs_c2 = np.array(diffs_c2).flatten()
        
        diffs = np.hstack((diffs_c1,
                           diffs_c2))
        print (diffs.shape)

        # same but shuffled female and male identity
        diffs_shuffled = []
        diffs_c2 = []
        diffs_c1 = []
        for p in range(100):
            for c1_fem in c1_females:
                for k in range(c1_fem.shape[0]):
                    idx = np.random.choice(np.arange(2))
                    if idx==0:
                        temp = c1_fem[k]-c1_male_ave[k]
                    else:
                        temp = c1_male_ave[k]-c1_fem[k]
                diffs_c1.append(temp)

            #
            for c2_fem in c2_females:
                for k in range(c2_fem.shape[0]):
                    idx = np.random.choice(np.arange(2))
                    if idx==0:
                        temp = c2_fem[k]-c2_male_ave[k]
                    else:
                        temp = c2_male_ave[k]-c2_fem[k]
                diffs_c2.append(temp)

        #
        diffs_shuffled = np.hstack((diffs_c1, diffs_c2))
            
        return diffs, diffs_shuffled

    def get_pups_adults_pairwise(self):

        #
        print ('behaviors: ', self.behaviors)

        #
        print (len(self.stack))
        print (self.stack[0].shape)
        print (self.stack[1].shape)

        #
        c1_females = self.stack[0][:2]
        c2_females = self.stack[0][2:5]
        print ("c1_females: ", c1_females.shape)
        print ("c2_females: ", c2_females.shape)

        # make male average for each cohort to subtraact against
        c1_male_ave = np.mean(self.stack[1][:2], axis=0)
        print ("c1 male average over dev: ", c1_male_ave.shape)
        c2_male_ave = self.stack[1][2]
        print ("c2 male average over dev: ", c2_male_ave.shape)

        # compute diffs over each cohort manually
        diffs_c1 = []
        for c1_fem in c1_females:
            temp = c1_fem - c1_male_ave
            #print ("c1 diff: ", temp)
            diffs_c1.append(temp)
        diffs_c1 = np.array(diffs_c1).flatten()

        #
        diffs_c2 = []
        for c2_fem in c2_females:
            temp = c2_fem - c2_male_ave
            #print ("c2 diff: ", temp)
            diffs_c2.append(temp)

        diffs_c2 = np.array(diffs_c2).flatten()
        
        diffs = np.hstack((diffs_c1,
                           diffs_c2))
        print ("diffs: ", diffs.shape)

        # same but shuffled female and male identity
        diffs_shuffled = []
        diffs_c2 = []
        diffs_c1 = []
        for p in range(100):
            for c1_fem in c1_females:
                for k in range(c1_fem.shape[0]):
                    idx = np.random.choice(np.arange(2))
                    if idx==0:
                        temp = c1_fem[k]-c1_male_ave[k]
                    else:
                        temp = c1_male_ave[k]-c1_fem[k]
                diffs_c1.append(temp)

            #
            for c2_fem in c2_females:
                for k in range(c2_fem.shape[0]):
                    idx = np.random.choice(np.arange(2))
                    if idx==0:
                        temp = c2_fem[k]-c2_male_ave[k]
                    else:
                        temp = c2_male_ave[k]-c2_fem[k]
                diffs_c2.append(temp)

        #
        diffs_shuffled = np.hstack((diffs_c1, diffs_c2))
            
        return diffs, diffs_shuffled

    #
    def get_pups_pairwise(self):

        #
        print ('behaviors: ', self.behaviors)

        #
        print (len(self.stack))
        print (self.stack[0].shape)
        print (self.stack[1].shape)

        #
        c1_females = self.stack[0][0][None]
        c2_females = self.stack[0][1:4]
        print ("c1_females: ", c1_females.shape)
        print ("c2_females: ", c2_females.shape)

        # make male average for each cohort to subtraact against
        c1_male_ave = np.mean(self.stack[1][:4], axis=0)
        print ("c1 male average over dev: ", c1_male_ave.shape)
        c2_male_ave = np.mean(self.stack[1][4:], axis=0)
        print ("c2 male average over dev: ", c2_male_ave.shape)

        # compute diffs over each cohort manually
        diffs_c1 = []
        for c1_fem in c1_females:
            temp = c1_fem - c1_male_ave
            #print ("c1 diff: ", temp)
            diffs_c1.append(temp)
        diffs_c1 = np.array(diffs_c1).flatten()

        #
        diffs_c2 = []
        for c2_fem in c2_females:
            temp = c2_fem - c2_male_ave
            #print ("c2 diff: ", temp)
            diffs_c2.append(temp)

        diffs_c2 = np.array(diffs_c2).flatten()
        
        diffs = np.hstack((diffs_c1,
                           diffs_c2))
        print ("diffs: ", diffs.shape)

        # same but shuffled female and male identity
        diffs_shuffled = []
        diffs_c1 = []
        diffs_c2 = []
        for p in range(100):
            for c1_fem in c1_females:
                for k in range(c1_fem.shape[0]):
                    idx = np.random.choice(np.arange(2))
                    if idx==0:
                        temp = c1_fem[k]-c1_male_ave[k]
                    else:
                        temp = c1_male_ave[k]-c1_fem[k]
                diffs_c1.append(temp)

            #
            for c2_fem in c2_females:
                for k in range(c2_fem.shape[0]):
                    idx = np.random.choice(np.arange(2))
                    if idx==0:
                        temp = c2_fem[k]-c2_male_ave[k]
                    else:
                        temp = c2_male_ave[k]-c2_fem[k]
                diffs_c2.append(temp)

        #
        diffs_shuffled = np.hstack((diffs_c1, diffs_c2))
            
        return diffs, diffs_shuffled

    #
    def intra_cohort_stats_pups(self):

        if 'exploration' in self.behaviors[0]:
            diffs, diffs_shuffled = self.get_pups_exploration_distance_food_water()
        elif 'distance' in self.behaviors[0]:
            diffs, diffs_shuffled = self.get_pups_exploration_distance_food_water()
        elif 'food' in self.behaviors[0]:
            diffs, diffs_shuffled = self.get_pups_exploration_distance_food_water()
        elif 'water' in self.behaviors[0]:
            diffs, diffs_shuffled = self.get_pups_exploration_distance_food_water()
        elif 'pairwise' in self.behaviors[0] and 'adult' not in self.behaviors[0]:
            diffs, diffs_shuffled = self.get_pups_pairwise()
        elif 'pairwise' in self.behaviors[0] and 'adult' in self.behaviors[0]:
            diffs, diffs_shuffled = self.get_pups_adults_pairwise()
        # elif ''

        # compute 2 sample ks test on diffs and diffs_shuffled
        ks_ = ks_2samp(diffs.flatten(), diffs_shuffled.flatten())
        print ("ks test: ", ks_)

        # also test diffs against a normal- zero mean distribution
        ttest = stats.ttest_1samp(diffs.flatten(), 0)
        print ("ttest: ", ttest)

        # plot diffs as a violin plot for each distribution with diffs at x=0 and diffs_shuffled at x=1
        plt.figure(figsize=(10,5))
        plt.subplot(111)
        plt.violinplot(diffs.flatten(), positions=[0], showmeans=True)
        plt.violinplot(diffs_shuffled.flatten(), positions=[1], showmeans=True)

 
        # label x axis with "real" and "shuffled"
        plt.xticks([0,1], ['real', 'shuffled'])

        # plot horizontal line at y=0
        plt.plot([-0.5,1.5], [0,0], 'k--')

        # show ks test result as title
        text1 = "{:.2e}".format(ks_[1])
        text2 = "{:.2e}".format(ttest[1])
        plt.title("ks test pval: "+text1 + "\n" + "ttest pval: "+text2)

        plt.suptitle(self.behaviors)

        #
        plt.ylabel("Intra cohort diffs (females-males @ every pday)")

        plt.show()
    #
    def intra_cohort_stats_adults(self):
        # 
        temp_stack = np.vstack(self.stack).copy()

        # females
        females = temp_stack[:3]
        males = temp_stack[3:]
        print ("adult females: ", females.shape, ", adult males: ", males.shape)

        # diffs
        diffs = females-males

        # same but shuffled female and male identity
        diffs_shuffled = []
        for p in range(100):
            for k in range(females.shape[0]):
                idx = np.random.choice(np.arange(2))
                if idx==0:
                    temp = females[k]-males[k]
                else:
                    temp = males[k]-females[k]

                diffs_shuffled.append(temp)

        #
        diffs_shuffled = np.array(diffs_shuffled)

        # compute 2 sample ks test on diffs and diffs_shuffled
        ks_ = ks_2samp(diffs.flatten(), diffs_shuffled.flatten())
        print ("ks test: ", ks_)

        # also test diffs against a normal- zero mean distribution
        ttest = stats.ttest_1samp(diffs.flatten(), 0)
        print ("ttest: ", ttest)

        # plot diffs as a violin plot for each distribution with diffs at x=0 and diffs_shuffled at x=1
        plt.figure(figsize=(10,5))
        plt.subplot(111)
        plt.violinplot(diffs.flatten(), positions=[0], showmeans=True)
        plt.violinplot(diffs_shuffled.flatten(), positions=[1], showmeans=True)

        # label x axis with "real" and "shuffled"
        plt.xticks([0,1], ['real', 'shuffled'])

        # plot horizontal line at y=0
        plt.plot([-0.5,1.5], [0,0], 'k--')

        # show ks test result as title
        text1 = "{:.2e}".format(ks_[1])
        text2 = "{:.2e}".format(ttest[1])
        plt.title("ks test pval: "+text1 + "\n" + "ttest pval: "+text2)

        plt.suptitle(self.behaviors)

        #
        plt.ylabel("Intra cohort diffs (females-males @ every pday)")

        plt.show()


    #
    def get_area_under_curve(self):

        # 
        temp_stack = np.vstack(self.stack)
        temp_starts = np.vstack(self.starts)

        # compute area under the curve for the first temp-stack
        auc_f = []
        for k in range(temp_starts[0][0], temp_starts[0][1],1):
            auc_f.append(np.trapz(temp_stack[k], dx=1))

        # same for the rest of the temp_stack
        auc_m = []
        for k in range(temp_starts[1][0], temp_starts[1][1],1):
            auc_m.append(np.trapz(temp_stack[k], dx=1))

        #
        auc_f = np.array(auc_f)
        auc_m = np.array(auc_m)

        #
        print ("auc_f mean: ", np.mean(auc_f), " ,std: ", np.std(auc_f))
        print ("auc_m mean: ", np.mean(auc_m), " ,std: ", np.std(auc_m))
        
        #
        self.auc_f = auc_f
        self.auc_m = auc_m

        self.ks_ = ks_2samp(auc_f, auc_m)

        # test the distributions for significance using ttest
        ttest = stats.ttest_ind(auc_f, auc_m)
        self.ttest = ttest

        # run anova test on data
        anova = scipy.stats.f_oneway(auc_f, auc_m)
        self.anova = anova
        
        text1 = "{:.2e}".format(self.ks_[1])
        text2 = "{:.2e}".format(self.ttest[1])
        text3 = "{:.2e}".format(self.anova[1])
        print ("ks test: ", text1)
        print ("ttest: ", text2)
        print ("anova: ", text3)


    #
    def plot_area_under_curve(self):    
        # plot scatter plot for each auc array separately in two columns
        plt.figure(figsize=(10,5))
        plt.subplot(111)

        plt.scatter(self.auc_f*0, self.auc_f,
                    c=self.clrs[0])
        plt.scatter(self.auc_m*0+1, self.auc_m,
                    c=self.clrs[1])
        
        # test the distributions for significance using 2 sample ks test
        ks_ = ks_2samp(auc_f, auc_m)
        print ("ks test: ", ks_)

        # test the distributions for significance using ttest
        ttest = stats.ttest_ind(auc_f, auc_m)
        print ("ttest: ", ttest)

        #

        plt.show()


    #
    def compute_dev_stages_histogram(self):

        # plot the two plots
        plt.figure()

        #
        img = self.dev_changes.copy()

        # find the first nonzero entry in each row of img
        idxs = []
        sums = []
        starts = []
        for k in range(img.shape[0]):
            idx = np.where(img[k,:]>0)[0]
            sums.append(np.sum(img[k,:]))
            if len(idx)>0:
                starts.append(idx[0])
            else:
                starts.append(np.nan)

        # sort idxs 
        starts = np.array(starts)
        idx = np.argsort(starts)
        starts = starts[idx]

        #
        sums = np.array(sums)
        sums = sums[idx]

        # bubble sort idxs but only when they have same sum
        change = True
        while change==True:
            change = False
            for k in range(len(starts)-1):

                #
                if starts[k]==starts[k+1]:
                    #print ("starts match")
                    if sums[k]>sums[k+1]:
                        #print ("sums diff")
                        temp = starts[k]
                        starts[k] = starts[k+1]
                        starts[k+1] = temp
                        change = True

                        # also change sums
                        temp = sums[k]
                        sums[k] = sums[k+1]
                        sums[k+1] = temp

                        # also change idx array
                        temp = idx[k]
                        idx[k] = idx[k+1]
                        idx[k+1] = temp

                        #
                        #print ("Moved k")
       #

        # reorder the img array
        img = img[idx,:]


        ###############
        # conver the image data to be 10 times larger and add empty rows every 10th row
        img2 = np.zeros((img.shape[0]*10, img.shape[1]))
        for k in range(img.shape[0]):
            img2[10*k:10*k+10,:] = img[k,:]
            img2[10*k]=0.1
    
        img = img2.copy()
        
        #
        plt.subplot(111)
        plt.imshow(img, 
                   aspect='auto',
                   interpolation='none',
                   # use extent to go from 0 to # of behaviors and then up to 14 days on x axis
                   extent = [0,14,0,len(self.behaviors)],
                   #cmap = 'binary'
                   )
        
        # label the y axis using the behavior names
        plt.yticks(np.arange(len(self.behaviors))+0.5, 
                   np.array(self.behaviors)[idx][::-1],
                   rotation=45,
                   fontsize=8)

        
        plt.title("dev changes")
        #plt.legend()

        # change xticks to start at 16
        labels = np.arange(16,30,1)
        plt.xticks(np.arange(14)+0.5,labels)

        plt.xlabel("dev PDay")

    #
    def plot_summary_devs(self):

        # ##########################################
        plt.subplot(111)

        # show the sum of the dev changes
        sums = np.sum(self.dev_changes,0)

        # find when sums idx is greater than 3
        idx = np.where(sums>=3)[0]
        # plot these locations in thick red
        plt.scatter(idx, sums[idx], linewidth=5,
                    c='red',
                    label='developmental stages')
        # same with <3
        idx = np.where(sums<3)[0]
        # plot these locations in blue
        plt.scatter(idx, sums[idx], linewidth=5,
                    c='blue',
                    label='stabilizing stages')
        

        
        # change xlabel to start at 16
        plt.xticks(np.arange(14),labels)

        plt.legend()

        #
        plt.show()

    #
    def show_dev_stage_similarties(self):

        # run PCA over the self.dev_changes array and plot it
        scaled_data = self.dev_changes.copy()
        print (scaled_data.shape)

        # Create a PCA instance with 2 components
        #num_components = 3
        pca = PCA(n_components=2)

        # Fit the PCA model and transform the data
        pca.fit(scaled_data)

        # transform the data
        data_pca = pca.transform(scaled_data)

        # plot
        plt.figure()
        plt.scatter(data_pca[:,0], 
                    data_pca[:,1],
                    c='black',
                    s=100)
        
        # label the points
        for k in range(len(self.behaviors)):
            plt.text(data_pca[k,0], data_pca[k,1], self.behaviors[k])

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        
        # show variance explained in the title
        text = "var exp: " + str(np.round(np.sum(pca.explained_variance_ratio_),2))
        plt.title(text)

        plt.show()

    #
    def get_sliding_window_dists(self):

        import numpy as np

        #
        t1_window = np.arange(0,14-self.sliding_window_size-1,1)
        t1_array = np.zeros(14)+np.nan

        # compute distribution for first window manually
        if self.interpolate_first_value:
            temp1 = self.data[self.behavior_id][:,0:self.sliding_window_size].copy()
            temp2 = self.data[self.behavior_id][:,self.sliding_window_size:2*self.sliding_window_size].copy()
            res = self.compute_signficance(temp1, temp2)

            t1_array[self.sliding_window_size] = res

        # compute dist for rest of distribution
        for t1 in t1_window:

            # grab chunk 1 of data
            temp1 = self.data[self.behavior_id][:,t1:t1+self.sliding_window_size].copy()
            temp2 = self.data[self.behavior_id][:,t1+self.sliding_window_size:t1+2*self.sliding_window_size].copy()
            #print ("temp1: ", temp1.shape)

            res = self.compute_signficance(temp1, temp2)

            t1_array[t1+self.sliding_window_size] = res
            #t1_array[t1+self.sliding_window_size-1:
            #         t1+self.sliding_window_size+1] = res

            # 
            #print (t1, t1+self.sliding_window_size, res)
            
        #
        return t1_array
    
    # 
    def compute_signficance(self, temp1, temp2):

        # flatten each chunk 
        temp1 = temp1.flatten()
        temp2 = temp2.flatten()

        #
        if self.dist_method == 'kl_div':
            # Perform kernel density estimation (KDE) 
            #samples1, samples2 = get_samples_from_kde(temp1, temp2, self.n_samples_kde)
            kl_divergence1 = entropy(temp1, temp2)
            #t1_array[t1+self.sliding_window_size] = kl_divergence1
            #
        
        elif self.dist_method =='t_test':
            print ("THIS needs correction ...")
            # Perform an independent two-sample t-test
            t_stat, p_value = stats.ttest_ind(temp1, temp2)
        # print (t1, t2, p_value)

            # Check the p-value to determine if the means are significantly different
            #t1_array[t1+self.sliding_window_size] = p_value
        
        #
        elif self.dist_method == '2sample_ks_test':
            
            # Perform a 2 sample ks test using scipy
            ks_ = ks_2samp(temp1, temp2)
            #t1_array[t1+self.sliding_window_size] = ks_[1]

        return ks_[1]

    #
    def process_data_circadian(self):



        #
        for behavior in self.behaviors:

            #
            adults = []
            pups = []
            pups_adults = []
    
            # 
            for c in range(2,5):
            
                #
                sub_dir = os.path.join(self.root_dir,
                                       'c' + str(c) + 
                                       "_ethogram_hourly_"+
                                       behavior)
                
                # for pairwise proximity we have many curves
                if 'pairwise' in behavior:
                    for a1 in range(6):
                        for a2 in range(6):
                            if a1==a2:
                                continue

                            #
                            fname = os.path.join(sub_dir,
                                                'ethogram_hourly_'+
                                                behavior + "_"+str(a1)+"_"+str(a2)
                                                +'.npy'
                                                )
                                            
                            #
                            d = np.load(fname)
                            
                            # remove first and last rows
                            d = d[1:-1,:]

                            #
                            if a1<2 and a2<2:
                                adults.append(d)

                            # this is pup-adult times
                            elif (a1<2 or a2<2) and (a1>2 or a2>2):
                                pups_adults.append(d)
                            #
                            else:
                                pups.append(d)
                #
                else:
                    for a1 in range(6):
                        fname = os.path.join(sub_dir,
                                            'ethogram_hourly_'+
                                            behavior + "_"+str(a1)
                                            +'.npy'
                                            )
                        #
                        d = np.load(fname)
                        
                        # remove first and last rows
                        d = d[1:-1,:]
                    
                        if a1<2:
                            adults.append(d)
                        else:
                            pups.append(d)

            #
            fname_out = os.path.join(self.root_dir,
                                     behavior + '_adults.npy')
            np.save(fname_out, np.array(adults))

            #
            fname_out = os.path.join(self.root_dir,
                                        behavior + '_pups.npy')
            np.save(fname_out, np.array(pups))

            #
            fname_out = os.path.join(self.root_dir,
                                        behavior + '_pups_adults.npy')
            np.save(fname_out, np.array(pups_adults))


    # Assuming you have three distributions A, B, and C as arrays
    def bhattacharyya_distance(self, p, q):
        mean_p = np.mean(p)
        mean_q = np.mean(q)
        var_p = np.var(p)
        var_q = np.var(q)

        if var_p == 0 and var_q == 0:
            return 0.0

        return 0.25 * np.log(0.25 * (var_p / var_q + var_q / var_p + 2.0)) + 0.25 * ((mean_p - mean_q) ** 2) / (var_p + var_q)

    #
    def compute_bhattacharyya_distance(self):

  #
        cl = self.selected_class

        t = np.arange(24)+0.5
        self.clusters = []
        for behavior in self.behaviors:
            #        
            cl_array = []
           
            # print ("self periods: ", self.periods)
            
            #
            if cl == 'pups_adults' and behavior != 'pairwise_proximity':
                continue

            #
            fname = os.path.join(self.root_dir, behavior + '_'+cl+'.npy')

            #
            data = np.load(fname)

            #
            d = np.mean(data, axis=0)

            # split d by row using self.period list of 3 tupples
            d_split = []
            for p in self.periods:

                # subtract 16 from p
                p = (p[0]-16, p[1]-16)

                #
                temp = d[p[0]:p[1]]
                #  print ("p: , temp.shape: ", p, temp.shape)

                #
                d_split.append(temp)

                #
                cl_array.append(data[:,p[0]:p[1]])

            #################################################
            # make the PCA array
            temp = []
            idxs = []
            start = 0
            for q in range(len(cl_array)):
                temp2 = np.mean(cl_array[q],axis=1)
                # print ("temp2: ", temp2.shape)
                temp.append(temp2)

                #
                idxs.append([start, start+temp2.shape[0]])
                start+= temp2.shape[0]

            #
            temp = np.vstack(temp)

            # Create a PCA instance with 2 components
            pca = PCA(n_components=3)

            # Fit the PCA model and transform the data
            pca.fit(temp)

            # transform the data
            data_pca = pca.transform(temp)

            #################################################################
            #################################################################
            #################################################################
         
            # loop over the developmental periods using hte idxs from above
            for idx in idxs:
                #
                temp = data_pca[idx[0]:idx[1]]

                
                if cl == self.selected_class:
                    self.clusters.append(temp)

                            
        #
        A,B,C = self.clusters

        # # Assuming you have three distributions A, B, and C as arrays
        distance_AB = self.bhattacharyya_distance(A, B)
        distance_AC = self.bhattacharyya_distance(A, C)
        distance_BC = self.bhattacharyya_distance(B, C)

        # # Calculate an overall distance (e.g., average)
        overall_distance = (distance_AB + distance_AC + distance_BC) / 3
       # print ("ave bhatta: ", overall_distance)

        return overall_distance


    def get_circadian_total_behavior_sums(self):
        #
        classes = [
            'adults',
            'pups',
            'pups_adults'
        ]

        #
        idx_day = np.zeros(24,'int32')
        idx_day[8:20] = 1
        idx_night = np.mod(idx_day+1,2)
        idx_day = np.where(idx_day==1)[0]
        idx_night = np.where(idx_night==1)[0]

        print (idx_day)
        print (idx_night)

        #
        plt.figure()
        t = np.arange(24)+0.5
        for behavior in self.behaviors:
            print ('')
            print ('')
            print ("Starting behavior: ", behavior)

            #
            for cl in classes:
                print ('')
                print ("starting: class: ",cl)
                
                #
                if cl == 'pups_adults' and behavior != 'pairwise_proximity':
                    continue

                #
                fname = os.path.join(self.root_dir, behavior + '_'+cl+'.npy')

                #
                data = np.load(fname)
                print ("data.shape: ", data.shape)

                # average overa all animals
                d = np.mean(data, axis=0)
                print ("d.shape: ", d.shape)
               
                # average over all days
                plt.plot(t,
                        np.mean(d,axis=0),
                        label=behavior+'_'+cl)
                        
                #
                dur_daytime = d[:,idx_day].sum()
                dur_daytime_std = np.std(d[:,idx_day].sum(axis=1))
                dur_nighttime = d[:,idx_night].sum()
                dur_nighttime_std = np.std(d[:,idx_night].sum(axis=1))

                #
                print ("dur_daytime: ", dur_daytime)
                print ("dur_nighttime: ", dur_nighttime)

                # show ratios also
                print ("light ratio: ", np.round(dur_daytime/(dur_nighttime+dur_daytime)*100,2),
                        " , std: ", np.round(dur_daytime_std/(dur_nighttime+dur_daytime)*100,2))
                print ("dark ratio: ", np.round(dur_nighttime/(dur_nighttime+dur_daytime)*100,2),
                        " , std: ", np.round(dur_nighttime_std/(dur_nighttime+dur_daytime)*100,2))


        #
        plt.legend()
        plt.show()

    def optimize_t1_t2(self):

        #
        res_matrix = np.zeros((30,30))
        for t1 in range(17,26):
            for t2 in range(t1+1,29):
                self.periods = [
                    [16, t1],
                    [t1, t2],
                    [t2, 30],
                ]
                res = self.compute_bhattacharyya_distance()
                res_matrix[t1,t2]=res

        #
        plt.figure()
        plt.imshow(res_matrix, 
                   cmap='jet', 
                   
                interpolation='none')
        plt.colorbar()
        plt.ylim(16,30)
        plt.xlim(16,30)
        plt.xlabel('t2')
        plt.ylabel('t1')
        plt.suptitle("Bhattacharyya distances optimizing for # "+
                    str(len(self.periods))+  " circadian stages, \nbased on behavior: "+str(self.behaviors[0])+
                    "\nDEV Epochs: P16 -> t1 -> t2 -> P30",
                    fontsize=10)
        #
        plt.show()
                
    #
    def circadian_plots(self):
        
        #
        classes = [
            'adults',
            'pups',
            'pups_adults'
        ]

        # make 3 colors that increase in intensity
        clrs = ['lightblue','blue','navy']


        #
        t = np.arange(24)+0.5
        self.clusters = []
        for behavior in self.behaviors:
            print ("")
            print ("Starting behavior: ", behavior)

            #
            for cl in classes:
                cl_array = []
                print ("")
                print ("starting: class: ",cl)
               
                # print ("self periods: ", self.periods)
                
                #
                if cl == 'pups_adults' and behavior != 'pairwise_proximity':
                    continue

                #
                fname = os.path.join(self.root_dir, behavior + '_'+cl+'.npy')

                #
                data = np.load(fname)
                print ("data.shape: ", data.shape)

                #
                d = np.mean(data, axis=0)

                # split d by row using self.period list of 3 tupples
                d_split = []
                for p in self.periods:
                    #tempp = p

                    # subtract 16 from p
                    p = (p[0]-16, p[1]-16)

                    #
                    temp = d[p[0]:p[1]]
                    print ("p: , temp.shape: ", p, temp.shape)

                    #
                    d_split.append(temp)

                    #
                    cl_array.append(data[:,p[0]:p[1]])

                #
                #d_split = np.array(d_split)
                #print (d_split.shape)

                ##########################################
                ##########################################
                ##########################################
                # plot each of the 3 periods
                plt.figure()
                for k in range(3):
                    mean = np.mean(d_split[k], axis=0)
                    plt.plot(t,mean,
                             label = str(self.periods[k]),
                             c=clrs[k],
                             linewidth = 5,
                            )
        
                    # plot sem standard error 
                    sem = np.std(d_split[k], axis=0)/np.sqrt(d_split[k].shape[0])
                    plt.fill_between(t, mean-sem, mean+sem,
                                        alpha=.2,
                                        color=clrs[k])
                    
                #          
                plt.title(behavior + ' ' + cl)
                
                # add background shading between 0 annd 8 and between 20 and 24
                plt.axvspan(0, 8, facecolor='gray', alpha=0.2)
                plt.axvspan(20, 24, facecolor='gray', alpha=0.2)
                plt.legend()
                #
                plt.xlim(0,24)
                
                # 
                plt.suptitle("Dev splits t1: "+str(self.dev_windows[0])+
                             " , t2: "+str(self.dev_windows[1])
                             +"\n"
                             + "stage 1: 16-"+str(self.dev_windows[0])
                             + ", stage 2: "+str(self.dev_windows[0])+"-"+str(self.dev_windows[1])
                             + ", stage 3: "+str(self.dev_windows[1])+"-30"
                            )

                #
                plt.show()

                #################################################
                #################################################
                #################################################
                # make the PCA array
                temp = []
                idxs = []
                start = 0
                #print ("# cl_array: ", len(cl_array))
                for q in range(len(cl_array)):
                    temp2 = np.mean(cl_array[q],axis=1)
                   # print ("temp2: ", temp2.shape)
                    temp.append(temp2)

                    #
                    idxs.append([start, start+temp2.shape[0]])
                    start+= temp2.shape[0]

                #
                print ("idxs: ", idxs)
                
                #
                temp = np.vstack(temp)
                print ("stacked cl array: ", temp.shape)

                # Create a PCA instance with 2 components
                pca = PCA(n_components=3)

                # Fit the PCA model and transform the data
                pca.fit(temp)

                # transform the data
                data_pca = pca.transform(temp)

                # print variance explained in 3 components
                print ("variance explained for 3pcs: ", np.sum(pca.explained_variance_ratio_))

                #################################################################
                #################################################################
                #################################################################
                # 
                if self.plot_3d==False:
                    plt.figure()
                    ctr=0
                    hulls = []
                    for idx in idxs:
    
                        #
                        temp = data_pca[idx[0]:idx[1]]
    
                        #
                        plt.scatter(temp[:,0], 
                                temp[:,1],
                                c=clrs[ctr],
                                s=100,
                                label = str(self.periods[ctr]))
                        
                        #
                    

                        # get the convex hull of temp
                        hull = ConvexHull(temp)

                        # plot the convex hull
                        for simplex in hull.simplices:
                            plt.plot(
                                temp[:,0][simplex], 
                                temp[:,1][simplex], 
                                color=clrs[ctr],
                                linewidth=5,
                                alpha=.5)
                            
                        # save the hull
                        hulls.append(temp[hull.vertices])

                        ctr+=1
                    plt.show()
                    ###############################################
                    ###############################################
                    ###############################################
                    # loop over hulls and find intersection area of each pair


                    inter_array = []
                    for k in range(len(hulls)):
                        inter = 0
                        for j in range(k+1,len(hulls)):
                            # compute the intersection area of the convex hulls
                            intersection = np.intersect1d(hulls[k], hulls[j])
                            inter+=intersection

                        print (k, "intersection: ", inter)
                        inter_array.append(inter)

                    # plot bar plot of overlap volumes
                    plt.figure()
                    for k in range(len(hulls)):
                        plt.bar(k, 
                                inter_array[k], 
                                0.9,
                                color=self.clrs[k],
                                label = str(self.periods[k]))
                                
                        
                    #
                    plt.legend()
                    

                # same plot for for 3d 
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    # use 3 types of markers every 1/3 the length of cl_array.shape[0]
                    markers = ['o','^','s']
     
                    # loop over the developmental periods using hte idxs from above
                    ctr=0
                    self.polygon_array = []
                    for idx in idxs:
                        #
                        temp = data_pca[idx[0]:idx[1]]

                        #
                        chunk = temp.shape[0]//3
                        for m in range(0, temp.shape[0], chunk):

                            #
                            if m==0:
                                ax.scatter(temp[m:m+chunk,0], 
                                    temp[m:m+chunk,1],
                                    temp[m:m+chunk,2],
                                    c=clrs[ctr],
                                    marker = markers[m//3],
                                    s=400,
                                    edgecolors='black',
                                    label = str(self.periods[ctr]) +
                                            "c1: o, c2: tri, c3: sq",
                                    alpha=1)
                            else:
                                ax.scatter(temp[m:m+chunk,0], 
                                    temp[m:m+chunk,1],
                                    temp[m:m+chunk,2],
                                    c=clrs[ctr],
                                    marker = markers[m//chunk],
                                    s=400,
                                    edgecolors='black',
                                    #label = str(self.periods[ctr]),
                                    alpha=1)

                        ##################################################
                        ##################################################
                        ##################################################


                        # repeat the last entry in temp and try again
                        #temp = np.vstack((temp,temp[-1]*1.01))
                        hull = ConvexHull(temp)
                        print ("class temp size: ", temp.shape)

                        if cl == 'pups':
                            self.clusters.append(temp)

                        #
                        for simplex in hull.simplices:
                            plt.plot(
                                temp[:,0][simplex], 
                                temp[:,1][simplex], 
                                temp[:,2][simplex], 

                                    color=clrs[ctr],
                                    linewidth=5,
                                    alpha=.5)
                            
                        # Step 2: Compute the convex hull
                        #hull = ConvexHull(temp)
                        vertices = temp[hull.vertices]  # Vertices of the convex hull

                        # get the ids of the vertices
                        # make a larger vertices array and fill in the missing vertices with zeros
                        vertices = np.zeros((temp.shape[0],3))
                        vertices[hull.vertices] = temp[hull.vertices]
                        #

                        # Step 3: Extract the valid edges and vertices from the convex hull
                        edges = []
                        for simplex in hull.simplices:
                            n_bad = 0
                            for v in simplex:
                                #print (v)
                                if v < 0 or v >= len(temp):
                                    n_bad += 1

                            if n_bad == 0:
                                edges.append(simplex)

                        # Step 5: Create a Poly3DCollection for the 3D polygon
                        poly3d = [[vertices[edge[0]], vertices[edge[1]], vertices[edge[2]]] for edge in edges]

                        ax.add_collection3d(Poly3DCollection(poly3d, 
                                                            facecolors=clrs[ctr], 
                                                            linewidths=1, 
                                                            edgecolors=clrs[ctr], 
                                                            alpha=0.5))
                    
                        ctr+=1  

                        # save the convex hull points only
                        #self.polygon_array.append(vertices[hull.vertices])
                        self.polygon_array.append(temp)

                # compute overlap between hulls
                #print ("len(self.hulls): ", len(self.hulls))
                #print ("len(self.clusters): ", len(self.clusters))

                
                #
                plt.xlabel("PC1")
                plt.ylabel("PC2")

                #
                plt.title(behavior + ' ' + cl)
                plt.legend()
                plt.show()
            

                ##############################################
                ##############################################
                ##############################################
                # Define your two polygons using their coordinates
                
                
                vedo.settings.default_font = 'Bongas'
                vedo.settings.use_depth_peeling = True

                #print ("# of hulls: ", len(self.polygon_array))

                pps = []
                vols = []
                # do 3d plot here
                import pyvista as pv
                pps_s = []
                for k in range(len(self.polygon_array)):
                    #print ("polygon: ", k, self.polygon_array[k].shape)
                    aa = vedo.ConvexHull(self.polygon_array[k])

                    polyhedron1 = pv.PolyData(self.polygon_array[k])
                    polyhedron1.triangulate()
                    pps_s.append(polyhedron1)

                    # 
                    pps.append(aa)

                    #
                    vol = aa.volume()
                    print (k, "vol: ", vol)
                    vols.append(vol)

                # find the intersection of the convex hulls and add them up
                perc_arrays = []
                for ctr1,pp1 in enumerate(pps):
                    temp_percent = 0
                    for ctr2,pp2 in enumerate(pps):
                        #
                        if ctr1==ctr2:
                            continue
                        #
                        cc = pp1.boolean("intersect", pp2)
                        ccc = vedo.ConvexHull(cc.points())

                        
                        # Compute the intersection of the two polyhedrons
#                        intersection = pps_s[ctr1].boolean_difference(pps_s[ctr2])

                        # Calculate the volume of the intersection
                       # intersection_volume = intersection.compute_cell_volumes().sum()

                        #print("Intersection volume:", intersection_volume)
                       # intersect = pps_s[ctr1].intersection(pps_s[ctr2])
                        print (ctr1, ctr2, ccc.volume()/vols[ctr1])
                        #print ("shapely intersection: ", intersect.volume)
                        #
                        if vols[ctr1]>0:
                            temp_percent = max(temp_percent, 
                                               ccc.volume()/vols[ctr1])
                    perc_arrays.append(temp_percent)

                # plot bar plot of overlap volumes
                plt.figure()
                for k in range(len(vols)):
                    plt.bar(k, 
                            perc_arrays[k], 
                            0.9,
                            color=self.clrs[k],
                            label = str(self.periods[k]))
                    
                #
                plt.legend()

                #
                #plt.ylim(0,1)

                plt.ylabel("ratio overlap with other behaviors")

                #
                plt.show()
  

        # plt.figure()
        # plt.imshow(d)

        # plt.show()



    #
    def load_data(self):

        #
        flip_behaviors = [

            #'food',
            #'water',
            #'pup_male_food',
            #'pup_male_water',
            #'pup_female_food',
            #'pup_female_water',
        ]

       #
        stack = []
        lens = []
        starts = []
        self.data = []
        ctr=0
        for behavior in self.behaviors:

            fname = os.path.join(self.root_dir,
                                behavior + '.npy')
            
            #
            d = np.load(fname)

            #
            if behavior in flip_behaviors:
                 d = d[:,::-1]
#
            # delete the pday = 3 values for cohort 4
            #if behavior =='food' or behavior == 'water' or behavior == 'distance' or behavior == 'exploration':
            # for p in range(4):
            #     d[8+p:,2] = d[8+p:,1]
                
            print ("behavior: ", behavior, d.shape)

            # smooth data
            if self.smooth:
                d = savgol_filter(d, 
                                  window_length=self.smooth_window,
                                  polyorder=self.smooth_order)
                
            #
            self.data.append(d)

            #
            stack.append(d)

            #
            lens.append(d.shape[0])

            #
            starts.append([ctr,ctr+d.shape[0]])

            #
            ctr+= d.shape[0]

        self.stack = stack
        self.starts = starts
    
    #
    def plot_mean_behavior(self):

        # plot the time series first
        plt.figure(figsize=(10,10))
        ctr=0
        for s in self.stack:

            #
            print ("s: ", s.shape)

            #
            mean = np.mean(s,axis=0)
            std = np.std(s,axis=0)

            #
            print ("mean: ", np.mean(s), " std: ", np.std(s))

            # get standard error
            sem = std/np.sqrt(s.shape[0])

            #
            t = np.arange(mean.shape[0])

            #
            plt.plot(t,mean, 
                    label=self.behaviors[ctr],
                    c=self.clrs[ctr],
                    linewidth=5,
                    alpha=.9)
       
            #
            #plt.plot(mean, c='black',label='beahvior')
            plt.fill_between(np.arange(len(mean)), mean-sem, mean+sem, 
                             alpha=.2,
                             color=self.clrs[ctr])
            #
            plt.title(self.behaviors[self.behavior_id])
            temp_changes = self.dev_changes[ctr]
            idx1 = np.where(temp_changes>0)[0]

            # plot a thick line over idx
            plt.scatter(t[idx1], mean[idx1], 
                        #linewidth=5, 
                        c=self.clrs[ctr],
                        marker='^',
                        s=self.size,
                        edgecolors='black',
                        #label='developmental stages'
                        )

            #
            ctr+=1

        #
        labels = np.arange(16,30,1)
        plt.xticks(np.arange(14),labels)

        plt.xlabel("dev PDay")
        plt.ylabel("kl divergence in sequential windows")

        #
        plt.legend()
                
        #
        plt.suptitle("")

        #
        plt.show()


    

        #
        plt.legend(fontsize=20)

        #
        plt.show()

    #
    def plot_pca_3d(self):

        #
        fig = plt.figure()

        #
        ax = fig.add_subplot(111, projection='3d')

        # First Principal Component
        polygon_array = []

        #
        for ctr,s in enumerate(self.stack):
            
            #
            temp = self.data_pca[self.starts[ctr][0]: self.starts[ctr][1], :]

            # Step 1: Fit a Gaussian model to the data
            if self.remove_outlier_point:
                data = temp.copy()
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)

                # Step 2: Calculate Mahalanobis distance for each data point
                mahalanobis_distances = np.abs(data - mean) / std

                # compute length of each mahalanobis distance
                lengths = np.sqrt(np.sum(mahalanobis_distances**2, axis=1))

                # find longest mahalanobis distance
                outlier_index = np.argmax(lengths)
                #
                idxs = np.arange(data.shape[0]) 
                print ("outlier index: ", outlier_index, )
                # print ("idxs: ", idxs)
                idxs = np.delete(idxs, outlier_index)

                temp = data[idxs].copy()
                print ("data after removing outliner: ", temp.shape)

            #
            # print ("temp polygon: ", temp)

            #
            ax.scatter( temp[:, 0],
                        temp[:,1],
                        temp[:,2],
                        s=self.scatter_size,
                        color=self.clrs[ctr],
                        #marker = markers[ctr],
                        label = self.behaviors[ctr] ,
                        alpha = 0.8,
                        # use black edges
                            edgecolors='k'
                        )
            
            # draw 3d convex hull

            if temp.shape[0]==1:
                continue
            if self.show_convex_hull:
                try:
                    hull = ConvexHull(temp)
                except:
                    # repeat the last entry in temp and try again
                    temp = np.vstack((temp,temp[-1]*1.01))
                    hull = ConvexHull(temp)

                # save convex hull points
                polygon_array.append(temp[hull.vertices])

                #
                for simplex in hull.simplices:
                    plt.plot(
                        temp[:,0][simplex], 
                        temp[:,1][simplex], 
                        temp[:,2][simplex], 

                            color=self.clrs[ctr],
                            linewidth=5,
                            alpha=.5)
                    
                # Step 2: Compute the convex hull
                hull = ConvexHull(temp)
                vertices = temp[hull.vertices]  # Vertices of the convex hull

                # get the ids of the vertices
                # make a larger vertices array and fill in the missing vertices with zeros
                vertices = np.zeros((temp.shape[0],3))
                vertices[hull.vertices] = temp[hull.vertices]
                #

                # Step 3: Extract the valid edges and vertices from the convex hull
                edges = []
                for simplex in hull.simplices:
                    n_bad = 0
                    for v in simplex:
                        #print (v)
                        if v < 0 or v >= len(temp):
                            n_bad += 1

                    if n_bad == 0:
                        edges.append(simplex)

                # Step 5: Create a Poly3DCollection for the 3D polygon
                poly3d = [[vertices[edge[0]], vertices[edge[1]], vertices[edge[2]]] for edge in edges]

                ax.add_collection3d(Poly3DCollection(poly3d, 
                                                    facecolors=self.clrs[ctr], 
                                                    linewidths=1, 
                                                    edgecolors=self.clrs[ctr], 
                                                    alpha=0.5))

        self.polygon_array = polygon_array

    #
    def plot_pca_2d(self):

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

        markers = ['o','*','^']
        polygon_array = []

        # First Principal Component
        for ctr,s in enumerate(self.stack):

            #
            temp = self.data_pca[self.starts[ctr][0]:
                                    self.starts[ctr][1], :]
            
            # triage to remove the most outliery points from temp
            # Step 1: Calculate the 5th percentile
            #temp2 = np.percentile(temp, self.)

            # Step 2: Create a new distribution with values in the bottom 95 percentile
            #print ("data going in: ",temp.shape)

            if self.remove_outlier_point:
                data = temp.copy()
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)

                # Step 2: Calculate Mahalanobis distance for each data point
                mahalanobis_distances = np.abs(data - mean) / std

                # compute length of each mahalanobis distance
                lengths = np.sqrt(np.sum(mahalanobis_distances**2, axis=1))

                # find longest mahalanobis distance
                outlier_index = np.argmax(lengths)
                #
                idxs = np.arange(data.shape[0]) 
                print ("outlier index: ", outlier_index, )
                # print ("idxs: ", idxs)
                idxs = np.delete(idxs, outlier_index)

                temp = data[idxs].copy()
                print ("data after removing outliner: ", temp.shape)

            
            chunk = int(temp.shape[0]/3)

            if chunk==0:
                chunk=1
            
            # make unique marker list for each cohort
            ctr2=0
            for m in range(0,temp.shape[0],chunk):
                #
                # show labels
                if ctr2==0:
                    ax.scatter( temp[m:m+chunk, 0],
                                temp[m:m+chunk,1],

                                s=self.scatter_size,
                                color=self.clrs[ctr],
                               # marker = markers[ctr2],
                                label = self.behaviors[ctr] ,
                                alpha = 0.8,
                        # use black edges
                            edgecolors='k'
                        )
                else:
                    ax.scatter( temp[m:m+chunk, 0],
                            temp[m:m+chunk,1],

                            s=self.scatter_size,
                        color=self.clrs[ctr],
                       # marker = markers[ctr2],
                        alpha = 0.8,
                        # use black edges
                            edgecolors='k'
                        )
                
                #
                ctr2+=1
            
            # also show the convex hull over the data
            # print ("behavior: ", self.behaviors[ctr])
            # print ("temp: ", temp.shape)
            # print ("temp: ", temp)

            if temp.shape[0]==1:
                polygon_array.append([])
                continue

            try:
                hull = ConvexHull(temp)
            except:
                # repeat the last entry in temp and try again
                temp = np.vstack((temp,temp[-1]*1.01))
                hull = ConvexHull(temp)

            for simplex in hull.simplices:
                plt.plot(self.data_pca[self.starts[ctr][0]:
                                self.starts[ctr][1], 0][simplex], 
                        self.data_pca[self.starts[ctr][0]:
                                self.starts[ctr][1], 1][simplex], 
                        color=self.clrs[ctr],
                        linewidth=5,
                        alpha=.5)

            # save convex hull points
            polygon_array.append(temp[hull.vertices])

        self.polygon_array = polygon_array

    #
    def run_pca(self):


        #
        clrs = []
        for ctr,s in enumerate(self.stack):
            temp = np.zeros(s.shape[0])+ctr

            clrs.extend(temp)

        #
        data = np.vstack(self.stack)
        print ("self.stack: ",  len(self.stack))
        
        #Standardize the data
        if self.standardize:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
        else:
            scaled_data = data.copy()

        # Create a PCA instance with 2 components
        #num_components = 3
        pca = PCA(n_components=self.n_dimensions)

        # Fit the PCA model and transform the data
        pca.fit(scaled_data)

        # transform the data
        self.data_pca = pca.transform(scaled_data)

        # 
        if self.n_dimensions == 2:
            self.plot_pca_2d()
    
                    
        # same but in 3d now:
        else:
            self.plot_pca_3d()


        ##########################################################
        # plot
        text = "var exp: " + str(np.round(np.sum(pca.explained_variance_ratio_),2))


        # add labels in order of
        # plot legend using title 'text'
        legend = plt.legend(fontsize=15, 
                   title = text,
                   )     
        plt.legend()

        plt.title(text)

        # Increase the size of the legend title
        title = legend.get_title()
        title.set_fontsize(16)
        plt.tight_layout()
        plt.show()

    #
    def find_overlaps(self):

        # Here we compute overlap in 3D/3PCs
        if self.n_dimensions == 3:

            vedo.settings.default_font = 'Bongas'
            vedo.settings.use_depth_peeling = True

            #print ("# of hulls: ", len(self.polygon_array))

            pps = []
            vols = []
            for k in range(len(self.polygon_array)):
                #print ("polygon: ", k, self.polygon_array[k].shape)
                aa = vedo.ConvexHull(self.polygon_array[k])
                pps.append(aa)

                #
                vol = aa.volume()
                vols.append(vol)

            # find the intersection of the convex hulls and add them up
            perc_arrays = []
            for ctr1,pp1 in enumerate(pps):
                temp_percent = 0
                for ctr2,pp2 in enumerate(pps):
                    #
                    if ctr1==ctr2:
                        continue
                    #
                    cc = pp1.boolean("intersect", pp2).c("red5")
                    ccc = vedo.ConvexHull(cc.points())
                   # print("intersection volume: ", ccc.volume())
                    #temp_vol = ccc.volume()*100

                    #
                    if vols[ctr1]>0:
                        temp_percent += ccc.volume()/vols[ctr1]
                perc_arrays.append(temp_percent)

        # compute the overlap in 2D
        else:

            # Define your two polygons using their coordinates
            pps = []
            vols = []
            for pol in self.polygon_array:
                temp = Polygon(pol)
                pps.append(temp)

                #
                vols.append(temp.area)

            # find the intersection of the convex hulls and add them up
            perc_arrays = []
            for ctr1,pp1 in enumerate(pps):
                temp_percent = 0
                for ctr2,pp2 in enumerate(pps):
                    #
                    if ctr1==ctr2:
                        continue

                    # Compute the intersection
                    intersection = pp1.intersection(pp2)

                    # Check if there is an intersection
                    if intersection.is_empty:
                        intersection_area = 0
                    else:
                        intersection_area = intersection.area

                    if vols[ctr1]>0:
                        temp_percent += intersection_area/vols[ctr1]
                
                #
                temp_percent = min(1.0, temp_percent)
                perc_arrays.append(temp_percent)

        # plot bar plot of overlap volumes
        plt.figure()
        for k in range(len(vols)):
            plt.bar(k, 
                    perc_arrays[k], 
                    0.9,
                    color=self.clrs[k],
                    label = self.behaviors[k])
            
        #
        plt.legend()

        #
        #plt.ylim(0,1)

        plt.ylabel("ratio overlap with other behaviors")

        #
        plt.show()

    #
    def compute_mahalanobis_distance_histograms(self):

        # Calculate the mean and covariance matrix of your distribution

        # 
        fig=plt.figure()
        for idx1 in range(len(self.starts)):
            #
            plt.subplot(2,2,idx1+1)
            #
            data1 = self.data_pca[self.starts[idx1][0]:self.starts[idx1][1]]
            print (data1.shape)
            mean = np.mean(data1, axis=0)
            covariance_matrix = np.cov(data1, rowvar=False)

            # Calculate the Mahalanobis distance
            dists1 = []
            for point in data1:
                mahalanobis_distance = mahalanobis(point, mean, 
                                            np.linalg.inv(covariance_matrix))
                dists1.append(mahalanobis_distance)

            min = np.min(dists1)*0.5
            max = np.max(dists1)*1.5
            bins = np.linspace(min,max,self.n_bins)
            y = np.histogram(dists1, bins=bins)
            plt.plot(y[1][:-1], y[0], 
                    color=self.clrs[idx1],
                    label=self.behaviors[idx1])
            #
            for idx2 in range(len(self.starts)):
                if idx1==idx2:
                    continue
                    
                #

                data2 = self.data_pca[self.starts[idx2][0]:self.starts[idx2][1]]
                print (data2.shape  )
                
                    #
                dists2 = []
                for point in data2:
                    mahalanobis_distance = mahalanobis(point, mean, 
                                                np.linalg.inv(covariance_matrix))
                    dists2.append(mahalanobis_distance)

                # show histograms of dists1 and dists2


                min = np.min(dists2)*0.5
                max = np.max(dists2)*1.5
                bins = np.linspace(min,max,self.n_bins)
                y = np.histogram(dists2, bins=bins)
                plt.plot(y[1][:-1], y[0],
                        # width=y[1][1]-y[1][0],
                        color=self.clrs[idx2],
                        label=self.behaviors[idx2])
            plt.legend()        


            #
            print ("")
     

        plt.show()

    #
    def compute_mahalanobis_distance_distributions(self):

        # Calculate the mean and covariance matrix of your distribution

        # 
        fig=plt.figure()
        for idx1 in range(len(self.starts)):
            #
            plt.subplot(2,2,idx1+1)
            #
            data1 = self.data_pca[self.starts[idx1][0]:self.starts[idx1][1]]
            mean = np.mean(data1, axis=0)
            covariance_matrix = np.cov(data1, rowvar=False)

            # Calculate the Mahalanobis distance
            dists1 = []
            for point in data1:
                mahalanobis_distance = mahalanobis(point, mean, 
                                            np.linalg.inv(covariance_matrix))
                dists1.append(mahalanobis_distance)

            #
            plt.scatter(np.zeros(len(dists1)), dists1, 
                    color=self.clrs[idx1],
                    label=self.behaviors[idx1])
            
            #
            ctr=0
            xticks= []
            xticks.append(self.behaviors[idx1])
            for idx2 in range(len(self.starts)):
                if idx1==idx2:
                    continue
                ctr+=1
                    
                #
                data2 = self.data_pca[self.starts[idx2][0]:self.starts[idx2][1]]
                dists2 = []
                for point in data2:
                    mahalanobis_distance = mahalanobis(point, mean, 
                                                np.linalg.inv(covariance_matrix))
                    dists2.append(mahalanobis_distance)

                # compute a 2sample ks test on dists1 and dists2
                #print ("")
                ks_ = ks_2samp(dists1, dists2)   
               # print ("ks test: ", ks_)             

                # convert ks_[1] to exponential notation
                text = "{:.2e}".format(ks_[1])
                #text = str(ks_[1])
                # show histograms of dists1 and dists2
                plt.scatter(np.zeros(len(dists2))+ctr, dists2, 
                        color=self.clrs[idx2],
                        label=self.behaviors[idx1] + " vs " + self.behaviors[idx2]+ " pval: " + text)
                
                
                #
                xticks.append(self.behaviors[idx2])

            plt.legend()        

            # convert the xticks to xticks list
            plt.xticks(np.arange(len(xticks)), xticks)

            #
            #print ("")
     

        plt.show()

    #
    def plot_overlap_volumes(self):


        # print the overlap volumes
        print ("polygon array: ", self.polygon_array)


        #
        import Geometry3D as g3d
        
        # plot 3d convex hull
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        polygons = []
        for polygon in self.polygon_array:
            
            # plto the polygon points in 3d
            ax.scatter( polygon[:, 0],
                        polygon[:,1],
                        polygon[:,2],
                        #s=self.scatter_size,
                        #color=self.clrs[ctr],
                        #marker = markers[ctr],
                        #label = self.behaviors[ctr] ,
                        alpha = 0.8,
                        # use black edges
                            edgecolors='k'
                        )
            


            # Compute the convex hull
            convex_hull = g3d.ConvexHull(polygon)

            # Get the vertices of the convex hull
            hull_vertices = convex_hull.get_vertices()

            pts = []
            for p in hull_vertices:
                pts.append(g3d.Point(p[0], p[1], p[2]))
       
            polygon1 = g3d.ConvexPolygon(pts)

            polygons.append(polygon1)

        #
        for k in range(len(polygons)):
            for p in range(len(polygons)):
                
                #
                inter = g3d.intersection(polygons[k],
                                     polygons[p])
                print("intersection: ", inter) # results I needed


















        # remove diagonal set to np.nan
        for i in range(len(self.overlap_volumes)):
            self.overlap_volumes[i][i] = np.nan

        plt.figure()

        # plot the overlap volumes as bar plots with unique values 
        # for each cohort
        for i in range(len(self.overlap_volumes)):
            #
            plt.bar(np.arange(len(self.overlap_volumes[i]))+i*0.2,
                    self.overlap_volumes[i],
                    width=0.2,
                    color=self.clrs[i],
                    label=self.behaviors[i])

        plt.show()


    def fit_gmm_pca(self):

        
        import matplotlib.pyplot as plt
        from sklearn.mixture import GaussianMixture


        # Step 2: Generate or load your 5D data
        # You should replace this with your actual data
        np.random.seed(0)
        data = np.vstack(self.stack)
        
        print (data.shape)

        # Create a PCA instance with 2 components
        #num_components = 3
        pca = PCA(n_components=self.n_dimensions)

        # Fit the PCA model and transform the data
        pca.fit(data)

        # transform the data
        data = pca.transform(data)
        print (data.shape)

        # Step 3: Fit a 2D Gaussian Mixture Model to the data
        n_components = 10  # You can choose the number of components
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data)

        # Step 4: Create a grid to visualize the GMM fit in 2D
        x, y = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 100),
                        np.linspace(data[:, 1].min(), data[:, 1].max(), 100))
        grid = np.c_[x.ravel(), y.ravel()]

        # Evaluate the GMM on the grid
        probs = gmm.score_samples(grid)
        probs = -np.exp(probs)
        probs = probs.reshape(x.shape)

        # Step 5: Plot the original data points and the GMM fit contours in 2D
        plt.figure(figsize=(8, 6))

        # Plot the data points
        plt.scatter(data[:, 0], data[:, 1], c='b', marker='o', label='Data Points')

        # Plot the GMM fit contours
        plt.contour(x, y, probs, levels=10, cmap='viridis')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Gaussian Mixture Model Fit')

        plt.legend()
        plt.show()
    #
    def run_umap(self):

        self.data = np.vstack(self.stack)


        # Create a UMAP model and fit it to your data
        umap_model = UMAP(n_neighbors=10,
                            min_dist=0.1, 
                        n_components=2)
        embedded_data = umap_model.fit_transform(self.data)

        #
        print (embedded_data.shape)
        markers = ['o','*','^']

        # Plot the first and second principal components
        fig = plt.figure(figsize=(10,10))

        # First Principal Component
        ax = fig.add_subplot(111)
        # First Principal Component
        for ctr,s in enumerate(self.stack):

            #
            for k in range(0,12,4):
                cc = [self.clrs[ctr]]*4
                #print (cc)
                ax.scatter( embedded_data[self.starts[ctr][0]+k:
                                    self.starts[ctr][0]+k+4, 0],
                        embedded_data[self.starts[ctr][0]+k:
                                self.starts[ctr][0]+k+4, 1],
                            s=350,
                        color=cc,
                        marker = markers[k//4],
                        label = self.behaviors[ctr] + " - c"+str(2+k//4)
                        )
        #
        #
        plt.legend()

        #
        plt.show()

#
def get_samples_from_kde(data1, 
                         data2,
                         n_samples = 100
                         ):
            
    #
    #print (data1.shape, data2.shape)

    #
    #num_samples = 100

    #
    data = data1.copy()
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit()  # Use "epa" for Epanechnikov kernel

    # Evaluate the estimated density at a set of values
    x = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)
    pdf = kde.evaluate(x)

    # get weighted samples from the KDE
    # Generate random samples from the estimated density
    num_samples = 1000
    samples1 = np.random.choice(x, size=num_samples, p=pdf/pdf.sum())
    
    #
    data = data2.copy()
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit()
    pdf = kde.evaluate(x)

    # get weighted samples from the KDE
    # Generate random samples from the estimated density
    samples2 = np.random.choice(x, size=num_samples, p=pdf/pdf.sum())

    
    return samples1, samples2

#
def replace_isolated_ones_with_zeros(temp,window_size):
    
    # This switches the first value if its followed by a 2+ sequence
    if False:
        start_idx = 1
        if temp[start_idx]!=temp[start_idx+1]:
            if temp[start_idx+1]==temp[start_idx+2]:
                temp[start_idx] = temp[start_idx+1]
    else:
        start_idx = window_size

    # rest of the values
    for k in range(start_idx,temp.shape[0]-1):
        #
        if temp[k]==temp[k-1]:
            continue
        elif temp[k]!=temp[k+1]:
            temp[k] = temp[k+1]

    # 
    return temp

     #
def generate_hourly_ethogram2(filename_in,
                              animal_id):

    #
    rectangle_ethogram = np.load(filename_in)
    print ("rectangle_ethogram: ", rectangle_ethogram.shape)

    #
    fname_out = os.path.join(os.path.split(filename_in)[0],
                            'hourly_ethogram_'+str(animal_id)+'.npy')
    
    #
    hourly_ethogram = np.zeros((16,24))
    
        #
    #print ("ethogram seconds: ", self.rectangle_ethogram.shape)
    
    # loop over each day per animals
    # make animal id based indexes into the array
    # use this if we have added empty rows
    if rectangle_ethogram.shape[0]>100:
        idxs = np.arange(1,rectangle_ethogram.shape[0],7)+(5-animal_id)
    else:
        idxs = np.arange(0,rectangle_ethogram.shape[0],6)+(5-animal_id)
    
    idxs = idxs[::-1]
    #print (idxs)

    ctr=0
    for idx in idxs:
        
        # print ("idx, temp.shpae: ", idx)
        temp = rectangle_ethogram[idx]
        # convert to boolean vals
        idx = np.where(temp>0)[0]
        temp[idx]=1                

        # split the data into 24 chunks
        day_total = 0
        for t in range(0,temp.shape[0],3600):
            temp_temp = temp[t:t+3600]
            #print ("idx, t, temp_temp, ctr: ", idx, t, temp_temp.shape, ctr)
            hourly_ethogram[ctr,t//3600] = np.nansum(temp_temp)
            
            #
            day_total += np.nansum(temp_temp)
            
        #
        #print ("anima: ", self.animal_id, " day: ", ctr, " daily total:", day_total)
        ctr+=1
        
    #
    np.save(fname_out,
            hourly_ethogram)