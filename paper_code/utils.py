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
    def get_dev_plots(self):


        #

        # 
        labels = np.arange(16,30,1)

        if self.show_plots:
            plt.figure()
        
        #
        dev_changes = np.zeros((len(self.behaviors),14))
        dev_stable = np.zeros((len(self.behaviors),14))
        for k in trange(len(self.behaviors)):

            #

            #
            self.behavior_id = k
            t1_m = self.get_sliding_window_dists()
            t = np.arange(t1_m.shape[0])

            # plot average of the behavior also
            mean = np.mean(self.data[self.behavior_id],0)
            std = np.std(self.data[self.behavior_id],0)

            #
            idx1 = np.where(t1_m<self.pval_thresh)[0]
            dev_changes[k,idx1] = 1

            #
            idx2 = np.where(t1_m>self.pval_thresh)[0]
            dev_stable[k,idx2] = 1

            if self.show_plots and k<4:

                plt.subplot(2,2,k+1)
                plt.plot(mean, c='black',label='beahvior')
                plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=.2)
                plt.title(self.behaviors[self.behavior_id])

                # plot t1_m and make it bold where it is larger thatn 0.05
                plt.plot(t,t1_m, label = 't_test - pvalue')
    

                # plot a thick line over idx
                plt.scatter(t[idx1], mean[idx1], linewidth=5, 
                            c='red',
                            label='developmental stages')


                # plot a thick line over idx
                plt.scatter(t[idx2], mean[idx2], linewidth=5, 
                            c='blue',
                            label='stabilizing stages')

                plt.xticks(np.arange(14),labels)

                plt.xlabel("dev PDay")
                plt.ylabel("kl divergence in sequential windows")

                plt.legend()
        #
        plt.suptitle("Automatic detection of developmental stages")

        #
        plt.show()

        self.dev_changes = dev_changes
        self.dev_stable = dev_stable

    #
    def compute_dev_stages_histogram(self):

        # plot the two plots
        plt.figure()

        #
        plt.subplot(211)
        plt.imshow(self.dev_changes, 
                   aspect='auto',
                   # use extent to go from 0 to # of behaviors and then up to 14 days on x axis
                   extent = [0,14,0,len(self.behaviors)],
                   #cmap = 'binary'
                   )
        
        # label the y axis using the behavior names
        plt.yticks(np.arange(len(self.behaviors))+0.5, 
                   self.behaviors[::-1],
                   rotation=45,
                   fontsize=8)

        
        plt.title("dev changes")
        #plt.legend()

        # change xticks to start at 16
        labels = np.arange(16,30,1)
        plt.xticks(np.arange(14)+0.5,labels)

        plt.xlabel("dev PDay")

        # 
        plt.subplot(212)

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
        t1_window = np.arange(0,12,1)
        t1_array = np.zeros(14)+np.nan

        for t1 in t1_window:

            # grab chunk 1 of data
            temp1 = self.data[self.behavior_id][:,t1:t1+self.sliding_window_size].copy()
            temp2 = self.data[self.behavior_id][:,t1+self.sliding_window_size:t1+2*self.sliding_window_size].copy()

            # flatten each chunk 
            temp1 = temp1.flatten()
            temp2 = temp2.flatten()


            #
            if self.dist_method == 'kl_div':
                # Perform kernel density estimation (KDE) 
                #samples1, samples2 = get_samples_from_kde(temp1, temp2, self.n_samples_kde)
                kl_divergence1 = entropy(temp1, temp2)
                t1_array[t1+self.sliding_window_size] = kl_divergence1
                #
            elif self.dist_method =='t_test':

                # Perform an independent two-sample t-test
                t_stat, p_value = stats.ttest_ind(temp1, temp2)
               # print (t1, t2, p_value)

                # Check the p-value to determine if the means are significantly different
                t1_array[t1+self.sliding_window_size] = p_value
            elif self.dist_method == '2sample_ks_test':
                
                # Perform a 2 sample ks test using scipy
                ks_ = ks_2samp(temp1, temp2)
                t1_array[t1+self.sliding_window_size] = ks_[1]

                
        

        #
        return t1_array
    
    #
    def load_data(self):

        #
        flip_behaviors = [

            'food',
            'water',
            'pup_male_food',
            'pup_male_water',
            'pup_female_food',
            'pup_female_water',
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
            mean = np.mean(s,axis=0)
            std = np.std(s,axis=0)

            plt.plot(mean, 
                    label=self.behaviors[ctr],
                    c=self.clrs[ctr],
                    linewidth=5)

            #
            ctr+=1

        # plot a horizontal line at y = 1.0
        # plt.plot([0,13],[1.0,1.0],'--',
        #         c='black',
        #         linewidth=4)

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