# Import modules

import numpy as np
from scipy import misc
import math

from files import read_faces
from files import read_scene

from plotting import plottingImagesPCA

from sklearn.cluster import KMeans
from plotting import plottingImagesKMean
from plotting import plottingElbowKMean

from scipy.spatial.distance import cdist


def getPCAImage():
    
    print('Implement PCA here ...')
    data_x = read_faces()
    print('X = ', data_x.shape)

    #do we need to center the data?
    covMat = np.cov(data_x.T)
    #print('covMat = ', covMat.shape)
    
    w, v = np.linalg.eigh(covMat)
    print('w, v = ', w.shape, v.shape)

    kLst = [3, 5, 10, 30, 50, 100, 150, 300]    # [3, 5, 10, 30, 50, 100, 150, 300]  #k largest eigenvalues
    ArrayLst = []
    for k in kLst:
        #visually inspect face 
        #get k eigenvectors vK corresponding to largest k eigenvalues  
        ncol = v.shape[1]
        vk = v[:, ncol-k: ncol+1]
        print('vk = ', vk.shape)
        #second row  reconstructed
        '''
        X1Recon = np.dot(np.dot(data_x[1, :], vk), vk.T)
        print('x1, X1Recon = ', data_x[1, :].shape, X1Recon.shape[0], math.sqrt(X1Recon.shape[0]), data_x[1, :], X1Recon)
    
        misc.imsave('../Figures/x1reconstruct_k'  + str(k) + '.jpg', np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0])))))
        '''
    
        XRecon = np.dot(np.dot(data_x, vk), vk.T)
        
        #get averagesquared reconstruction error
        errorK = math.sqrt(np.mean(data_x-XRecon))
        print ("reconstruction error for different k ", k, errorK)
        #memory consumed by XRecon
        compressionRate = (np.dot(data_x, vk).nbytes + vk.nbytes)/data_x.nbytes
        print ("compression Rate for different k ", k, compressionRate)

        X1Recon = XRecon[1, :]
        #print('x1, X1Recon = ', data_x[1, :].shape, X1Recon.shape[0], math.sqrt(X1Recon.shape[0]), data_x[1, :], X1Recon)
    
        ArrayLst.append(X1Recon)
        #misc.imsave('../Figures/x1reconstruct_ka'  + str(k) + '.jpg', np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0])))))
    
    plottingImagesPCA(kLst, ArrayLst, "../Figures/x1reconstructImages", "Face")

def KMeanCompress():
    
    print('Implement k-means here ...')
    data_x = read_scene()
    print('X = ', data_x.shape)
    
    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    print('Flattened image = ', flattened_image.shape)

    kLst = [2, 5, 10, 25, 50, 75, 100, 200]         #k clusters

    ArrayLst = []

    sumSquareErros = []
    for k in kLst:
        kmeans = KMeans(n_clusters = k, random_state=0).fit(flattened_image)
        clustCenter = kmeans.cluster_centers_
        labels = kmeans.labels_
        #print('labels dim = ', k, clustCenter.shape, labels.shape)
        #sqError = sum(np.min(cdist(flattened_image, clustCenter, 'euclidean'), axis=1))
        #print('sqError = ', k, sqError, math.log(sqError, 10))

        #sumSquareErros.append(math.log(sqError))
    
        #replace its data point with its centroid data value where it belong to
        flattenedImagesReconstructed = np.zeros(flattened_image.shape)        
        for i in np.arange(flattened_image.shape[0]):
            flattenedImagesReconstructed[i] = clustCenter[labels[i]]
        
        reconstructed_image = flattenedImagesReconstructed.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
        print('Reconstructed image = ', k, reconstructed_image.shape)
        
        
        errorK = math.sqrt(np.mean(abs(data_x-reconstructed_image)))
        print ("reconstruction error for different k ", k, errorK)
        
        compressionRate = (k*3*32+reconstructed_image.shape[0]*3*math.ceil(math.log(k, 2)))/(data_x.shape[0]*data_x.shape[1]*24)
        print ("compression Rate for different k ", k, compressionRate)
        
        
        ArrayLst.append(reconstructed_image)
        

    plottingImagesKMean(kLst, ArrayLst, "../Figures/TimeSquarereconstructImages", "Times_square", data_x.shape)
   # plottingElbowKMean(kLst, sumSquareErros, "../Figures/TimeSquareKMeansElbow", "Elbow plot of kMeans on Time_square image")

if __name__ == '__main__':
	
	################################################
	# PCA
    #getPCAImage()
	
	################################################
	# K-Means
    KMeanCompress()
   
