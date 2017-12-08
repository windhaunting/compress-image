# Import modules

import numpy as np
from scipy import misc
import math

from files import read_faces
from files import read_scene
from plotting import plottingPCAImage

def getPCAImage():
    data_x = read_faces()
    print('X = ', data_x.shape)

    print('Implement PCA here ...')
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
        #print ("error for different k ", k, errorK)
        #memory consumed by XRecon
        compressionRate = (np.dot(data_x, vk).nbytes + vk.nbytes)/data_x.nbytes
        
        print ("compressionRate for different k ", k, compressionRate)

        X1Recon = XRecon[1, :]
        #print('x1, X1Recon = ', data_x[1, :].shape, X1Recon.shape[0], math.sqrt(X1Recon.shape[0]), data_x[1, :], X1Recon)
    
        ArrayLst.append(X1Recon)
        #misc.imsave('../Figures/x1reconstruct_ka'  + str(k) + '.jpg', np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0])))))
    
    plottingPCAImage(kLst, ArrayLst, "../Figures/x1reconstructImages")

def KMeanCompress():
    
    data_x = read_scene()
    print('X = ', data_x.shape)

    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    print('Flattened image = ', flattened_image.shape)

    print('Implement k-means here ...')

    reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    print('Reconstructed image = ', reconstructed_image.shape)
    

if __name__ == '__main__':
	
	################################################
	# PCA
    getPCAImage()
	
	
	################################################
    
	# K-Means
    KMeanCompress()
   
