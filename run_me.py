# Import modules

import numpy as np

from files import read_faces
from files import read_scene


def getPCAImage():
    data_x = read_faces()
    print('X = ', data_x.shape)

    print('Implement PCA here ...')
    #do we need to center the data?
    
    covMat = np.cov(data_x.T)
    print('covMat = ', covMat.shape)
    
    w, v = np.linalg.eigh(covMat)
    print('w, v = ', w.shape, v.shape,  v[:, -1])
    
    k = 3    # [3, 5, 10, 30, 50, 100, 150, 300]  #k largest eigenvalues
    #visually inspect face 
    #get k eigenvectors vK corresponding to largest k eigenvalues  
    ncol = v.shape[1]
    vk = v[:, ncol-k: ncol+1]
    print('vk = ', vk.shape, vk)
    #second row  reconstructed
    X1Recon = np.dot(np.dot(data_x[1, :], vk), vk.T)
    print('x1, X1Recon = ', data_x[1, :], X1Recon)
    
if __name__ == '__main__':
	
	################################################
	# PCA
    getPCAImage()
	
	
	################################################
    
	# K-Means

    data_x = read_scene()
    print('X = ', data_x.shape)

    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    print('Flattened image = ', flattened_image.shape)

    print('Implement k-means here ...')

    reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    print('Reconstructed image = ', reconstructed_image.shape)

