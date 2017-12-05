# Import modules

import numpy as np

from files import read_faces
from files import read_scene


def getPCAImage():
    data_x = read_faces()
    print('X = ', data_x.shape)

    print('Implement PCA here ...')
    
    
    
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

