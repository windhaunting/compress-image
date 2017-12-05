#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:35:58 2017

@author: fubao
"""
from scipy import misc


def read_scene():
	data_x = misc.imread('../../Data/Scene/times_square.jpg')

	return (data_x)

def read_faces():
	nFaces = 100
	nDims = 2500
	data_x = np.empty((0, nDims), dtype=float)

	for i in np.arange(nFaces):
		data_x = np.vstack((data_x, np.reshape(misc.imread('../../Data/Faces/face_%s.png' % (i)), (1, nDims))))

	return (data_x)
