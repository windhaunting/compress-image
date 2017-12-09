#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:57:50 2017

@author: fubao
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt



def plottingImagesPCA(kLst, ArrayLst, outImagePdf, titlePart):
    plt.figure(1, figsize=(6,4))  # figsize=(6,4))

    subPlotNums = len(ArrayLst)
    plt.gray()
    for i in range(331, subPlotNums+331):

        plt.subplot(i)
        X1Recon = ArrayLst[i-331]
        X1Recon = np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0]))))
        #misc.imsave('../Figures/x1reconstruct_ka'  + str(kLst[i-331]) + '.jpg', np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0])))))
        plt.imshow(X1Recon, interpolation='nearest', cmap='gray')

        #plt.xlabel("k: " + str(kLst[i-331]))
        #plt.ylabel(ylabel)
        plt.title("k= " + str(kLst[i-331]))
        
    plt.suptitle(titlePart + "reconstruction image with kMeans")
    plt.savefig(outImagePdf + ".pdf")
    
    

def plottingImagesKMean(kLst, ArrayLst, outImagePdf, titlePart, dataXShape):

    plt.figure(2, figsize=(8,9)) # figsize=(6,4))

    subPlotNums = len(ArrayLst)
    plt.gray()
    for i in range(331, subPlotNums+331):

        plt.subplot(i)
        reconImage = ArrayLst[i-331]
        reconImage = reconImage.ravel().reshape(dataXShape)
        #misc.imsave('../Figures/x1reconstruct_ka'  + str(kLst[i-331]) + '.jpg', np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0])))))
        plt.imshow(reconImage.convert('L'), interpolation='nearest', cmap=plt.get_cmap('gray'))

        #plt.xlabel("k: " + str(kLst[i-331]))
        #plt.ylabel(ylabel)
        plt.title("k= " + str(kLst[i-331]))
        
    plt.suptitle(titlePart + "reconstruction image with kMeans")
    plt.savefig(outImagePdf + ".pdf")
    

def plottingElbowKMean(kLst, sumSquareErros, outImagePdf, titlePart):
    
    plt.figure(3, figsize=(6, 8))  # figsize=(6,4))
    plt.plot(kLst, sumSquareErros, 'bx-')
    plt.xlabel("k value")
    plt.ylabel("Sum of squared error")
    plt.title(titlePart)
    plt.savefig(outImagePdf + ".pdf")
