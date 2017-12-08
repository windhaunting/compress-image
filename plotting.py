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
from scipy import misc



def plottingPCAImage(kLst, ArrayLst, outImagePdf):
    plt.figure(1, figsize=(6,4))  # figsize=(6,4))

    subPlotNums = len(ArrayLst)
    
    for i in range(331, subPlotNums+331):

        plt.subplot(i)
        X1Recon = ArrayLst[i-331]
        X1Recon = np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0]))))
        #misc.imsave('../Figures/x1reconstruct_ka'  + str(kLst[i-331]) + '.jpg', np.reshape(X1Recon, (int(math.sqrt(X1Recon.shape[0])), int(math.sqrt(X1Recon.shape[0])))))
        plt.imshow(X1Recon, interpolation='nearest', cmap='gray')

        plt.xlabel("k: " + str(kLst[i-331]))
        plt.ylabel("Face reconstructed")
        plt.title("face 1 (k= " + str(kLst[i-331]) + ")")
        
    plt.savefig(outImagePdf + ".pdf")