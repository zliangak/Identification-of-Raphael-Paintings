# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:22:27 2017

@author: ZhicongLiang
"""

import numpy as np
import pickle
from processing_data import *
import matplotlib.pyplot as plt

def dist(F,g):
    '''return the distance from the center 0 of g, given F'''
    g = g[1:]
    g = g[F]
    dst = sum([i**2 for i in g])
    return dst


def ROC(F):
    '''Given list F (index of feature that we choose)
    return the TPR and FPR of different threshold p'''
    dst = []
    step = 500
    TPR = []
    FPR = []
    T = sum(G[:,0])
    N = G.shape[0] - T
    
    for g in G:
        dst += [dist(F,g)]
    top   = max(dst)
    floor = min(dst)
    p = np.linspace(floor,top,step)
    
    for k in range(step):
        TP = 0
        FP = 0
        for i in range(len(dst)):
            if (dst[i]<=p[k] and G[i][0]==1):TP += 1
            if (dst[i]<=p[k] and G[i][0]==0):FP += 1
        TPR += [TP/T]
        FPR += [FP/N]
#    plt.plot(FPR,TPR)
    return TPR,FPR
    

def AUC(F):
    '''return the area under the ROC curve of given F'''
    TPR,FPR = ROC(F)
    area = 0
    for i in range(1,len(FPR)):
        area += (FPR[i]-FPR[i-1])*TPR[i]
    return area
    
def ft_selection(ft_num):
    '''return the index of feature we choose
    ft_num means that number of feature we want'''
    ft_slt = []
    for k in range(ft_num):
        print('seletcting',k,'-th feature')
        area = []
        for j in range(54):
            if (j not in ft_slt):
                F = ft_slt + [j]
                area += [AUC(F)]
            else:
                area += [-1]
        idx = area.index(max(area))
        ft_slt += [idx]
    return ft_slt


N = pickle.load(open('data/tight_frame_N.p','rb'))
T = pickle.load(open('data/tight_frame_T.p','rb'))
    
'''centre of the genuine painting is 0, 
since we have normalize the data'''
nol_N = normalize(N)
nol_T = normalize(T)

'''adding label to the first row'''
lab_T = np.column_stack(([1 for i in range(nol_T.shape[0])],nol_T))
lab_N = np.column_stack(([0 for i in range(nol_N.shape[0])],nol_N))
G = np.row_stack((lab_T,lab_N))

del N,T,lab_T,lab_N,nol_N,nol_T



            