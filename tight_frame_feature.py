# -*- coding: utf-8 -*-

"""
Created on Thu Jul 20 11:01:15 2017

@author: ZhicongLiang
"""

from processing_data import *
import numpy as np
import pickle

def normalize(A):
    '''normalize a whole matrix by column'''
    A = np.array(A)
    A = A - A.mean(axis=0)
    A = A / A.std(axis=0)
    return A

def ft_img(image):
    '''calculting the 18 feature-images of every painting'''
    n,m = image.shape
    # use ft_img to store the feature-images
    ft_img = []
    #t is the size of the neighbour
    t = 2
    for k in range(18):
        fil = np.zeros((n-2*t,m-2*t))
        for i in range(t,n-t):
            for j in range(t,m-t):
                fil[i-t,j-t] = sum(sum(tao[k]*image[i-t:i+t+1:t,j-t:j+t+1:t]))
            # just to show the process
            if i%20==0:print(i,'row for',k,'ft_img')
        # store the feature-image in ft_img
        ft_img += [fil]
        del fil
    return ft_img

def stat_1(image):
    '''renturn the mean of the image'''
    return image.mean()

def stat_2(image):
    '''return the std of the image'''
    return np.std(image)

def stat_3(image):
    '''reutrn the precent of the tail'''
    n,m = image.shape
    mean = stat_1(image)
    std  = stat_2(image)
    count = 0
    for i in range(n):
        for j in range(m):
            if (abs(image[i,j]-mean)>std):
                count+=1
    return count/(m*n)

def feature_set(S):
    '''extracting 54 feature for every painting in a whole set,
    S = N,T,D'''
    for k in len(S):
        image = S[k]
        ft = ft_img(image)
        single_S = []
        for loop in range(18):
            st_1 = stat_1(ft[loop])
            st_2 = stat_2(ft[loop])
            st_3 = stat_3(ft[loop])
            single_S += [st_1,st_2,st_3]
        # save the 54 features
        pickle.dump(single_T,open('data/S_'+str(k)+'.p','wb'))


if __name__ == '__main__':
    
    '''loading the gray-scale and edge-truncated painting data'''
    raphael_D = pickle.load(open('data/truncated_edge_D.p','rb'))
    raphael_T = pickle.load(open('data/truncated_edge_T.p','rb'))
    raphael_N = pickle.load(open('data/truncated_edge_N.p','rb'))
    
    
    '''setting the 18 filters'''
    tao = []
    tao_0 = np.array([[1,2,1],      [2,4,2],    [1,2,1]])/16
    tao_1 = np.array([[1,0,-1],     [2,0,-2],   [1,0,-1]])/16
    tao_2 = np.array([[1,2,1],      [0,0,0],    [-1,-2,-1]])/16
    tao_3 = np.array([[1,1,0],      [1,0,-1],   [0,-1,-1]])*(2**0.5)/16
    tao_4 = np.array([[0,1,1],      [-1,0,1],   [-1,-1,0]])*(2**0.5)/16
    tao_5 = np.array([[1,0,-1],     [0,0,0],    [-1,0,1]])*(7**0.5)/24
    tao_6 = np.array([[-1,2,-1],    [-2,4,-2],  [-1,2,-1]])/48
    tao_7 = np.array([[-1,2,-1],    [2,4,2],    [-1,-2,-1]])/48
    tao_8 = np.array([[0,0,-1],     [0,2,0],    [-1,0,0]])/12
    tao_9 = np.array([[-1,0,0],     [0,2,0],    [0,0,-1]])/12
    tao_10 = np.array([[0,1,0],     [-1,0,-1],  [0,1,0]])*(2**0.5)/12
    tao_11 = np.array([[-1,0,1],    [2,0,-2],   [-1,0,1]])*(2**0.5)/16
    tao_12 = np.array([[-1,2,-1],   [0,0,0],    [1,-2,1]])*(2**0.5)/16
    tao_13 = np.array([[1,-2,1],    [-2,4,-2],  [1,-2,1]])/48
    tao_14 = np.array([[0,0,0],     [-1,2,-1],  [0,0,0]])*(2**0.5)/12
    tao_15 = np.array([[-1,2,-1],   [0,0,0],    [-1,2,-1]])*(2**0.5)/24
    tao_16 = np.array([[0,-1,0],    [0,2,0],    [0,-1,0]])*(2**0.5)/12
    tao_17 = np.array([[-1,0,-1],   [2,0,2],    [-1,0,-1]])*(2**0.5)/24
    tao += [tao_0,tao_1,tao_2,tao_3,tao_4,tao_5,tao_6,tao_7,tao_8,tao_9,\
            tao_10,tao_11,tao_12,tao_13,tao_14,tao_15,tao_16,tao_17]
    pickle.dump(tao,open('data/tao.p','wb'))
#    tao = pickle.load(open('gray/tao.p','rb'))

    '''extracting 54 feature for every painting set: N,T,D'''
    feature_set(T)
    feature_set(D)
    feature_set(N)
   
    ''' manually collecting feature of painting into set T,N,D'''
    T_2 = pickle.load(open('data/T_2.p','rb'))
    T_3 = pickle.load(open('data/T_3.p','rb'))
    T_4 = pickle.load(open('data/T_4.p','rb'))
    T_5 = pickle.load(open('data/T_5.p','rb'))
    T_6 = pickle.load(open('data/T_6.p','rb'))
    T_8 = pickle.load(open('data/T_8.p','rb'))
    T_9 = pickle.load(open('data/T_9.p','rb'))
    T_21 = pickle.load(open('data/T_21.p','rb'))
    T_22 = pickle.load(open('data/T_22.p','rb'))
    T_24 = pickle.load(open('data/T_24.p','rb'))
    T_27 = pickle.load(open('data/T_27.p','rb')) 
    T = []
    T += [T_2,T_3,T_4,T_5,T_6,T_8,T_9,T_21,T_22,T_24,T_27]
    
    D_1 = pickle.load(open('data/D_1.p','rb'))
    D_7 = pickle.load(open('data/D_7.p','rb'))
    D_10 = pickle.load(open('data/D_10.p','rb'))
    D_20 = pickle.load(open('data/D_20.p','rb'))
    D_23 = pickle.load(open('data/D_23.p','rb'))
    D_25 = pickle.load(open('data/D_25.p','rb'))
    D_26 = pickle.load(open('data/D_26.p','rb'))
    D = []
    D += [D_1,D_7,D_10,D_20,D_23,D_25,D_26]
    
    N_11 = pickle.load(open('data/N_11.p','rb'))
    N_12 = pickle.load(open('data/N_12.p','rb'))
    N_13 = pickle.load(open('data/N_13.p','rb'))
    N_14 = pickle.load(open('data/N_14.p','rb'))
    N_15 = pickle.load(open('data/N_15.p','rb'))
    N_16 = pickle.load(open('data/N_16.p','rb'))
    N_17 = pickle.load(open('data/N_17.p','rb'))
    N_18 = pickle.load(open('data/N_18.p','rb'))
    N_19 = pickle.load(open('data/N_19.p','rb'))
    N = []
    N += [N_11,N_12,N_13,N_14,N_15,N_16,N_17,N_18,N_19]
    
    pickle.dump(T,open('data/T.p','wb'))
    pickle.dump(N,open('data/N.p','wb'))
    pickle.dump(D,open('data/D.p','wb'))
