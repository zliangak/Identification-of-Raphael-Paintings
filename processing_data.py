# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:13:25 2017

@author: ZhicongLiang
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pickle

def normalize(A):
    '''normalize the features'''
    A = np.array(A)
    A = A - A.mean(axis=0)
    A = A / A.std(axis=0)
    return A



def tran_gray(image):
    '''transforming image to grayscale'''
    image_1 = np.array(image,dtype=int)
    image_2 = np.zeros(image_1.shape[0:2])
    image_2 = image_1[:,:,0]*299+image_1[:,:,1]*587+image_1[:,:,2]*114
    image_2 = image_2//1000
    return image_2

def show_gray(image):
    '''show grayscale image'''
    plt.imshow(image,cmap='Greys_r')
    return
  

def truncate(image_1):
    '''cutting the eages'''
    '''n  is  the num of pixel cutting from each eage'''
    n = 100
    i,j = image_1.shape
    image_2 = image_1[n:i-n,n:j-n]
    return image_2





if __name__=='__main__':
    '''laooding data manually,since they are of differnet forms(tif,tiff,jpg)
     D means the disputed data
     N means non-raphael data
     T mean Raphael's data
     '''
    raphael_D = []
    raphael_1 = mpimg.imread('painting/1.D.tif')
    raphael_7 = mpimg.imread('painting/7.D.tiff')
    raphael_10 = mpimg.imread('painting/10.D.tif')
    raphael_20 = mpimg.imread('painting/20.D.tif')
    raphael_23 = mpimg.imread('painting/23.D.tif')
    raphael_25 = mpimg.imread('painting/25.D.tif')
    raphael_26 = mpimg.imread('painting/26.D.tif')
    raphael_D = raphael_D + [raphael_1,raphael_7,raphael_10,raphael_20,\
                         raphael_23,raphael_25,raphael_26]
    print('laoding raphael_D done!')

    raphael_T = []
    raphael_2 = mpimg.imread('painting/2.T.tif')
    raphael_3 = mpimg.imread('painting/3.T.tif')
    raphael_4 = mpimg.imread('painting/4.T.tiff')
    raphael_5 = mpimg.imread('painting/5.T.tiff')
    raphael_6 = mpimg.imread('painting/6.T.tiff')
    raphael_8 = mpimg.imread('painting/8.T.tif')
    raphael_9 = mpimg.imread('painting/9.T.tif')
    raphael_21 = mpimg.imread('painting/21.T.jpg')
    raphael_22 = mpimg.imread('painting/22.T.jpg')
    raphael_24 = mpimg.imread('painting/24.T.tif')
    raphael_27 = mpimg.imread('painting/27.T.tiff')
    raphael_T = raphael_T + [raphael_2,raphael_3,raphael_4,raphael_5,\
                         raphael_6,raphael_8,raphael_9,raphael_21,\
                         raphael_22,raphael_24,raphael_27]
    print('laoding raphael_T done!')


    raphael_N = []
    raphael_11 = mpimg.imread('painting/11.N.jpg')
    raphael_12 = mpimg.imread('painting/12.N.jpg')
    raphael_13 = mpimg.imread('painting/13.N.jpg')
    raphael_14 = mpimg.imread('painting/14.N.jpg')
    raphael_15 = mpimg.imread('painting/15.N.jpg')
    raphael_16 = mpimg.imread('painting/16.N.jpg')
    raphael_17 = mpimg.imread('painting/17.N.jpg')
    raphael_18 = mpimg.imread('painting/18.N.jpg')
    raphael_19 = mpimg.imread('painting/19.N.jpg')
    raphael_N = raphael_N + [raphael_11,raphael_12,raphael_13,raphael_14,\
                         raphael_15,raphael_16,raphael_17,raphael_18,\
                         raphael_19]
    print('laoding raphael_N done!')


    ''' transforming painting into grayscale'''
    raphael_D_2 = []
    for i in raphael_D:
        raphael_D_2 += [tran_gray(i)]  
        
    raphael_N_2 = []
    for j in raphael_N:
        raphael_N_2 += [tran_gray(j)]
        
    raphael_T_2 = []
    for k in raphael_T:
        raphael_T_2 += [tran_gray(k)]
    print('transformation done!')

    ''' truncating the edges of the gray scale paintings'''
    raphael_D_3 = []
    for i in raphael_D_2:
        raphael_D_3 += [truncate(i)]
    
    raphael_N_3 = []
    for j in raphael_N_2:
        raphael_N_3 += [truncate(j)]

    raphael_T_3 = []
    for k in raphael_T_2:
        raphael_T_3 += [truncate(k)]
    print('truncation done!')
    
    '''21.T need special truncation'''
    image = raphael_T_3[7]
    show_gray(image)
    i,j = image.shape
    image_2 = image[100:i-100,100:j-100]
    show_gray(image_2)
    raphael_T_3[7] = image_2
    
    '''17.N need special truncation'''
    image = raphael_N_3[6]
    show_gray(image)
    i,j = image.shape
    image_2 = image[1020:i-1020,80:j-80]
    show_gray(image_2)
    raphael_N_3[6] = image_2

    # saving the processed data:
    pickle.dump(raphael_D_3,open('data/truncated_edge_D.p','wb'))
    pickle.dump(raphael_N_3,open('data/truncated_edge_N.p','wb'))
    pickle.dump(raphael_T_3,open('data/truncated_edge_T.p','wb'))


    



