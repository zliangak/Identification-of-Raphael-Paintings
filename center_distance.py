# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:20:51 2017

@author: ZhicongLiang
"""

import numpy as np
from feature_selection import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def distance(x):
    '''return a square distance of 
    a particular sample'''
    dst = []
    for i in range(x.shape[0]):
        dst+= [sum([j**2 for j in x[i,:]])]
    return dst

def thsd(train_x,train_y):
    '''Given training data train_x and its label train_y,
    return the best threshold(along with the highest classification accuracy)
    of this training set'''
    acc = []
    dst = distance(train_x)
    for i in range(len(dst)):
        p = dst[i]
        # count is the number of accurate classified sample
        count = 0
        for j in range(len(dst)):
            if (dst[j]<=p and train_y[j]==1):count+=1
            if (dst[j]>p and train_y[j]==0):count+=1 
        acc += [count/len(train_y)]
    # idx is the index of the threshold with best accuracy
    idx = acc.index(max(acc))
    return dst[idx]

def validate(ft_num=3):
    '''print out the leave-one-out cross validationg accuracy
    of model with feature number:ft_num'''
    count = 0
    for k in range(len(y)):
        index = [j for j in range(len(y)) if j!=k]
        x_train = x[index,:]
        y_train = y[index]
        p = thsd(x_train,y_train)
        if (sum([t**2 for t in x[k,:]])<=p and y[k]==1):count+=1
        if (sum([t**2 for t in x[k,:]])>p and y[k]==0):count+=1
    print('cv_acc=',count/len(y))
    
    
def dpt_predict():
    '''return the prediction of the diputed paintings'''
    D = pickle.load(open('data/tight_frame_D.p','rb'))
    nol_D = normalize(D)
    # ft_slt is a list of features we choose
    test_D = nol_D[:,ft_slt]
    p = thsd(x,y)
    pred = []
    for k in range(test_D.shape[0]):
        if (sum([t**2 for t in test_D[k,:]])<p):pred+=[1]
        if (sum([t**2 for t in test_D[k,:]])>=p):pred+=[0]
    return pred

def plot(ft_num=3):
    '''plot the 3D or 2D figure of the all the sample'''
    if ft_num==3:
        '''plot the scatter figure with the 3 feature we choose
        red points represent genuine paintings and blue for non-raphael'''
        # ploting the 3D figure    
        ax = plt.figure()
        ax = ax.add_subplot(111,projection='3d')
        # the first 11 paintings are known as Raphael
        xs_T = x[0:11,0]
        ys_T = x[0:11,1]
        zs_T = x[0:11,2]
        ax.scatter(xs_T,ys_T,zs_T,c='r',marker='^')
        # the remmaining paintings are non-raphael's
        xs_N = x[11:,0]
        ys_N = x[11:,1]
        zs_N = x[11:,2]
        ax.scatter(xs_N,ys_N,zs_N,c='b',marker='o')
        plt.savefig('3D.jpg')
    elif ft_num==2:
        '''ploting the scatter figure with the first 2 features
        according to the graph, we can see that there are 4 points
        mis-classifed'''
        p = thsd(x,y)
        plt.scatter(x[0:11,0],x[0:11,1],c='r')
        plt.scatter(x[11:,0],x[11:,1],c='b')
        cent_T = x.mean(axis=0)
        # plot the classification circle
        a,b = cent_T
        r = np.sqrt(p)
        theta = np.arange(0, 2*np.pi, 0.01)
        u = a + r * np.cos(theta)
        v = b + r * np.sin(theta)
        plt.scatter(u,v,marker='o')
        plt.savefig('2D.jpg')
    else:
        print('Oops:Dimension larger than 3!')
    return

        
    

if __name__ == '__main__':    
    #ft_slt = [6,33,2], the three features we choose
    ft_num = 3
    ft_slt = ft_selection(ft_num) 
    # G comes from the file: tight_frame_feature_selection
    # it contains all the 20 sample and the first column is
    # their label
    x = G[:,1:]  
    x = x[:,ft_slt]
    y = G[:,0]
    validate(ft_num)
    predict = dpt_predict()
    plot(ft_num)
 







    
