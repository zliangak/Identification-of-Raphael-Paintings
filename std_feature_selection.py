# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:46:59 2017

@author: ZhicongLiang
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def scale(A):
    A = np.array(A)
    tt = []
    for i in range(A.shape[1]):
        tt += [max([abs(i) for i in A[:,i]])]
    A = A/tt
    return A


def load():
    N = pickle.load(open('data/tight_frame_N.p','rb'))
    T = pickle.load(open('data/tight_frame_T.p','rb'))
    scale_N = scale(N)
    scale_T = scale(T)
    lab_T = np.column_stack(([1 for i in range(scale_T.shape[0])],scale_T))
    lab_N = np.column_stack(([0 for i in range(scale_N.shape[0])],scale_N))
    G = np.row_stack((lab_T,lab_N))
    cent_T = scale_T.mean(axis=0)
    return G,scale_T,scale_N,cent_T

def ft_selection(ft_num):
    
    std_T = scale_T.std(axis=0)
    std_N = scale_N.std(axis=0)
    std_ratio = std_T/std_N
    
    ft_slt = []
    std = []
    for i in range(ft_num):
        idx = np.where(std_ratio==std_ratio.min())
        ft_slt += [int(idx[0])]
        std += [std_ratio.min()]
        std_ratio[idx[0]] = std_ratio.max()+1
    return ft_slt,std

def distance(x):
    dst = []
    for i in range(x.shape[0]):
        diff = (x[i,ft_slt] - cent_T[ft_slt])*weights
        dst+= [sum([j**2 for j in diff])]
    return dst

def thsd(train_x,train_y):
    acc = []
    dst = distance(train_x)
    for i in range(len(dst)):
        p = dst[i]
        count = 0
        for j in range(len(dst)):
            if (dst[j]<=p and train_y[j]==1):count+=1
            if (dst[j]>p and train_y[j]==0):count+=1 
        acc += [count/len(train_y)]
    idx = acc.index(max(acc))
    return dst[idx]
    
def validate():
    '''print out the leave-one-out cross validationg accuracy
    of model with feature number:ft_num'''
    count = 0
    for k in range(len(y)):
        index = [j for j in range(len(y)) if j!=k]
        x_train = x[index,:]
        y_train = y[index]
        p = thsd(x_train,y_train)
        diff = (x[k,ft_slt] - cent_T[ft_slt])*weights
        if (sum([t**2 for t in diff])<=p and y[k]==1):count+=1
        elif (sum([t**2 for t in diff])>p and y[k]==0):count+=1
        else:print(k)
    print('cv_acc=',count/len(y))

def dpt_predict():
    '''return the prediction of the diputed paintings'''
    D = pickle.load(open('data/tight_frame_D.p','rb'))
    scale_D = scale(D)
    p = thsd(x,y)
    pred = []
    for k in range(scale_D.shape[0]):
        diff = (scale_D[k,ft_slt] - cent_T[ft_slt])*weights
        if (sum([t**2 for t in diff])<=p):pred+=[1]
        if (sum([t**2 for t in diff])>p):pred+=[0]
    return pred

def plot():
    '''ploting the scatter figure with the first 2 features
    according to the graph, we can see that there are 4 points
    mis-classifed'''
    if ft_num!=2:
        return
    p = thsd(x,y)
    plt.axis('equal')
    plt.scatter(x[0:11,ft_slt[0]],x[0:11,ft_slt[1]],c='r')
    plt.scatter(x[11:,ft_slt[0]],x[11:,ft_slt[1]],c='b')
    a,b = cent_T[ft_slt]
    r = np.sqrt(p)
    theta = np.arange(0, 2*np.pi, 0.01)
    u = a + r * np.cos(theta)/weights[0]
    v = b + r * np.sin(theta)/weights[1]
    plt.scatter(u,v,marker='.')
    plt.savefig('2D_scale.jpg')
    return
    
    

if __name__ == '__main__':
    ft_num = 2
    G,scale_T,scale_N,cent_T = load()
    x = G[:,1:]
    y = G[:,0]
    ft_slt,std = ft_selection(ft_num)
    weights = [np.exp(-i) for i in range(ft_num)]
    validate()
    pred = dpt_predict()
    plot()

    