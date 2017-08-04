# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:15:29 2017

@author: ZhicongLiang
"""

import numpy as np
from tight_frame_feature_selection import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tight_frame_neural_net import *

def neural(k,dspt=False):
    '''t is the threshold when of re-classified a painting in Neural network'''
    t = 0.7
    if dspt == False:
        index = [j for j in range(len(y)) if j!=k]
        x_train = neural_x[index,:]
        y_train = neural_y[index]
        model = fit(x_train,y_train)
        prob = model.predict(neural_x[k,:].reshape(1,neural_x.shape[1]))
        print(k,':',1 - prob)
        pred = threshold(prob,1-t)[0][0]
        print(pred)
        del model
    else:
        model = fit(neural_x,neural_y)
        prob = model.predict(nol_D[k,:].reshape(1,nol_D.shape[1]))
        print(k,':',1-prob)
        pred =threshold(prob,1-t)[0][0]
        print(pred)
    return pred

def distance(x):
    dst = []
    for i in range(x.shape[0]):
        dst+= [sum([j**2 for j in x[i,:]])]
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
        dst = 0
        index = [j for j in range(len(y)) if j!=k]
        x_train = x[index,:]
        y_train = y[index]
        p = thsd(x_train,y_train)
        dst  = sum([t**2 for t in x[k,:]])
        if (dst>p):
            '''if a painting is predict as non-raphael by disatance dsicrimiant
            analysis, we would predict it again in neural network
            If it has a probability large than 80%, than we predict it as genuine'''
            pred = neural(k)
            if pred==1:
                dst=0         
        if (dst<=p and y[k]==1):
            count+=1
            print(k,'times : count=',count)
        elif (dst>p and y[k]==0):
            count+=1
            print(k,'times : count=',count)
    cv_acc= count/len(y)
    print('cv_acc=',cv_acc)
    return cv_acc
    
    
def dpt_predict():
    '''return the prediction of the diputed paintings'''
    test_D = nol_D[:,ft_slt]
    p = thsd(x,y)
    pred = []
    for k in range(test_D.shape[0]):
        dst = sum([t**2 for t in test_D[k,:]]) 
        if dst>p:
            '''use the dspt mode, which means that our neural network will
            train on all the 20 known samples instead of some 19 of them'''
            re_pred = neural(k,dspt=True)
            if re_pred==1:
                dst=0 
        if (dst<p):pred+=[1]
        if (dst>=p):pred+=[0]
    return pred

def plot(ft_num=3):
    if ft_num==3:
        '''plot the scatter figure with the first 3 features
        red points represent genuine paintings and blue for non-raphael'''
        # ploting the 3D figure    
        ax = plt.figure()
        ax = ax.add_subplot(111,projection='3d')
        xs_T = x[0:11,0]
        ys_T = x[0:11,1]
        zs_T = x[0:11,2]
        ax.scatter(xs_T,ys_T,zs_T,c='r',marker='^')

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
    #ft_slt = [6,33,2,12,39]
    ft_num = 3
    ft_slt = ft_selection(ft_num)  
    x = G[:,1:]  # G comes from the file: tight_frame_feature_selection
    x = x[:,ft_slt]
    y = G[:,0]
    neural_x = G[:,1:]
    neural_y = G[:,0]
    D = pickle.load(open('data/tight_frame_D.p','rb'))
    nol_D = normalize(D)
    plot(ft_num)
    '''repeat the process 10 time and calculate the average'''
    cr_acc = []
    predict = np.zeros(len(D))
    num_iter = 20
    for i in range(num_iter):
        print('----------------the',i,'time---------------')
        cr_acc += [validate()]
        predict = np.add(predict,dpt_predict())
    avg_pred = predict/num_iter
    avg_cr_acc = np.mean(cr_acc)
    