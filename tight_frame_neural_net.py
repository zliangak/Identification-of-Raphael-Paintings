# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:36:59 2017

@author: ZhicongLiang
"""

import numpy as np
import pickle
from tight_frame_feature import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

def load():
    ''' loading data and transform to array'''
    D = pickle.load(open('data/tight_frame_D.p','rb'))
    T = pickle.load(open('data/tight_frame_T.p','rb'))
    N = pickle.load(open('data/tight_frame_N.p','rb'))

    ''' normalize the training data'''
    nol_D = normalize(D)
    nol_T = normalize(T)
    nol_N = normalize(N)

    ''' adding label to the first row'''
    lab_T = np.column_stack(([1 for i in range(nol_T.shape[0])],nol_T))
    lab_N = np.column_stack(([0 for i in range(nol_N.shape[0])],nol_N))
    train = np.row_stack((lab_T,lab_N))


    '''shuffling training data'''
    index_1 = np.random.permutation(train.shape[0])
    index_2 = np.random.permutation(train.shape[0])
    index_2 = index_1[index_2]
    train = train[index_2,:]
    x = train[:,1:]
    y = train[:,0]
    return nol_D,x,y
    


def fit(train_x,train_y):
    '''constructing neural network and fit with the given data'''
    model = Sequential()
    model.add(Dense(7,input_dim=train_x.shape[1],kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('tanh'))
    model.add(Dense(1,input_dim=7))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='sgd')
    model.fit(train_x,train_y,epochs=50,batch_size=1,verbose=False)
    return model


def threshold(pred,t):
    '''to control the threshold of the classificaiton,
    instead of the default 0.5'''
    for i in range(len(pred)):
        if pred[i]<t:
            pred[i]=1
        else:
            pred[i]=0
    return pred


def validate(t,num_iter):
    '''leave-one-out cross_validation'''
    cv_acc = []
    # store the TPR and TNR of 20 tests
    TPR = []
    TNR = []
    F_score = []      
    # t control the threshold
    # when using relu, t should be set as 0
    for k in range(num_iter):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(x.shape[0]):
            print('The',i,'th sample out of',x.shape[0],'iter=',k)
            # rule out the sample that we want to test this time
            index =[j for j in range(x.shape[0]) if j!=i]
            train_x = x[index,:]
            train_y = y[index]
            # fit the remaining data and return the model as mdl
            mdl = fit(train_x,train_y)
            # predict the probability of the i sample
            prob = mdl.predict(x[i,:].reshape(1,x.shape[1]))
            pred = threshold(prob,t)
            # just to compare the prediction and real label
            print('pred=',pred)
            print('y=',y[i])
            if (pred[0][0]==1 and y[i]==1):
                TP += 1
            if (pred[0][0]==1 and y[i]==0):
                FP +=1
            if (pred[0][0]==0 and y[i]==1):
                FN += 1
            if (pred[0][0]==0 and y[i]==0):
                TN += 1
            del mdl
        # store the evaluation of this test
        cv_acc += [(TP+TN)/y.shape[0]]
        TPR += [TP/(TP+FN)]
        TNR += [TN/(TN+FP)]
        F_score += [2*TP/(2*TP+FP+FN)]
    return cv_acc,TPR,TNR,F_score

def dpt_pred(t,num_iter):
    '''predict the disputed paintings
    return the probability of a painting being genuine
    'num_iter' is the number of repeated experiments'''
    tol = [0 for i in range(nol_D.shape[0])]
    for i in range(num_iter):
        print('The',i,'time out of',num_iter)
        # fit the model with all 20 known samples
        mdl = fit(x,y)
        disputed_prob = mdl.predict(nol_D)
        disputed_pred = threshold(disputed_prob,t)
        for j in range(len(tol)):
            tol[j] += disputed_pred[j][0]
        del mdl,disputed_pred,disputed_prob
    return [i/num_iter for i in tol]



if __name__ == '__main__':
    
    '''load the data'''
    nol_D,x,y = load()


    '''training on n-1 paintings and calculate the accuracy
        of the laeve-one-out cross validation.
        since neural network is not stable, we repeat the process
        by num_iter times and calculate the average accuracy
    '''
    # t is the classification threshold
    t = 0.5
    num_iter = 1
    cv_acc,TPR,TNR,F_score = validate(t,num_iter)
    avg_cv_acc = sum(cv_acc)/num_iter
    avg_TPR = sum(TPR)/num_iter
    avg_TNR = sum(TNR)/num_iter
    avg_F_score = sum(F_score)/num_iter

    '''predict the disputed paintings:
        repeating 100 times and calculating the average probability
    '''
    predict_avg = dpt_pred(t,100)                
