#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:05:37 2018

@author: mac
"""
#manipulate AID_h5
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Load data from AID.h5
def load_dataset():
    
    # Load dictionaries from AID1_15.h5 and AID16_30.h5
    data_set1 = h5py.File('/mnt/AID1_15.h5', 'r')
    data_set2 = h5py.File('/mnt/AID16_30.h5', 'r')
    
    # Get data and labels from dictionaries
    X1 = data_set1['data'][:]
    Y1 = data_set1['labels'][:]
    
    X2 = data_set2['data'][:]
    Y2 = data_set2['labels'][:]
    
    data_set1.close()
    data_set2.close()
    
    # Vertically stack X1 and X2
    X = np.vstack((X1, X2))
    
    del X1,X2
    
    # Vertically stack Y1 and Y2
    Y = np.vstack((Y1, Y2))
    
    del Y1,Y2
    
    # Initialize classes
    #classes = 30
    
    # Randomly split X, Y into X_train, X_test, Y_train, Y_test
    # 80% data as training, 20% data as testing data
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    #del X,Y
    
    # Dimension should be:
    # X_train: (8000, 600, 600, 3)
    # Y_train: (8000, 30)
    # X_test: (2000, 600, 600, 3)
    # Y_test: (2000, 30)
    #return X_train, Y_train, X_test, Y_test, classes
    return X,Y