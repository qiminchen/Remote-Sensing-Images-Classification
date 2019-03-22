#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 22:29:11 2018

@author: mac
"""

#manipulate NWPU_h5
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Load data from NWPU.h5
def load_dataset():
    
    # Load dictionaries from NWPU.h5, get data and labels from dictionaries
    data_set1 = h5py.File('/mnt/NWPU1_10.h5', 'r')
    X1 = data_set1['data'][:]
    Y1 = data_set1['labels'][:]
    data_set1.close()
    
    data_set2 = h5py.File('/mnt/NWPU11_20.h5', 'r')
    X2 = data_set2['data'][:]
    Y2 = data_set2['labels'][:]
    data_set2.close()
    
    data_set3 = h5py.File('/mnt/NWPU21_30.h5', 'r')
    X3 = data_set3['data'][:]
    Y3 = data_set3['labels'][:]
    data_set3.close()
    
    data_set4 = h5py.File('/mnt/NWPU31_40.h5', 'r')
    X4 = data_set4['data'][:]
    Y4 = data_set4['labels'][:]
    data_set4.close()
    
    data_set5 = h5py.File('/mnt/NWPU41_45.h5', 'r')
    X5 = data_set5['data'][:]
    Y5 = data_set5['labels'][:]
    data_set5.close()
    
    # Vertically stack X1 and X2
    X = np.vstack((X1, X2, X3, X4, X5))
    del X1, X2, X3, X4, X5
    
    # Vertically stack Y1 and Y2
    Y = np.vstack((Y1, Y2, Y3, Y4, Y5))
    del Y1, Y2, Y3, Y4, Y5
    
    # Initialize classes
    classes = 45
    
    # Randomly split X, Y into X_train, X_test, Y_train, Y_test
    # 90% data as training, 20% data as testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    del X,Y
    
    # Dimension should be:
    # X_train: (25200, 224, 224, 3)
    # Y_train: (25200, 45)
    # X_test: (6300, 224, 224, 3)
    # Y_test: (6300, 45)
    return X_train, Y_train, X_test, Y_test, classes