# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:16:49 2020

@author: rmbp
"""


import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


from SVR_parameter import load_subsidence


# Implementation of kernel ridge with kfold                                 
def  kernelRidgeSkLearnCV(kfold = 5):

    df = pd.read_excel("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/all_features.xlsx")
    df = df.dropna()
    df = shuffle(df)
    
    X = df.iloc[0:,[5,6,7,8,9,10,11]].to_numpy()
    y = df["MCDA"].to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)


    #test_x =  StandardScaler().fit_transform(test_x)
    
    # Perhaps create a degree list as well.
    alphaParaLst = [1, 0.001, 0.0001]
    gammaParaLst = [None, 1, 0.001]
    kernelParaLst = ["rbf", "polynomial"]
    degreeLst = [3,4,5]
    
    mseErrorSmallest = 2**32
    #mseErrorLst = []
    for alpha in alphaParaLst:
        for gamma in gammaParaLst:
            for kernel in kernelParaLst:
                for degree in degreeLst:
                    clf = KernelRidge(alpha=alpha, gamma = gamma, degree=degree, kernel=kernel)
                    mseError = -1*np.mean(cross_val_score(clf, x_train, y_train, cv=kfold, scoring="neg_mean_squared_error"))
                    R2 = np.mean(cross_val_score(clf, x_train, y_train, cv=kfold, scoring="r2"))
                    print ("mseError: ", alpha, gamma, kernel, degree,  mseError)
                    print ("R2: ", alpha, gamma, kernel, degree,  R2)
                    #mseErrorLst.append(mseError)
                    if mseError < mseErrorSmallest:
                        mseErrorSmallest = mseError
                        paramtersBest = [alpha, gamma, kernel, degree]
                
    print ("best mseError: ", kfold, paramtersBest,  mseErrorSmallest)

    #train whole data
    clf = KernelRidge(alpha=paramtersBest[0], gamma = paramtersBest[1], degree=paramtersBest[3], kernel=paramtersBest[2])
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(y_pred)
    
#a = kernelRidgeSkLearnCV()


"""

Again, prepare the data set, use the optimal value from the function
kernelRidgeSkLearnCV()

"""
    
df = pd.read_excel("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/all_features.xlsx")
df = df.dropna()
df = shuffle(df)

X = df.iloc[0:,[5,6,7,8,9,10,11]].to_numpy()
y = (df.iloc[0:,[12]].to_numpy()).round(2)
y = df[["MCDA","Lat","Long"]].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

    # Make copy of original features:
Y_train_feats = y_train.copy()
Y_test_feats = y_test.copy()
    
    # Remove latitude and longitude and store for later use
y_train = np.delete(Y_train_feats,np.s_[1:3],axis=1)
y_test = np.delete(Y_test_feats,np.s_[1:3],axis=1)


"""
Applying the optimal parameter values 
"""
clf = KernelRidge(alpha=0.001, gamma = 0.001, degree=4, kernel="polynomial")
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
latlong = Y_test_feats[:,[1,2]]

# Add latitude and longitude of the choosen point data
together = np.hstack((latlong,y_pred))
#np.savetxt("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/prediction_sets/KRR_opt_pred.csv", together, delimiter=",")



def trainKernelRidgeExtra():
    '''
    try different kernel ridge model or even different kernel
    how to select effective kernel
    '''
    
    x_train, y_train, x_test, y_test  = load_subsidence()
    print('Train=', x_train.shape, type(x_train))
    print('Test=', x_test.shape)
  
    #[1, 0.01, 0.001, 0.0001]

    kfoldLst = range(3,12)
    smallestError = 2**32
    
    for kfold in kfoldLst:
        parameters = {'kernel':('rbf', 'polynomial', 'sigmoid', 'laplacian', 'chi2'), 'alpha': np.linspace(0.001, 0.1, 100), 'gamma':[0.001, 1, 100]}
        
        clf = GridSearchCV(KernelRidge(), parameters, cv=kfold, n_jobs=8)   #scoring= "neg_mean_squared_error" )
        clf.fit(x_train, y_train)
        meanTestError = clf.cv_results_['mean_test_score']
        bestPara = clf.best_estimator_
        
        if clf.best_score_ < smallestError:
            smallestError = clf.best_score_
            paramtersBest = [bestPara.alpha, bestPara.gamma, bestPara.degree,  bestPara.kernel, kfold, clf.best_score_]
            
        print ("trainKernelRidgeExtra Result : ", bestPara.alpha, bestPara.gamma, bestPara.degree,  bestPara.kernel, clf.best_score_, meanTestError,)
       
       # kwargs = {'n_neighbors': bestPara.n_neighbors}

        clf = KernelRidge(alpha=bestPara.alpha, gamma = bestPara.gamma, degree=bestPara.degree, kernel=bestPara.kernel)
        clf.fit(x_train, y_train)
        predY = clf.predict(x_test)
