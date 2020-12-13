# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:33:49 2020

@author: rmbp
"""

import time
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn import datasets, linear_model,svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import Normalize
from sklearn.model_selection import GridSearchCV
import seaborn as sns

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR


df = pd.read_excel("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/all_features.xlsx")
df = df.dropna()
df = shuffle(df)
X = df.iloc[0:,[5,6,7,8,9,10,11]].to_numpy()
y = (df["MCDA"].to_numpy()).ravel()


# Measuring runtime by grid searching the regularization and gamma parameter
train_size = 3000
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s"
      % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)





# Calculating runtime for KRR and SVR based on training size

sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for name, reg in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
                                           gamma=0.1),
                        "SVR": SVR(kernel='rbf', C=1e1, gamma=0.1)}.items():
    
    train_time = []; test_time = []
    
   
    for m in sizes:
        x_train,x_test,y_train,y_test = train_test_split(X,y, train_size = m)
        t0 = time.time()
        reg.fit(x_train,y_train)
        train_time.append(time.time() - t0)
        t0 = time.time()
        reg.predict(x_test)
        test_time.append(time.time() - t0)
    plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
             label="%s (train)" % name)
    plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
             label="%s (test)" % name)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title('Execution Time')
plt.legend(loc="best")
    
plt.figure()


def learning_curve(X,y):
    """
    Setting up the learning curve for a fixed hyperparameter value.
    
    Takes the design matrix (X) and target value (y) as input.
    
    Default test size is defined as 30% of the data set

    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    test_error1, test_error2 = [], []  # For MSE

    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    

    model_1 = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    model_2 = SVR(kernel='rbf', C=1e1, gamma=0.1)
    
    for m in range(1,len(X_train)):
        model_1.fit(X_train[:m], y_train[:m])
        y_test_pred_1 = model_1.predict(X_test)
        test_error1.append(mean_squared_error(y_test, y_test_pred_1))

        model_2.fit(X_train[:m], y_train[:m])
        y_test_pred_2 = model_2.predict(X_test)
        test_error2.append(mean_squared_error(y_test, y_test_pred_2))
        
    
    plt.plot(np.sqrt(test_error1), "r-", linewidth=2, label="KRR")
    plt.plot(np.sqrt(test_error2), "b-", linewidth=2, label="SVR")
    plt.legend()
    plt.title("Learning curve")
    plt.ylabel("MSE")
    plt.xlabel("Training set size")
    

learning_curve(X, y)



