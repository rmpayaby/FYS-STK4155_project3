# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:03:05 2020

@author: rmbp
"""

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


# Processing thre data set
def load_subsidence():
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
    Y_train_feats = y_train.copy()
    Y_test_feats = y_test.copy()
    y_train = np.delete(Y_train_feats,np.s_[1:3],axis=1)
    y_test = np.delete(Y_test_feats,np.s_[1:3],axis=1)
    
    return x_train, x_test, y_train, y_test

"""Optimal values after using grid search in svr_basics.py"""
#C=0.1, epsilon=0.1, gamma= 'auto', kernel='rbf'

def test_SVR_rbf(*data):
  '''
  Test the prediction performance of Gaussian kernel SVR with the influence of higher
  gamma parameter
  '''
  X_train,X_test,y_train,y_test=data
  gammas=range(1,20)
  train_scores=[]
  test_scores=[]
  for gamma in gammas:
    regr=svm.SVR(kernel='rbf',gamma=gamma)
    regr.fit(x_train,y_train)
    y_pred = regr.predict(x_test)
    y_tilde = regr.predict(x_train)
    train_scores.append(mean_squared_error(y_train,y_tilde))
    test_scores.append(mean_squared_error(y_test, y_pred))
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  ax.plot(gammas,train_scores,label="Training MSE ",marker='+' )
  ax.plot(gammas,test_scores,label= " Testing MSE ",marker='o' )
  ax.set_title( "SVR RBF")
  ax.set_xlabel(r"$\gamma$")
  ax.set_ylabel("score")
  ax.set_ylim(-1,1)
  ax.legend(loc="best",framealpha=0.5)
  plt.show()
   
 # Call test_SVR_rbf
x_train,x_test,y_train,y_test=load_subsidence()

#test_SVR_rbf(x_train,x_test,y_train,y_test) 



# Create heat maps to find optimal R2-score and MSE based on gamma and C-parameter
C_range = np.array([-2.0, 1.0,3.0])
gamma_range = np.array([-9.0, -5.0,-2.0])
nr_averages = 1
acc_mat = np.zeros((len(C_range), len(gamma_range)))
area_mat = np.zeros((len(C_range), len(gamma_range)))
cnt = 0
for k in range(len(gamma_range)):
    for j in range(len(C_range)):
        acc = 0
        area = 0
        

        x_train, x_test, y_train, y_test = load_subsidence()
        regr = svm.SVR(epsilon=0.5,kernel='rbf',gamma=10**gamma_range[k],C=10**C_range[j])
        regr.fit(x_train, y_train.ravel())
        y_pred = regr.predict(x_test)
        acc += r2_score(y_test, y_pred)
        area += mean_squared_error(y_test, y_pred)

        #acc_mat[j,k] = acc
        area_mat[j,k] = area
        
        cnt = cnt + 1
        print(cnt)
        
fig, ax = plt.subplots(1, 1, figsize=(10,6))
plt.title("Subsidence, MSE")
sns.heatmap(area_mat, ax=ax, annot=True, fmt=".2f", vmin=0.0, vmax=1.0, xticklabels=gamma_range, yticklabels=C_range, cmap="Greens", square=True)
ax.set_xlabel("Log10(Gamma)")
ax.set_ylabel("Log10(C)")
plt.ylim(0, len(C_range));
ax.set_yticklabels(C_range, rotation = 45, ha="right")
plt.tight_layout()
