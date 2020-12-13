# -*- coding: utf-8 -*-


from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from numpy.linalg import inv
import pandas as pd



# Preparing the data set
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


# Defining the parameters used in grid search
param_grid = {
    "kernel": ['poly','rbf','sigmoid'],
   "degree": [3,4,5],
     "C": [0.1,1.0,10.0],
     "gamma": ['scale','auto'],
     "epsilon": [0.01, 0.1, 0.5, 1.0]
 }


# Grid search
regressor = SVR()
grid_search = GridSearchCV(regressor,param_grid, cv=5)
#grid_search.fit(x_train,y_train.ravel())
#print(grid_search.best_params_)


"""
Optimal value:
    
    C=0.1, epsilon=0.5, gamma= 'auto', kernel='rbf'
"""

# Running program 40 times to calculate average r2 and mse
"""
scores = []; mse_avg = []
for _ in range(40):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    
    Y_train_feats = y_train.copy()
    Y_test_feats = y_test.copy()
    
    y_train = np.delete(Y_train_feats,np.s_[1:3],axis=1)
    y_test = np.delete(Y_test_feats,np.s_[1:3],axis=1)
    Y_train_feats = y_train.copy()
    Y_test_feats = y_test.copy()
    y_train = np.delete(Y_train_feats,np.s_[1:3],axis=1)
    y_test = np.delete(Y_test_feats,np.s_[1:3],axis=1)
    
    
    regressor = SVR(C=0.1, epsilon=0.1, gamma= 'auto', kernel='rbf')
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    scores.append(r2)
    mse_avg.append(mse)

print(scores)
print(np.mean(scores)); print(np.mean(mse_avg))
"""

regressor = SVR(C=0.1, epsilon=0.4, gamma= 'auto', kernel='rbf')
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

# Count the number of support vectors in the analysis
print(regressor.support_vectors_.shape)
print(r2_score(y_test,y_pred))
y_pred = y_pred.reshape((2841,1))
print(y_pred.shape)
latlong = Y_test_feats[:,[1,2]]
together = np.hstack((latlong,y_pred))

#np.savetxt("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/prediction_sets/SVR_opt_pred.csv", together, delimiter=",")


