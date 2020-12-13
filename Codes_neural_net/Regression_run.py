# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, roc_auc_score as AUC_score, r2_score
from sklearn import preprocessing
from sklearn.utils import shuffle

from Regression import Regression
from neural_net_regression import NeuralNetwork

import seaborn as sns

mpl.rcdefaults()
plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18




df = pd.read_excel("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/all_features.xlsx")
df = df.dropna()
df = shuffle(df)

#X = df.iloc[0:,[5,6,7,8,9,10,11]].to_numpy()
#y = (df.iloc[0:,[12]].to_numpy()).round(2)


X = df.iloc[0:,[8,9,10]].to_numpy()
y = (df.iloc[0:,[12]].to_numpy()).round(2)
y = df[["MCDA","Lat","Long"]].to_numpy()



hidden_neuron_list = [5]
epochs = 100
runs = 1
lr_rate = 0.001
lmbd = 0.0001
# Calling the class function containing activaion and cost function
reg = Regression(hidden_activation='sigmoid',output_activation="sigmoid")


# Initialize storing values
r2_test_runs = np.zeros((runs,epochs))
r2_train_runs = np.zeros((runs,epochs))
r2_end_test = np.zeros(runs)
r2_end_train = np.zeros(runs)

MAPE_test_runs = np.zeros((runs,epochs))
MAPE_train_runs = np.zeros((runs,epochs))
MAPE_test_end = np.zeros(runs)
MAPE_train_end = np.zeros(runs)



for run in tqdm(range(runs)):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.3)
    Scaler = preprocessing.StandardScaler()
    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)
    
    
    # Make copy of original features:
    Y_train_feats = Y_train.copy()
    Y_test_feats = Y_test.copy()
    
    # Remove latitude and longitude and store for later use
    Y_train = np.delete(Y_train_feats,np.s_[1:3],axis=1)
    Y_test = np.delete(Y_test_feats,np.s_[1:3],axis=1)


    
    
    nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem=reg,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=1,
                        epochs=epochs,
                        batch_size=10,
                        lr_rate=lr_rate,
                        lmbd=lmbd)
    
    nn.SGD(metric=['r2'],test_data=(X_test_scaled,Y_test),train_data=(X_train_scaled,Y_train))
    r2_test_runs[run,:] = nn.r2_test
    r2_train_runs[run,:] = nn.r2_train
    r2_end_test[run] = nn.r2_test[-1]
    r2_end_train[run] = nn.r2_train[-1]
    
    
    """Used to run the MAPE-values"""
    #MAPE_test_runs[run,:]= nn.MAPE_test
    #MAPE_train_runs[run,:] = nn.MAPE_train
    #MAPE_test_end[run] = nn.MAPE_test[-1]
    #MAPE_train_end[run] = nn.MAPE_train[-1]
    
    
    """Applying latitude and longitude back to the data set"""
    pred = nn.predict_proba(X_test)
    latlong = Y_test_feats[:,[1,2]]
    
    together = np.hstack((latlong,pred))
    
    #np.savetxt("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/prediction_sets/sigmoid_NN.csv", together, delimiter=",")
    
    print(together)


r2_mean_test = np.mean(r2_end_test)
r2_mean_train = np.mean(r2_end_train)
print(r2_mean_test)

MAPE_mean_test = np.mean(MAPE_test_end); MAPE_mean_train = np.mean(MAPE_train_end)
fig,ax = plt.subplots()
for i in range(runs):
    ax.plot(r2_train_runs[i,:],color='black',label='train, mean = {:.2f}'.format(r2_mean_train))
    ax.plot(r2_test_runs[i,:],color='green',label='test, mean = {:.2f}'.format(r2_mean_test))
    #ax.plot(MAPE_train_runs[i,:],color='black',label='train, mean = {:.2f}'.format(MAPE_mean_train))
    #ax.plot(MAPE_test_runs[i,:],color='green',label='test, mean = {:.2f}'.format(MAPE_mean_test))
    if i == 0:
        ax.legend(loc=4)
ax.set_ylabel('R2 score')
ax.set_xlabel('Epochs')
ax.set_ylim(0,1) # 0 to 1 if R2
fig.tight_layout()

plt.title("R2 of data set")
print('epochs',epochs,'runs',runs)
print('lr_rate',lr_rate,' lambda ',lmbd,' neuron list ',hidden_neuron_list)


print(min(nn.predict_proba(X_test)))








"""


eta_vals = np.logspace(-5, -1, 5)
lmbd_vals = np.logspace(-5, 1, 7)
# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

# grid search
for i, lr_rate in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem=reg,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=1,
                        epochs=epochs,
                        batch_size=20,
                        lr_rate=lr_rate,
                        lmbd=lmbd)
        nn.SGD(test_data=(X_test_scaled,Y_test),train_data=(X_train_scaled,Y_train))
        
        DNN_numpy[i][j] = nn
        
        test_predict = nn.predict_proba(X_test_scaled)

        print("Learning rate  = ", lr_rate)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", r2_score(Y_test,test_predict))
        print()


sns.set()

test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        nn = DNN_numpy[i][j]
        
        test_pred = nn.predict_proba(X_test_scaled)
        test_r2[i][j] = r2_score(Y_test, test_pred)

        
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_r2, annot=True, ax=ax, cmap="viridis")
ax.set_title("Grid search of R2-score, Sigmoid")
ax.set_ylabel("Learning rate")
ax.set_xlabel("Regularization parameter")
plt.show()


#print(y_predict)

#print(MAPE(Y_test,y_predict))




"""