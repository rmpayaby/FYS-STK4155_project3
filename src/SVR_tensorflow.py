# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class SVR(object):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        
    def fit(self, X, y, epochs=100, learning_rate=0.1):
        self.sess = tf.Session()
        
        feature_len = X.shape[-1] if len(X.shape) > 1 else 1
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        
        
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, feature_len))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        
        self.W = tf.Variable(tf.random_normal(shape=(feature_len, 1)))
        self.b = tf.Variable(tf.random_normal(shape=(1,)))
        
        self.y_pred = tf.matmul(self.X, self.W) + self.b

        """Defining the loss function"""
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_pred))
        #self.loss = tf.reduce_mean(tf.cond(self.y_pred - self.y < self.epsilon, lambda: 0, lambda: 1))
        
        # Second part of following equation, loss is a function of how much the error exceeds a defined value, epsilon
        # Error lower than epsilon = no penalty.
        self.loss = tf.norm(self.W)/2 + tf.reduce_mean(tf.maximum(0., tf.abs(self.y_pred - self.y) - self.epsilon))
        self.loss = tf.reduce_mean(tf.maximum(0., tf.abs(self.y_pred - self.y) - self.epsilon))
        
        # Check other optimizers as well
        #opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        #opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        opt_op = opt.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        
        errors = []
        for i in range(epochs):
            loss = self.sess.run(
                self.loss, 
                {
                    self.X: X,
                    self.y: y
                }
            )
            print("{}/{}: loss: {}".format(i + 1, epochs, loss))
            errors.append(loss)
        
 
            
            self.sess.run(
                opt_op, 
                {
                    self.X: X,
                    self.y: y
                }
            )
        print(X.shape)
        error_array = np.array(errors)
        """Used to store predicted data by choosing different epsilon values"""
        #np.savetxt("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/prediction_sets/errors_0_1.csv", error_array, delimiter=",")
            
        return self
            
    def predict(self, X, y=None):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        y_pred = self.sess.run(
            self.y_pred, 
            {
                self.X: X 
            }
        )
        return y_pred
    
# Define epsilon value   
model = SVR(epsilon=0)

# Prepare the data set
df = pd.read_excel("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/all_features.xlsx")
df = df.dropna()
df = shuffle(df)

X = df.iloc[0:,[5,6,7,8,9,10,11]].to_numpy()
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

model.fit(x_train, y_train,epochs=10000, learning_rate=0.001)
y_pred = model.predict(x_test)

r2 = r2_score(y_test,y_pred)
print(r2)


# Plot loss vs. epochs based on different epsilon values
df = pd.read_excel("C:/Users/rmbp/GIS-project/FYS-STK4155_project3/prediction_sets/epsilon_vals.xlsx")
df.info()


plt.title("Loss function with changing epsilon-value")
plt.xlabel("Epochs");plt.ylabel("Loss")
plt.plot(df["eps_05"],label=r"$\epsilon = 0.5$")
plt.plot(df["eps_1"],label=r"$\epsilon = 1.0$")
plt.plot(df["eps_01"],label=r"$\epsilon = 0.01$")
plt.plot(df["eps_0_1"],label=r"$\epsilon = 0.1$")
plt.legend()
