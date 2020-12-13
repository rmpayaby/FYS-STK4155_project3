# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import roc_auc_score as AUC_score, r2_score, mean_squared_error

class NeuralNetwork:

    def __init__(
            
            self,
            X_data,
            Y_data,
            problem,    
            n_hidden_neurons_list =[2],    
            n_output_neurons=10,    
            epochs=10,
            batch_size=100,
            lr_rate=0.1,
            lmbd=0.0):
        
        """
            Initializes the values for the neural networks.
            
            The problem-parameter calls a class function containing
            the activation functions and cost functions for the regression
            problem
            
            n_hidden neurons_list is a list of numbers of neurons in each layer
            The number of layers is defined by the lenghth of the list
        """
            

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_layers = len(n_hidden_neurons_list)
        self.n_hidden_neurons_list = n_hidden_neurons_list
        self.n_output_neurons = n_output_neurons

        self.Problem = problem
        
         
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.lr_rate = lr_rate
        self.lmbd = lmbd
        
         
        # Regression metrics
        self.r2_test = np.zeros(epochs)
        self.r2_train = np.zeros(epochs)
        self.MAPE_test = np.zeros(epochs)
        self.MAPE_train = np.zeros(epochs)
        

        self.init_layers()

    def init_layers(self):
        """
        Initializes the weights and biases for the hidden layers and output layer.
        Appending output layer with weights and biases. 
        Setting weights for l = [1,L-1]
        The dimension of layers l are dependent on the last layer l-1
        
        """
        n_hidden = self.n_hidden_neurons_list
        
        # Add 0.01 to bias to avoid exploding gradients
        self.bias_list = [np.zeros(n)+0.01 for n in n_hidden]
        self.bias_list.append(np.zeros(self.n_output_neurons)+0.01)
        
        self.weights_list = [np.random.normal(loc=0.0,scale=0.05,size=(self.n_features,n_hidden[0]))]
        for i in range(1,self.n_layers):
            self.weights_list.append(np.random.normal(loc=0.0,scale=0.05,size=(n_hidden[i-1],n_hidden[i])))
                
        self.weights_list.append(np.random.normal(loc=0.0,scale=0.05,size=(n_hidden[-1], self.n_output_neurons)))
       
            
            
    def FeedForward(self):
        """
        The feed forward algorithm itself. The program
        avoids using weights and bias for input.
        The layers are loop through to store the weighted sums and
        activations. 
        
        Then the last entry in a_list is overwrited to use the
        output activation function.
        """

        problem = self.Problem
        self.a_list = [self.X_data]; self.z_list = []
        for w,b in zip(self.weights_list,self.bias_list):
            self.z_list.append(np.matmul(self.a_list[-1],w)+b)
            self.a_list.append(problem.hidden_activation(self.z_list[-1]))
            
        self.a_list[-1] = problem.output_activation(self.z_list[-1])

    def FeedForward_out(self, X):
        """
        Feed forward loop used for predictions on trained network on input data X
        """
        problem = self.Problem
        a_list = [X]; z_list = []

        for w,b in zip(self.weights_list,self.bias_list):
            z_list.append(np.matmul(a_list[-1],w)+b)
            a_list.append(problem.hidden_activation(z_list[-1]))

        a_list[-1] = problem.output_activation(z_list[-1])
        return a_list[-1]

    def Backpropagation(self):
        """
        Performs the back propagation algorithm, with output from forward pass
        Uses the expressions from the given Problem class
        to compute the output error.
        
        First of all, the output error is found by using
        the defined cost function.
        
        Then the output error is propagated back in the hidden lauers to find
        the error from each layer. Gradients are calculated for each
        layer looping over all layers. Gradients are defined
        differently if regularization is used. 
        """
        problem = self.Problem
        
        error_list = [];grad_w_list = [];grad_b_list = []
        

        output_error = problem.output_error(self.a_list[-1],self.Y_data)
        error_list.append(output_error)
        
        L = self.n_layers   
        
        for l in range(2,L+2): 
            prev_error = error_list[-1]
            prev_w = self.weights_list[-l+1]
            current_z = self.z_list[-l]
            error_hidden = np.matmul(prev_error,prev_w.T)*problem.hidden_activation(current_z,deriv=True)
            error_list.append(error_hidden)

        error_list.reverse()

        for l in range(L+1):
            grad_b_list.append(np.sum(error_list[l],axis=0))
            grad_w_list.append(np.matmul(self.a_list[l].T,error_list[l]))

            if self.lmbd > 0.0: 
                grad_w_list[l] += self.lmbd * self.weights_list[l]
            
            self.weights_list[l] -= self.lr_rate*grad_w_list[l]
            self.bias_list[l] -= self.lr_rate*grad_b_list[l]
            

    def SGD(self,metric=False,test_data=False,train_data=False):
        """
        Perform a stochastic gradient descent algorithm, looping over epochs and saturating each minibatch
        Stores R_score after each epoch, to visualize the learning rate
        """
        if test_data != False:
            X_test,Y_test = test_data
        if train_data != False:
            X_train,Y_train = train_data

        data_idx = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                datapoints = np.random.choice(
                    data_idx, size=self.batch_size, replace=False
                )

                self.X_data = self.X_data_full[datapoints]
                self.Y_data = self.Y_data_full[datapoints]

                self.FeedForward()
                self.Backpropagation()
            
            if metric != False:
                if test_data != False:
                    pred_test = self.predict_proba(X_test)
                if train_data != False:
                    pred_train = self.predict_proba(X_train)

                if 'r2' in metric:   #   r2 score
                    if test_data != False:
                        self.r2_test[i] = r2_score(Y_test,pred_test)
                    if train_data != False:
                        self.r2_train[i] = r2_score(Y_train,pred_train)
                        
                if 'MAPE' in metric:
                    if test_data != False:
                        self.MAPE_test[i] = self.MAPE(Y_test,pred_test)
                    if train_data != False:
                        self.MAPE_train[i] = self.MAPE(Y_train,pred_train)
                
            

    """Get prediction data"""                    
    def predict(self, X):
        probabilities = self.FeedForward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        probabilities = self.FeedForward_out(X)
        return probabilities
    
    def MAPE(self, actual, pred): 
        # Investigate MAPE for regression
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual)) * 100
    

    
    

