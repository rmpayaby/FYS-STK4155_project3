# -*- coding: utf-8 -*-


import numpy as np

class Regression():


    def __init__(self,hidden_activation='ReLU',output_activation='linear',cost_func='MSE'):

        self.h_a = hidden_activation
        self.o_a = output_activation
        self.cost = cost_func

    def hidden_activation(self,x,deriv=False):
        """
        Returns the appropriate activation function for hidden layers given string from initialization.
        The respective derivatives also defined
        """
        if self.h_a == 'ReLU':
            if deriv:
                return self._ReLU_deriv(x)
            else:
                return self._ReLU(x)
        elif self.h_a == 'sigmoid':
            if deriv:
                return self._sigmoid_deriv(x)
            else:
                return self._sigmoid(x)
        elif self.h_a == "leaky_ReLU":
            if deriv:
                return self._leaky_ReLU_deriv(x)
            else:
                return self._leaky_ReLU(x)
    
    def output_activation(self,z,deriv=False):
        """
        Returns the appropriate activation function for the output layer given string from initialization.
        """

        if self.o_a == 'linear':
            if deriv:
                return 1
            else:
                return z
        if self.o_a == 'ReLU':
            if deriv:
                return self._ReLU_deriv(z)
            else:
                return self._ReLU(z)
        elif self.o_a == 'sigmoid':
            if deriv:
                return self._sigmoid_deriv(z)
            else:
                return self._sigmoid(z)
        elif self.o_a == 'leaky_ReLU':
            if deriv:
                return self._leaky_ReLU_deriv(z)
            else:
                return self._leaky_ReLU(z)

    def cost_function(self,a,t):
        """ Returns value of appropriate cost function """
        if self.cost == 'cross_entropy':
            return self._cross_entropy_cost(a,t)
        if self.cost == 'MSE':
            return self._MSE(a,t)

    def output_error(self,a,t,z=None):
        """
        Computes the delta^L value, error from output layer.
        """
        if self.cost == 'cross_entropy':
            return (a-t)
        if self.cost == 'MSE':
            return (a-t)/len(a)*self.output_activation(z,deriv=True)


    # Cost functions
    def _cross_entropy_cost(self,a,t):
        """
        Computes cross entropy for an output a with target output t
        Uses np.nan_to_num to handle log(0) cases
        """
        return -np.sum(np.nan_to_num(t*np.log(a)-(1-t)*np.log(1-a)))

    def _quadratic_cost(self,a,t):
        return 1/2/len(a)*np.sum((a-t)**2)
    
    """
    These section returns the activation functions for 
    the hidden layers and their respective derivatives
    """
    _sigmoid = lambda self, x: 1/(1+np.exp(-x))
    _sigmoid_deriv = lambda self, x: self._sigmoid(x)*(1 - self._sigmoid(x))

    _leaky_ReLU = lambda self, x: np.where(x > 0, x, x * 0.01)
    _leaky_ReLU_deriv = lambda self, x: np.where(x > 0, 1, 0.01)

    _ReLU = lambda self,x: np.where(x<0,0,x)
    _ReLU_deriv = lambda self, x: np.where(x<0,0,1)
        
