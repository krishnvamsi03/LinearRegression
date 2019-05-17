# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:00:48 2019

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

#Data Generation Experience vs Salary (Random Data)
X = np.arange(5, 20, 0.2)
m  = len(X)
i = 0
y = []
while i < m:
    temp_y = np.random.randint(100, 500)
    if temp_y not in y:
        y.append(temp_y)
        i += 1
        
y = np.array(y)
y = np.sort(y)

#Visualizing 
plt.scatter(X, y, c = 'red', marker = 'x', s = 100)
plt.xlabel('Experience')
plt.ylabel('Salary in k Dollars')
plt.show()


class Linear_regression:
    
    def __init__(self, weight, bias, alpha):
        self.weight = weight
        self.bias = bias
        self.alpha = alpha
        self.cost_ = []
        
        
        
    def cost_function(self, y_hat, y, m):
        ''' Cost function is calculated using sum of squared Error'''
        cost = 0
        
        for i in range(m):
            temp = (y_hat[i] - y[i]) ** 2
            cost += temp
        
        cost = cost / 2 * m 
        self.cost_.append(cost) 
        
        
    def Gradient_Descent(self, y_hat, y, X, m):
        '''self.aplha is learning rate'''
        
        weight = 0
        bias = 0
        
        for i in range(m):
            weight += 2 * X[i] * (y_hat[i] - y[i]) 
            bias += 2 * (y_hat[i] - y[i]) 
        
        self.bias -= (bias / float(m)) * self.alpha
        self.weight -= (weight / float(m)) * self.alpha 
        
    def predict(self, X):
        y_pred = []
        for i in X:
            y_pred.append(i * self.weight + self.bias)
        
        return y_pred
        
    def fit(self, X, y):
        y_hat = []
        m = len(X)
        for i in range(m):
            temp = X[i] * self.weight + self.bias
            y_hat.append(temp)
        
        self.cost_function(y_hat, y, m)
        self.Gradient_Descent(y_hat, y, X, m)

    def cost(self):
        
        return self.cost_
    
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)        

'''Bias and weight needs to be hand picked since data is randomly generated,
    it may vary according to data generated
'''        
bias = -10 
weight = 20
learning_rate = 0.003
regressr = Linear_regression(weight, bias, learning_rate)
epoch = 20 

#Optimization        
for i in range(epoch):
    regressr.fit(X_train, y_train)

#Getting Prediction  
y_pred = regressr.predict(X_test)          

'''Error is calculated by mean_absolute Error, which is 
    (sum |y - y_hat|) /n'''
    
mae = mean_absolute_error(y_test, y_pred) 
#print(mae)


#Visualizing Prediction
plt.scatter(X_test, y_test, c = 'red')
plt.plot(X_test, y_pred, c = 'blue')   
plt.show()

#Visualizing Cost 
cost = regressr.cost()
plt.plot([i for i in range(epoch)], cost)
plt.show()

'''Comparing with sklearn model ''' 
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)
y_ = regressor.predict(X_test.reshape(-1, 1))

mae_sk = mean_absolute_error(y_test, y_) 
#print(mae_sk)
     
plt.scatter(X_test, y_test, c = 'red')
plt.plot(X_test, y_, c = 'blue')   
plt.show()     
        
