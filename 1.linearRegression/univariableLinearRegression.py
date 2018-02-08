#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Loading data
data = np.loadtxt("datasets/ex1data1.txt",delimiter=",")
data = np.array(data)

#seperate the input (X) and output (Y)
X = data[::,:1]
Y = data[::,1:]

# Just plotting the dataset
# plt.scatter(X.transpose(),Y.transpose(),40,color="red",marker="x")
# plt.xlabel("population in 10000's")
# plt.ylabel("profit in 10000 $")
# plt.show()

# introduce weights of hypothesis (randomly initialize)
Theta = np.random.rand(1,2)
# m is total example set , n is number of features
m,n = X.shape
# add bias to input matrix by simple make X0 = 1 for all
X_bias = np.ones((m,2))
X_bias[::,1:] = X


#define function to find cost
def fitness(X_bias,Y,Theta):
    m,n = X.shape
    hypothesis = X_bias.dot(Theta.transpose())
    return (1/(2.0*m))*((np.square(hypothesis-Y)).sum(axis=0))



#function gradient descent algorithm from minimizing theta
def gradientDescent(X_bias,Y,Theta,iterations,alpha):
    count = 1
    fitness_log = np.array([])
    while(count <= iterations):
        hypothesis = X_bias.dot(Theta.transpose())
        temp0 = Theta[0,0] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,0:1])).sum(axis=0)
        temp1 = Theta[0,1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,-1:])).sum(axis=0)
        Theta[0,0] = temp0
        Theta[0,1] = temp1
        fitness_log = np.append(fitness_log,fitness(X_bias,Y,Theta))
        count = count + 1
    # plt.plot(np.linspace(1,iterations,iterations,endpoint=True),fitness_log)
    # plt.title("Iteration vs Fitness graph ")
    # plt.xlabel("Number of iteration")
    # plt.ylabel("Fitness function")
    # plt.show()
    return Theta

alpha = 0.01
iterations = 2000
Theta = gradientDescent(X_bias,Y,Theta,iterations,alpha)
print("Theta: ", Theta)

# predict the profit for city with 35000 and 75000 people
X_test = np.array([[1,4.0],[1,7.0]])
hypothesis = X_test.dot(Theta.transpose())
print ('profit from 40000 people city is ',hypothesis[0,0]*10000,'$')
print ('profit from 70000 people city is ',hypothesis[1,0]*10000,'$')


# Plot showing hypothesis 
plt.scatter(X.transpose(),Y.transpose(),40,color="red",marker="x")
X_axis = X
Y_axis = X_bias.dot(Theta.transpose())
plt.plot(X_axis,Y_axis)
plt.title('Simple Linear Regression - Predicting Food Truck Profit vs City Population')
plt.xlabel("population (10000)")
plt.ylabel("profit (10000) $")
plt.show()

print("##################")
print("Predicted Formula:")
print("h_theta(x) = theta_0 + theta_1*x")
print("h_theta(x) = 0.36 + 5.08*x")
