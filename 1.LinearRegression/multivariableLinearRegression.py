#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import locale
locale.setlocale( locale.LC_ALL, '' )

np.set_printoptions(suppress=True) 

#Loading data
data = np.loadtxt("datasets/ex1data2.txt",dtype=np.float64, delimiter=",")
# data = np.array(data)

#seperate the input (X) and output (Y)
X = data[::,0:2]
Y = data[::,-1:]


# Just plotting the dataset
# plt.figure(figsize = (15,4),dpi=100)
# plt.subplot(121)
# plt.scatter(X[::,0:1],Y)
# plt.xlabel("Size of house (X1)")
# plt.ylabel("Price (Y)")
# plt.subplot(122)
# plt.scatter(X[::,-1:],Y)
# plt.xlabel("Number of Bedrooms (X2)")
# plt.ylabel("Price (Y)")
# plt.show()

# introduce weights of hypothesis (randomly initialize)
Theta = np.random.rand(1,3)
# m is total example set , n is number of features
m,n = X.shape
# add bias to input matrix by simple make X0 = 1 for all
X_bias = np.ones((m,n+1))
X_bias[::,1:] = X

#feature scaling
# it also protect program from overflow error
mean_size = np.mean(X_bias[::,1:2])
mean_bedroom = np.mean(X_bias[::,2:])
size_std = np.std(X_bias[::,1:2])
bedroom_std = np.std(X_bias[::,2:])
X_bias[::,1:2] = (X_bias[::,1:2] - mean_size)/ (size_std) 
X_bias[::,2:] = (X_bias[::,2:] - mean_bedroom)/ (bedroom_std)
X_bias[0:5,::]



#define function to find cost
def fitness(X_bias,Y,Theta):
    np.seterr(over="raise")
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
        temp1 = Theta[0,1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,1:2])).sum(axis=0)
        temp2 = Theta[0,2] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,-1:])).sum(axis=0)
        Theta[0,0] = temp0
        Theta[0,1] = temp1
        Theta[0,2] = temp2
        fitness_log = np.append(fitness_log,fitness(X_bias,Y,Theta))
        count = count + 1
    # plt.plot(np.linspace(1,iterations,iterations,endpoint=True),fitness_log)
    # plt.title("Iteration vs Fitness graph ")
    # plt.xlabel("Number of iteration")
    # plt.ylabel("Fitness function")
    # plt.show()
    return Theta

alpha = 0.3
iterations = 100
Theta = gradientDescent(X_bias,Y,Theta,iterations,alpha)

# predict the price of a house 1with 1650 feet and 3 rooms
X_predict = np.array([[1.0, 1650.0, 3.0]])
#Normalizing data
X_predict[0][1] = (X_predict[0][1] - mean_size)/ (size_std) 
X_predict[0][2] = (X_predict[0][2]- mean_bedroom)/ (bedroom_std)

hypothesis = X_predict.dot(Theta.transpose())
print ('price of a house with 1650 feets and 3 rooms is: ',locale.currency(hypothesis[0][0], grouping=True))


# Plot showing hypothesis 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X[::,0:1], X[::,1:2], Y, c='b')

ax.set_xlabel('Square Feets')
ax.set_ylabel('Rooms Number')
ax.set_zlabel('Price')

plt.show()



price = Theta[0][0]*X_predict[0][0] + Theta[0][1]*X_predict[0][1] + Theta[0][2]*X_predict[0][2]
print("Price: ", locale.currency(price, grouping=True))


print("\n\n##################")
print("Predicted Formula:")
print("h_theta(x) = theta_0*x_0 + theta_1*x_1 + theta_2*x_2")
print("h_theta(x) = ", Theta[0][0], " + ", Theta[0][1], "* x_1 + ", Theta[0][2], "* x_2")
