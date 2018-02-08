![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg)   ![Problem Regression](https://img.shields.io/badge/Problem-Univariable%20Regression-orange.svg)   ![Problem Regression](https://img.shields.io/badge/Problem-Multivariable%20Regression-orange.svg) ![Normal Equation](https://img.shields.io/badge/Equation-Normal%20Equation-orange.svg)
# Machine Learning

Exercises from Coursera Machine Learning

## Simple Linear Regression

Univariable Linear regression to predict profits for a food truck in a new City. It's based on population size.


![screen 1](https://raw.githubusercontent.com/moraisaugusto/machineLearning/master/1.LinearRegression/univariableLinearRegression.png)


## Multivariable Linear Regression

Multivariable Linear regression to predict house prices based on square feet and number of rooms.


![screen 2](https://raw.githubusercontent.com/moraisaugusto/machineLearning/master/1.LinearRegression/multivariableLinearRegression.png)


## Equation: Normal Equation for Theta

Normal equation to find the theta values (optimum global) with out ITERATION. This can be better sometimes




```
Theta_0 is: -3.89578087831191
Theta_1 is: 1.1930336441896
```

:boom: _With the normal equation, computing the inversion has complexity **O(n3)**. So
if we have a very large number of features, the normal equation will be slow.
  In practice, when n exceeds **10,000** it might be a good time to go from a
  normal solution to an iterative process._


###### Normal Equation Noninvertibility When implementing the normal equation
in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv'
function will give you a value of θ even if XTX is not invertible.

If X^TX is _noninvertible_, the common causes might be having :

Redundant features, where two features are very closely related (i.e. they are
linearly dependent) Too many features (e.g. m ≤ n). In this case, delete some
features or use "regularization" (to be explained in a later lesson).
Solutions to the above problems include deleting a feature that is linearly
dependent with another or deleting one or more features when there are too many
features.

