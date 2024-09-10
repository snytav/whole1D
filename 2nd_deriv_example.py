# -*- coding: utf-8 -*-
"""ODE 2 example.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/snytav/Neural-Networks-for-Differential-Equations/blob/master/ODE%202%20example.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import autograd.numpy as np
import torch.nn
from autograd import grad
import autograd.numpy.random as npr

from matplotlib import pyplot as plt
# %matplotlib inline

nx = 3
dx = 1. / nx

from PoisNet import PoisNet

pn = PoisNet(nx)

def f(x, psy, dpsy):
    '''
        d2(psy)/dx2 = f(x, dpsy/dx, psy)
        This is f() function on the right
    '''
    return -1./5. * np.exp(-x/5.) * np.cos(x) - 1./5. * dpsy - psy


def psy_analytic(x):
    '''
        Analytical solution of current problem
    '''
    return np.exp(-x/5.) * np.sin(x)

x_space = np.linspace(0, 2, nx)
y_space = psy_analytic(x_space)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

def psy_trial(xi, net_out):
    return xi + xi**2 * net_out

psy_grad = grad(psy_trial)
psy_grad2 = grad(psy_grad)

def loss_function(W, x):
    loss_sum = 0.

    for xi in x:
        net_out = neural_network(W, xi)[0][0]

        net_out_d = grad(neural_network_x)(xi)
        net_out_dd = grad(grad(neural_network_x))(xi)

        psy_t = psy_trial(xi, net_out)

        gradient_of_trial = psy_grad(xi, net_out)
        second_gradient_of_trial = psy_grad2(xi, net_out)

        func = f(xi, psy_t, gradient_of_trial) # right part function

        err_sqr = (second_gradient_of_trial - func)**2
        loss_sum += err_sqr

    return loss_sum

W = [npr.randn(1, nx), npr.randn(nx, 1)]
lmb = 0.001

y = neural_network(W,x_space[1])
x = torch.from_numpy(x_space).float()
yt = torch.matmul(x,pn.fc1.weight.T)

# 1st layer



pn.fc1.weight = torch.nn.Parameter(torch.from_numpy(W[0]))
#pn.fc1.bias   = torch.nn.Parameter(torch.zeros(pn.fc1.bias.shape[0]).double())
pn.fc2.weight = torch.nn.Parameter(torch.from_numpy(W[1]).reshape(1,3))
pn.fc2.bias   = torch.nn.Parameter(torch.zeros(pn.fc2.bias.shape[0]).double())

y1 = np.dot(x_space[1],W[0])
y1t = pn.fc1(torch.from_numpy(x_space).float()[1])

s1 = sigmoid(y1)
s1t = torch.sigmoid(y1t)

y2 = np.dot(s1,W[1])
y2t = pn.fc2(s1t)


yt = pn.forward(torch.ones(1))



for i in range(50):
    loss_grad =  grad(loss_function)(W, x_space)

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]

print( loss_function(W, x_space))
res = [psy_trial(xi, neural_network(W, xi)[0][0]) for xi in x_space]


plt.figure()
plt.plot(x_space, y_space)
plt.plot(x_space, res)
plt.show()





