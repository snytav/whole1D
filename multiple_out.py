####import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian,hessian
from deriv_example import neural_network_x,neural_network,nx
from deriv_example import W,x_space,pn,sigmoid


x0 =x_space[0]


y = neural_network(W,x0)
yt = pn.forward(x0)
dy = np.abs(y-yt.item())

net_out_arr = []
for xi in x_space:
    net_out = neural_network(W, xi)[0][0]
    net_out_arr.append(net_out)
net_out_arr = np.array(net_out_arr)


qq = 0
from multiply import Multiply
ml = Multiply(nx,nx)
w = torch.from_numpy(W[0])
w3=torch.cat((w,w,w))
x = torch.from_numpy(x_space)
ml.weight = nn.Parameter(w3.T)
y1t = ml(x)

y1 = [np.dot(xi,W[0]) for xi in x_space]
y1 = sigmoid(np.array(y1))
y1t = torch.sigmoid(y1t)
w1 = torch.from_numpy(W[1])
yn = torch.inner(torch.from_numpy(y1[0]),w1.T)  # comparable to neural network result from numpy
y1t = y1t.T
yt = torch.inner(y1t[0],w1.T)
dy2 = np.abs(yn-yt.item())
yt_all = torch.matmul(y1t,w1.reshape(3,1))
y_all = [neural_network(W,xi)[0][0] for xi in x_space]
y_all = np.array(y_all)
y_all = y_all.reshape(3,1)
# yn_all = np.array(y_all)
d_y_all = np.max(np.abs(y_all-yt_all.detach().numpy()))

# 2nd layer
fc = nn.Linear(3,1)
fc.weight.shape
# torch.Size([1, 3])
fc.weight = torch.nn.Parameter(w1.float().reshape(fc.weight.shape))
y = fc(y1t.float())
yn = [neural_network(W,xi)[0][0] for xi in x_space]
yn = np.array(yn)
yt = torch.matmul(y1t,w1)
fc.weight = torch.nn.Parameter(w1.T.float())
# fc.weight.shape
# torch.Size([1, 3])
fc.bias = nn.Parameter(torch.zeros(3,1).float())
y1t = torch.sigmoid(y1t)
yt = fc(y1t.float())
d_y_final = np.max(np.abs(yn-yt.T.detach().numpy()))



qq = 0