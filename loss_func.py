import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian,hessian
from deriv_example import neural_network_x,neural_network,nx
from deriv_example import W,x_space,pn


x0 =x_space[0]


y = neural_network(W,x0)
yt = pn.forward(x0)
dy = np.abs(y-yt.item())

qq = 0