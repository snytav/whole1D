import torch
import torch.nn as nn
from torch.autograd.functional import jacobian,hessian
from deriv_example import neural_network_x,neural_network,nx
from deriv_example import W,x_space,pn,sigmoid
import autograd.numpy as np
from WholePoisNet import WholePoisNet,w,w1,xt
from autograd import grad



if __name__ == "__main__":
    yn = np.array([neural_network(W, xi)[0][0] for xi in x_space])
    wh = WholePoisNet(nx)
    wh.fc1.set_weight(w)   #weight = nn.Parameter(w3.T)
    wh.fc2.weight = nn.Parameter(w1)
    yt = wh(xt)
    d_yn_dx = []
    for xi in x_space:
        #net_out = neural_network(W, xi)[0][0]

        net_out_d = grad(neural_network_x)(xi)
        d_yn_dx.append(net_out_d)

    d_yn_dx = np.array(d_yn_dx)
    from torch.autograd.functional import jacobian

    jac = jacobian(wh.forward,inputs=xt)
    d_yt_dx = jac[:, 0, :].diag()

    dy = np.max(np.abs(d_yn_dx - d_yt_dx.detach().numpy()))

    qq = 0





qq = 0