import torch
import torch.nn as nn
from torch.autograd.functional import jacobian,hessian
from deriv_example import neural_network_x,neural_network,nx,psy_trial
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
    y_all = []
    d_yn_dx = []
    d2_yn_dx2 = []
    psy_t_all = []
    for xi in x_space:
        net_out = neural_network(W, xi)[0][0]
        psy_t = psy_trial(xi, net_out)
        psy_t_all.append(psy_t)
        y_all.append(net_out)

        net_out_d = grad(neural_network_x)(xi)
        d_yn_dx.append(net_out_d)

        net_out_dd = grad(grad(neural_network_x))(xi)
        d2_yn_dx2.append(net_out_dd)
    d_yn_dx = np.array(d_yn_dx)
    d2_yn_dx2 = np.array(d2_yn_dx2)
    psy_t_all = np.array(psy_t_all)
    from torch.autograd.functional import jacobian,hessian

    jac = jacobian(wh.forward,inputs=xt)
    d_yt_dx = jac[:, 0, :].diag()

    dy = np.max(np.abs(d_yn_dx - d_yt_dx.detach().numpy()))

    psy_t_all_torch = psy_trial(xt,yt)

    d_psy_t = np.max(np.abs(psy_t_all - psy_t_all_torch.diag().detach().numpy()))

   
    hes = torch.func.hessian(wh.forward)(xt)


    # dimension of "hes" is (3,1,3,3), so reshaping it to eliminate 2nd dimension
    h3 = hes.reshape(hes.shape[0], hes.shape[2], hes.shape[3])
    # so called "diagonal" of 3rd rank tensor
    qd = torch.diagonal(h3, 0)
    # qd is a matrix, so this time te real diagonal
    d2_yt_dx2 = qd.diag()
    dh = np.max(np.abs(d2_yn_dx2-d2_yt_dx2.detach().numpy()))

    f = lambda x: psy_trial(x[0], x[1])
    xt = xt.reshape(3,1)
    tt = torch.cat((xt, yt), 1)

    psy_t_all_torch = f(tt.T)

    d_psy_t = np.max(np.abs(psy_t_all - psy_t_all_torch.detach().numpy()))


    qq = 0





qq = 0