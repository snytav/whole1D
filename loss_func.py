import torch
import torch.nn as nn
from torch.autograd.functional import jacobian,hessian
from deriv_example import neural_network_x,neural_network,nx,psy_trial
from deriv_example import W,x_space,pn,sigmoid
import autograd.numpy as np
from WholePoisNet import WholePoisNet,w,w1,xt
from autograd import grad




def loss_numpy_torch(W,x_space,nx,w1,xt,wh):

    if wh.numpy_check:
       yn = np.array([neural_network(W, xi)[0][0] for xi in x_space])

       wh.fc1.set_weight(w)   #weight = nn.Parameter(w3.T)
       wh.fc2.weight = nn.Parameter(w1)

    yt = wh(xt)
    
    y_all = []
    d_yn_dx = []
    d2_yn_dx2 = []
    psy_t_all = []
    gradients_of_trial = []
    all_second_gradient_of_trial = []
    func_all = []
    err_sqr_all = []
    for xi in x_space:
        net_out = neural_network(W, xi)[0][0]
        psy_t = psy_trial(xi, net_out)
        psy_t_all.append(psy_t)
        y_all.append(net_out)

        net_out_d = grad(neural_network_x)(xi)
        d_yn_dx.append(net_out_d)

        net_out_dd = grad(grad(neural_network_x))(xi)
        d2_yn_dx2.append(net_out_dd)
        from deriv_example import psy_grad
        gradient_of_trial = psy_grad(xi, net_out)
        gradients_of_trial.append(gradient_of_trial)
        from deriv_example import psy_grad2,f
        second_gradient_of_trial = psy_grad2(xi, net_out)
        all_second_gradient_of_trial.append(second_gradient_of_trial)
        func = f(xi, psy_t, gradient_of_trial)  # right part function
        func_all.append(func)
        err_sqr = (second_gradient_of_trial - func) ** 2
        err_sqr_all.append(err_sqr)

    d_yn_dx = np.array(d_yn_dx)
    d2_yn_dx2 = np.array(d2_yn_dx2)
    psy_t_all = np.array(psy_t_all)
    gradients_of_trial = np.array(gradients_of_trial)
    all_second_gradient_of_trial = np.array(all_second_gradient_of_trial)
    func_all = np.array(func_all)
    err_sqr_all = np.array(err_sqr_all)

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

    f1 = lambda x: psy_trial(x[0], x[1])
    xt = xt.reshape(3,1)
    tt = torch.cat((xt, yt), 1)

    psy_t_all_torch = f1(tt.T)

    d_psy_t = np.max(np.abs(psy_t_all - psy_t_all_torch.detach().numpy()))

    gradients_of_trial_torch = jacobian(f1,inputs=tt.T)
    g = gradients_of_trial_torch
    g0 = g[:,0,:]
    gt = g0.diag()

    d_grad = np.max(np.abs(gt.detach().numpy()-gradients_of_trial))

    # only d_psy_dx2 found at h[0,0,0,0,0]
    h = torch.func.hessian(f1)(tt.T)
    # grad2_of_trial_torch
    # with psy-function defined here: y = lambda x :x + x**2*yt
    # all OK !!!
    #redefining psy_trial
    psy = lambda x: x + x ** 2 * yt
    h_psy = torch.func.hessian(psy)(xt)
    hh = h_psy.reshape(h_psy.shape[0], h_psy.shape[2], h_psy.shape[4])
    h_2D_diag = torch.einsum('iii->i', hh)
    d_psy_hesssian = np.max(np.abs(h_2D_diag.detach().numpy()-all_second_gradient_of_trial))

    # Right-hand-side
    func_torch = f(xt, psy(xt), gt)
    d_func = np.max(np.abs(func_all-func_torch.diag().detach().numpy()))


    # now we must be ready to evaluate loss function
    err_sqr_torch = torch.pow(func_torch.diag()-h_2D_diag,2.0)
    d_err_sqr = np.max(np.abs(err_sqr_torch.detach().numpy()-err_sqr_all))
    return torch.sum(err_sqr_torch)






if __name__ == "__main__":
    wh = WholePoisNet(nx,True)
    y = loss_numpy_torch(W, x_space, nx, w1, xt,wh)
    qq = 0