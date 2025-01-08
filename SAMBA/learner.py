import  torch
from    torch import nn
from    torch import Tensor
from    torch.nn import init
from    torch.nn import functional as F
from    torch.nn.parameter import Parameter

import  numpy as np
# from    meta import PhaseShifter    no Matryoshka allowed :-P


#%% complex PhaseShifter -> adding noise -> compute power
class PhaseShifter_zy(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    scale: float
    theta: Tensor

    def __init__(self, args, norm_factor = None) -> None:
        super(PhaseShifter_zy, self).__init__()
        self.in_dim = args.n_antenna
        self.in_features = 2 * args.n_antenna
        self.out_features = args.n_wb        # n_widebeams
        self.theta = Parameter(torch.Tensor(self.in_dim, self.out_features))
        self.scale = np.sqrt(args.n_antenna)
        # self.noise_power = args.noise_power
        # self.norm_factor = norm_factor
        
        # self.theta = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.compute_power = ComputePower(self.out_features)
    
    def forward(self, inputs, theta):         # theta is input from outside (fast weight or backward(), step())
        # theta (n_antenna, n_wide_beam), 64 x 12 for instance
        self.real_kernel = (1 / self.scale) * torch.cos(theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(theta)  # 

        cat_kernels_4_real = torch.cat(
            (self.real_kernel, -self.imag_kernel), dim = -1)
        cat_kernels_4_imag = torch.cat(
            (self.imag_kernel, self.real_kernel), dim = -1)
        # 2*n_antenna x 2*n_wide_beam
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag), dim = 0) 
        
        output = torch.matmul(inputs, cat_kernels_4_complex.transpose(1, 0))
        bf_power = self.compute_power(output)
        return bf_power.float()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )    

    # def get_weights(self) -> torch.Tensor:
    #     with torch.no_grad():
    #         real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
    #         imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
    #         beam_weights = real_kernel + 1j*imag_kernel
    #     return beam_weights


#%% calculate power of complex signal sequence   ！！！
class ComputePower(nn.Module):
    def __init__(self, in_feature):
        super(ComputePower, self).__init__()
        self.in_feature = in_feature
        # self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[:, :self.in_feature]
        imag_part = x[:, self.in_feature:]
        sq_real = torch.pow(real_part, 2)
        sq_imag = torch.pow(imag_part, 2)
        abs_values = sq_real + sq_imag
        return abs_values


class Learner(nn.Module):

    def __init__(self, config, args, norm_factor):
        super(Learner, self).__init__()

        self.config = config
        self.scale = np.sqrt(args.n_antenna)
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()                          # ***********************
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        self.codebook = PhaseShifter_zy(args, norm_factor)                                
        # self.compute_power = ComputePower(args.n_wb)

        for _, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'conv1d':
                # [ch_out, ch_in, kernelsz]
                w = nn.Parameter(torch.ones(*param[:3]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                # self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'complexnn':
                theta = nn.Parameter(torch.zeros(*param))       # ***********************
                # init.kaiming_normal_(theta)
                init.uniform_(theta, a = 0, b = 2*np.pi)
                self.vars.append(theta)                         # ***********************
                # self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'linear':
                # w [24, 12] [36, 24] [128, 36]
                w = nn.Parameter(torch.ones(*param))            
                # gain=1 according to cbfinn's implementation
                init.kaiming_normal_(w)
                self.vars.append(w)
                # b [24] [36] [128]  b is initialized as 0
                self.vars.append(nn.Parameter(torch.zeros(param[0])))   # b = nn.Parameter(torch.zeros(param[0]))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'max_pool1d', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError


    def forward(self, x, vars=None, bn_training=True):
        # x is x_spt / s_qry
        if vars is None:
            vars = self.vars        # *********************** self.vars only relies on outside backward() step()

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride = param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'conv1d':
                w = vars[idx]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv1d(x, w)
                idx += 1
                # print(name, param, '\tout:', x.shape)
            elif name == 'complexnn':
                theta = vars[idx]
                x = self.codebook(x, theta)
                idx += 1
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'max_pool1d':
                x = F.max_pool2d(x, param[0], param[1])
            
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def get_theta(self) -> torch.Tensor:
        with torch.no_grad():
            #  vars[0] is theta
            real_kernel = (1 / self.scale) * torch.cos(self.vars[0])  #
            imag_kernel = (1 / self.scale) * torch.sin(self.vars[0])  #        
            beam_weights = real_kernel + 1j*imag_kernel
        return beam_weights
    
    def get_codebook(self) -> np.ndarray:
        return self.get_theta().cpu().detach().clone().numpy()

    def get_weight(self) -> torch.Tensor:
        with torch.no_grad():
            w1 = self.vars[1].cpu().detach().clone().numpy()
            w2 = self.vars[3].cpu().detach().clone().numpy()
            w3 = self.vars[5].cpu().detach().clone().numpy()
            w = [w1, w2, w3]
        return w

    def get_bias(self) -> torch.Tensor:
        with torch.no_grad():
            b1 = self.vars[2].cpu().detach().clone().numpy()
            b2 = self.vars[4].cpu().detach().clone().numpy()
            b3 = self.vars[6].cpu().detach().clone().numpy()
            b = [b1, b2, b3]
        return b
    
    def get_MLP(self) -> np.ndarray:
        # weight = [], bias = []
        # weight[0], weight[1], weight[2] = self.get_weight()
        # bias[0], bias[1], bias[2] = self.get_bias()
        return self.get_weight(), self.get_bias()

    def zero_grad(self, vars = None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars