import math
import warnings
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.functional import pad
import g_d6 as d6
import numpy as np
from numpy import load


class gConv2d(Module):
    def __init__(self,in_channels,out_channels,h,shells=3,deep=None):
        super(gConv2d,self).__init__()
        self.deep=deep
        self.shells=shells
        self.kernel_size=7

        self.out_channels=out_channels
        self.in_channels = in_channels

        self.kernel_e=[]
        self.bias_e=[]
        self.h=h

        if deep==None:
            raise ValueError('Specify deep')

        self.theta,self.I,self.J= d6.padding_basis(self.h,1)

        #deep vs 1st layer
        #if in_channels==self.shells:
        if self.deep==0:
            self.weight=Parameter(torch.Tensor(out_channels,in_channels,self.kernel_size))
            self.basis_e_h=d6.basis_expansion(self.deep)
            self.basis_e_t=0

        elif self.deep==1: #and (in_channels % 12)==0:
            self.weight = Parameter(torch.Tensor(out_channels,in_channels,12,self.kernel_size))
            self.basis_e_t, self.basis_e_h = d6.basis_expansion(self.deep)

        else:
            print("input_dim=%d" % in_channels)
            raise ValueError('The number of channels in deep layer '
                             'not divisible by order of group '
                             'something is wrong')

        self.bias = Parameter(torch.Tensor(out_channels))
        self.bias_basis= d6.bias_basis(out_channels)

        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self,input):
        #with torch.no_grad():
        #self.kernel_e=torch.rand(self.kernel_e.shape)
        self.bias_e = self.bias[self.bias_basis]
        self.kernel_e=d6.basis_mul(self.weight,self.basis_e_h ,self.deep, basis1=self.basis_e_t)
        out=F.conv2d(input.float(),self.kernel_e.float(),bias=self.bias_e)
        out=pad(out,(1,1,1,1))
        return d6.gpad(out,self.theta,self.I,self.J,self.deep)

class opool(Module):
    def __init__(self,in_channels):
        super(opool,self).__init__()
        self.in_channels=in_channels

    def forward(self,input):
        batch_size=input.shape[0]
        H=input.shape[-2]
        W=input.shape[-1]
        D=12
        input_pool=torch.zeros(batch_size,self.in_channels,H,W, requires_grad=False)
        pooled, ind=F.max_pool3d(input,[12,H,W],return_indices=True)
        #inds=np.asarray(ind.detach().reshape(batch_size*self.in_channels).numpy())
        inds = np.asarray(ind.detach().reshape(batch_size * self.in_channels).cpu().numpy())
        subs = np.asarray(np.unravel_index(inds, [D*self.in_channels, H, W]))
        subs = subs.reshape(3,batch_size,self.in_channels)
        for b in range(0,batch_size):
            input_pool[b,:,:,:]=input[b,subs[0,b,:],:,:].clone()
        return input_pool.cuda()
