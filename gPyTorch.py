import math
import warnings
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.module import Module
import g_d6 as d6
import numpy as np
from numpy import load


class gConv2d(Module):
    def __init__(self,in_channels,out_channels,shells=3,deep=None):
        super(gConv2d,self).__init__()
        self.deep=deep
        self.shells=shells
        self.kernel_size=7

        self.out_channels=out_channels
        self.in_channels = in_channels

        self.kernel_e=[]
        self.bias_e=[]

        if deep==None:
            raise ValueError('Specify deep')

        #deep vs 1st layer
        #if in_channels==self.shells:
        if self.deep==0:
            #self.deep=0
            self.weight=Parameter(torch.Tensor(out_channels,in_channels,self.kernel_size))

        elif self.deep==1: #and (in_channels % 12)==0:
            #real_in=int(in_channels/12)
            self.weight = Parameter(torch.Tensor(out_channels,in_channels,12,self.kernel_size))
        else:
            print("input_dim=%d" % in_channels)
            raise ValueError('The number of channels in deep layer '
                             'not divisible by order of group '
                             'something is wrong')

        self.bias = Parameter(torch.Tensor(out_channels))



        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self,input):
        #with torch.no_grad():
        self.kernel_e= d6.conv2d(1,self.weight,self.deep)
        #self.kernel_e=torch.rand(self.kernel_e.shape)
        self.bias_e=d6.expand_bias(self.bias)
        return d6.gpad(F.conv2d(input.float(),self.kernel_e.float(),padding=(1,1)),self.deep)
        #return F.conv2d(input.float(), self.kernel_e.float(), padding=(1, 1))
