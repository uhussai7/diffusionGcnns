import math
import warnings
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.module import Module
import group_d6 as d6
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

        if deep==None:
            raise ValueError('Specify deep')

        #deep vs 1st layer
        #if in_channels==self.shells:
        if self.deep==0:
            #self.deep=0
            self.weight=Parameter(torch.Tensor(out_channels,in_channels,self.kernel_size))
            self.weight_d=self.weight.detach()

        elif self.deep==1: #and (in_channels % 12)==0:
            #real_in=int(in_channels/12)
            self.weight = Parameter(torch.Tensor(out_channels,in_channels,12,self.kernel_size))
            self.weight_d = self.weight.detach()
        else:
            print("input_dim=%d" % in_channels)
            raise ValueError('The number of channels in deep layer '
                             'not divisible by order of group '
                             'something is wrong')
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self,input):

        self.kernel_e=np.copy( d6.conv2d(1,self.weight_d.numpy(),self.deep))
        self.kernel_e=torch.from_numpy(self.kernel_e)
        self.kernel_e.requires_grad_(True)
        return d6.gpad(F.conv2d(input,self.kernel_e,padding=(1,1)),self.deep)

## testing
X_train=load('K:\\Datasets\\DiffusionIcosahedron\\X_train.npy')
Y_train=load('K:\\Datasets\\DiffusionIcosahedron\\Y_train.npy')
input=X_train[0:1,:,:,:]
input=np.moveaxis(input,-1,1)
input=torch.from_numpy(input)

gnn=gConv2d(2,3,deep=0)
kernel=gnn.weight_d.numpy()

out=gnn(input)
#gnn1=gConv2d(3,2,deep=1)


test=torch.zeros(3,2,4,3)
for a in range(0,3):
    for b in range(0,2):
        for c in range(0,4):
            for d in range(0,3):
                test[a,b,c,d]=a*1000+b*100+c*10+d
