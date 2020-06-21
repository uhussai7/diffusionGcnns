import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init



class simpleLayer(Module):
    def __init__(self,N,N_e):
        super(simpleLayer,self).__init__()

        self.weight=Parameter(torch.Tensor(N))
        self.weight_e=torch.zeros([N,N_e],requires_grad=False)

        self.reset_parameters()

        for i in range(0,N_e):
            for j in range(0,N):
                 k=(j-i) % N
                 self.weight_e[j,i]=self.weight.as_strided([1],[0],k)

        self.weight_e=self.weight.expand(N_e)

    def reset_parameters(self):
        init.uniform_(self.weight)


    def forward(self,input):
        return torch.matmul(input,self.weight)



N=10
N_e=5
b=32

net=simpleLayer(N,N_e)


X=torch.rand(b,N)
Y=torch.rand(b,N,N_e)


