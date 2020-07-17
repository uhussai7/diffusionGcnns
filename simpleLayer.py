import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
from torch import nn
import torch.optim as optim



class simpleLayer(Module):
    def __init__(self,N,N_e):
        super(simpleLayer,self).__init__()

        self.weight=Parameter(torch.Tensor(N))
        self.basis=torch

        self.reset_parameters()

        for i in range(0,N_e):
            for j in range(0,N):
                k=(i+j)%N
                self.weight_e[j,i]=self.weight[k].view(1,1)


    def reset_parameters(self):
        init.uniform_(self.weight)


    def forward(self,input):
        return torch.matmul(input,self.weight_e)



N=10
N_e=5
b=32

net=simpleLayer(N,N_e)

optimizer=optim.Adam(net.parameters(),lr=0.001)
optimizer.zero_grad()


X=torch.rand(b,N)
Y=torch.rand(b,N_e)


output=net(X)

criterion=nn.L1Loss()
loss=criterion(output,Y)


for epoch in range(0,10):
    #print(n)
    X = torch.rand(b, N)
    Y = torch.rand(b, N_e)

    optimizer.zero_grad()

    output=net(X)

    loss=criterion(output,Y)
    #loss.backward(retain_graph=True)
    loss.backward()
    #optimizer.step()
    print(loss.item())




#test
h=5
strip=np.array([1,2,3,4])
strip=strip.reshape(1,h-1)
test=strip
for c in range(0,4):
    test=np.column_stack((test,strip+(c+1)*(h+1)))
    #test=(test,test+h+1)

for t in range(0,12):
    for i in range(0,6):
        for j in range(0,30):
            test[t,i,j]=(1000*t+100*i+j)