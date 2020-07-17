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
from gPyTorch import gConv2d
from torch.nn import Linear
import torch.optim as optim
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d

from torch import nn


class Net(Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=gConv2d(3,1,5,deep=0)
        self.conv2=gConv2d(1,1,5,deep=1)
        #self.conv3 = gConv2d(1, 1, deep=1)
        #self.conv3 = gConv2d(2, 2, deep=1)
        #self.conv4 = gConv2d(2, 1, deep=1)
        self.mx=MaxPool2d([1,2])
        self.fc1=Linear(12,3)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        #x=F.relu(self.conv2(x))
        #x = self.conv3(x)
        #x = self.conv4(x)
        x=self.mx(x)
        x=x.view(-1,12)
        x=F.relu(self.fc1(x))
        return x


net=Net()
net.cpu()
print(net)

X_train=load('X_train.npy')
Y_train=100*load('Y_train.npy')


bn=BatchNorm2d(3)
bnf=BatchNorm1d(3)


N_data=1000
inputs=X_train[0:N_data,:,:,:]
target=Y_train[0:N_data,2:5]
inputs=np.moveaxis(inputs,-1,1)
inputs=torch.from_numpy(inputs.astype(np.float32))
inputs=bn(inputs)
input=inputs.detach()
targets=torch.from_numpy(target.astype(np.float32))
targets=bnf(targets)
target=targets.detach()

#output=net(input)

criterion=nn.MSELoss()

#loss=criterion(output,target)

optimizer=optim.Adam(net.parameters(),lr=0.1)
optimizer.zero_grad()

#loss.backward()
#optimizer.step()

shape1 = input.shape
shape2 = target.shape
running_loss=0
for i in range(0,N_data):
    #optimizer.zero_grad()

    inputs=input[i,:,:,:]
    inputs=inputs.view(1,3,6,30)

    targets=target[i,:]
    targets = targets.view(1,3)

    output=net(inputs)
    loss=criterion(output,targets)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i%10==0:
        print('[%d, %5d] loss: %.3f' %
              ( 1, i + 1, running_loss / 10))
        running_loss = 0.0





gnn=gConv2d(1,1,deep=0)
test_input=input[0,0,:,:].clone()
test_input=test_input.view([1,1,6,30])

optimizer = optim.SGD(gnn.parameters(), lr=0.1)
criterion = nn.MSELoss()
for i in range(0,10):
    output = gnn(test_input)
    target=torch.rand_like(output)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    params=list(gnn.parameters())
    print(params[0][0,0,0])
    optimizer.step()
    print(params[0][0,0,0])

lin = nn.Linear(10, 2)
with torch.no_grad():
    lin.weight[0][0] = 1.

x = torch.randn(1, 10)
output = lin(x)
output.mean().backward()