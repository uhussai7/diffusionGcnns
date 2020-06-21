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
from gPyTorch import opool
from torch.nn import Linear
import torch.optim as optim
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d
from torch.nn import Conv2d

from torch import nn


class Net(Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=gConv2d(1,1,deep=0)
        self.conv1=Conv2d(1,32,kernel_size=3,padding=[1,1])
        self.conv2=Conv2d(32,16,kernel_size=3,padding=[1,1])
        self.conv3=Conv2d(16,8,kernel_size=3,padding=[1,1])

        #self.conv2=gConv2d(1,1,deep=1)
        #self.conv1 = Conv2d(3,125,3)
        #self.conv2 = Conv2d(125,100,3)
        self.pool=opool(1)

        #self.conv3 = gConv2d(2, 1, deep=1)
        #self.conv3 = gConv2d(2, 2, deep=1)
        #self.conv4 = gConv2d(2, 1, deep=1)
        self.mx=MaxPool2d([1,2])
        self.fc1=Linear(336,100)
        self.fc2 = Linear(100, 3)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(self.conv2(self.mx(x)))
        #x =F.relu(self.conv3(x))
        #x = self.conv4(x)
        #x=self.pool(x)
        x=self.mx(F.relu(self.conv3(x)))
        x=x.view(-1,336)
        x= self.fc1(x)
        x =self.fc2(x)
        return x


net=Net()
net.cpu()
print(net)

X_train=load('K:\\Datasets\\DiffusionIcosahedron\\X_train.npy')
Y_train=load('K:\\Datasets\\DiffusionIcosahedron\\Y_train.npy')


bn=BatchNorm2d(1)
bnf=BatchNorm1d(1)


N_data=10000
inputs=np.asarray(1/(1+X_train[0:N_data,:,:,2]))
inputs=inputs.reshape(N_data,6,30,1)
inputs=np.moveaxis(inputs,-1,1)
inputs=torch.from_numpy(inputs.astype(np.float32))
#inputs=bn(inputs)
input=inputs.detach()

target=Y_train[0:N_data,1:4]
#target=target.reshape(N_data,1,1)
targets=torch.from_numpy(target.astype(np.float32))
#targets=bnf(targets)
target=targets.detach()
target=10000*target
#output=net(input)

criterion=nn.MSELoss()

#loss=criterion(output,target)

optimizer=optim.SGD(net.parameters(),lr=0.000001)
optimizer.zero_grad()

#loss.backward()
#optimizer.step()

shape1 = input.shape
shape2 = target.shape
running_loss=0

for epoch in range(0,5):
    for i in range(0,N_data):
        #optimizer.zero_grad()

        inputs=input[i,:,:,:]
        inputs=inputs.view(1,1,6,30)

        targets=target[i]
        targets = targets.view(1,3)

        output=net(inputs)
        #print(output.shape)
        loss=criterion(output,targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%100==0:
            print('[%d, %5d] loss: %.3f' %
                  ( 1, i + 1, running_loss / 100))
            running_loss = 0.0


N_test=25
inputs=np.asarray(X_train[N_data:N_data+N_test,:,:,:])
#inputs[np.isinf(inputs)]=0
target=Y_train[N_data:N_data+N_test,1:4]
inputs=np.moveaxis(inputs,-1,1)
print(inputs.shape)
inputs=torch.from_numpy(inputs.astype(np.float32))
#inputs=bn(inputs)
input=inputs.detach()
targets=torch.from_numpy(target.astype(np.float32))
#targets=bnf(targets)
target=targets.detach()
target=10000*target
#output=net(input)

test=[]
for i in range(0,N_test):
    #optimizer.zero_grad()
    print(i)
    inputs=input[i,:,:,:]
    inputs=inputs.view(1,1,6,30)
    print(inputs[0,1,3,5])

    targets=target[i,:]
    targets = targets.view(1,3)

    print(net(inputs))
    test.append(net(inputs))




import matplotlib.pyplot as plt

plt.imshow(input[0,0,:,:])



# net=Net()
# print(net.conv1.weight[0,0,:])
# net(input)
# print(net.conv1.kernel_e[0:12,0,:,:])
#
# print(net.conv2.weight[0,0,0,:])
# net(input)
# for i in range(0,12):
#     print(i)
#     print(net.conv2.kernel_e[i,i-1,:,:])
#
#
# weights=torch.zeros(7,12,1)
# for t in range(0,12):
#     for w in range(0,7):
#         weights[w,t,0]=t*10+w