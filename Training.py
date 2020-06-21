import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
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
from torch.utils.data import DataLoader


##Get data##
X_train=load('K:\\Datasets\\DiffusionIcosahedron\\X_train.npy')
Y_train=load('K:\\Datasets\\DiffusionIcosahedron\\Y_train.npy')
X_test=load('K:\\Datasets\\DiffusionIcosahedron\\X_test.npy')
Y_test=load('K:\\Datasets\\DiffusionIcosahedron\\Y_test.npy')
N_train=500
##Tensor flow timing##
#Prepare data
X_train_t=np.copy(10/X_train[0:N_train,:,:,:])
Y_train_t=np.copy(Y_train[0:N_train,1:4])
X_train_t[np.isinf(X_train_t)]=0

# #model
#model = Sequential() #model
# model.add(Conv2D(8,kernel_size=3,activation='linear', input_shape= (6,30,3)))
# model.add(Conv2D(16,kernel_size=3,activation='linear'))
# #model.add(Conv2D(32,kernel_size=3,activation='relu'))
# #model.add(Conv2D(32,kernel_size=3,activation='relu'))
# #model.add(Conv2D(32,kernel_size=3,activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# #model.add(Dense(20,activation='linear'))
# model.add(Dense(3,activation='linear'))
# #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
# model.fit(X_train_t,100*Y_train_t, epochs=50)


##Pytorch timing##
#Prepare data
class Net(Module):
    def __init__(self):
        super(Net,self).__init__()
        self.flat=2160
        self.conv1=gConv2d(3,4,deep=0)
        self.conv2 = gConv2d(4, 4, deep=1)
        #self.conv1=Conv2d(3,32,kernel_size=3,padding=[1,1])
        #self.conv2=Conv2d(32,16,kernel_size=3,padding=[1,1])
        #self.conv3=Conv2d(16,8,kernel_size=3,padding=[1,1])

        #self.conv2=gConv2d(1,1,deep=1)
        #self.conv1 = Conv2d(3,125,3)
        #self.conv2 = Conv2d(125,100,3)
        self.pool=opool(4)

        #self.conv3 = gConv2d(2, 1, deep=1)
        #self.conv3 = gConv2d(2, 2, deep=1)
        #self.conv4 = gConv2d(2, 1, deep=1)
        self.mx=MaxPool2d([2,2])
        #self.fc1=Linear(336,100)
        self.fc2 = Linear(int(self.flat/12), 3)

    def forward(self,x):
        x=self.conv1(x)
        #x=self.conv2(x)
        x =self.conv2(x)
        #x = self.conv4(x)
        x=self.pool(x)
        #x=self.mx(F.relu(self.conv3(x)))
        x=self.mx(x)
        x=x.view(-1,int(self.flat/12))
        #x= self.fc1(x)
        #x =F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

#data
X_train_p=np.copy(10/X_train[0:N_train,:,:,:])
Y_train_p=np.copy(10000*Y_train[0:N_train,1:4])
X_train_p[np.isinf(X_train_p)]=0


inputs=np.moveaxis(X_train_p,-1,1)
inputs=torch.from_numpy(inputs.astype(np.float32))
input=inputs.detach()

target=Y_train_p
targets=torch.from_numpy(target.astype(np.float32))
target=targets.detach()

#net
net=Net()

#criterion=nn.MSELoss()
criterion=nn.L1Loss()

optimizer=optim.Adam(net.parameters(),lr=0.001)
optimizer.zero_grad()

running_loss=0

train=torch.utils.data.TensorDataset(input,target)
trainloader=DataLoader(train,batch_size=32)

train_loader_iter = iter(trainloader)
imgs, labels = next(train_loader_iter)


for epoch in range(0,3):
    for n,(inputs,targets) in enumerate(trainloader,0):
        #print(n)

        optimizer.zero_grad()

        output=net(inputs)

        loss=criterion(output,targets)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
    else:
        print(running_loss/len(trainloader))
    #if i%N_train==0:
    #    print('[%d, %5d] loss: %.3f' %
    #          ( 1, i + 1, running_loss / 100))
    running_loss = 0.0








