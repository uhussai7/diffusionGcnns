import numpy as np
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
from torch.nn import GroupNorm
from torch.nn import Conv2d
from torch import nn
from torch.utils.data import DataLoader
import math as m
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from dipy.core.sphere import cart2sphere
from dipy.core.sphere import sphere2cart
import plotly.express as px

from mathutils.geometry import intersect_ray_tri
from mathutils import Vector

#network stuff
N_train = 10000
start = 4
end = 7
last = 8



##Get data##
X_train = load('X_train.npy')
Y_train = load('Y_train.npy')
X_test = load('X_test.npy')
Y_test = load('Y_test.npy')

def checker(x,y,z):
    vertices = [[0.894427, 0.000000, 0.447214],
                [0.000000, 0.000000, 1.000000],
                [0.276393, 0.850651, 0.447214],
                [0.723607, 0.525731, -0.447214],
                [-0.276393, 0.850651, -0.447214],
                [-0.000000, 0.000000, -1.000000],
                [0.276393, -0.850651, 0.447214],
                [0.723607, -0.525731, -0.447214],
                [-0.276393, -0.850651, -0.447214],
                [-0.723607, -0.525731, 0.447214],
                [-0.894427, 0.000000, -0.447214],
                [-0.723607, 0.525731, 0.447214]]
    faces=[[4,3,2],[3,4,5],
         [10, 4, 11], [4, 10, 5],
         [8, 10, 9], [10, 8, 5],
         [7, 8, 6], [8, 7, 5],
         [3, 7, 0], [7, 3, 5]
           ]

    for f in range(0,10):
        v1= Vector(vertices[faces[f][0]])
        v2= Vector(vertices[faces[f][1]])
        v3= Vector(vertices[faces[f][2]])
        ray=Vector([x,y,z])
        orig=Vector([0,0,0])
        check=intersect_ray_tri(v1,v2,v3,ray,orig)
        if check != None:
            return 1
    return 0


#mess with data
X_train_p = np.copy(1000/X_train[0:N_train, :, :, :])
Y_train_p = (np.copy(Y_train[0:N_train, start:end]))
# Y_train_p[:,1:4]=np.copy(1000*Y_train[:,1:4])
X_train_p[np.isinf(X_train_p)] = 0

temp=np.zeros([N_train,2])
for i in range(0,N_train):
    x=Y_train_p[i,0]
    y=Y_train_p[i,1]
    z=Y_train_p[i,2]
    # if z<0:
    #     Y_train_p[i,:]=-Y_train_p[i,:]
    if checker(x,y,z)==1:
        Y_train_p[i,:]=-Y_train_p[i,:]
    XsqPlusYsq = x ** 2 + y ** 2
    temp[i,0] = (m.atan2(z, m.sqrt(XsqPlusYsq)) + np.pi/2)  # theta
    temp[i, 1] = (m.atan2(y, x) + np.pi)
    if temp[i,0] > np.pi/2:
        temp[i,0]=np.pi - temp[i,0]
        temp[i,1]=(temp[i,1] + np.pi) % 2*np.pi
    temp[i,0]=temp[i,0]/(np.pi/2)
    temp[i, 1] = temp[i, 1] / (2*np.pi )

#Y_train_p=temp


#move stuff around
inputs = np.moveaxis(X_train_p, -1, 1)
inputs = torch.from_numpy(inputs.astype(np.float32))
input = inputs.detach()
input = input.cuda()

target = Y_train_p
targets = torch.from_numpy(target.astype(np.float32))
target = targets.detach()
target = target.cuda()



##Pytorch timing##
# Prepare data
h=5
H=h+1
W=5*H
last=32
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flat = 2160
        # self.conv1 = gConv2d(3, 8, h, deep=0)
        # self.gn1 = GroupNorm(8, 8 * 12)
        # self.conv2 = gConv2d(8, 16, h, deep=1)
        # self.gn2 = GroupNorm(16, 16 * 12)
        # self.conv3 = gConv2d(16, 32, h, deep=1)
        # self.gn3 = GroupNorm(32, 32 * 12)
        # self.gn4 = GroupNorm(8, 8 * 12)
        # self.conv4 = gConv2d(8, 8, h, deep=1)
        # self.gn5 = GroupNorm(8, 8 * 12)
        # self.conv5 = gConv2d(8, 8, h, deep=1)
        # self.gn6 = GroupNorm(8, 8 * 12)
        # self.conv6 = gConv2d(8, 8, h, deep=1)

        self.conv1=Conv2d(3,8,kernel_size=3,padding=[1,1])
        self.gn1=BatchNorm2d(8)
        self.conv2=Conv2d(8,16,kernel_size=3,padding=[1,1])
        self.gn2 = BatchNorm2d(16)
        self.conv3=Conv2d(16,32,kernel_size=3,padding=[1,1])
        self.gn3 = BatchNorm2d(32)

        #self.conv2=gConv2d(1,1,deep=1)
        # self.conv1 = Conv2d(3,125,3)
        # self.conv2 = Conv2d(125,100,3)
        self.pool = opool(last)
        # self.conv3 = gConv2d(2, 1, deep=1)
        # self.conv3 = gConv2d(2, 2, deep=1)
        # self.conv4 = gConv2d(2, 1, deep=1)
        self.mx = MaxPool2d([2, 2])
        self.fc1=Linear(int(last * H * W / 1),3)#,end - start-1)
        self.fc2=Linear(3,3)
        self.fc3=Linear(100,100)
        self.fc4 = Linear(3, end - start)


    def forward(self, x):

        x = F.relu( self.conv1(x))
        x = self.gn1(x)
        x =F.relu( self.conv2(x))
        x = self.gn2(x)
        x =F.relu(self.conv3(x))
        x=self.gn3(x)
        #x =F.relu(self.conv4(x))
        # x=self.gn4(x)
        # x = F.relu(self.conv5(x))
        # x = self.gn5(x)
        # x = F.relu(self.conv6(x))
        # x = self.gn6(x)
        # # x = self.bn1(x)
        #x = self.pool(x)
        #x=self.mx(F.relu(x))
        #x=self.mx(x)
        x = x.view(-1, int(last * H * W / 1))
        x= F.relu(self.fc1(x))
        # x =F.relu(self.fc2(x))
        #x = self.fc2(x)
        #x = self.fc3(x)
        x = self.fc4(x)

        return x


# data
# X_train_p=np.copy(10/X_train[0:N_train,:,:,:])
# Y_train_p=np.copy(10*Y_train[0:N_train,1:4])
# X_train_p[np.isinf(X_train_p)]=0



# net
net = Net().cuda()


def Myloss(output,target):
    x=output
    y=target
    norm=x.norm(dim=-1)
    norm=norm.view(-1,1)
    norm=norm.expand(norm.shape[0],3)
    x=x/norm
    loss=x*y
    loss=1-loss.sum(dim=-1)
    return loss.mean().abs()

#criterion = nn.MSELoss()
#criterion=nn.SmoothL1Loss()
#criterion=nn.CosineSimilarity()
criterion=Myloss


optimizer = optim.Adamax(net.parameters(), lr=0.01)#, weight_decay=0.001)
optimizer.zero_grad()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)



running_loss = 0

train = torch.utils.data.TensorDataset(input, target)
trainloader = DataLoader(train, batch_size=16)

train_loader_iter = iter(trainloader)
imgs, labels = next(train_loader_iter)

for epoch in range(0, 120):
    print(epoch)
    for n, (inputs, targets) in enumerate(trainloader, 0):
        # print(n)

        optimizer.zero_grad()

        output = net(inputs.cuda())

        loss = criterion(output, targets)
        loss=loss.sum()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(running_loss / len(trainloader))
    # if i%N_train==0:
    #    print('[%d, %5d] loss: %.3f' %
    #          ( 1, i + 1, running_loss / 100))
    scheduler.step(running_loss)
    running_loss = 0.0

torch.save(net.state_dict(), 'net')

#mess with data
N_test=2500
X_test_p = np.copy(1000 / X_test[0:N_test, :, :, :])
Y_test_p = (np.copy(Y_test[0:N_test, start:end]))
# Y_train_p[:,1:4]=np.copy(1000*Y_train[:,1:4])
X_test_p[np.isinf(X_test_p)] = 0

temp=np.zeros([N_test,2])
for i in range(0,N_test):
    x=Y_test_p[i,0]
    y=Y_test_p[i,1]
    z=Y_test_p[i,2]
    # if z < 0:
    #     Y_test_p[i, :] = -Y_test_p[i, :]
    if checker(x, y, z) == 1:
        Y_test_p[i, :] = -Y_test_p[i, :]
    XsqPlusYsq = x ** 2 + y ** 2
    temp[i,0] = m.atan2(z, m.sqrt(XsqPlusYsq)) + np.pi/2  # theta
    temp[i, 1] = m.atan2(y, x) + np.pi
    if temp[i,0] > np.pi/2:
        temp[i,0]=np.pi - temp[i,0]
        temp[i,1]=(temp[i,1] + np.pi) % 2*np.pi
    temp[i,0]=temp[i,0]/(np.pi/2)
    temp[i, 1] = temp[i, 1] / (2*np.pi )

#Y_test_p=temp

#move stuff around
inputs_test = np.moveaxis(X_test_p, -1, 1)
inputs_test = torch.from_numpy(inputs_test.astype(np.float32))
input_test = inputs_test.detach()
input_test = input_test.cuda()

target_test = Y_test_p
targets_test = torch.from_numpy(target_test.astype(np.float32))
target_test = targets_test.detach()
target_test = target_test.cuda()

pred_test=net(input_test.contiguous())
norm=pred_test.norm(dim=-1)
norm=norm.view(-1,1)
norm=norm.expand(norm.shape[0],3)
pred_test=pred_test/norm

# X_test_p=np.copy(10/X_test[0:N_train,:,:,:])
# Y_test_p=np.copy(10000*Y_test[0:N_train,0:4])
# X_test_p[np.isinf(X_test_p)]=0


# inputs=np.moveaxis(X_test_p,-1,1)
# inputs=torch.from_numpy(inputs.astype(np.float32))
# inputs=inputs.detach().cuda()

# target=Y_test_p
# targets=torch.from_numpy(target.astype(np.float32))
# targets=targets.detach().cuda()

# pred=net(inputs)
# true=targets

pred=pred_test
true=target_test

fig=plt.figure()
test=pred*true
test=test.sum(-1)
test=test.abs()
plt.hist(test.cpu().detach().numpy())

test_noabs=pred*true
test_noabs=test_noabs.sum(-1)

fig, axs = plt.subplots(3)
s=-1.2
t=1.2#2*np.pi

for i in range(0,N_test):
    if test_noabs[i] <0:
        pred[i] = -pred[i]

for r in range(0,3):
   axs[r].set_aspect('equal')
   #axs[r].set_xlim([s,t])
   #axs[r].set_ylim([s, t])
   axs[r].scatter(true[:,r].detach().cpu().numpy(), pred[:,r].detach().cpu().numpy(),s=10,alpha=0.3)
   axs[r].plot([s,t],[s,t])


##plot the "images"
fig=px.imshow(1000/X_train[0,:,:,0])
fig.show()




##Tensor flow timing##
# Prepare data
# X_train_t = np.copy(10 / X_train[0:N_train, :, :, :])
# Y_train_t = np.copy(10 * Y_train[0:N_train, start:end])
# X_train_t[np.isinf(X_train_t)] = 0
#



# #model
# model = Sequential() #model
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


