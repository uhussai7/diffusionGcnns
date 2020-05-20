import groupConv
import numpy as np
import group_d6 as d6
from numpy import load
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


inn=3
out=1
g=groupConv.groupConv(1)
g.build([2,6,30,inn])

# inr=int(inn/12)
# weights=np.zeros([7,12,inr,out])
# for th in range(0,12):
#     for o in range(0,out):
#         for i in range(0,inr):
#             weights[0:6,th,i,o]= (i+1)*10+np.linspace(1,6,6)
#             weights[6,th,i,o]=th#10*th+(o+1)
#
# g.kernel=K.variable(weights)

X_train=load('K:\\Datasets\\DiffusionIcosahedron\\X_train.npy')
input=X_train[0:2 ,:,:,:]
input=K.variable(input)
#shape=input.shape.as_list()
#shape=[2,]+shape
#input=K.reshape(input,shape)

output=d6.conv2d(input,g.kernel,g.deep)
#d6= groupConv.group_d6()
# weights=np.zeros([9,1])
# weights[0:6,0]=np.linspace(1,6,6)

# ker=np.zeros([3,3,12,2,3])
# for out in range(0,3):
#     for inn in range(0,2):
#         for theta in range(0,12):
#             weights[-3]=theta
#             weights[-2]=inn
#             weights[-1]=out
#             ker[:,:,theta,inn,out]=np.reshape(d6.unproject(weights),[3,3])
# ker=K.variable(ker)



   # ker=K.reshape(ker,[3,3,1,1])
#print(d6.reflect(ker,0))



# #test out the usual convolution and the sizes
# #load some training data extracted before
# X_train=load('K:\\Datasets\\DiffusionIcosahedron\\X_train.npy')
# Y_train=load('K:\\Datasets\\DiffusionIcosahedron\\Y_train.npy')
# X_test=load('K:\\Datasets\\Diffug.sionIcosahedron\\X_test.npy')
# Y_test=load('K:\\Datasets\\DiffusionIcosahedron\\Y_test.npy')
#
# topleft=np.zeros([3,3,1,1])
# topleft[0,0,0,0]=1
# ker_tensor=K.variable(topleft)
# input = np.zeros([1,6,30,1])
# input[0,:,:,0]=np.copy(X_train[0,:,:,0])
# input_tensor=K.variable(input)
# conv_result=K.conv2d(input,ker_tensor)
#
# plt.figure(1)
# plt.imshow(input_tensor[0,:,:,0])
# plt.figure(2)
# plt.imshow(conv_result[0,:,:,0])