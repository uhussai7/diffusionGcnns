import groupConv
import numpy as np
import group_d6 as d6
from numpy import load
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from tensorflow import executing_eagerly
import tensorflow as tf
from tensorflow.keras import Input
import gConv


executing_eagerly()
inn=3
out=1
# g=groupConv.groupConv(1)
# g.build([2,6,30,inn])

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
Y_train=load('K:\\Datasets\\DiffusionIcosahedron\\Y_train.npy')
input=X_train[0:2,:,:,:]
input=K.variable(input)
g=groupConv.groupConv(2)
g.build(input.shape)
output=g.call(input)
#output=d6.conv2d(input,g.kernel,g.deep)



# g1 = groupConv.groupConv(3)
# g1.build(output.shape)
# output1=d6.onv2d(output,g1.kernel,g1.deep)
#
# output1=tf.py_function(d6.conv2d,(output,g1.kernel,g1.deep),Tout=tf.float32)

# s=Input(batch_shape=(100,6,30,3))
#
# model = Sequential()
# model.add(gConv.gConv(3,input_shape=(6,30,3)))
# model.add(Flatten(data_format="channels_last"))
# model.add(Dense(3))
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
# #
# model.fit(10/X_train,100*Y_train[:,4:7],batch_size=500)

# input1=Input(shape=(6,30,3))
# input2=Input(shape=(6,30,3*12))
model2 = Sequential()
model2.add(groupConv.groupConv(3,input_shape=(6,30,3)))
model2.add(Flatten(data_format="channels_last"))
model2.add(Dense(3))
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
#
# #
# #
# #
# model1 = Sequential()
# model1.add(Conv2D(3,kernel_size=3,input_shape=(6,30,3)))
# model1.add(Flatten(data_format="channels_last"))
# model1.add(Dense(3))
# model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# #
# #
model2.fit(X_train,Y_train[:,4:7],epochs=50)

#shape=input.shape.as_list()
#shape=[2,]+shape
#input=K.reshape(input,shape)

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