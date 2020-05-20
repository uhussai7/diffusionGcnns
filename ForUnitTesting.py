import groupConv
import numpy as np
import group_d6 as d6
from numpy import load
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


#for scalar
inn=3
out=1
H=6 #height
W=30 #width
b=1 #batch size
g=groupConv.groupConv(out) #this makes a layer with one filter
g.build([b,H,W,inn]) #say this is the input size it expects

print(g.kernel.shape) #will give (7,3,1) 7 for hexogonal weights 3 for input channels and 1 for output channel/ number of filters
expanded_kernel=d6.conv2d(1,g.kernel,g.deep) #conv2d modified to return expanded kernel, g.deep is assigned based on input size
print(expanded_kernel.shape) #will give (3,3,3,12) 3,3 for matrix form of hexagonal filter, 3 input channels, and 12 orientation channels


#for regular or deep
inn=12 #minimum orientation channels for deep layer (because a single 0 layer channel generates 12 channels for next layer)
out=1
H=6
W=30
b=1 #batch size
g1=groupConv.groupConv(out) #this makes a layer with one filter
g1.build([b,H,W,inn]) #say this is the input size it expects
print(g1.kernel.shape) #will give (7,12,1,1) 7 for usual hexagon and
                       # 12 new weights for orientation channels,
                       # 1 for input channel (its 12 inputs but
                       # 1 is the "true" channel 12
                       #are orientaion and last one is for 1 filter)
expanded_kernel1=d6.conv2d(1,g1.kernel,g1.deep)
print(expanded_kernel1.shape) #will [3,3,12,1,12] [3,3,12] are 3d weights, 1 is the true input channel, 12 are the different rotations and reflections