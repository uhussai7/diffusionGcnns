from gPyTorch import gConv2d
import numpy as np
import g_d6 as d6
import torch
import plotly.express as px


conv1=gConv2d(1,1,5,deep=1)

#make a mock function and then pad it

s_mock=np.zeros([12,6,30])
for channels in range(0,12):
    for i in range(0,6):
        for j in range(0,30):
            s_mock[channels,i,j]=channels*1000+ 100*i+j


s_mock=torch.from_numpy(s_mock)
s_mock[:,:,0]=0
s_mock[:,:,-1]=0
s_mock[:,0,:]=0
s_mock[:,-1,:]=0


s_mock_padded=s_mock[conv1.theta,conv1.I,conv1.J]

fig=px.imshow(s_mock_padded[1,:,:])
fig.show()