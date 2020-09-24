import somemath
from somemath import isomesh
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import matplotlib.cm as cm
from dipy.core.sphere import cart2sphere
from dipy.core.sphere import sphere2cart
from gPyTorch import gConv2d




# use this script to make some 3d visualizations of the icosahedron and the projective nature
# of the padding. Keeping this visualization as a reference one can check that the padding in
# flat icosahedron map is correct. Note that this is only one chart and the padding will not
# only be for all charts but also over all orientations -- not so simple.

iso=isomesh()
iso.get_icomesh()
#iso.makeFlat()
#plt.imshow(iso.s_flat)
#plt.colorbar()
iso.plot_icomesh()
c=0
m= iso.m
N= m+1
height = N+1
width = 5*(N+1)
ii = np.linspace(0, N - 1, N)
jj = np.linspace(0, N - 1, N)
jjj, iii = np.meshgrid(ii, jj)
jjj = np.flip(jjj)
iii = np.transpose(iii)
jjj = np.transpose(jjj)
color=(1,1,1)

offset=[0.05,0,0.05,0.03,0.00]
colors=[(0.75,0.75,0.75),(0.5,0.5,0.5),(0,0,0),(0.3,0.3,0.3),(1,1,1)]
col_ind=0
for ct in range(0,5):
    c=4-ct
    if c==0 or c==3 or c==2 or c==4 or c==1:
        x=[]
        y=[]
        z=[]
        for ii in range(0,N):
            for jj in range(0,N):
                I=ii
                J=(N+1)*c + jj + 1
                i = iii[ii, jj]
                j = jjj[ii, jj]
                theta,phi = iso.cij2thetaphi(c,i,j)
                xp,yp,zp=sphere2cart(1,theta,phi)
                x.append(xp)
                y.append(yp)
                z.append(zp)
                print(col_ind)
                mlab.points3d(x, y, z, scale_factor=0.08,color=colors[col_ind])#(1/(d*2),1/d,1/(1+d)))
                mlab.text(x[-1], y[-1],'%d, %d, %d' % (c, I ,J), z=float(z[-1]+offset[c]),width=0.3,opacity=0.8,color=(0,1,0))
        col_ind=col_ind+1

x=[]
y=[]
z=[]
for c in range(0,5):
    for t in range(0,4):
        for point in iso.Points[c,t]:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
mlab.points3d(x,y,z,scale_factor=0.02,color=(1,1,1),opacity=0.1)

mlab.points3d(0,0,0,scale_factor=0.08,color=(1,0,0))

# #the next two loops join the edge points but we want to go over
# for ii_p in range(0,N):
#     x = []
#     y = []
#     z = []
#     jj=N-1
#     ii=ii_p
#     c=0
#     I = N
#     J = (N + 1) * c + jj + 1
#     i = iii[ii, jj]
#     j = jjj[ii, jj]
#     theta, phi = iso.cij2thetaphi(c, i, j)
#     xp, yp, zp = sphere2cart(1, theta, phi)
#     x.append(xp)
#     y.append(yp)
#     z.append(zp)
#
#     c=3
#     jj=N-1-ii_p
#     ii=N-1
#     I = N
#     J = (N + 1) * c + jj + 1
#     i = iii[ii, jj]
#     j = jjj[ii, jj]
#     theta, phi = iso.cij2thetaphi(c, i, j)
#     xp, yp, zp = sphere2cart(1, theta, phi)
#     x.append(xp)
#     y.append(yp)
#     z.append(zp)

    #mlab.plot3d(x,y,z,opacity=0.2,color=(0,1,0),tube_radius=0.01)

# for ii_p in range(0,N):
#     x = []
#     y = []
#     z = []
#
#     jj=ii_p
#     ii=N-1
#     c=0
#     I = N
#     J = (N + 1) * c + jj + 1
#     i = iii[ii, jj]
#     j = jjj[ii, jj]
#     theta, phi = iso.cij2thetaphi(c, i, j)
#     xp, yp, zp = sphere2cart(1, theta, phi)
#     x.append(xp)
#     y.append(yp)
#     z.append(zp)
#
#     c=2
#     jj=N-1
#     ii=N-1-ii_p
#     I = N
#     J = (N + 1) * c + jj + 1
#     i = iii[ii, jj]
#     j = jjj[ii, jj]
#     theta, phi = iso.cij2thetaphi(c, i, j)
#     xp, yp, zp = sphere2cart(1, theta, phi)
#     x.append(xp)
#     y.append(yp)
#     z.append(zp)
#
#     #mlab.plot3d(x,y,z,opacity=0.2,color=(0,0,1),tube_radius=0.01)
#
# # #make the lines connecting the points one over
# c=0
# x = []
# y = []
# z = []
#
# oneover=[2,5,7]
# ii_p=0
# for i in oneover:
#     xc=[]
#     yc=[]
#     zc=[]
#     print(i)
#     xp= iso.Points[c-1,2][i][0]
#     yp = iso.Points[c - 1, 2][i][1]
#     zp = iso.Points[c - 1, 2][i][2]
#     print(xp,yp,zp)
#     x.append(xp)
#     y.append(yp)
#     z.append(zp)
#     mlab.points3d(x, y, z, scale_factor=0.08, color=color)  # (1/(d*2),1/d,1/(1+d)))
#
#     xc.append(xp)
#     yc.append(yp)
#     zc.append(zp)
#
#     cc=2
#     jj = N - 2
#     ii = N - 1 - ii_p
#     ii_p+=1
#     I = N
#     J = (N + 1) * cc + jj + 1
#     i = iii[ii, jj]
#     j = jjj[ii, jj]
#     theta, phi = iso.cij2thetaphi(cc, i, j)
#     xx, yy, zz = sphere2cart(1, theta, phi)
#     xc.append(xx)
#     yc.append(yy)
#     zc.append(zz)
#
#     #mlab.plot3d(xc, yc, zc, opacity=0.2, color=(0, 0, 1), tube_radius=0.01)
#
#
# for i in range(9,10):
#     xc = []
#     yc = []
#     zc = []
#     xp= iso.Points[c-1,3][i][0]
#     yp = iso.Points[c - 1, 3][i][1]
#     zp = iso.Points[c - 1, 3][i][2]
#     x.append(xp)
#     y.append(yp)
#     z.append(zp)
#     mlab.points3d(x, y, z, scale_factor=0.08, color=color)  # (1/(d*2),1/d,1/(1+d)))
#     xc.append(xp)
#     yc.append(yp)
#     zc.append(zp)
#     cc = 2
#     jj = N - 2
#     ii = 1
#     ii_p += 1
#     I = N
#     J = (N + 1) * cc + jj + 1
#     i = iii[ii, jj]
#     j = jjj[ii, jj]
#     theta, phi = iso.cij2thetaphi(cc, i, j)
#     xx, yy, zz = sphere2cart(1, theta, phi)
#     xc.append(xx)
#     yc.append(yy)
#     zc.append(zz)

    #mlab.plot3d(xc, yc, zc, opacity=0.2, color=(0, 0, 1), tube_radius=0.01)

#plot all the points with low opacity


# # #make the lines connecting the points one over
# c=0
# x = []
# y = []
# z = []
#
# ii_p=0
# oneover=[4,5,6]
# for i in oneover:
#     xc = []
#     yc = []
#     zc = []
#     print(i)
#     xp= iso.Points[c,2][i][0]
#     yp = iso.Points[c, 2][i][1]
#     zp = iso.Points[c, 2][i][2]
#     print(xp,yp,zp)
#     x.append(xp)
#     y.append(yp)
#     z.append(zp)
#     mlab.points3d(x, y, z, scale_factor=0.08, color=color)  # (1/(d*2),1/d,1/(1+d)))
#     xc.append(xp)
#     yc.append(yp)
#     zc.append(zp)
#     cc = 3
#
#     jj = ii_p+2
#     ii = N-2
#     ii_p += 1
#     I = N
#     J = (N + 1) * cc + jj + 1
#     i = iii[ii, jj]
#     j = jjj[ii, jj]
#     theta, phi = iso.cij2thetaphi(cc, i, j)
#     xx, yy, zz = sphere2cart(1, theta, phi)
#     xc.append(xx)
#     yc.append(yy)
#     zc.append(zz)
#     #mlab.plot3d(xc, yc, zc, opacity=0.2, color=(0, 1, 0), tube_radius=0.01)
#
#
# xc = []
# yc = []
# zc = []
# i=2
# print(i)
# xp= iso.Points[c,3][i][0]
# yp = iso.Points[c, 3][i][1]
# zp = iso.Points[c, 3][i][2]
# print(xp,yp,zp)
# x.append(xp)
# y.append(yp)
# z.append(zp)
# xc.append(xp)
# yc.append(yp)
# zc.append(zp)
# mlab.points3d(x, y, z, scale_factor=0.08, color=color)  # (1/(d*2),1/d,1/(1+d)))
# cc = 3
#
# jj = 1
# ii = N - 2
# ii_p += 1
# I = N
# J = (N + 1) * cc + jj + 1
# i = iii[ii, jj]
# j = jjj[ii, jj]
# theta, phi = iso.cij2thetaphi(cc, i, j)
# xx, yy, zz = sphere2cart(1, theta, phi)
# xc.append(xx)
# yc.append(yy)
# zc.append(zz)
# #mlab.plot3d(xc, yc, zc, opacity=0.2, color=(0, 1, 0), tube_radius=0.01)
#
#
# mlab.points3d(0,0,0,scale_factor=0.08,color=(1,0,0))
# mlab.show()
#
#
#
#
#


