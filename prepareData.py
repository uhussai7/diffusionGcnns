import numpy as np
import diffusion
import random
from numpy import save
import matplotlib.pyplot as plt


#fresh testing
dvol=diffusion.diffVolume()
dvol.getVolume("/home/uzair/Datasets/101006/Diffusion")
dvol.shells()
dti=diffusion.dti()
dti.load("/home/uzair/Datasets/101006/Diffusion/dti")
mask=dvol.mask.get_data()
i,j,k=np.where(mask ==1)

Ntrain=2000
Ntest=200
N=Ntrain+Ntest


all_inds=random.sample(range(N),N)
train_inds=all_inds[0:Ntrain]
test_inds=all_inds[Ntrain:N]

H=4
W=20
X_train=np.empty([Ntrain,H,W,3])
X_test=np.empty([Ntest,H,W,3])
Y_train=np.empty([Ntrain,10])  #V1x3, V2X3, FA, Lx3
Y_test=np.empty([Ntest,10])
t=0
for ind in train_inds:
    print(ind)
    p = [i[ind], j[ind], k[ind]]
    for shell in range(1,4):
        X_train[t,:,:,shell-1]=dvol.makeFlatHemisphere(p,shell)
    t=t+1
t = 0
for ind in test_inds:
    print(ind)
    p = [i[ind], j[ind], k[ind]]
    for shell in range(1, 4):
        X_test[t, :, :, shell - 1] = dvol.makeFlatHemisphere(p, shell)
    t = t + 1
t=0
for ind in train_inds:
    #p = [i[ind], j[ind], k[ind]]
    Y_train[t, 0] = dti.FA[i[ind], j[ind], k[ind]]
    Y_train[t, 1] = dti.L1[i[ind], j[ind], k[ind]]
    Y_train[t, 2] = dti.L2[i[ind], j[ind], k[ind]]
    Y_train[t, 3] = dti.L3[i[ind], j[ind], k[ind]]
    Y_train[t, 4] = dti.V1[i[ind], j[ind], k[ind],0]
    Y_train[t, 5] = dti.V1[i[ind], j[ind], k[ind],1]
    Y_train[t, 6] = dti.V1[i[ind], j[ind], k[ind],2]
    Y_train[t, 7] = dti.V2[i[ind], j[ind], k[ind],0]
    Y_train[t, 8] = dti.V2[i[ind], j[ind], k[ind],1]
    Y_train[t, 9] = dti.V2[i[ind], j[ind], k[ind],2]
    t=t+1
t=0
for ind in test_inds:
    #p = [i[ind], j[ind], k[ind]]
    Y_test[t, 0] = dti.FA[i[ind], j[ind], k[ind]]
    Y_test[t, 1] = dti.L1[i[ind], j[ind], k[ind]]
    Y_test[t, 2] = dti.L2[i[ind], j[ind], k[ind]]
    Y_test[t, 3] = dti.L3[i[ind], j[ind], k[ind]]
    Y_test[t, 4] = dti.V1[i[ind], j[ind], k[ind],0]
    Y_test[t, 5] = dti.V1[i[ind], j[ind], k[ind],1]
    Y_test[t, 6] = dti.V1[i[ind], j[ind], k[ind],2]
    Y_test[t, 7] = dti.V2[i[ind], j[ind], k[ind],0]
    Y_test[t, 8] = dti.V2[i[ind], j[ind], k[ind],1]
    Y_test[t, 9] = dti.V2[i[ind], j[ind], k[ind],2]
    t=t+1

save('X_train_small.npy',X_train)
save('Y_train_small.npy',Y_train)
save('X_test_small.npy',X_test)
save('Y_test_small.npy',Y_test)
#
# import stripy
#
# N=5
# height = N+1
# width = 5*(N+1)
# ii = np.linspace(0, N - 1, N)
# jj = np.linspace(0, N - 1, N)
# jjj, iii = np.meshgrid(ii, jj)
# jjj = np.flip(jjj)
# iii = np.transpose(iii)
# jjj = np.transpose(jjj)
# for c in range(0, 5):
#     print(c)
#     for ii in range(0, N):
#         for jj in range(0, N):
#             I = ii
#             J = (N + 1) * c + jj + 1
#             i = iii[ii, jj]
#             j = jjj[ii, jj]
#             theta, phi = iso.cij2thetaphi(c, i, j)
#             phi=phi+3.141592654
#             print(theta % np.pi, phi % 2*np.pi)
#             val1 = interpolator(theta % np.pi, phi % 2*np.pi)
#             print((np.pi - theta) % np.pi, (phi + np.pi) % 2*np.pi, "\n")
#             val2 = interpolator((np.pi - theta) % np.pi, (phi + np.pi) % 2*np.pi)
#             print(0.5 * (val1 + val2), "\n\n")
#             iso.s_flat[I, J] = 0.5 * (val1 + val2)
#             test[I,J]=(val1)
#
#
# stripy.spherical._ssrfpack.