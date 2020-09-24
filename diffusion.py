import ioFunctions
import numpy as np
from scipy import interpolate
import dipy
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import SmoothSphereBivariateSpline
from scipy.interpolate import LSQSphereBivariateSpline
import torch
import somemath
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class diff2d():
    def __init__(self):
        self.vox=[]
        self.signal=[]

    def loadData(self,dvol=None):
        if dvol is None:
            raise ValueError("please call with a diffVolume object")
        img=dvol.vol.getData()
        shape=img.shape
        img=img.reshape((shape[0]*shape[1]*shape[2],shape[3]),order='F')


class dti():
    def __init__(self):
        self.FA=[]
        self.L1=[]
        self.L2=[]
        self.L3=[]
        self.V1=[]
        self.V2=[]
        self.V3=[]
        self.MD = []
        self.MO = []
        self.S0 = []
        self.mask= []

    def load(self,pathprefix):
        if pathprefix is None:
            raise ValueError("Please provide path including prefix for dti data, prefix=...")
        self.FA = ioFunctions.loadgetVol(pathprefix+"_FA.nii.gz")
        self.L1 = ioFunctions.loadgetVol(pathprefix + "_L1.nii.gz")
        self.L2 = ioFunctions.loadgetVol(pathprefix + "_L2.nii.gz")
        self.L3 = ioFunctions.loadgetVol(pathprefix + "_L3.nii.gz")
        self.V1 = ioFunctions.loadgetVol(pathprefix + "_V1.nii.gz")
        self.V2 = ioFunctions.loadgetVol(pathprefix + "_V2.nii.gz")
        self.V3 = ioFunctions.loadgetVol(pathprefix + "_V3.nii.gz")

    #def y(self,p):
     #   v=[]



class diffVolume():
    def __init__(self):
        """
        Class for storing gridded volume data
        """
        self.vol = []
        self.interpExists = 0
        self.interpolator = []
        self.bvals = []
        self.bvecs = []
        self.bvecs_hemi_cart=[]
        self.bvecs_hemi_sphere=[]
        self.bvecs_hemi_cart_kdtree=[]
        self.inds= []
        self.gtab = []
        self.img=[]
        self.sgrad_x=[]
        self.sgrad_y = []
        self.sgrad_z = []
        self.current_signal=[]
        self.mask=[]


    def getVolume(self, folder=None):
        """
        Gets volume data
        :param filename: Path of volume file
        :return:
        """
        self.vol, self.gtab =ioFunctions.loadDiffVol(folder=folder)
        self.img = self.vol.get_data()
        self.mask = ioFunctions.loadVol(filename=folder+"/nodif_brain_mask.nii.gz")

    def makeInterpolator(self):
        """
        Makes a linear interpolator
        :return: Fills out self. interpolator and sets self.interpExists = 1 after interpolator is calculated
        """
        shape = self.vol.shape
        print(shape)
        img = self.vol.get_data()
        #TODO other shapes like scalars most impot
        if  len(shape) > 3:
            if shape[3] == 3:
                i = np.linspace(0, shape[0] - 1, num=shape[0])
                j = np.linspace(0, shape[1] - 1, num=shape[1])
                k = np.linspace(0, shape[2] - 1, num=shape[2])
                self.interpolator = [interpolate.RegularGridInterpolator((i, j, k), img[:, :, :, f]) for f in range(shape[3])]
                self.interpExists=1
            if shape[3]==1:
                i = np.linspace(0, shape[0] - 1, num=shape[0])
                j = np.linspace(0, shape[1] - 1, num=shape[1])
                k = np.linspace(0, shape[2] - 1, num=shape[2])
                self.interpolator = interpolate.RegularGridInterpolator((i, j, k), img[:, :, :,0])
                self.interpExists = 1
        else:
            i = np.linspace(0, shape[0] - 1, num=shape[0])
            j = np.linspace(0, shape[1] - 1, num=shape[1])
            k = np.linspace(0, shape[2] - 1, num=shape[2])
            self.interpolator = interpolate.RegularGridInterpolator((i, j, k), img[:, :, :])
            self.interpExists = 1

    def shells(self):
        tempbvals=[]
        tempbvals=np.round(self.gtab.bvals,-2)
        inds_sort=np.argsort(tempbvals)
        bvals_sorted=self.gtab.bvals[inds_sort]
        bvecs_sorted=self.gtab.bvecs[inds_sort]
        tempbvals=np.sort(tempbvals)
        gradbvals=np.gradient(tempbvals)
        inds_shell_cuts=np.where(gradbvals!=0)
        shell_cuts=[]
        for i in range(int(len(inds_shell_cuts[0]) / 2)):
            shell_cuts.append(inds_shell_cuts[0][i * 2])
        shell_cuts.insert(0,-1)
        shell_cuts.append(len(bvals_sorted))
        print(shell_cuts)
        print(bvals_sorted.shape)
        temp_bvals=[]
        temp_bvecs=[]
        temp_inds=[]
        for t in range(int(len(shell_cuts)-1)):
            print(shell_cuts[t]+1,shell_cuts[t + 1])
            temp_bvals.append(bvals_sorted[shell_cuts[t]+1:1+shell_cuts[t+1]])
            temp_bvecs.append(bvecs_sorted[shell_cuts[t]+1:1+shell_cuts[t+1]])
            temp_inds.append(inds_sort[shell_cuts[t]+1:1+shell_cuts[t+1]])
        self.bvals=temp_bvals
        self.bvecs=temp_bvecs
        self.inds=temp_inds
        self.inds=np.asarray(self.inds)


        pi=3.14159265
        for bvecs in self.bvecs: #this is shells
            temp_bvec = []
            temp_vec = []

            for bvec in bvecs: #this is each vector in shell
                r, theta, phi=dipy.core.sphere.cart2sphere(bvec[0],bvec[1],bvec[2])
                #if theta > pi/2: #this is the anitpodal port becareful whether this is on or off
                #    theta= pi- theta
                #    phi=phi+3.14159265
                phi=(phi)%(2*pi)
                x,y,z=dipy.core.sphere.sphere2cart(1,theta,phi)
                temp_vec.append([x,y,z])
                temp_bvec.append([r,theta,phi])
            self.bvecs_hemi_cart_kdtree.append(KDTree(temp_vec,10))
            self.bvecs_hemi_sphere.append(temp_bvec)
            self.bvecs_hemi_cart.append(temp_vec)
        self.bvecs_hemi_cart=np.asarray(self.bvecs_hemi_cart)
        self.bvecs_hemi_sphere=np.asarray(self.bvecs_hemi_sphere)




    def makeFlatHemisphere(self,p1,shell):
        s0 = []
        s1 = []
        th = []
        ph = []
        x=[]
        y=[]
        z=[]
        interpolator=[]
        i = 0
        for ind in self.inds[shell]:
            x.append(self.bvecs_hemi_cart[shell][i][0])
            y.append(self.bvecs_hemi_cart[shell][i][1])
            z.append(self.bvecs_hemi_cart[shell][i][2])
            s1.append(self.img[p1[0], p1[1], p1[2], ind])
            th.append(self.bvecs_hemi_sphere[shell][i][1])
            ph.append(self.bvecs_hemi_sphere[shell][i][2])
            i = i + 1
        th = np.asarray(th)
        ph = np.asarray(ph)
        s1 = np.asarray(s1)
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)


        for ind in self.inds[0]:
            s0.append(self.img[p1[0], p1[1], p1[2], ind])
        norm=sum(s0)/len(s0)


        thph=np.column_stack((th,ph))
        xyz = np.column_stack((x, y, z))
        #interpolator=LinearNDInterpolator(thph,s1/norm)
        #interpolator = NearestNDInterpolator(thph, s1 / norm)
        #interpolator = NearestNDInterpolator(xyz, s1/norm)
        #interpolator = LinearNDInterpolator(xyz, s1/norm)
        #interpolator = SmoothSphereBivariateSpline(thph[:,0],thph[:,1],s1/norm,s=0.1)
        #interpolator = LSQSphereBivariateSpline(thph[:,0],thph[:,1],s1/norm)
        interpolator=self.bvecs_hemi_cart_kdtree[shell]


        iso=somemath.isomesh()
        iso.get_icomesh()
        iso.makeFlat(interpolator,s1/norm)
        #print(interpolator(th,ph))
        return iso.s_flat

    def plotSignal(self,p1,shell):
        N=64
        sphere_sig=somemath.sphereSig()
        theta = np.linspace(0, np.pi, N)
        phi = np.linspace(0, 2 * np.pi, N)
        theta, phi = np.meshgrid(theta,phi)
        s1 = []
        th = []
        ph = []
        i=0
        for ind in self.inds[shell]:
            s1.append(self.img[p1[0], p1[1], p1[2], ind])
            th.append(self.bvecs_hemi_sphere[shell][i][1])
            ph.append(self.bvecs_hemi_sphere[shell][i][2])
            i = i + 1
        th = np.asarray(th)
        ph = np.asarray(ph)
        s1 = np.asarray(s1)
        print(s1)
        ss1 = griddata((th, ph), 1/s1, (theta, phi), method='nearest')  # , fill_value=-1)
        sphere_sig.grid = np.real(ss1)
        sphere_sig.N=N
        #sphere_sig.plot()
        self.current_signal=sphere_sig
        plt.imshow(sphere_sig.grid)
