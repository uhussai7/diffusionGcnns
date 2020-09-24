import somemath
import diffusion
import scipy
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import mathutils
from mathutils.geometry import intersect_ray_tri

iso=somemath.isomesh()
iso.get_icomesh()
iso.plot_icomesh()

# dvol=diffusion.diffVolume()
# dvol.getVolume("/home/uzair/Datasets/101006/Diffusion")
# dvol.shells()
#

vertices=np.zeros([12,3])
faces=np.zeros([20,3])

for v in range(0,12):
    for i in range(0,3):
        vertices[v,i]=iso.vertices[v][i]

for v in range(0, 20):
    for i in range(0, 3):
        faces[v, i] = iso.faces[v][i]

tri_iso=trimesh.Trimesh(vertices=vertices, faces=faces)

tri_iso.show()