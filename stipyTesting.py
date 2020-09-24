import stripy
import numpy as np
from mayavi import mlab
mesh = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=3, include_face_points=False, tree=True)

x=mesh.points[:,0]
y=mesh.points[:,1]
z=mesh.points[:,2]

mlab.triangular_mesh(x,y,z,mesh.simplices,representation='wireframe')
mlab.show()

eqs = np.genfromtxt("EQ-M5.5-IRIS-ALL.txt", usecols=(2,3,4,10), delimiter='|', comments="#")

lons = np.radians(eqs[:,1])
lats = np.radians(eqs[:,0])
depths = eqs[:,2]
depths[np.isnan(depths)] = -1.0

distances, vertices = mesh.nearest_vertices(lons, lats, k=10)
norm = distances.sum(axis=1)

hit_countid = np.zeros_like(mesh.lons)

for i in range(0,distances.shape[0]):
    hit_countid[vertices[i,:]] += distances[i,:] / norm[i]
