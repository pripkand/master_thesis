import numpy as np
from psdhermitian import PsdHermitian
import numpy as np

dictionary_x = {"constant":np.array([[[11,0],[0,-23]],np.zeros((2,2))]),"y1":np.array([[[10,4],[4,0]],np.zeros((2,2))]),"y2":np.array([[[0,0],[0,-8]],np.zeros((2,2))]),"y3":np.array([[[0,-8],[-8,-2]],np.zeros((2,2))])}
dictionary_z = {"constant":np.array([[[11,0],[0,-23]],np.zeros((2,2))]),"y1":np.array([[[10,4],[4,0]],np.zeros((2,2))]),"y2":np.array([[[0,0],[0,-8]],np.zeros((2,2))])}
chart = {"constant":False,"y1":False,"y2":False,"y3":False}
x = PsdHermitian(dictionary_x,chart,name = "X")
print(x.shape)
y = x+x
print(y.shape)
z = PsdHermitian(dictionary_z,chart,name = "Z")
print(z.shape)
z_dir_x = z.direct_sum(x)
print(z_dir_x.shape)
