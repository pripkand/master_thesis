import numpy as np
#from psdhermitian import PsdHermitian

dictionary = {"constant":np.array([[[11,0],[0,-23]],np.zeros((2,2))]),"y1":np.array([[[10,4],[4,0]],np.zeros((2,2))]),"y2":np.array([[[0,0],[0,-8]],np.zeros((2,2))]),"y3":np.array([[[0,-8],[-8,-2]],np.zeros((2,2))])}
chart = {"constant":False,"y1":False,"y2":False,"y3":False}
#x = PsdHermitian(dictionary,chart,name = "X")

for d in ({"a":2},{"b":3}):
    print(d)
