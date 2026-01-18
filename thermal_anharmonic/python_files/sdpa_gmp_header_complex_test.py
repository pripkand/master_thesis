
from psdhermitian import PsdHermitian,PSDConstraint,Problem
import numpy as np

Problem.set_gmp_path("/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/sdpa-gmp")
Problem.set_in_file("/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/sdpa_files/")
Problem.set_out_file("/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/sdpa_files/")

dictionary_x = {"constant":np.zeros((2,2),dtype=complex),
                "z11":np.array([[1+1j*0,0+1j*0],[0+1j*0,0+1j*0]]),
                "z22":np.array([[0+1j*0,0+1j*0],[0+1j*0,1+1j*0]]),
                "x":np.array([[0+1j*0,1+1j*0],[1+1j*0,0+1j*0]]),
                "y":np.array([[0+1j*0,0+1j*1],[0-1j*1,0+1j*0]])}
chart = {"constant":False,"z11":False,"z22":False,"x":False,"y":False}
x = PsdHermitian(dictionary_x,chart,name = "X")
con = [PSDConstraint(x)]
print(con)
targ = {"x":1}
pr = Problem(targ,con,"complex_example_redone.dat","complex_example_redone.out")
pr.solve()