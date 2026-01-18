"""
This file uses the psdhermitian header file to solve example 1 from the SDPA files.
"""
from psdhermitian import PsdHermitian,PSDConstraint,Problem
import numpy as np

Problem.set_gmp_path("/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/sdpa-gmp")
Problem.set_in_file("/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/sdpa_files")
Problem.set_out_file("/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/sdpa_files")

dictionary_x = {"constant":np.array([[-11+1j*0,0+1j*0],[0+1j*0,23+1j*0]]),"y1":np.array([[10+1j*0,4+1j*0],[4+1j*0,0+1j*0]]),"y2":np.array([[0+1j*0,0+1j*0],[0+1j*0,-8+1j*0]]),"y3":np.array([[0+1j*0,-8+1j*0],[-8+1j*0,-2+1j*0]])}
chart = {"constant":False,"y1":False,"y2":False,"y3":False}
x = PsdHermitian(dictionary_x,chart,name = "X")
con = [PSDConstraint(x)]
print(con)
targ = {"y1":48,"y2":-8,"y3":20}
pr = Problem(targ,con,"example1_redone.dat","example1_redone.out")
pr.solve()