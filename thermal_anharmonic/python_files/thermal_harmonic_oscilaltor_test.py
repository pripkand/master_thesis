import cvxpy as cp
import numpy as np
import h5py
import json

from cvxpy import length

from header_file import build_matrix,run_sdp

in_file = "/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/Harmonic_output_for_l=4_m=3_k=3.json"
out_folder = "python_outputs"

temperature_range=np.arange(0.02,0.6,0.02)
run_sdp(np.array([1/T for T in temperature_range]),in_file,out_folder)


