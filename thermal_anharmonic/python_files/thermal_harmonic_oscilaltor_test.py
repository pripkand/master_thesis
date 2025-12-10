import cvxpy as cp
import numpy as np
import h5py
import json

from cvxpy import length
from cvxpy.reductions.solvers.conic_solvers import CLARABEL

from header_file import build_matrix,run_sdp

in_file = "/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/harmonic_output_for_l=4_m=3_k=3.json"
out_folder = "python_outputs"

beta_range=np.arange(.7,1,.1)
run_sdp(beta_range,in_file,out_folder)


