import numpy as np
import h5py
import cvxpy as cp
from cvxpy import installed_solvers

from header_file import run_sdp

#========================================================
# This script runs the SDP for a temperature range and for the length listed. The corresponding json files need to have been
# created using the mathematica script beforehand.
# The run_sdp function will then export the energy, temperature and status for each temperature into an HDF5 file.
#========================================================

# Define the folder to output the results.
out_folder = "python_outputs"

for L in [4]:
    # Define the input json files.
    print(installed_solvers())
    input_file = f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/harmonic_output_for_l={L}_m=3_k=3.json"
    run_sdp(np.array([1e-3]),input_file,out_folder,verbose=True)


    def theoretical_curve(T):
        return - 1 / 2 + 1 / (-np.exp(-1 / T) + 1)
    print(theoretical_curve(1e-3))
    with h5py.File(f"python_outputs/Harmonic_L=10_m=3_k=3",'r') as f:
       print(f["energy"][:])
