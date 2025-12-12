import numpy as np

from header_file import run_sdp

#========================================================
# This script runs the SDP for a temperature range and for the length listed. The corresponding json files need to have been
# created using the mathematica script beforehand.
# The run_sdp function will then export the energy, temperature and status for each temperature into an HDF5 file.
#========================================================

# Define the folder to output the results.
out_folder = "python_outputs"

for L in [6,8,10]:
    # Define the input json files.
    input_file = f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/harmonic_output_for_l={L}_m=3_k=3.json"
    run_sdp(np.arange(0.1,0.8,0.1),input_file,out_folder,verbose=True)

