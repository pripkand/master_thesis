import numpy as np

from header_file import run_sdp

#========================================================
# This script runs the SDP for a temperature range and for the length listed. The corresponding json files need to have been
# created using the mathematica script beforehand.
# The run_sdp function will then export the energy, temperature and status for each temperature into an HDF5 file.
#========================================================

# Define the folder to output the results.
out_folder = "python_outputs"

# Define Temperature Range
temperature_range = np.arange(0.1,0.8,0.1)

for L in [8]:
    for m in [3,4]:
        # Define the input JSON files.
        input_file = f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/harmonic_output_for_l={L}_m={m}_k={m}.json"

        scs_parameters = {"max_iters":30000,"eps":1e-6,"eps_abs":1e-7,"eps_rel":1e-7,"acceleration_lookback":10,"normalize":True,"scale":1.0}
        sdpa_parameters = {"maxIterations":1000,"print":"display"}
        run_sdp(temperature_range,input_file,out_folder,verbose=True,maximize=False,solver_parameters= sdpa_parameters
                ) # Find Lower Bound
        run_sdp(temperature_range,input_file,out_folder,verbose=True,maximize=True,solver_parameters= sdpa_parameters
                )  # Find Upper Bound

