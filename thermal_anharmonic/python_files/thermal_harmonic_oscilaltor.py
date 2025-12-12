import numpy as np

from header_file import run_sdp


out_folder = "python_outputs"

for L in [6,8,10]:
    input_file = f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/harmonic_output_for_l={L}_m=3_k=3.json"
    run_sdp(np.arange(0.1,0.8,0.1),input_file,out_folder)

