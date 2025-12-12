import matplotlib
from cvxpy import installed_solvers

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

#===============================================
# This script plots the data gathered from other scripts using the hdf5 files they exported. There is no need to rerun
# the corresponding scripts if the needed file is present.
#===============================================

for L in [6,8,10]:
    with h5py.File(f"python_outputs/Harmonic_L={L}_m=3_k=3") as f:
        energy=f["energy"][()]
        status=f["status"][()]
        temperature_range=f["temperatures"][()]


        for i,beta in enumerate(temperature_range):
            if i%2==0:
                print(beta,status[i],energy[i])

        plt.plot(temperature_range, energy, label=f"L={L} and (m,k)=(3,3)")


def theoretical_curve(T):
    return 1 / 2 + np.exp(1 / T) / (np.exp(1 / T) - 1)
plt.plot(temperature_range, [theoretical_curve(beta) for beta in temperature_range],label=r"Theoretical:$\langle E \rangle=\frac{1}{2}+\frac{e^{1/T}}{e^{1/T}-1}$")

plt.title("Thermal Harmonic Oscillator: Bootstrap VS Theory")
plt.xlabel("Temperature (Natural Units)")
plt.ylabel(r"$\langle E \rangle$ (Natural Units)")
plt.grid(True,alpha=0.3)
plt.legend()
plt.show()