import matplotlib
from cvxpy import installed_solvers

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import h5py

#===============================================
# This script plots the data gathered from other scripts using the hdf5 files they exported. There is no need to rerun
# the corresponding scripts if the needed file is present.
#===============================================
fig, ax =plt.subplots(figsize=(8,5))
axins = inset_axes(ax, width="30%", height="30%", loc="lower right")

for L in [6]:
    for k in [3,4]:
        colors = {6:{4:"red",3:"orange"},4:{4:"blue",3:"cyan"}}
        formats = {4:"-",6:"--"}
        with h5py.File(f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/python_outputs/Harmonic_L={L}_m=3_k={k}_upper.hdf5") as f:
            with h5py.File(f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/python_outputs/Harmonic_L={L}_m=3_k={k}_lower.hdf5") as g:
                energy_upper = f["energy"][()]
                energy_lower = g["energy"][()]
                #status=f["status"][()]
                temperature_range=f["temperatures"][()]

                ax.plot(temperature_range, energy_upper, label=f"L={L} and (m,k)=({k},{k})",color=colors[L][k],linestyle=formats[L])
                ax.plot(temperature_range, energy_lower,color=colors[L][k],linestyle=formats[L])
                axins.plot(temperature_range, energy_upper,color=colors[L][k],linestyle=formats[L])
                axins.plot(temperature_range, energy_lower,color=colors[L][k],linestyle=formats[L])
                #plt.fill_between(temperature_range,energy_upper,energy_lower,color=colors[L][k],alpha=0.5)

# Plot Theoretical For Harmonic
def theoretical_curve(T):
    return -1/2+1/(1-np.exp(-1/T))
ax.plot(temperature_range, [theoretical_curve(beta) for beta in temperature_range],label=r"Theoretical:$\langle E \rangle=-\frac{1}{2}+\frac{e^{1/T}}{e^{1/T}-1}$",color="black",linestyle='dashdot')
axins.plot(temperature_range, [theoretical_curve(beta) for beta in temperature_range],color="black",linestyle='dashdot')
# Plot Approximate points from literature
#plt.plot(temperature_range[:-1],[1.060,1.060,1.060,1.064,1.073,1.085],label="Approximate Paper Curve for L=10 m,k=3")


# Define zoom region
x1, x2 = 1*1e-6+4.1953e-1, 3.5*1e-6+4.1953e-1
y1, y2 = 28.5*1e-6+6.025e-1, 32.5*1e-6+6.025e-1

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.set_xticks([])
axins.set_yticks([])

# Optional: remove ticks for cleaner look
axins.grid(which='both',alpha=0.3)
# ----------------------
# Draw box + connecting lines
# ----------------------
ax.indicate_inset_zoom(axins, edgecolor="black")

ax.set_title("Thermal Anharmonic Oscillator: Bootstrap VS Theory",fontsize=16)
ax.set_xlabel("Temperature (Natural Units)",fontsize=14)
ax.set_ylabel(r"$\langle E \rangle$ (Natural Units)",fontsize=14)
ax.grid(True,which='both',alpha=0.3)
ax.legend(fontsize=14)
plt.show()