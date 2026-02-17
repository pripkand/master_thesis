import matplotlib
from thermal_bootstrap_header_file import anharmonic_thermal_energy
matplotlib.use("Qt5Agg") # This is needed for the IDE I am working in.
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py

#===============================================
# This script plots the data gathered from other scripts using the hdf5 files they exported. There is no need to rerun
# the corresponding scripts if the needed file is present.
#===============================================
fig, ax =plt.subplots(figsize=(8,5))
axins = inset_axes(ax, width="30%", height="30%", loc="upper right")

for L in [10]:
    for k in [3]:
        colors = {10:{4:"red",3:"orange"},6:{4:"blue",3:"cyan"}}
        formats = {6:"-",10:"--"}
        with h5py.File(f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/python_outputs/Anharmonic_L={L}_m={k}_k={k}_upper.hdf5") as f:
            with h5py.File(f"/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/python_files/python_outputs/Anharmonic_L={L}_m={k}_k={k}_lower.hdf5") as g:
                energy_upper = f["energy"][()]
                energy_lower = g["energy"][()]
                #status=f["status"][()]
                temperature_range=f["temperatures"][()]

                ax.plot(temperature_range, energy_upper, label=f"L={L} and (m,k)=({k},{k})",color=colors[L][k],linestyle=formats[L])
                ax.plot(temperature_range, energy_lower,color=colors[L][k],linestyle=formats[L])
                axins.plot(temperature_range, energy_upper,color=colors[L][k],linestyle=formats[L])
                axins.plot(temperature_range, energy_lower,color=colors[L][k],linestyle=formats[L])


ax.plot(temperature_range,[anharmonic_thermal_energy(1/T) for T in temperature_range],color="black",linestyle='dashdot',label="Theoretical Line")
axins.plot(temperature_range,[anharmonic_thermal_energy(1/T) for T in temperature_range],color="black",linestyle='dashdot')



# Define zoom region
x1, x2 = 0.107, 0.114
y1, y2 = 0.000175+1.06, 0.000375+1.06

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.set_xticks([])
axins.set_yticks([])

# Optional: remove ticks for cleaner look
axins.grid(which='both',alpha=0.3)

ax.indicate_inset_zoom(axins, edgecolor="black")

ax.set_title("Thermal Quartic Oscillator: Bootstrap VS Theory",fontsize=16)
ax.set_xlabel("Temperature (Natural Units)",fontsize=14)
ax.set_ylabel(r"$\langle E \rangle$ (Natural Units)",fontsize=14)
ax.grid(True,which='both',alpha=0.3)
ax.legend(fontsize=14)
plt.show()