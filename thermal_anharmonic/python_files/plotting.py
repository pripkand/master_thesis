import matplotlib
from cvxpy import installed_solvers

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File("python_outputs/Harmonic_L=4_m=3_k=3") as f:
    energy=f["energy"][()]
    status=f["status"][()]

    temperature_range=np.arange(0.02,0.6,0.02)

    for i,beta in enumerate(temperature_range):
        if i%2==0:
            print(beta,status[i],energy[i])

    def theoretical_curve(T):
        return 1/2+np.exp(1/T)/(np.exp(1/T)-1)


    plt.plot(temperature_range,[theoretical_curve(beta) for beta in temperature_range],label="theoretical")

    plt.plot(temperature_range,energy,label="experimental")
    plt.ylim(0,2)
    plt.legend()
    plt.show()