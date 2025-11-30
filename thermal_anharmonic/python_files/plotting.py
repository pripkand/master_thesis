import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File("python_outputs/harmonic_L=10_n=1_k=4") as f:
    energy=f["energy"][()]
    status=f["status"][()]

    beta_range = np.arange(10,60,0.5)

    for i,beta in enumerate(beta_range):
        if i%2==0:
            print(beta,status[i],energy[i])

    def theoretical_curve(beta):
        return 1/2+np.exp(beta)/(np.exp(beta)-1)


    plt.plot(beta_range,[theoretical_curve(beta) for beta in beta_range],label="theoretical")

    plt.plot(beta_range,energy,label="experimental")

    plt.legend()
    plt.show()