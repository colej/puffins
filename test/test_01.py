import numpy as np
import matplotlib.pyplot as plt

import puffins as pf
from pythia.timeseries.lombscargle import LS_periodogram

if __name__ == '__main__':
    t,f = np.loadtxt('ugru.dat', unpack=True)
    # f -= np.median(f)

    X, betas, reconstructed_y = pf.solver.generate_synthetic_data(t, f, 1.8805, 35)

    plt.plot(t, f, 'k.')
    plt.plot(t, reconstructed_y, 'r-')

    res = f - reconstructed_y
    # plt.plot(t, res, 'b-')
    plt.show()
    
    print(res)

    nu, amp = LS_periodogram(t, res-np.median(res), max=40)
    fig, ax = plt.subplots(1,1)
    ax.plot(nu, amp)
    plt.show()