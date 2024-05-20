import numpy as np
from scipy import signal
from pythia.timeseries.lombscargle import LS_periodogram

def calculate_autocorrelation(x, y, type='time', mode='same', fmax=None):

    if type == 'periodogram':
        y -= np.median(y)
        # Calculate the periodogram
        xx, yy = LS_periodogram(x, y, max=fmax)
    elif type == 'time':
        xx, yy = x, y
    else:
        raise ValueError(f'Unknown type {type}')
        
    dx = xx[10]-xx[9] # arbitrary

    acorr = signal.correlate(yy, yy, mode=mode)
    lags = signal.correlation_lags(len(yy), len(yy), mode=mode)
    idx = np.where(lags >= 100)

    lags = lags[idx] * dx
    acorr = acorr[idx] 
    return lags, acorr / acorr.max()