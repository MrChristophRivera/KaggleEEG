# Functions for transforming the data
import numpy as np
import pandas as pd
from scipy import signal


def scale(x):
    """scales a Pandas series by subtracting the mean and dividing by std
    Parameters:
        x(pd.Series): The series to scale
    Returns:
        x_scaled(pd.Series)"""
    return (x - np.mean(x)) / np.std(x)


########################################################################################################################
# FFT transform related
########################################################################################################################

def psd(x, index=1):
    """ calculate the FFT periodiogram of time series x  and returns as a df"""
    freqs, ppx = signal.periodogram(x, fs=400)
    name = 'channel %d ' % index

    return pd.DataFrame({name: ppx}, index=freqs, )


def transform_psd(data, detrend=True):
    """Generates the FFT powerspectrum using the scipy.signal periodiogram function with sampling rate of 400
    Parameters:
        data(pd.DataFrame): The data with the time series
        detrend(bool): if true, detrend with the mean
    Return:
        psd_df(pd.DataFrame): a data frame with the PSD
    """

    # copy the data
    data = data.copy()

    if detrend:
        data.apply(lambda x: x - x.mean())

    # calculate the psd
    return pd.concat([psd(data.iloc[:, i], i + 1) for i in range(16)], axis=1)
