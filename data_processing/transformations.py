# Functions for transforming the data
import numpy as np
import pandas as pd
from outliers import smirnov_grubbs as grubbs
from scipy import signal
from scipy.stats import iqr

def scale(x):
    """scales a Pandas series by subtracting the mean and dividing by std
    Parameters:
        x(pd.Series): The series to scale
    Returns:
        x_scaled(pd.Series)"""
    return (x - np.mean(x)) / np.std(x)


########################################################################################################################
# imputation
########################################################################################################################
def impute_zeros(ts):
    """Replaces the zeroes with the the mean of the surrounding values"""
    ts = list(ts)
    m = len(ts) / 2
    if ts[m] != 0:
        return ts[m]

    else:
        ts.pop(m)
        return np.mean(ts)


def impute_time_series(time_series_df, window=3):
    """ replaces all zeors with the average value for the times series with rolling_apply"""

    # get the first and last row
    first_row = time_series_df.iloc[0, :]
    last_row = time_series_df.iloc[-1, :]

    # do the imputation
    imputed = time_series_df.rolling(center=True, window=window).apply(func=impute_zeros)

    # replace the rows
    imputed.iloc[0, :] = first_row
    imputed.iloc[-1, :] = last_row
    return imputed


def interpolate_zeros(df):
    """Replaces zero values using linear interpolation
    Parameters:
        df(pd.DataFrame): a data frame with numeric values
    Returns:
        df
    """

    return df.replace(to_replace=0.0, value=np.nan).interpolate(method='linear')


def replace_outliers_grubs(x, alpha_input=0.001, val=0.0):
    """Detect the outliers of a series using Smirnov_grubbs test and replace the outliers with zeros
       Please install the outlier-utils library first
       pip install outlier-utils
    """
    outliers_indices = grubbs.two_sided_test_indices(x, alpha=alpha_input)
    x.loc[outliers_indices] = val
    return x


def replace_outliers_with_zeros(df, alpha_input=0.001):
    """Detect the outliers of a dataframe using Smirnov_grubbs test and replace the outliers with zeros"""

    return df.apply(replace_outliers_grubs)


def replace_outliers(s, factor=2.0, replacement=0.0):
    """ convert outliers to zero
    Parameters:
        s(pd.Series): the input series
        factor(float): the factor to multiply the interquartile range
    Returns:
        replacement(float): the value for which to replace
    """
    s = s.copy()
    iqr_ = iqr(s)
    outliers = s.between(-factor * iqr_, factor * iqr_) == False
    s.loc[outliers] = replacement
    return s


########################################################################################################################
# FFT transform related
########################################################################################################################

def psd(x, index=1):
    """ calculate the FFT periodiogram of time series x  and returns as a df"""
    freqs, ppx = signal.periodogram(x, fs=400)
    name = 'channel %d ' % index

    return pd.DataFrame({name: ppx}, index=freqs, )


def transform_psd(data, detrend=True):
    """Generates the FFT power spectrum using the scipy.signal periodiogram function with sampling rate of 400
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
