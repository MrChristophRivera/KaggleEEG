import numpy as np
import numpy.linalg as la
import pandas as pd

from transformations import transform_psd


def extract_correlations(data):
    """Calculate a correlation matrix with its associated eigenvalues and the eigenvalues
    Parameters:
        data(pd.DataFrame): Data frame with the time series data
    Returns:
        corr(pd.Series): The correlations the time series
        eigs(pd.Series): The eigenvalues.

    """

    def format_tuple(t):
        """Helper function to arrange tuples."""
        # convet to list and sort
        t = sorted(list(t))
        return 'corr(%d,%d)' % (t[0], t[1])

    # calculate the correlation matrix and its eigenvalues
    corr = data.corr()
    eigs = pd.Series(la.eigvals(corr))

    # manipulate the corr data
    corr.values[np.tril_indices_from(corr)] = np.nan
    corr = corr.unstack()
    corr = corr[corr.isnull() == False]

    # rename the indexes for the correlation
    corr.index = map(format_tuple, corr.index.tolist())

    # change the name of the eigs
    eigs.index = map(lambda x: 'eig_%d' % (x + 1), list(eigs.index))

    return pd.DataFrame(pd.concat([corr, eigs])).T


########################################################################################################################
# FFT transform related
########################################################################################################################


def psi(psd_series, index=0):
    """Gets thePower Spectral Intensity of the standard bands for one time series
    This is using the standard bands of:
        δ(0.5–4Hz), θ(4–7Hz), α(8–12Hz), β(12–30Hz), and γ(30–100Hz)
    Parameters:
        psd_df(pd.DataFrame): a data frame with index frequencies and power.
        index(int): an index for labeling
    Returns:
        bands_df(pd.DataFrame): a data frame with the band sums
    """

    # compute the sums

    bands_df = pd.DataFrame({'channel %d δ(0.5–4Hz)' % (index + 1): psd_series[psd_series.between(0, 0.5)].sum(),
                             'channel %d θ(4–7Hz) ' % (index + 1): psd_series[psd_series.between(4, 7)].sum(),
                             'channel %d α(8–12Hz)' % (index + 1): psd_series[psd_series.between(8, 12)].sum(),
                             'channel %d  β(12–30Hz)' % (index + 1): psd_series[psd_series.between(12, 30)].sum(),
                             'channel %d γ(30–100Hz)' % (index + 1): psd_series[psd_series.between(30, 100)].sum()},
                            index=[1])
    return bands_df


def get_psi(psd_df):
    """Gets thePower Spectral Intensity of the standard bands for all time series.
    This is using the standard bands of:
        δ(0.5–4Hz), θ(4–7Hz), α(8–12Hz), β(12–30Hz), and γ(30–100Hz)
    Parameters:
        psd_df(pd.DataFrame): a data frame with index frequencies and power.
    Returns:
        psi_df(pd.DataFrame): a data frame with the band sums
    """
    # do the calcs
    psi_df = [psi(psd_df.iloc[:, i], i) for i in range(16)]
    return pd.concat(psi_df, axis=1)


class FFT_Features(object):
    """ Class for extracting multiple features from a fft
        These features include:
            1. PowerSpectral Intensity
            2. Relative Intensity.
            3. Correlation of the PSD
            4. eigen values of the psd
    """

    def __init__(self, data):
        # transform the psd
        self.psd = transform_psd(data)
