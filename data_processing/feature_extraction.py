import numpy as np
import numpy.linalg as la
import pandas as pd

from transformations import transform_psd


def extract_correlations(data, frequency_domain=False):
    """Calculate a correlation matrix with its associated eigenvalues and the eigenvalues.
    Used for both caluculating frequency domain and non frequency domain.
    Parameters:
        data(pd.DataFrame): Data frame with the time series data
    Returns:
        corr(pd.Series): The correlations the time series
        eigs(pd.Series): The eigenvalues.
        frequency_domain(bool): affects how the ouput data is labeled. If True, its in frequency space

    """

    def format_tuple(t, frequency):
        """Helper function to arrange tuples."""
        # convert to list and sort
        t = sorted(list(t))
        if frequency:
            return 'corr(%s,%s)' % (t[0], t[1])
        return 'corr(%d,%d)' % (t[0], t[1])

    # calculate the correlation matrix and its eigenvalues
    corr = data.corr()
    eigs = pd.Series(la.eigvals(corr))

    # manipulate the corr data
    corr.values[np.tril_indices_from(corr)] = np.nan
    corr = corr.unstack()
    corr = corr[corr.isnull() == False]

    # rename the indexes for the correlation
    corr.index = [format_tuple(index, frequency_domain) for index in corr.index.tolist()]

    # change the name of the eigs
    if frequency_domain:
        eigs.index = map(lambda x: 'PSD eig_%d' % (x + 1), list(eigs.index))
    else:
        eigs.index = map(lambda x: 'eig_%d' % (x + 1), list(eigs.index))
    return pd.DataFrame(pd.concat([corr, eigs])).T


########################################################################################################################
# FFT transform related
########################################################################################################################


def psi_rir(psd_series, index=0):
    """Gets thePower Spectral Intensity of the standard bands for one time series
    This is using the standard bands of:
        δ(0.5–4Hz), θ(4–7Hz), α(8–12Hz), β(12–30Hz), and γ(30–100Hz)
    Parameters:
        psd_df(pd.DataFrame): a data frame with index frequencies and power.
        index(int): an index for labeling
    Returns:
        bands_df(pd.DataFrame): a data frame with the band sums
    """

    # compute the values for the psi and place into an array
    psis = np.array([psd_series[psd_series.between(0.5, 4)].sum(),
                     psd_series[psd_series.between(4, 7)].sum(),
                     psd_series[psd_series.between(8, 12)].sum(),
                     psd_series[psd_series.between(12, 30)].sum(),
                     psd_series[psd_series.between(30, 70)].sum(),
                     psd_series[psd_series.between(70, 180)].sum()
                     ])

    # compute the values for the rir
    rirs = psis / psis.sum()

    # set up the data frame
    bands_df = pd.DataFrame({'channel %d δ(0.5–4Hz)' % (index + 1): psis[0],
                             'channel %d θ(4–7Hz) ' % (index + 1): psis[1],
                             'channel %d α(8–12Hz)' % (index + 1): psis[2],
                             'channel %d  β(12–30Hz)' % (index + 1): psis[3],
                             'channel %d low γ(30–70Hz)' % (index + 1): psis[4],
                             'channel %d high γ(70–180Hz)' % (index + 1): psis[5],
                             'channel %d RIR1' % (index + 1): rirs[0],
                             'channel %d RIR2' % (index + 1): rirs[1],
                             'channel %d RIR3' % (index + 1): rirs[2],
                             'channel %d RIR4' % (index + 1): rirs[3],
                             'channel %d RIR5' % (index + 1): rirs[4],
                             'channel %d RIR6' % (index + 1): rirs[5]
                             }, index=[1])

    return bands_df


def compute_psi_and_rir(psd_df):
    """computes the Power Spectral Intensity and the Relative Intenstiy Ratio of the standard bands for all time series.
    This is using the standard bands of:
        δ(0.5–4Hz), θ(4–7Hz), α(8–12Hz), β(12–30Hz), and γ(30–100Hz)
    Parameters:
        psd_df(pd.DataFrame): a data frame with index frequencies and power.
    Returns:
        psi_rir_df(pd.DataFrame): a data frame with the band sums and the relative ratio
    """

    # create
    psi_df = [psi_rir(psd_df.iloc[:, i], i) for i in range(16)]
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
        # transform the time sereis to compute the psd
        self.psd = transform_psd(data)


########################################################################################################################
# time domain extractions
########################################################################################################################

def petrosian_fd(ts):
    """ Computes the Fractal Dimension using one of Petrosian's Methods. It is a simple yet inexact method for computing
    the fractal dimension. This algo uses method c) as per Esteller et al 2001.

    It estimates the fractal dimension as:
    PFD =log10(N)/(log10(N) +log10(N/(N+0.4Ndelta)))
    where: N is the length of the series, (number of time points), Ndelta is the number of sign changes.
    Parameters:
        ts(np.array or pd.Series): the array
    Returns:
        pfd(float): the petrosian_fd
    """

    # get the number of total points
    n = len(ts)

    # compute the number of sign changes of the binary sequence
    # convert to binary sequence by subtracting the consecutive time points,
    # and converting to +1 or 0 if postive or negative
    binary = (ts.diff() >= 0) + 0

    # compute the diff again to determine the transition points,and compute the N delta
    n_delta = np.sum(np.abs(binary.diff()))

    # compute and return the pfd
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))


def extract_petrosian_fd(time_series):
    """Computes the Petrosian Fractal Dimension for each time_series in a data frame
    Parameters:
        time_series(pd.DataFrame): the data frame with the time series
    Returns:
        pfd_df(pd.DataFrame): a row data frame with the PFD
    """

