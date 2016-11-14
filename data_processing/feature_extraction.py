#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.stats import kurtosis, skew

from transformations import transform_psd


def extract_correlations(data, frequency_domain=False):
    """Calculate a correlation matrix with its associated eigenvalues and the eigenvalues.
    Used for both caluculating frequency domain and non frequency domain.
    Parameters:
        data(pd.DataFrame): Data frame with the time series data
        frequency_domain(bool)
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
        psd_series(pd.DataFrame): a data frame with index frequencies and power.
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


def psi(psd_series, index=0):
    """Gets thePower Spectral Intensity of the standard bands for one time series
    This is using the standard bands of:
        δ(0.5–4Hz), θ(4–7Hz), α(8–12Hz), β(12–30Hz), and γ(30–100Hz)
    Parameters:
        psd_series(pd.DataFrame): a data frame with index frequencies and power.
        index(int): an index for labeling
    Returns:
        bands_df(pd.DataFrame): a data frame with the band sums
    """

    # compute the values for the psi_ and place into an array
    psi_ = pd.Series([psd_series[psd_series.between(0.5, 4)].sum(),
                      psd_series[psd_series.between(4, 7)].sum(),
                      psd_series[psd_series.between(8, 12)].sum(),
                      psd_series[psd_series.between(12, 30)].sum(),
                      psd_series[psd_series.between(30, 70)].sum(),
                      psd_series[psd_series.between(70, 180)].sum()
                      ])

    psi_.index = ['channel %d δ(0.5–4Hz)' % (index + 1),
                  'channel %d θ(4–7Hz)' % (index + 1),
                  'channel %d α(8–12Hz)' % (index + 1),
                  'channel %d  β(12–30Hz)' % (index + 1),
                  'channel %d low γ(30–70Hz)' % (index + 1),
                  'channel %d high γ(70–180Hz)' % (index + 1)]
    return pd.DataFrame(psi_)


def rir(psis_df, index=0):
    """computes the Relative Intensity Ratio given Spectral bands as a data frame"""
    # make a copy
    rir_df_ = psis_df.copy()

    # compute the rir, update the index name and return
    rir_df_ = rir_df_ / rir_df_.sum()

    rir_df_.index = ['channel %d RIR %d ' % (index + 1, i + 1) for i in range(len(rir_df_))]
    return rir_df_


def spectral_entropy(rir):
    """ computes the spectal etropy given relative intensity ratios"""
    return np.sum(rir.apply(lambda x: x * np.log(x)))


class FFT_Features(object):
    """ Class for extracting multiple features from a fft and places into a dataframe
        These features include:
            1. PowerSpectral Intensity at several different bands (done)
            2. Relative Intensity Ratio. (done)
            3. Correlation of the PSD (done)
            4. eigen values of the psd (done)
            5. Spectral Entropy at several bands (done)
    """

    def __init__(self, data=None):
        # set up the attributes of interests
        self.num = None
        self.psd = None

        # for the data
        self.psi = None
        self.rir = None
        self.correlation = None
        self.spectral_entropy = None
        self.features = None
        if data is not None:
            self.psd = transform_psd(data)
            self.num = len(data.columns)
            self.compute_features()

    def compute_features(self, data=None):
        """Computes the fetures as defined by the class and returns the values in a 1 row df"""
        if data is not None:
            self.psd = transform_psd(data)
            self.num = len(data.columns)
            self.psd = transform_psd(data)

        # compute some features
        self.psi = self.__compute_psis()
        self.rir = self.__compute_rirs()
        self.correlation = extract_correlations(self.psd, frequency_domain=True)
        self.spectral_entropy = self.__compute_spectral_entropy()

        # organize the features and return
        self.__format_data()

    def __compute_psis(self):
        """computes the psis and places into list of data frames"""
        # get the number of columns

        return [psi(self.psd.iloc[:, i], i) for i in range(self.num)]

    def __compute_rirs(self):
        """ computes the spectral intensity rations"""
        return [rir(self.psi[i], i) for i in range(self.num)]

    def __compute_spectral_entropy(self):
        """computes the spectral entropys"""
        # compute the spectral entropy for each and organize into a data frame
        df = pd.Series([spectral_entropy(self.rir[i]) for i in range(self.num)])
        df.index = ['SpectralEntropy %d' % (i + 1) for i in range(self.num)]
        return pd.DataFrame(df).T

    def __format_data(self):
        """ A helper function to format the features into a data frame"""
        psi_df = pd.concat(self.psi).T
        rir_df = pd.concat(self.rir).T
        features = pd.concat([psi_df, rir_df, self.correlation, self.spectral_entropy], axis=1)
        self.features = features.apply(lambda x: float(x))


def extract_fft_features(time_series):
    '''Given ts returns a df of features from freq domain'''
    return FFT_Features(time_series).features


########################################################################################################################
# time domain extractions
########################################################################################################################

def petrosian_fd(ts):
    """ Computes the Fractal Dimension using one of Petrosians Methods. It is a simple yet inexact method for
    computing the fractal dimension. This algo uses method c) as per Esteller et al 2001.

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

    # compute the values, place in a data frame and return
    pfd_df = pd.DataFrame(time_series.apply(petrosian_fd))
    pfd_df.index = ['PFD %d' % (i + 1) for i in range(len(time_series.columns))]
    return pfd_df.T


def extract_means(time_series, num=16):
    '''gets the means'''
    if num is None:
        num = len(time_series.columns)
    df = pd.DataFrame(time_series.mean())
    df.index = ['Mean %d' % (i + 1) for i in range(num)]
    return df.T


def extract_var(time_series, num=16):
    '''gets the vars'''
    if num is None:
        num = len(time_series.columns)
    df = pd.DataFrame(time_series.var())
    df.index = ['Variance %d' % (i + 1) for i in range(num)]
    return df.T


def extract_kurtosis(time_series, num=16):
    '''gets the kurtosis'''
    if num is None:
        num = len(time_series.columns)
    df = pd.DataFrame(time_series.apply(kurtosis))
    df.index = ['Kurtosis %d' % (i + 1) for i in range(num)]
    return df.T


def extract_skew(time_series, num=16):
    '''gets the vars'''
    if num is None:
        num = len(time_series.columns)
    df = pd.DataFrame(time_series.apply(skew))
    df.index = ['Kurtosis %d' % (i + 1) for i in range(num)]
    return df.T


def extract_hfd_features(df, Kmax=5):
    """ Compute Hjorth Fractal Dimension of a data frame with 16 time series data, kmax
     is an HFD parameter

    df --- an input dataframe including the signals for 16 channels (each column is for each channel)

    return: a one-row dataframe with 16 features named hfd_1, hfd_2, ..., hfd_16.
    """
    index = 0
    base = "hfd"
    df_features = pd.DataFrame(index=[0], columns=[(base + str(i)) for i in range(1, 17)])
    df_features
    for column in df:
        X = df[column].tolist()
        L = []
        x = []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(float(1) / k), 1])

        (p, r1, r2, s) = np.linalg.lstsq(x, L)

        df_features.iloc[0, index] = p[0]
        index += 1

    return df_features


def extract_time_domain_features(time_series, num=16):
    """Extract features for all the time domain"""
    return pd.concat([extract_var(time_series, 16),
                      extract_skew(time_series, 16),
                      extract_kurtosis(time_series, 16),
                      extract_correlations(time_series),
                      extract_hfd_features(time_series),
                      extract_petrosian_fd(time_series)
                      ], axis=1)
