# Functions for transforming the data
import numpy as np
import pandas as pd


def scale(x):
    """scales a Pandas series by subtracting the mean and dividing by std
    Parameters:
        x(pd.Series): The series to scale
    Returns:
        x_scaled(pd.Series)"""
    return (x - np.mean(x)) / np.std(x)


def extract_correlation(df):
    """Calculate the correlation matrix for time series with its associated eigenvalues and the eigenvalues
    Parameters:
        df(pd.DataFrame): Data frame with the time series data
    Returns:
        corr(pd.Series): The correlations the time series
        eigs(pd.Series): The eigenvalues.

    """

    # calcualte the correlation matrix, convert the lower left triangle into a np.nan.
    corr = data.corr()
    # calculate the eigenvalues
    eigs = pd.Series(LA.eigvals(corr))
    corr.values[np.tril_indices_from(corr)] = np.nan

    # Unstack, return the values that are not null and return
    corr = corr.unstack()
    return corr[corr.isnull() == False], eigs
