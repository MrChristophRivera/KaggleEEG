# Functions for transforming the data
import numpy as np
import numpy.linalg as la
import pandas as pd


def scale(x):
    """scales a Pandas series by subtracting the mean and dividing by std
    Parameters:
        x(pd.Series): The series to scale
    Returns:
        x_scaled(pd.Series)"""
    return (x - np.mean(x)) / np.std(x)
