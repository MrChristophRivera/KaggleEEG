# Functions for handling data processing

import pandas as pd
from scipy.io import loadmat


def convert_index_to_timedelta(index, sampling_rate=400):
    """converts the index to time delta"""
    index = [i * 1.0 / sampling_rate for i in index]
    return pd.to_timedelta(index, 's')


def load_data(path, convert_index=True):
    """converts the data to a pandas object
    Parameters:
        path(str): absolute path to the m file
        convert_index(bool): if True, convert the index to a time delta
    Returns:
        (data,sampling_rate,sequence):(pd.DataFrame, int, int)
    """
    # load the matlab file and extract the data
    data = loadmat(path)['dataStruct']

    # get the sampling rate and cast to int
    sampling_rate = int(data['iEEGsamplingRate'][0][0])

    # extract the iEEG traces and electrode channel data and place into a data frame
    traces = data['data'][0][0]
    channels = data['channelIndices'][0][0][0]
    df = pd.DataFrame(traces, columns=channels)

    if convert_index:
        df.index = convert_index_to_timedelta(df.index, sampling_rate)

    # get the sequence collection number if present (not present in test)
    sequence = -1
    if 'sequence' in data.dtype.names:
        sequence = int(data['sequence'])

    return df, sampling_rate, sequence
