# Functions for handling data processing

import pandas as pd
from scipy.io import loadmat
import dask.multiprocessing
from dask import compute, delayed
from .transformations import normalize

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

class Processor(object):

    def __init__(self, list_of_functions, dtrend=None):
        self.list_of_functions = list_of_functions
        self.dtrend = dtrend

    def process_folder(train_path, function_name):
        """ Apply function to all files in
        """
        seizure_df = pd.DataFrame()
        failures = []
        results = []
        print(train_path)
        for patient_path in train_path:
            # This is how I speed up processing 4x by making full use of all cores in the CPUs.
            values = [delayed(function_name)('\\'.join([patient_path, f])) for f in listdir(patient_path) if isfile('/'.join([patient_path, f]))]
            result = compute(*values, get=dask.multiprocessing.get)
            results.append(result)
        return results

    def process_file(self, fname):
        """ Apply list of functions to file.

        Each function should return a dataframe with x columns and 1 row. The number of columns is equal to features
        extracted.

        :param fname:
        :param list_of_functions:
        :param dtrend (string):  Must be  'None', 'mean', 'median'
        :return:
        """
        base_path, target_path, f = fname.split('\\')
        path = '\\'.join([base_path, target_path])
        print('processing', path, f)
        df, sampling_rate, sequence = load_data(join(path, f))
        df = normalize(df, dtrend)
        # Determine if this is an inter or preictal dataset and put in corresponding bucket.
        split_string = f.replace('.', '_').split('_')
        feature_df_list = []
        for func in self.list_of_functions:
            # Process function and append index columns
            func_result_df = func(df)
            feature_df_list.append(func_result_df)
        feature_df = pd.concat(feature_df_list, 1)
        feature_df = append_index(feature_df, split_string)
        return feature_df

    def append_index(df, split_string):
        """ Append data set identifier and set index to identifier"""
        df['patient'] = split_string[0]
        df['dataset_id'] = split_string[1]
        df['pre_ictal'] = split_string[2]
        return df

    def normalize(self):
        """ Normalize data frame.
        """
        pass
