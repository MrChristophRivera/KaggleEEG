# Functions for handling data processing

from os import listdir
from os.path import join, split, isfile

import pandas as pd
from dask import delayed, compute
import dask
from scipy.io import loadmat


def convert_index_to_timedelta(index, sampling_rate=400):
    """converts the index to time delta"""
    index = [i * 1.0 / sampling_rate for i in index]
    return pd.to_timedelta(index, 's')


def load_data(path, convert_index=False):
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


def extract_base_stats(data, patient=1, number=1, condition=0):
    """extracts the mean, variance, skew and kurtosis of a given data set of data set
    data(dataframe): a given data set
    patient(int): the patient
    number(int): the trace number
    conditon(int): 0 if interictal or 1 if ictal
    """
    mean = data.mean()
    var = data.var()
    skew = data.skew()
    kurtosis = data.kurtosis()

    l = len(mean)

    # format the namea

    def create_tuple(name):
        return [(name, str(i + 1)) for i in xrange(l)]

    columns = create_tuple('mean') + create_tuple('variance') + create_tuple('skew') + create_tuple('kurtosis')

    res = pd.DataFrame(pd.concat([mean, var, skew, kurtosis])).T
    res.columns = pd.MultiIndex.from_tuples(columns, names=['Statistic', 'Channel'])
    res.index = pd.MultiIndex.from_tuples([(patient, number, condition)], names=['Patient', 'TraceNumber', 'Condition'])
    return res


def parse_filename(filename, split_file=False):
    """Parses m filename to get the pertinent information"""
    if split_file:
        filename = split(filename)[1]

    # strip out the .mat
    filename = filename.replace('.mat', '')

    # parse the remaing part
    return [int(part) for part in filename.split('_')]


def count_files(directory):
    """counts the number of files per condition for each patient in a directory"""

    # parse the files
    files = [parse_filename(f) for f in listdir(directory) if f.endswith('.mat')]

    # go through each file and count
    interictal = {}
    postictal = {}

    for f in files:
        if f[2] == 0:
            interictal.setdefault(f[0], 0)
            interictal[f[0]] += 1
        else:
            postictal.setdefault(f[0], 0)
            postictal[f[0]] += 1
    return interictal, postictal


def get_files():
    """gets the file names"""
    base = '/Users/crivera5/Documents/NonIntuitProjects/Kaggle/KaggleEEG'
    path1 = join(base, 'train_1')
    path2 = join(base, 'train_2')
    path3 = join(base, 'train_3')

    def get_fs(path):
        return [join(path, f) for f in listdir(path) if f.endswith('.mat')]

    return get_fs(path1) + get_fs(path2) + get_fs(path3)


def get_stats_from_one_file(filename):
    """gets the stats from one file
    Parameters:
        filename(str): absolute path to the file
    Return(df): 1 row data frame with the stats

    """
    try:
        # get the data
        data = load_data(filename)[0]
        # get the file name parts
        patient, number, condition = parse_filename(split(filename)[1])

        # get the stats and return
        return extract_base_stats(data, patient, number, condition)
    except ValueError:
        return None


def get_stats():
    """function to get the stats and put into a data frame """
    # get the files names
    files = get_files()

    # use a slow for loop to get the res
    return pd.concat(map(get_stats_from_one_file, files))


class Processor(object):
    def __init__(self, list_of_functions, dtrend=None):
        self.list_of_functions = list_of_functions
        self.dtrend = dtrend

    def process_folder(self, train_path, function_name):
        """ Apply function to all files in
        """
        seizure_df = pd.DataFrame()
        failures = []
        results = []
        print(train_path)
        for patient_path in train_path:
            # This is how I speed up processing 4x by making full use of all cores in the CPUs.
            values = [delayed(function_name)('\\'.join([patient_path, f])) for f in listdir(patient_path) if
                      isfile('/'.join([patient_path, f]))]
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
        df = self.normalize(df)
        # Determine if this is an inter or preictal dataset and put in corresponding bucket.
        split_string = f.replace('.', '_').split('_')
        feature_df_list = []
        for func in self.list_of_functions:
            # Process function and append index columns
            func_result_df = func(df)
            feature_df_list.append(func_result_df)
        feature_df = pd.concat(feature_df_list, 1)
        feature_df = self.append_index(feature_df, split_string)
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
