# Functions for handling data processing

from os import listdir
from os.path import join, split, isfile
import dask.multiprocessing
from dask import compute, delayed

import numpy as np
import pandas as pd
from scipy.io import loadmat

from transformations import impute_time_series

def convert_index_to_timedelta(index, sampling_rate=400):
    """converts the index to time delta"""
    index = [i * 1.0 / sampling_rate for i in index]
    return pd.to_timedelta(index, 's')


def parse_filename(filename, split_file=False):
    """Parses m filename to get the pertinent information"""
    if split_file:
        filename = split(filename)[1]

    # strip out the .mat
    filename = filename.replace('.mat', '')

    # parse the remaing part
    return [int(part) for part in filename.split('_')]


def get_data_files(list_o_paths):
    """This gets the data matlab files"""
    file_names = []
    for path in list_o_paths:
        files = [join(path, f) for f in listdir(path) if f.endswith('.mat')]
        file_names.extend(files)

    return file_names

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


def detect_whole_file_dropout(df):
    """ Detect dataframe that is entirely full of dropout values

    To do this sum up the entire dataframe.

    :param df: Data Dataframe to calculate
    :return: df: Dataframe with sum of sums of entire dataframe.
    """
    new_df = pd.DataFrame({'df_sum': [df.sum().sum()]})
    return new_df


def drop_nd_rows(df):
    """ Drop rows that have all 0 across all columns

    :param df: Data Dataframe to calculate
    :return: df: Data Dataframe with dropped row.
    """
    dropout_rows = df[(df.sum(axis=1) == 0)].index
    df.drop(dropout_rows)
    return df


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


def parse_filename(filename):
    """Parses m filename to get the pertinent information"""

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

    def get_data(self, folders, functions=None):
        if functions is None:
            functions = [...]

        da_data = []
        for folder in folders:
            results = [df for df in self.process_folder(folder) if df is not None]
            da_data.extend(results)

        return pd.concat(da_data)

    def process_folder(self, list_of_directories):
        """ Apply function to all files in
        """
        seizure_df = pd.DataFrame()
        failures = []
        results = []
        print(train_path)
        for patient_path in train_path:
            # This is how I speed up processing 4x by making full use of all cores in the CPUs.
            values = [delayed(self.process_file)('\\'.join([patient_path, f])) for f in listdir(patient_path) if
                      isfile('/'.join([patient_path, f]))]
            result = compute(*values, get=dask.multiprocessing.get)[0]
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
        df, _, _ = load_data(join(fname))
        # df = self.normalize(df)
        df = self.pre_process(df)
        if not df.empty:
            # Determine if this is an inter or preictal dataset and put in corresponding bucket.
            split_string = fname.split('/').pop().replace('.', '_').split('_')
            feature_df_list = []
            for func in self.list_of_functions:
                # Process function and append index columns
                func_result_df = func(df)
                feature_df_list.append(func_result_df)
            feature_df = pd.concat(feature_df_list, 1)
            feature_df = self.append_index(feature_df, split_string)
            return feature_df
        return None

    def append_index(self, df, split_string):
        """ Append data set identifier and set index to identifier"""
        path, patient_id = split_string[0].split('\\')
        df['patient'] = patient_id
        df['dataset_id'] = split_string[1]
        df['pre_ictal'] = split_string[2]

        return df

    def normalize(self, df):
        """ Normalize data frame.
        """
        return df

    def pre_process(self, df):
        df = drop_nd_rows(df)

        if not df.empty:
            df = impute_time_series(df)
        return df


########################################################################################################################
# stuff
########################################################################################################################



def map_functions(data, functions):
    """maps a list of functions to data and returns as a list of results
    Parameters:
        data: data to be computed on
        functions(list): a list of functions
    Returns:
        results(list): a list of the results
    """
    return [fun(data) for fun in functions]


def process_data(file_name, functions=None):
    """Processes one file at a time for extracting features
    Parameters:
        file_name(str): the file name
        functions(list): a list of functions for extracting features
    Returns:
        res(pd.DataFrame): a one row data frame with the features in the columns
    """

    if functions is None:
        functions = [extract_time_domain_features, extract_fft_features]

    # get the time series and parse the filename for the info
    time_series = load_data(file_name, True)[0]
    patient, number, condition = parse_filename(file_name, True)

    # create an index and prefix df
    index = pd.MultiIndex.from_tuples([(patient, number, condition)],
                                      names=['Patient', 'TraceNumber', 'Condition'])

    prefix_df = pd.DataFrame({'Patient': patient,
                              'TraceNumber': number,
                              'Condition': condition},
                             index=[0]
                             )

    # create a list two hold the data frames, call the functions and then concatenate the resulting dataframes
    res = [prefix_df]
    res.extend(map_functions(time_series, functions))
    res = pd.concat(res, axis=1)
    res.index = index
    return res


def process_multiple_data(files):
    """uses dask to process many files in parallel"""
    # set up the compute graph
    graph = delayed([delayed(process_data)(file_) for file_ in files])
    # compute the graph
    results = compute(graph)

    return pd.concat([results[0][i] for i in range(len(files))])
