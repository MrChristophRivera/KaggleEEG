from


def plot_ts(tsdata, columns=-1):
    """plots the timeseries
    Parameters:
        tsdata(pd.DataFrame): the data
        column: the columns to use
    """

    if columns > 0:
        tsdata = tsdata.iloc[:, columns]

    # scale the data
    scaled = tsdata.apply(scale)

    # unstack the data, reset the columns and change the names
    ts = pd.DataFrame(scaled.unstack())
    ts.reset_index(inplace=True)
    ts.columns = ['Position', 'Time', 'Signal']

    # plot the data

    ax = sns.tsplot(time="Time", value="Signal", data=ts, condition='Position', unit='Position')

    sns.despine()
    return ax
