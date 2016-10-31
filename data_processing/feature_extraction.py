def extract_correlations(data):
    """Calculate the correlation matrix for time series with its associated eigenvalues and the eigenvalues
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
