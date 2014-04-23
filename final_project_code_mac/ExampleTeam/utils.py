def unpickle(file):
    """Loads and returns a pickled data structure in the given `file` name
    Example usage:
        data = unpickle('output/U_20_std')
    """
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data

def pickle(data, file):
    """Dumps data to a file
    Example usage:
        pickle(U, 'output/U_20_std')
    """
    fo = open(file,'wb')
    cPickle.dump(data,fo)
    fo.close()