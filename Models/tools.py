import numbers
import math
import pandas as pd


def categorical_qcut(series, q, start_value):
    """Computes categorical quantiles of a pandas.Series objects."""
    bin_freq = 1 / q
    value_counts = series.value_counts(normalize=True).sort_index()
    
    bins = {}
    n_distinct = {} # Count the number of distinct values per bin
    encoding = {}
    mapping = {}
    values_in_bin = []
    freq_in_bin = []
    cum_freq = 0
    
    value = start_value
    for i, (val, freq) in enumerate(value_counts.iteritems()):
        if len(values_in_bin)==0:
            left = i
        values_in_bin.append(val)
        freq_in_bin.append(freq)
        cum_freq += freq
        encoding[val] = value
        if cum_freq >= bin_freq or (i+1) == len(value_counts):
            values_in_bin = sorted(values_in_bin)
            n_distinct[value] = dict()
            for i,v in enumerate(values_in_bin):
                bins[v] = value
                n_distinct[value][v]=freq_in_bin[i]/cum_freq
            values_in_bin = []
            freq_in_bin = []
            cum_freq = 0
            value+=1

    return n_distinct, encoding



def discretize_series(series: pd.Series, n_mcv, n_bins, is_continous=False, continuous_bins=None, drop_na=True) -> pd.Series:
    """
    Map every value to category, binning the small categories if there are more than n_mcv categories.
    Map intervals to categories for efficient model learning
    return:
    s: discretized series
    n_distinct: number of distinct values in a mapped category (could be empty)
    encoding: encode the original value to new category (will be empty for continous attribute)
    mapping: map the new category to pd.Interval (for continuous attribute only)
    """
    s = series.copy()
    n_distinct = dict()
    encoding = dict()
    mapping = dict()

    if is_continous or (s.nunique() >= len(s) / 10 and isinstance(s.iloc[0], numbers.Number)):
        # Under this condition, we can assume we are dealing with continuous data
        # Histogram for continuous data
        if not continuous_bins:
            continuous_bins = n_bins * 2
        domains = (s.min(), s.max())
        temp = pd.qcut(s, q=continuous_bins, duplicates='drop')
        categ = dict()
        val = 0
        # map interval object to int category for efficient training
        for interval in sorted(list(temp.unique()), key=lambda x: x.left):
            categ[interval] = val
            mapping[val] = interval
            val += 1

        s = temp.cat.rename_categories(categ)

        if drop_na:
            s = s.cat.add_categories(int(val))
            s = s.fillna(val)  # Replace np.nan with some integer that is not in encoding
        return s, None, None, mapping, domains
        
    
    # Remove trailing whitespace
    if s.dtype == 'object':
        s = s.str.rstrip()
    domains = list(s.unique())
    # Treat most common values
    value_counts = s.value_counts()
    n_mcv = len(value_counts) if n_mcv == -1 else n_mcv
    n_largest = value_counts.nlargest(n_mcv)
    most_common_vals = set(n_largest.index)
    most_common_mask = s.isin(most_common_vals)
    
    #encoding most common categories to int categories
    val = 0
    for i in n_largest.index:
        encoding[i] = val
        val+=1
    
    # Treat least common values
    n_least_common = s[~most_common_mask].nunique()
    n_bins = min(n_least_common, n_bins)
    if n_least_common > 0:
        #encoding the least common string category to int category
        n_distinct, nl_encoding = categorical_qcut(s[~most_common_mask], q=n_bins, start_value=val)
        encoding.update(nl_encoding)
        
    #map the original value to encoded value
    temp = series.copy()
    for i in s.unique():
        if not math.isnan(i):
            temp[s==i] = encoding[i]
    del s
    
    if drop_na:
        #temp = temp.cat.add_categories(int(n_mcv+n_bins+1))
        temp = temp.fillna(n_mcv+n_bins+1)  # Replace np.nan with some integer that is not in encoding

    return temp, n_distinct, encoding, None, domains

