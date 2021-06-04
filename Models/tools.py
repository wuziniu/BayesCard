import numbers
import math
import copy
import numpy as np
import pandas as pd
from enum import Enum
import logging

logger = logging.getLogger(__name__)


def categorical_qcut(series, q, start_value, fanout=0, fanout_values=[], fanout_sum=[]):
    """Computes categorical quantiles of a pandas.Series objects."""
    bin_freq = 1 / q
    value_counts = series.value_counts(normalize=True).sort_index()
    
    bins = {}
    bin_width = {}
    n_distinct = {}    # Count the number of distinct values per bin
    encoding = {}
    values_in_bin = []
    freq_in_bin = []
    cum_freq = 0
    
    value = start_value
    for i, (val, freq) in enumerate(value_counts.iteritems()):
        if len(values_in_bin) == 0:
            left = i
        values_in_bin.append(val)
        freq_in_bin.append(freq)
        cum_freq += freq
        encoding[val] = value
        if cum_freq >= bin_freq or (i+1) == len(value_counts):
            if fanout == 1:
                fanout_values.append(np.sum(np.asarray(values_in_bin) * np.asarray(freq_in_bin) / cum_freq))
                fanout_sum.append(cum_freq)
            elif fanout == 2:
                values_copy = copy.deepcopy(np.asarray(values_in_bin))
                values_copy[values_copy == 0] = 1
                fanout_values.append(np.sum(1 / values_copy * np.asarray(freq_in_bin) / cum_freq))
                fanout_sum.append(cum_freq)
            values_in_bin = sorted(values_in_bin)
            n_distinct[value] = dict()
            bin_width[value] = cum_freq
            for j, v in enumerate(values_in_bin):
                bins[v] = value
                n_distinct[value][v] = freq_in_bin[j]/cum_freq
            values_in_bin = []
            freq_in_bin = []
            cum_freq = 0
            value += 1

    return n_distinct, bin_width, encoding, fanout_values, fanout_sum



def discretize_series(series: pd.Series, n_mcv, n_bins, is_continous=False, continuous_bins=None, drop_na=True,
                      fanout=0):
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
    bin_width = dict()
    encoding = dict()
    mapping = dict()

    if is_continous or (s.nunique() >= len(s) / 30 and isinstance(s.iloc[0], numbers.Number)):
        # Under this condition, we can assume we are dealing with continuous data
        # Histogram for continuous data
        if not continuous_bins:
            continuous_bins = n_bins * 2
        domains = (s.min(), s.max())
        temp = pd.qcut(s, q=continuous_bins, duplicates='drop')
        categ = dict()
        val = 0

        # store fanout values for fast computation
        fanout_values = []
        fanout_sum = []

        # map interval object to int category for efficient training
        for interval in sorted(list(temp.unique()), key=lambda x: x.left):
            categ[interval] = val
            mapping[val] = interval
            if fanout != 0:
                curr_values = np.asarray(temp[temp == interval].index)
                fanout_sum.append(len(curr_values))
                if fanout == 1:
                    fanout_values.append(np.nanmean(curr_values))
                elif fanout == 2:
                    curr_values[curr_values == 0] = 1
                    fanout_values.append(np.nanmean(1/curr_values))
            val += 1

        s = temp.cat.rename_categories(categ)

        if drop_na:
            s = s.cat.add_categories(int(val))
            s = s.fillna(val)  # Replace np.nan with some integer that is not in encoding
        return s, None, None, None, mapping, domains, fanout_values, fanout_sum
        
    
    # Remove trailing whitespace
    if s.dtype == 'object':
        s = s.str.strip()
    domains = list(s.unique())

    #store fanout values for fast computation
    fanout_values = []
    fanout_sum = []
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
        if fanout == 1:
            fanout_values.append(i)
            fanout_sum.append(n_largest[i])
        elif fanout == 2:
            fanout_values.append(1/max(i, 1))
            fanout_sum.append(n_largest[i])
        val += 1
    
    # Treat least common values
    n_least_common = s[~most_common_mask].nunique()
    n_bins = min(n_least_common, n_bins)
    if n_least_common > 0:
        #encoding the least common string category to int category
        n_distinct, bin_width, nl_encoding, fanout_values, fanout_sum = categorical_qcut(s[~most_common_mask], n_bins,
                                                              val, fanout, fanout_values, fanout_sum)
        encoding.update(nl_encoding)
        
    #map the original value to encoded value
    temp = series.copy()
    for i in s.unique():
        temp[s == i] = encoding[i]
    del s
    
    if drop_na:
        #temp = temp.cat.add_categories(int(n_mcv+n_bins+1))
        temp = temp.fillna(n_mcv+n_bins+1)  # Replace np.nan with some integer that is not in encoding

    return temp, n_distinct, bin_width, encoding, None, domains, np.asarray(fanout_values), np.asarray(fanout_sum)


class MetaType(Enum):
    REAL = 1
    BINARY = 2
    DISCRETE = 3


class Type(Enum):
    REAL = (1, MetaType.REAL)
    INTERVAL = (2, MetaType.REAL)
    POSITIVE = (3, MetaType.REAL)
    CATEGORICAL = (4, MetaType.DISCRETE)
    ORDINAL = (5, MetaType.DISCRETE)
    COUNT = (6, MetaType.DISCRETE)
    BINARY = (7, MetaType.BINARY)

    def __init__(self, enum_val, meta_type):
        self._enum_val = enum_val
        self._meta_type = meta_type

    @property
    def meta_type(self):
        return self._meta_type


META_TYPE_MAP = {
    MetaType.REAL: [Type.REAL, Type.INTERVAL, Type.POSITIVE],
    MetaType.BINARY: [Type.BINARY],
    MetaType.DISCRETE: [Type.CATEGORICAL, Type.ORDINAL, Type.COUNT],
}


