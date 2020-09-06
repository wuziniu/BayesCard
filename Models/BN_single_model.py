import pomegranate
import time
import bz2
import pickle
import logging
import numpy as np

from Models.tools import discretize_series

logger = logging.getLogger(__name__)

class BN_Single():
    """
    Build a single Bayesian Network for a single table.
    Initialize with an appropriate table_name.
    """

    def __init__(self, table_name, meta_info=None, method='Pome', debug=True):
        self.table_name = table_name
        if meta_info is None:
            self.fanout_attr = []
            self.fanout_attr_inverse = []
            self.fanout_attr_positive = []
            self.null_values = []
            self.n_distinct_mapping = dict()
        else:
            self.fanout_attr = meta_info['fanout_attr']
            self.fanout_attr_inverse = meta_info['fanout_attr_inverse']
            self.fanout_attr_positive = meta_info['fanout_attr_positive']
            self.null_values = meta_info['null_values']
            self.n_distinct_mapping = meta_info['n_distinct_mapping']
        self.n_in_bin = dict()
        self.encoding = dict()
        self.mapping = dict()
        self.domain = dict()
        self.fanouts = dict()
        self.max_value = dict()
        self.method = method
        self.model = None
        self.structure = None
        self.debug = debug

    def __str__(self):
        return f"bn{self.table_name}.{self.algorithm}-{self.max_parents}-{self.root}-{self.n_mcv}-{self.n_bins}"

    def build_discrete_table(self, data, n_mcv, n_bins, drop_na=True, ignore_cols=[]):
        """
        Discretize the entire table use bining (This is using histogram method for continuous data)
        ::Param:: table: original table
                  n_mcv: for categorical data we keep the top n most common values and bin the rest
                  n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
                  drop_na: if True, we drop all rows with nan in it
                  ignore_cols: drop the unnessary columns for example id attribute
        """
        table = data.copy()
        if drop_na:
            table = table.dropna()
        for col in table.columns:
            if col in ignore_cols:
                table = table.drop(col, axis=1)
            else:
                f = 0
                if col in self.fanout_attr_inverse:
                    f = 2
                elif col in self.fanout_attr_positive:
                    f = 1
                table[col], self.n_in_bin[col], self.encoding[col], self.mapping[col], self.domain[col], \
                self.fanouts[col] = discretize_series(
                    table[col],
                    n_mcv=n_mcv,
                    n_bins=n_bins,
                    is_continous=self.attr_type[col] == "continuous",
                    drop_na=not drop_na,
                    fanout=f
                )
                self.max_value[col] = int(table[col].max()) + 1
        self.node_names = list(table.columns)
        return table

    def is_numeric(self, val):
        if isinstance(val, int):
            return True
        if isinstance(val, float):
            return True

    def get_attr_type(self, dataset, threshold=3000):
        attr_type = dict()
        for col in dataset.columns:
            n_unique = dataset[col].nunique()
            if n_unique == 2:
                attr_type[col] = 'boolean'
            elif n_unique >= len(dataset)/20 or (self.is_numeric(dataset[col].iloc[0]) and n_unique > threshold):
                attr_type[col] = 'continuous'
            else:
                attr_type[col] = 'categorical'
        return attr_type

    def apply_encoding_to_value(self, value, col):
        """ Given the original value in the corresponding column and return its encoded value
            Note that every value of all col in encoded.
        """
        if col not in self.encoding:
            return None
        else:
            if type(value) == list:
                enc_value = []
                for val in value:
                    if val not in self.encoding[col]:
                        enc_value.append(None)
                    else:
                        enc_value.append(self.encoding[col][val])
                return enc_value
            else:
                if value not in self.encoding[col]:
                    return None
                else:
                    return self.encoding[col][value]

    def apply_ndistinct_to_value(self, enc_value, value, col):
        # return the number of distinct value in the bin
        if col not in self.n_in_bin:
            return 1
        else:
            if type(enc_value) != list:
                enc_value = [enc_value]
                value = [value]
            else:
                assert len(enc_value) == len(value), "incorrect number of values"
            n_distinct = []
            for i, enc_val in enumerate(enc_value):
                if enc_val not in self.n_in_bin[col]:
                    n_distinct.append(1)
                elif type(self.n_in_bin[col][enc_val]) == int:
                    n_distinct.append(1 / self.n_in_bin[col][enc_val])
                elif value[i] not in self.n_in_bin[col][enc_val]:
                    n_distinct.append(1)
                else:
                    n_distinct.append(self.n_in_bin[col][enc_val][value[i]])
            return np.asarray(n_distinct)

    def learn_model_structure(self, dataset, nrows=None, attr_type=None, rows_to_use=500000, n_mcv=30, n_bins=60,
                              ignore_cols=['id'], algorithm="greedy", drop_na=True, max_parents=2, root=None,
                              n_jobs=8, return_model=False, return_dataset=False, discretized=False):
        """ Build the Pomegranate model from data, including structure learning and paramter learning
            ::Param:: dataset: pandas.dataframe
                      attr_type: type of attributes (binary, discrete or continuous)
                      rows_to_use: subsample the number of rows to use to learn structure
                      n_mcv: for categorical data we keep the top n most common values and bin the rest
                      n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
            for other parameters, pomegranate gives a detailed explaination:
            https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
        """
        if nrows is None:
            self.nrows = len(dataset)
        else:
            self.nrows = nrows
        self.algorithm = algorithm
        self.max_parents = max_parents
        self.n_mcv = n_mcv
        self.n_bins = n_bins
        self.root = root

        if attr_type is None:
            self.attr_type = self.get_attr_type(dataset)
        else:
            self.attr_type = attr_type
        t = time.time()
        if not discretized:
            discrete_table = self.build_discrete_table(dataset, n_mcv, n_bins, drop_na, ignore_cols)
            logger.info(f'Discretizing table takes {time.time() - t} secs')
            logger.info(f'Learning BN optimal structure from data with {self.nrows} rows and'
                        f' {len(self.node_names)} cols')
            print(f'Discretizing table takes {time.time() - t} secs')
        t = time.time()
        if len(discrete_table) <= rows_to_use:
            model = pomegranate.BayesianNetwork.from_samples(discrete_table,
                                                         algorithm=algorithm,
                                                         state_names=self.node_names,
                                                         max_parents=max_parents,
                                                         n_jobs=n_jobs,
                                                         root=self.root)
        else:
            model = pomegranate.BayesianNetwork.from_samples(discrete_table.sample(n=rows_to_use),
                                                         algorithm=algorithm,
                                                         state_names=self.node_names,
                                                         max_parents=max_parents,
                                                         n_jobs=n_jobs,
                                                         root=self.root)
        logger.info(f'Structure learning took {time.time() - t} secs.')
        print(f'Structure learning took {time.time() - t} secs.')

        self.structure = model.structure

        if return_model:
            if return_dataset:
                return model, discrete_table
            else:
                return model
        elif return_dataset:
            return discrete_table

        return None

    def build_from_data(self, dataset):
        raise NotImplemented

    def save(self, path, compress=False):
        if compress:
            with bz2.BZ2File(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def query(self, query):
        raise NotImplemented


def load_BN_single(path):
    """Load BN ensembles from pickle file"""
    with open(path, 'rb') as handle:
        bn = pickle.load(handle)
    return bn
