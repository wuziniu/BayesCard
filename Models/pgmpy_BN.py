from Pgmpy.models import BayesianModel
import numpy as np
import copy
import logging
import itertools
import time
from Models.BN_single_model import BN_Single

logger = logging.getLogger(__name__)


class Pgmpy_BN(BN_Single):
    """
    Build a single Bayesian Network for a single table using pgmpy
    """

    def __init__(self, table_name, method='Pome', debug=True, infer_algo=None):
        """
        infer_algo: inference method, choose between 'exact', 'BP'
        """
        BN_Single.__init__(self, table_name, method, debug)
        self.infer_algo = infer_algo

    def build_from_data(self, dataset, attr_type=None, sample_size=1000000, n_mcv=30, n_bins=60, ignore_cols=['id'],
                        algorithm="chow-liu", drop_na=True, max_parents=-1, root=0, n_jobs=8):
        """ Build the Pomegranate model from data, including structure learning and paramter learning
            ::Param:: dataset: pandas.dataframe
                      attr_type: type of attributes (binary, discrete or continuous)
                      sample_size: subsample the number of rows to use to learn structure
                      n_mcv: for categorical data we keep the top n most common values and bin the rest
                      n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
            for other parameters, pomegranate gives a detailed explaination:
            https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
        """
        self.algorithm = algorithm
        if algorithm != "junction":
            discrete_table = self.learn_model_structure(dataset, attr_type, sample_size,
                                                        n_mcv, n_bins, ignore_cols, algorithm,
                                                        drop_na, max_parents, root, n_jobs, return_dataset=True)
        else:
            discrete_table = self.learn_model_structure(dataset, attr_type, sample_size,
                                                        n_mcv, n_bins, ignore_cols, 'exact',
                                                        drop_na, max_parents, root, n_jobs, return_dataset=True)
        spec = []
        orphans = []
        for i, parents in enumerate(self.structure):
            for p in parents:
                spec.append((self.node_names[p], self.node_names[i]))
            if not parents:
                orphans.append(self.node_names[i])
        if self.debug:
            logger.info(f"Model spec{spec}")
        self.model = BayesianModel(spec)
        for o in orphans:
            self.model.add_node(o)
        logger.info('calling pgm.BayesianModel.fit...')
        t = time.time()
        if len(discrete_table) > 10 * sample_size:
            self.model.fit(discrete_table.sample(n=10 * sample_size))
        else:
            self.model.fit(discrete_table)
        if algorithm == "junction":
            try:
                self.model = self.model.to_junction_tree()
            except:
                self.model = self.model
                logger.warning(
                    "This BN is not able to transform into junction tree, probably because it's not connected, just use BN")
        logger.info(f"done, took {time.time() - t} secs.")
        self.init_inference_method()

    def init_inference_method(self):
        """
        Initial the inference method for query
        """
        if self.infer_algo is None:
            if self.algorithm == "chow-liu":
                self.infer_algo = "exact"
            else:
                self.infer_algo = "BP"

        if self.infer_algo == "exact":
            from Pgmpy.inference import VariableElimination
            self.infer_machine = VariableElimination(self.model)
        elif self.infer_algo == "BP":
            try:
                # Belief Propogation won't work if the model graph is not connected
                from Pgmpy.inference import BeliefPropagation
                self.infer_machine = BeliefPropagation(self.model)
                self.infer_machine.calibrate()
                logger.warning("Graph is not connected, we have automatically set the "
                               "inference algorithm to exact")
            except:
                from Pgmpy.inference import VariableElimination
                self.infer_machine = VariableElimination(self.model)
        elif self.infer_algo == "sampling":
            from Pgmpy.sampling import BayesianModelSampling
            self.infer_machine = BayesianModelSampling(self.model)
            logger.warning("Using sampling as an inference algorithm is very inefficient")
            raise NotImplemented

    def continuous_range_map(self, col, range):
        def cal_coverage(l, r, target):
            tl = target.left
            tr = target.right
            if l >= tr: return 0
            if r <= tl: return 0
            if r > tr:
                if l < tl:
                    return 1
                else:
                    return (tr - l) / (tr - tl)
            else:
                if l > tl:
                    return (r - l) / (tr - tl)
                else:
                    return (r - tl) / (tr - tl)

        def binary_search(i, j):
            # binary_search to find a good starting point
            if i == j:
                return i
            m = int((j - i) / 2)
            interval = self.mapping[col][m]
            if left > interval.right:
                binary_search(m, j)
            elif right < interval.left:
                binary_search(i, m)
            else:
                return m

        (left, right) = range
        if left is None: left = -np.Inf
        if right is None: right = np.Inf
        query = []
        coverage = dict()
        start_point = binary_search(0, len(self.mapping[col]))
        start_point_left = start_point
        start_point_right = start_point + 1
        indicator_left = True
        indicator_right = True
        while (start_point_left >= 0 and start_point_right < len(self.mapping[col])
               and (indicator_left or indicator_right)):
            if indicator_left:
                cover = cal_coverage(left, right, self.mapping[col][start_point_left])
                if cover != 0:
                    query.append(start_point_left)
                    coverage[start_point_left] = cover
                    start_point_left -= 1
                else:
                    indicator_left = False

            if indicator_right:
                cover = cal_coverage(left, right, self.mapping[start_point_right])
                if cover != 0:
                    query.append(start_point_right)
                    coverage[start_point_right] = cover
                    start_point_right += 1
                else:
                    indicator_right = False
        return query, coverage

    def one_iter_of_infer(self, query, n_distinct):
        """Performance a BP in random order.
           This adapts the BP implemented in pgympy package itself.
        """
        copy_query = copy.deepcopy(query)
        sampling_order = copy.deepcopy(self.node_names)
        np.random.shuffle(sampling_order)

        p_estimate = 1
        for attr in sampling_order:
            if attr in query:
                val = copy_query.pop(attr)
                probs = self.infer_machine.query([attr], evidence=copy_query).values
                if any(np.isnan(probs)):
                    p_estimate = 0
                    break
                p = probs[val] / (np.sum(probs)) * n_distinct[attr]
                p_estimate *= p

        return p_estimate

    def query_sampling(self, query, sample_size=10000):
        gen = self.infer_machine.forward_sample(size=sample_size)
        query_str = ""
        n = 0
        for attr in query:
            query_str += attr
            if type(query[attr]) != list:
                query_str += (attr + " == " + str(query[attr]))
            elif len(query[attr]) == 1:
                query_str += (attr + " == " + str(query[attr][0]))
            else:
                query_str += (attr + " in " + str(query[attr]))
            if n != len(query)-1:
                query_str += " and "
        card = len(gen.query(query_str))
        return card/sample_size

    def query(self, query, num_samples=1, coverage=None, return_prob=False, sample_size=10000):
        """Probability inference using Loopy belief propagation. For example estimate P(X=x, Y=y, Z=z)
           ::Param:: query: dictionary of the form {X:x, Y:y, Z:z}
                     x,y,z can only be a single value
                     num_samples: how many times to run inference, only useful for approaximate algo
                     an approaximation, we might to run it for multiple times and take the average.
                     coverage: the same as ndistinct for continous data
                     return_prob: if true, return P(X=x, Y=y, Z=z)
                                  else return P(X=x, Y=y, Z=z)*nrows
        """
        assert self.infer_algo is not None, "must call .init_inference_method() first"
        if self.infer_algo == "sampling":
            p_estimate = self.query_sampling(query, sample_size)
            if return_prob:
                return (p_estimate, self.nrows)
            return p_estimate * self.nrows

        nrows = self.nrows
        n_distinct = dict()
        for attr in query:
            if self.attr_type[attr] == 'continuous':
                n_distinct[attr] = coverage[attr]
            else:
                encode_value = self.apply_encoding_to_value(query[attr], attr)
                if encode_value is None or (encode_value == []):
                    if return_prob:
                        return (0, nrows)
                    return 0
                n_distinct[attr] = self.apply_ndistinct_to_value(encode_value, query[attr], attr)
                query[attr] = encode_value
            query[attr] = []

        if self.infer_algo == "exact" or num_samples == 1:
            # Using topological order to infer probability
            sampling_order = []
            while len(sampling_order) < len(self.structure):
                for i, deps in enumerate(self.structure):
                    if i in sampling_order:
                        continue  # already ordered
                    if all(d in sampling_order for d in deps):
                        sampling_order.append(i)
            sampling_order = [self.node_names[i] for i in sampling_order]

            p_estimate = 1
            for attr in sampling_order:
                if attr in query:
                    val = query.pop(attr)
                    probs = self.infer_machine.query([attr], evidence=query).values
                    if any(np.isnan(probs)):
                        p_estimate = 0
                        break
                    p = np.sum(probs[val] / (np.sum(probs)) * n_distinct[attr])
                    p_estimate *= p

        else:
            p_estimates = []
            for i in range(num_samples):
                p_estimates.append(self.one_iter_of_infer(query, n_distinct))
            p_estimate = sum(p_estimates) / num_samples

        print(p_estimate)
        if return_prob:
            return (p_estimate, nrows)
        return round(p_estimate * nrows)