import numpy as np
import copy
import random
import itertools
import logging

from Structure.BN_single_model import BN_Single

logger = logging.getLogger(__name__)


class Pome_BN(BN_Single):
    """
    Build a single Bayesian Network for a single table using pomegranate
    """

    def __init__(self, table_name, method='Pome'):
        BN_Single.__init__(self, table_name, method)

    def build_from_data(self, dataset, attr_type=None, n_mcv=30, n_bins=60, ignore_cols=['id'],
                        algorithm="chow-liu", drop_na=True, max_parents=-1, root=0, n_jobs=8):
        """ Build the Pomegranate model from data, including structure learning and paramter learning
            ::Param:: dataset: pandas.dataframe
                      attr_type: type of attributes (binary, discrete or continuous)
                      n_mcv: for categorical data we keep the top n most common values and bin the rest
                      n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
            for other parameters, pomegranate gives a detailed explaination:
            https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
        """
        discrete_table = self.build_discrete_table(dataset, n_mcv, n_bins, drop_na, ignore_cols)
        self.model = self.learn_model_structure(discrete_table, attr_type, n_mcv, n_bins, ignore_cols, algorithm,
                                   drop_na, max_parents, root, n_jobs, return_model=True)

    def loopy_belief_propagation(self, evidence, n_distinct):
        """Performance a LBP in random order.
           This adapts the LBP implemented in pomegranate package itself.
        """
        index = list(range(len(self.node_names)))
        p_estimate = 1

        while len(index) != 0:
            i = random.choice(index)
            val = evidence[i]
            if val is not None:
                evidence[i] = None
                dist = self.model.predict_proba(evidence)
                p = dist[i].parameters[0][val] * n_distinct[i]
                p_estimate *= p
            index.remove(i)
        return p_estimate

    def infer_point_query_LBP(self, query, num_samples=1, return_prob=False, coverage=None):
        """Probability inference using Loopy belief propagation. For example estimate P(X=x, Y=y, Z=z)
           ::Param:: query: dictionary of the form {X:x, Y:y, Z:z}
                     x,y,z can only be a single value
                     num_samples: how many times to run inference. Since Loopy belief propagation is sometime
                     an approaximation, we might to run it for multiple times and take the average.
                     return_prob: if true, return P(X=x, Y=y, Z=z)
                                  else return P(X=x, Y=y, Z=z)*nrows
        """
        nrows = self.nrows

        evidence = [None] * len(self.node_names)
        n_distinct = [1] * len(self.node_names)
        for attr in query:
            ind = self.node_names.index(attr)
            if self.attr_type[attr] == 'continuous':
                evidence[ind] = query[attr]
                n_distinct[ind] = coverage[attr]
            else:
                evidence[ind] = self.apply_encoding_to_value(query[attr], attr)
                n_distinct[ind] = self.apply_ndistinct_to_value(evidence[ind], query[attr], attr)

        if num_samples == 1:
            # Using topological order to infer probability
            sampling_order = []
            while len(sampling_order) < len(self.model.structure):
                for i, deps in enumerate(self.model.structure):
                    if i in sampling_order:
                        continue  # already ordered
                    if all(d in sampling_order for d in deps):
                        sampling_order.append(i)

            p_estimate = 1
            for i in sampling_order:
                val = evidence[i]
                if val is not None:
                    evidence[i] = None
                    dist = self.model.predict_proba(evidence)
                    p = dist[i].parameters[0][val] * n_distinct[i]
                    p_estimate *= p

        else:
            p_estimates = []
            for i in range(num_samples):
                copy_evidence = copy.deepcopy(evidence)
                p_estimates.append(self.loopy_belief_propagation(copy_evidence, n_distinct))
            p_estimate = sum(p_estimates) / num_samples

        if return_prob:
            return (p_estimate, nrows)
        return int(p_estimate * nrows)

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
            #binary_search to find a good starting point
            if i==j:
                return i
            m = int((j-i)/2)
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
        start_point_right = start_point+1
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

    def infer_range_query_LBP(self, query, num_samples=1, return_prob=False):
        """Probability inference using Loopy belief propagation. For example estimate P(X=x, Y=y, Z=z)
           ::Param:: query: dictionary of the form {X:[x], Y:[y], Z:[z]}
                     x,y,z can only be set of single value
                     num_samples: how many times to run inference. Since Loopy belief propagation is sometime
                     an approaximation, we might to run it for multiple times and take the average.
                     return_prob: if true, return P(X=x, Y=y, Z=z)
                                  else return P(X=x, Y=y, Z=z)*nrows
           LBP for estimating range query can be really slow
        """

        def cartesian_product(d):
            target_list = []
            res = dict()
            for key in d:
                val = d[key]
                if type(val) != list:
                    if self.attr_type[key] == "continuous":
                        val, res[key] = self.continuous_range_map(key, val)
                    else:
                        val = [val]
                target_list.append(val)
            return itertools.product(*target_list), res

        p_estimate = 0
        all_queries, coverage = cartesian_product(query)
        for query_tuple in all_queries:
            point_query = dict()
            i = 0
            cover = dict()
            for attr in query:
                point_query[attr] = query_tuple[i]
                if self.attr_type[attr] == 'continuous':
                    cover[attr] = coverage[attr][query_tuple[i]]
                i += 1
            p_estimate += self.infer_point_query_LBP(point_query,  num_samples=num_samples,
                                                     return_prob=True, coverage=cover)[0]

        if return_prob:
            return p_estimate, self.nrows
        return p_estimate * self.nrows
