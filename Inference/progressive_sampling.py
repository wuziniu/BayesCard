#This fill is adapted from https://github.com/naru-project/naru/blob/master/estimators.py
import collections
import json

import time

import numpy as np
import pandas as pd

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.
        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))

def FillInUnqueriedColumns(table, columns, operators, vals):
    """Allows for some columns to be unqueried (i.e., wildcard).
    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.
    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs

class BayesianNetwork(CardEst):
    """Progressive sampling with a pomegranate bayes net."""

    def build_discrete_mapping(self, table, discretize, discretize_method):
        assert discretize_method in ["equal_size",
                                     "equal_freq"], discretize_method
        self.max_val = collections.defaultdict(lambda: None)
        if not discretize:
            return {}
        table = table.copy()
        mapping = {}
        for col_id in range(len(table[0])):
            col = table[:, col_id]
            if max(col) > discretize:
                if discretize_method == "equal_size":
                    denom = (max(col) + 1) / discretize
                    fn = lambda v: np.floor(v / denom)
                elif discretize_method == "equal_freq":
                    per_bin = len(col) // discretize
                    counts = collections.defaultdict(int)
                    for x in col:
                        counts[int(x)] += 1
                    assignments = {}
                    i = 0
                    bin_size = 0
                    for k, count in sorted(counts.items()):
                        if bin_size > 0 and bin_size + count >= per_bin:
                            bin_size = 0
                            i += 1
                        assignments[k] = i
                        self.max_val[col_id] = i
                        bin_size += count
                    assignments = np.array(
                        [assignments[i] for i in range(int(max(col) + 1))])

                    def capture(assignments):

                        def fn(v):
                            return assignments[v.astype(np.int32)]

                        return fn

                    fn = capture(assignments)
                else:
                    assert False

                mapping[col_id] = fn
        return mapping

    def apply_discrete_mapping(self, table, discrete_mapping):
        table = table.copy()
        for col_id in range(len(table[0])):
            if col_id in discrete_mapping:
                fn = discrete_mapping[col_id]
                table[:, col_id] = fn(table[:, col_id])
        return table

    def apply_discrete_mapping_to_value(self, value, col_id, discrete_mapping):
        if col_id not in discrete_mapping:
            return value
        return discrete_mapping[col_id](value)

    def __init__(self,
                 dataset,
                 num_samples,
                 algorithm="greedy",
                 max_parents=-1,
                 topological_sampling_order=True,
                 use_pgm=True,
                 discretize=None,
                 discretize_method="equal_size",
                 root=None):
        CardEst.__init__(self)

        from pomegranate import BayesianNetwork
        self.discretize = discretize
        self.discretize_method = discretize_method
        self.dataset = dataset
        self.original_table = self.dataset.tuples.numpy()
        self.algorithm = algorithm
        self.topological_sampling_order = topological_sampling_order
        self.num_samples = num_samples
        self.discrete_mapping = self.build_discrete_mapping(
            self.original_table, discretize, discretize_method)
        self.discrete_table = self.apply_discrete_mapping(
            self.original_table, self.discrete_mapping)
        print('calling BayesianNetwork.from_samples...', end='')
        t = time.time()
        self.model = BayesianNetwork.from_samples(self.discrete_table,
                                                  algorithm=self.algorithm,
                                                  max_parents=max_parents,
                                                  n_jobs=8,
                                                  root=root)
        print('done, took', time.time() - t, 'secs.')

        def size(states):
            n = 0
            for state in states:
                if "distribution" in state:
                    dist = state["distribution"]
                else:
                    dist = state
                if dist["name"] == "DiscreteDistribution":
                    for p in dist["parameters"]:
                        n += len(p)
                elif dist["name"] == "ConditionalProbabilityTable":
                    for t in dist["table"]:
                        n += len(t)
                    if "parents" in dist:
                        for parent in dist["parents"]:
                            n += size(dist["parents"])
                else:
                    assert False, dist["name"]
            return n

        self.size = 4 * size(json.loads(self.model.to_json())["states"])

        # print('json:\n', self.model.to_json())
        self.json_size = len(self.model.to_json())
        self.use_pgm = use_pgm
        #        print(self.model.to_json())

        if topological_sampling_order:
            self.sampling_order = []
            while len(self.sampling_order) < len(self.model.structure):
                for i, deps in enumerate(self.model.structure):
                    if i in self.sampling_order:
                        continue  # already ordered
                    if all(d in self.sampling_order for d in deps):
                        self.sampling_order.append(i)
                print("Building sampling order", self.sampling_order)
        else:
            self.sampling_order = list(range(len(self.model.structure)))
        print("Using sampling order", self.sampling_order, str(self))

        if use_pgm:
            from pgmpy.models import BayesianModel
            data = pd.DataFrame(self.discrete_table.astype(np.int64))
            spec = []
            orphans = []
            for i, parents in enumerate(self.model.structure):
                for p in parents:
                    spec.append((p, i))
                if not parents:
                    orphans.append(i)
            print("Model spec", spec)
            model = BayesianModel(spec)
            for o in orphans:
                model.add_node(o)
            print('calling pgm.BayesianModel.fit...', end='')
            t = time.time()
            model.fit(data)
            print('done, took', time.time() - t, 'secs.')
            self.pgm_model = model

    def __str__(self):
        return "bn-{}-{}-{}-{}-bytes-{}-{}-{}".format(
            self.algorithm,
            self.num_samples,
            "topo" if self.topological_sampling_order else "nat",
            self.size,
            # self.json_size,
            self.discretize,
            self.discretize_method if self.discretize else "na",
            "pgmpy" if self.use_pgm else "pomegranate")

    def Query(self, columns, operators, vals):
        if len(columns) != len(self.dataset.table.columns):
            columns, operators, vals = FillInUnqueriedColumns(
                self.dataset.table, columns, operators, vals)

        self.OnStart()
        ncols = len(columns)
        nrows = self.discrete_table.shape[0]
        assert ncols == self.discrete_table.shape[1], (
            ncols, self.discrete_table.shape)

        def draw_conditional_pgm(evidence, col_id):
            """PGM version of draw_conditional()"""

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = self.apply_discrete_mapping_to_value(
                    self.dataset.table.val_to_bin_funcs[col_id](vals[col_id]),
                    col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if val == 0 and operators[col_id] == "<":
                        val += 1
                    elif val == self.max_val[col_id] and operators[
                            col_id] == ">":
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in enumerate(distribution):
                    if op(k, val):
                        p += v
                return p

            from pgmpy.inference import VariableElimination
            model_inference = VariableElimination(self.pgm_model)
            xi_distribution = []
            for row in evidence:
                e = {}
                for i, v in enumerate(row):
                    if v is not None:
                        e[i] = v
                result = model_inference.query(variables=[col_id], evidence=e)
                xi_distribution.append(result[col_id].values)

            xi_marginal = [prob_match(d) for d in xi_distribution]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in enumerate(d):
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        def draw_conditional(evidence, col_id):
            """Draws a new value x_i for the column, and returns P(x_i|prev).
            Arguments:
                evidence: shape [BATCH, ncols] with None for unknown cols
                col_id: index of the current column, i
            Returns:
                xi_marginal: P(x_i|x0...x_{i-1}), computed by marginalizing
                    across the range constraint
                match_rows: the subset of rows from filtered_rows that also
                    satisfy the predicate at column i.
            """

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = self.apply_discrete_mapping_to_value(
                    self.dataset.table.val_to_bin_funcs[col_id](vals[col_id]),
                    col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if val == 0 and operators[col_id] == "<":
                        val += 1
                    elif val == self.max_val[col_id] and operators[
                            col_id] == ">":
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in distribution.items():
                    if op(k, val):
                        p += v
                return p

            xi_distribution = self.model.predict_proba(evidence,
                                                       max_iterations=1,
                                                       n_jobs=-1)
            xi_marginal = [
                prob_match(d[col_id].parameters[0]) for d in xi_distribution
            ]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in d[col_id].parameters[0].items():
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        p_estimates = [1. for _ in range(self.num_samples)]
        evidence = [[None] * ncols for _ in range(self.num_samples)]
        for col_id in self.sampling_order:
            if self.use_pgm:
                xi_marginal, xi_samples = draw_conditional_pgm(evidence, col_id)
            else:
                xi_marginal, xi_samples = draw_conditional(evidence, col_id)
            for ev_list, xi in zip(evidence, xi_samples):
                ev_list[col_id] = xi
            for i in range(self.num_samples):
                p_estimates[i] *= xi_marginal[i]

        self.OnEnd()
        return int(np.mean(p_estimates) * nrows)
