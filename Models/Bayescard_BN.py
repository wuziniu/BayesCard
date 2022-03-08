from Pgmpy.models import BayesianModel
import numpy as np
import logging
import time
from collections import deque
import copy
from Models.BN_single_model import BN_Single
from DeepDBUtils.rspn.algorithms.ranges import NominalRange, NumericRange
from DataPrepare.StatisticalTypes import MetaType
from Pgmpy.factors.discrete.CPD import TabularCPD
logger = logging.getLogger(__name__)


def build_meta_info(column_names, null_values):
    meta_info = dict()
    fanout_attr = []
    fanout_attr_inverse = []
    fanout_attr_positive = []
    meta_info['null_values'] = dict()
    for i, col in enumerate(column_names):
        if col is not None and 'mul_' in col:
            fanout_attr.append(col)
            if '_nn' in col:
                fanout_attr_inverse.append(col)
            else:
                fanout_attr_positive.append(col)
        if null_values:
            meta_info['null_values'][col] = null_values[i]
        else:
            meta_info['null_values'][col] = None
    meta_info['fanout_attr'] = fanout_attr
    meta_info['fanout_attr_inverse'] = fanout_attr_inverse
    meta_info['fanout_attr_positive'] = fanout_attr_positive
    meta_info['n_distinct_mapping'] = dict()
    #For continuous variables, pandas uses a default equal-width interval, so some values are undercounted/overcounted
    #Thus the following two lines are hard coded for JOB to account for this effect.
    #This should be very easy to make it automatic in the future.
    meta_info['n_distinct_mapping']['movie_keyword.keyword_id']={117: 8, 8200: 10, 398: 5, 7084: 20}
    meta_info['n_distinct_mapping']['movie_companies.company_id']={22956: 30}
    return meta_info


def multi_dim_index(a, index, new_value):
    assert a.ndim == len(index) == new_value.ndim
    new_index = []
    n = len(index)
    for i, ind in enumerate(index):
        ind = np.asarray(ind)
        if i != n-1:
            new_shape = tuple([-1] + [1]*(n-i-1))
        else:
            new_shape = -1
        new_index.append(ind.reshape(new_shape))
    a[tuple(new_index)] = new_value
    return a


def _literal_list(condition):
    _, literals = condition.split('(', 1)
    return [value.strip(' "\'') for value in literals[:-1].split(',')]


def _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True):
    matching_none_intervals = [idx for idx, single_range in enumerate(ranges[:, attribute_index]) if
                               single_range is None]
    if lower_than:
        for idx, single_range in enumerate(ranges):
            if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][1] <= literal:
                continue
            ranges[idx, attribute_index].ranges[0][1] = literal
            ranges[idx, attribute_index].inclusive_intervals[0][1] = inclusive

        ranges[matching_none_intervals, attribute_index] = NumericRange([[-np.inf, literal]],
                                                                        inclusive_intervals=[[False, inclusive]])

    else:
        for idx, single_range in enumerate(ranges):
            if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][0] >= literal:
                continue
            ranges[idx, attribute_index].ranges[0][0] = literal
            ranges[idx, attribute_index].inclusive_intervals[0][0] = inclusive
        ranges[matching_none_intervals, attribute_index] = NumericRange([[literal, np.inf]],
                                                                        inclusive_intervals=[[inclusive, False]])
    return ranges


class Bayescard_BN(BN_Single):
    """
    Build a single Bayesian Network for a single table using pgmpy
    """

    def __init__(self, schema_graph, relationship_list=[], table_set=set(), column_names=None,
                 full_join_size=None, table_meta_data=None, meta_types=None, null_values=None,
                 meta_info=None, method='Pome', debug=True, infer_algo=None):
        """
        schema_graph: contain the information of the schema
        relationship_list: which relations are this BN built on
        table_set: which set of tables are this BN built on
        column_names: the name of the columns
        table_meta_data: the information about the tables
        meta_types: the information about attribute types
        full_join_size: full outer join size of the data this BN is built on
        infer_algo: inference method, choose between 'exact', 'BP'
        """
        BN_Single.__init__(self, table_set, meta_info, method, debug)
        self.schema_graph = schema_graph
        self.relationship_set = set()
        self.table_set = table_set

        self.relationship_set = set()
        if relationship_list is None:
            relationship_list = []
        for relationship in relationship_list:
            assert (self.schema_graph.relationship_dictionary.get(relationship) is not None)
            self.relationship_set.add(relationship)

        for relationship in relationship_list:
            relationship_obj = self.schema_graph.relationship_dictionary.get(relationship)
            self.table_set.add(relationship_obj.start)
            self.table_set.add(relationship_obj.end)
        self.table_meta_data = table_meta_data
        self.meta_types = meta_types
        self.null_values = null_values
        self.column_names = column_names
        self.full_join_size = full_join_size

        self.nrows = full_join_size
        self.infer_algo = infer_algo
        self.infer_machine = None
        self.cpds = None

    def _parse_conditions(self, conditions, group_by_columns=None, group_by_tuples=None):
        """
        Translates string conditions to NumericRange and NominalRanges the SPN understands.
        """
        assert self.column_names is not None, "For probability evaluation column names have to be provided."
        group_by_columns_merged = None
        if group_by_columns is None or group_by_columns == []:
            ranges = np.array([None] * len(self.column_names)).reshape(1, len(self.column_names))
        else:
            ranges = np.array([[None] * len(self.column_names)] * len(group_by_tuples))
            group_by_columns_merged = [table + '.' + attribute for table, attribute in group_by_columns]

        for (table, condition) in conditions:

            table_obj = self.schema_graph.table_dictionary[table]

            # is an nn attribute condition
            if table_obj.table_nn_attribute in condition:
                full_nn_attribute_name = table + '.' + table_obj.table_nn_attribute
                # unnecessary because column is never NULL
                if full_nn_attribute_name not in self.column_names:
                    continue
                # column can become NULL
                elif condition == table_obj.table_nn_attribute + ' IS NOT NULL':
                    attribute_index = self.column_names.index(full_nn_attribute_name)
                    ranges[:, attribute_index] = NominalRange([1])
                    continue
                elif condition == table_obj.table_nn_attribute + ' IS NULL':
                    attribute_index = self.column_names.index(full_nn_attribute_name)
                    ranges[:, attribute_index] = NominalRange([0])
                    continue
                else:
                    raise NotImplementedError

            # for other attributes parse. Find matching attr.
            matching_fd_cols = [column for column in list(self.table_meta_data[table]['fd_dict'].keys())
                                if column + '<' in table + '.' + condition or column + '=' in table + '.' + condition
                                or column + '>' in table + '.' + condition or column + ' ' in table + '.' + condition]
            matching_cols = [column for column in self.column_names if column + '<' in table + '.' + condition or
                             column + '=' in table + '.' + condition or column + '>' in table + '.' + condition
                             or column + ' ' in table + '.' + condition]
            assert len(matching_cols) == 1 or len(matching_fd_cols) == 1, "Found multiple or no matching columns"
            if len(matching_cols) == 1:
                matching_column = matching_cols[0]

            elif len(matching_fd_cols) == 1:
                matching_fd_column = matching_fd_cols[0]

                def find_recursive_values(column, dest_values):
                    source_attribute, dictionary = list(self.table_meta_data[table]['fd_dict'][column].items())[0]
                    if len(self.table_meta_data[table]['fd_dict'][column].keys()) > 1:
                        logger.warning(f"Current functional dependency handling is not designed for attributes with "
                                       f"more than one ancestor such as {column}. This can lead to error in further "
                                       f"processing.")
                    source_values = []
                    for dest_value in dest_values:
                        if not isinstance(list(dictionary.keys())[0], str):
                            dest_value = float(dest_value)
                        source_values += dictionary[dest_value]

                    if source_attribute in self.column_names:
                        return source_attribute, source_values
                    return find_recursive_values(source_attribute, source_values)

                if '=' in condition:
                    _, literal = condition.split('=', 1)
                    literal_list = [literal.strip(' "\'')]
                elif 'NOT IN' in condition:
                    literal_list = _literal_list(condition)
                elif 'IN' in condition:
                    literal_list = _literal_list(condition)

                matching_column, values = find_recursive_values(matching_fd_column, literal_list)
                attribute_index = self.column_names.index(matching_column)

                if self.meta_types[attribute_index] == MetaType.DISCRETE:
                    condition = matching_column + 'IN ('
                    for i, value in enumerate(values):
                        condition += '"' + value + '"'
                        if i < len(values) - 1:
                            condition += ','
                    condition += ')'
                else:
                    min_value = min(values)
                    max_value = max(values)
                    if values == list(range(min_value, max_value + 1)):
                        ranges = _adapt_ranges(attribute_index, max_value, ranges, inclusive=True, lower_than=True)
                        ranges = _adapt_ranges(attribute_index, min_value, ranges, inclusive=True, lower_than=False)
                        continue
                    else:
                        raise NotImplementedError

            attribute_index = self.column_names.index(matching_column)

            if self.meta_types[attribute_index] == MetaType.DISCRETE:

                val_dict = self.table_meta_data[table]['categorical_columns_dict'][matching_column]

                if '=' in condition:
                    column, literal = condition.split('=', 1)
                    literal = literal.strip(' "\'')

                    if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
                        ranges[:, attribute_index] = NominalRange([val_dict[literal]])
                    else:
                        matching_group_by_idx = group_by_columns_merged.index(matching_column)
                        # due to functional dependencies this check does not make sense any more
                        # assert val_dict[literal] == group_by_tuples[0][matching_group_by_idx]
                        for idx in range(len(ranges)):
                            literal = group_by_tuples[idx][matching_group_by_idx]
                            ranges[idx, attribute_index] = NominalRange([literal])

                elif 'NOT IN' in condition:
                    literal_list = _literal_list(condition)
                    single_range = NominalRange(
                        [val_dict[literal] for literal in val_dict.keys() if not literal in literal_list])
                    if self.null_values[attribute_index] in single_range.possible_values:
                        single_range.possible_values.remove(self.null_values[attribute_index])
                    if all([single_range is None for single_range in ranges[:, attribute_index]]):
                        ranges[:, attribute_index] = single_range
                    else:
                        for i, nominal_range in enumerate(ranges[:, attribute_index]):
                            ranges[i, attribute_index] = NominalRange(
                                list(set(nominal_range.possible_values).intersection(single_range.possible_values)))

                elif 'IN' in condition:
                    literal_list = _literal_list(condition)
                    single_range = NominalRange([val_dict[literal] for literal in literal_list])
                    if all([single_range is None for single_range in ranges[:, attribute_index]]):
                        ranges[:, attribute_index] = single_range
                    else:
                        for i, nominal_range in enumerate(ranges[:, attribute_index]):
                            ranges[i, attribute_index] = NominalRange(list(
                                set(nominal_range.possible_values).intersection(single_range.possible_values)))

            elif self.meta_types[attribute_index] == MetaType.REAL:
                if '<=' in condition:
                    _, literal = condition.split('<=', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True)

                elif '>=' in condition:
                    _, literal = condition.split('>=', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=False)
                elif '=' in condition:
                    _, literal = condition.split('=', 1)
                    literal = float(literal.strip())

                    def non_conflicting(single_numeric_range):
                        assert single_numeric_range[attribute_index] is None or \
                               (single_numeric_range[attribute_index][0][0] > literal or
                                single_numeric_range[attribute_index][0][1] < literal), "Value range does not " \
                                                                                        "contain any values"

                    map(non_conflicting, ranges)
                    if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
                        ranges[:, attribute_index] = NumericRange([[literal, literal]])
                    else:
                        matching_group_by_idx = group_by_columns_merged.index(matching_column)
                        assert literal == group_by_tuples[0][matching_group_by_idx]
                        for idx in range(len(ranges)):
                            literal = group_by_tuples[idx][matching_group_by_idx]
                            ranges[idx, attribute_index] = NumericRange([[literal, literal]])

                elif '<' in condition:
                    _, literal = condition.split('<', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=True)
                elif '>' in condition:
                    _, literal = condition.split('>', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=False)
                else:
                    raise ValueError("Unknown operator")

                def is_invalid_interval(single_numeric_range):
                    assert single_numeric_range[attribute_index].ranges[0][1] >= \
                           single_numeric_range[attribute_index].ranges[0][0], \
                        "Value range does not contain any values"

                map(is_invalid_interval, ranges)

            else:
                raise ValueError("Unknown Metatype")

        if group_by_columns_merged is not None:
            for matching_group_by_idx, column in enumerate(group_by_columns_merged):
                if column not in self.column_names:
                    continue
                attribute_index = self.column_names.index(column)
                if self.meta_types[attribute_index] == MetaType.DISCRETE:
                    for idx in range(len(ranges)):
                        literal = group_by_tuples[idx][matching_group_by_idx]
                        if not isinstance(literal, list):
                            literal = [literal]

                        if ranges[idx, attribute_index] is None:
                            ranges[idx, attribute_index] = NominalRange(literal)
                        else:
                            updated_possible_values = set(ranges[idx, attribute_index].possible_values).intersection(
                                literal)
                            ranges[idx, attribute_index] = NominalRange(list(updated_possible_values))

                elif self.meta_types[attribute_index] == MetaType.REAL:
                    for idx in range(len(ranges)):
                        literal = group_by_tuples[idx][matching_group_by_idx]
                        assert not isinstance(literal, list)
                        ranges[idx, attribute_index] = NumericRange([[literal, literal]])
                else:
                    raise ValueError("Unknown Metatype")

        return ranges

    def compute_mergeable_relationships(self, query, start_table):
        """
        Compute which relationships are merged starting from a certain table (Application B)
        """

        relationships = []
        queue = deque()
        queue.append(start_table)

        while queue:
            # BFS
            table = queue.popleft()

            # list neighbours
            table_obj = self.schema_graph.table_dictionary[table]

            for relationship in table_obj.incoming_relationships:
                # only mergeable if part of SPN and still to be merged in query
                if relationship.identifier in self.relationship_set and \
                        relationship.identifier in query.relationship_set and \
                        relationship.identifier not in relationships:
                    relationships.append(relationship.identifier)
                    queue.append(relationship.start)

            for relationship in table_obj.outgoing_relationships:
                # only mergeable if part of SPN and still to be merged in query
                if relationship.identifier in self.relationship_set and \
                        relationship.identifier in query.relationship_set and \
                        relationship.identifier not in relationships:
                    relationships.append(relationship.identifier)
                    queue.append(relationship.end)

        return relationships

    def relevant_conditions(self, query, merged_tables=None):
        """Compute conditions for E(1/multiplier * 1_{c_1 Λ… Λc_n}) (Application A)."""

        condition_dict = query.table_where_condition_dict
        conditions = []

        if merged_tables is None:
            merged_tables = query.table_set.intersection(self.table_set)

        # Conditions from Query
        for table in condition_dict.keys():
            if table in merged_tables:
                for condition in condition_dict[table]:
                    conditions.append((table, condition))

        # We have to require the tables of the query to be not null
        # This can happen since we learn the SPN on full outer join
        for table in merged_tables:
            table_obj = self.schema_graph.table_dictionary[table]
            condition = table_obj.table_nn_attribute + ' IS NOT NULL'
            conditions.append((table, condition))

        return conditions

    def compute_multipliers(self, query):
        """Compute normalizing multipliers for E(1/multiplier * 1_{c_1 Λ… Λc_n}) (Application A).

        Idea: Do a BFS tree search. Only a relevant multiplier if relationship is from
        higher degree to lower degree. So we store each table in dict.
        """

        queue = deque()
        depth_dict = dict()
        # Usually for BFS we would need a set of visited nodes. This is not required here.
        # We can simply use depth_dict

        for table in query.table_set:
            queue.append(table)
            depth_dict[table] = 0

        depth_dict = self.compute_depths(queue, depth_dict)
        norm_multipliers = []

        # evaluate for every relationship if normalization is necessary
        for relationship in self.relationship_set:
            if relationship not in query.relationship_set:
                relationship_obj = self.schema_graph.relationship_dictionary[relationship]
                # only queries directed to query have to be included
                if depth_dict[relationship_obj.start] > depth_dict[relationship_obj.end]:
                    norm_multipliers.append((relationship_obj.end, relationship_obj.multiplier_attribute_name_nn))

        return norm_multipliers

    def compute_depths(self, queue, depth_dict):
        """
        Do a BFS to compute min-distance of tables to set of tables already in queue.
        """

        # while not empty
        while queue:
            # BFS
            table = queue.popleft()

            # list neighbours
            table_obj = self.schema_graph.table_dictionary[table]

            for relationship in table_obj.incoming_relationships:
                # only consider if part of relationships of combine-SPN
                if relationship.identifier in self.relationship_set:

                    # potentially new table
                    potential_new_table = relationship.start
                    if potential_new_table not in depth_dict.keys():
                        queue.append(potential_new_table)
                        depth_dict[potential_new_table] = depth_dict[table] + 1

            for relationship in table_obj.outgoing_relationships:
                # only consider if part of relationships of combine-SPN
                if relationship.identifier in self.relationship_set:

                    # potentially new table
                    potential_new_table = relationship.end
                    if potential_new_table not in depth_dict.keys():
                        queue.append(potential_new_table)
                        depth_dict[potential_new_table] = depth_dict[table] + 1

        return depth_dict

    def realign(self, encode_value, n_distinct):
        """
        Discard the invalid and duplicated values in encode_value and n_distinct and realign the two
        """
        if type(encode_value) != list and type(n_distinct) != list:
            return encode_value, n_distinct

        assert len(encode_value) == len(n_distinct)
        res_value = []
        res_n_distinct = []
        for i, c in enumerate(encode_value):
            if c is not None:
                if c in res_value:
                    index = res_value.index(c)
                    res_n_distinct[index] += n_distinct[i]
                    res_n_distinct[index] = min(res_n_distinct[index], 1)
                else:
                    res_value.append(c)
                    res_n_distinct.append(n_distinct[i])
        return res_value, res_n_distinct

    def build_from_data(self, dataset, attr_type=None, sample_size=1000000, n_mcv=30, n_bins=60, ignore_cols=['id'],
                        algorithm="chow-liu", drop_na=True, max_parents=-1, root=0, n_jobs=8, discretized=False):
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
            discrete_table = self.learn_model_structure(dataset, self.nrows, attr_type, sample_size,
                                                        n_mcv, n_bins, ignore_cols, algorithm,
                                                        drop_na, max_parents, root, n_jobs,
                                                        return_dataset=True, discretized=discretized)
        else:
            discrete_table = self.learn_model_structure(dataset, self.nrows, attr_type, sample_size,
                                                        n_mcv, n_bins, ignore_cols, 'chow-liu',
                                                        drop_na, max_parents, root, n_jobs,
                                                        return_dataset=True, discretized=discretized)
        self.data_length = len(discrete_table)
        if self.nrows is None:
            self.nrows = len(discrete_table)
            self.full_join_size = len(discrete_table)
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
        self.model.fit(discrete_table)
        if algorithm == "junction":
            try:
                self.model = self.model.to_junction_tree()
            except:
                self.model = self.model
                logger.warning(
                    "This BN is not able to transform into junction tree, probably because "
                    "it's not connected, just use BN")
        logger.info(f"done, took {time.time() - t} secs.")
        print(f"done, parameter learning took {time.time() - t} secs.")
        self.legitimacy_check()
        #self.init_inference_method()

    def update_from_data(self, dataset):
        """
        Preserve the structure and only incrementally update the parameters of BN.
        Currently only implemented data insertion. Data deletion can be done in a similar way.
        """
        t = time.time()
        self.insert_len = len(dataset)
        self.n_in_bin_update = copy.deepcopy(self.n_in_bin)
        self.encoding_update = copy.deepcopy(self.encoding)
        self.mapping_update = copy.deepcopy(self.mapping)

        discrete_table = self.process_update_dataset(dataset)
        print(f"Discretizing table took {time.time() - t} secs.")
        t = time.time()
        incremental_model = copy.deepcopy(self.model)
        incremental_model.fit(discrete_table)

        # incremental parameter updating
        for i, cpd in enumerate(self.model.cpds):
            new_cpd = incremental_model.cpds[i]
            assert set(cpd.state_names.keys()) == set(new_cpd.state_names.keys()), "cpd attribute name mismatch"
            assert cpd.variable == new_cpd.variable, "variable mismatch"
            self.model.cpds[i] = self.update_cpd_table(cpd, new_cpd)

        # changing meta-info
        self.nrows += self.insert_len
        self.full_join_size += self.insert_len
        self.mapping = self.mapping_update
        self.encoding = self.encoding_update
        self.n_in_bin = self.n_in_bin_update
        self.legitimacy_check()

        print(f"done, incremental parameter updating took {time.time() - t} secs.")
        self.init_inference_method()


    def update_cpd_table(self, old_cpd, new_cpd):
        """
        Incrementally update the value of one cpd table
        """
        var = old_cpd.variable
        ret_cpd_variable = var
        ret_cpd_evidence = []
        ret_cpd_evidence_card = []
        ret_cpd_state_names = dict()
        ret_values_shape = []
        for col in old_cpd.state_names:
            if self.attr_type[col] == "continuous":
                ret_cpd_state_names[col] = list(self.mapping_update[col].keys())
            else:
                ret_cpd_state_names[col] = list(set(self.encoding_update[col].values()))
            if col == var:
                ret_cpd_variable_card = len(ret_cpd_state_names[col])
            else:
                ret_cpd_evidence.append(col)
                ret_cpd_evidence_card.append(len(ret_cpd_state_names[col]))
            ret_values_shape.append(len(ret_cpd_state_names[col]))
        ret_values_old = np.zeros(tuple(ret_values_shape))
        old_index = []
        for col in old_cpd.state_names:
            old_index.append([ret_cpd_state_names[col].index(x) for x in old_cpd.state_names[col]])
        ret_values_old = multi_dim_index(ret_values_old, old_index, old_cpd.values)

        ret_values_new = np.zeros(tuple(ret_values_shape))
        new_index = []
        for col in old_cpd.state_names:
            new_index.append([ret_cpd_state_names[col].index(x) for x in new_cpd.state_names[col]])
        ret_values_new = multi_dim_index(ret_values_new, new_index, new_cpd.values)

        ret_values = self.nrows * ret_values_old + self.insert_len * ret_values_new
        ret_values = ret_values.reshape((ret_values.shape[0], -1))

        ret_cpd = TabularCPD(ret_cpd_variable, ret_cpd_variable_card, ret_values, ret_cpd_evidence,
                             ret_cpd_evidence_card, state_names=ret_cpd_state_names)
        ret_cpd.normalize()
        return ret_cpd



    def init_inference_method(self, algorithm=None):
        """
        Initial the inference method for query
        """
        if algorithm:
            self.infer_algo = algorithm
        if self.infer_algo is None:
            if self.algorithm == "chow-liu":
                self.infer_algo = "exact"
            else:
                self.infer_algo = "BP"

        if self.infer_algo == "exact-jit":
            assert self.algorithm == "chow-liu", "Currently JIT only supports CLT"
            from Pgmpy.inference import VariableEliminationJIT
            cpds, topological_order, topological_order_node = self.align_cpds_in_topological()
            self.cpds = cpds
            self.topological_order = topological_order
            self.topological_order_node = topological_order_node
            self.infer_machine = VariableEliminationJIT(self.model, cpds, topological_order, topological_order_node,
                                                       self.fanouts)
        elif self.infer_algo == "exact-jit-torch":
            assert self.algorithm == "chow-liu", "Currently JIT only supports CLT"
            from Pgmpy.inference import VariableEliminationJIT_torch
            cpds, topological_order, topological_order_node = self.align_cpds_in_topological()
            self.cpds = cpds
            self.topological_order = topological_order
            self.topological_order_node = topological_order_node
            self.infer_machine = VariableEliminationJIT_torch(self.model, cpds, topological_order, topological_order_node,
                                                             self.fanouts)
        elif self.infer_algo == "exact":
            from Pgmpy.inference import VariableElimination
            self.infer_machine = VariableElimination(self.model)
        elif self.infer_algo == "BP":
            try:
                # Belief Propogation won't work if the model graph is not connected
                from Pgmpy.inference import BeliefPropagation
                self.infer_machine = BeliefPropagation(self.model)
                self.infer_machine.calibrate()
            except:
                logger.warning("Graph is not connected, we have automatically set the "
                               "inference algorithm to exact. If you would like to use BP,"
                               "please manually connect the graph.")
                from Pgmpy.inference import VariableElimination
                self.infer_algo = "exact"
                self.infer_machine = VariableElimination(self.model)
        elif self.infer_algo == "sampling":
            from Pgmpy.sampling import BayesianModelSampling
            self.infer_machine = BayesianModelSampling(self.model)
            logger.warning("Using sampling as an inference algorithm is very inefficient")
        elif self.infer_algo == "progressive_sampling":
            cpds, topological_order, topological_order_node = self.align_cpds_in_topological()
            self.cpds = cpds
            self.topological_order = topological_order
            self.topological_order_node = topological_order_node
        else:
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
            m = int(i + (j - i) / 2)
            interval = self.mapping[col][m]
            if left >= interval.right:
                return binary_search(m, j)
            elif right <= interval.left:
                return binary_search(i, m)
            else:
                return m

        (left, right) = range
        if left is None: left = -np.Inf
        if right is None: right = np.Inf
        query = []
        coverage = []
        min_val = min(list(self.mapping[col].keys()))
        max_val = max(list(self.mapping[col].keys()))
        if left >= self.mapping[col][max_val].right or right <= self.mapping[col][min_val].left:
            print(left, self.mapping[col][max_val].right, right, self.mapping[col][min_val].left)
            return None, None
        start_point = binary_search(min_val, max_val)
        start_point_left = start_point
        start_point_right = start_point + 1
        indicator_left = True
        indicator_right = True
        while (start_point_left >= min_val and start_point_right < max_val
               and (indicator_left or indicator_right)):
            if indicator_left:
                cover = cal_coverage(left, right, self.mapping[col][start_point_left])
                if cover != 0:
                    query.append(start_point_left)
                    coverage.append(cover)
                    start_point_left -= 1
                else:
                    indicator_left = False
            if indicator_right:
                cover = cal_coverage(left, right, self.mapping[col][start_point_right])
                if cover != 0:
                    query.append(start_point_right)
                    coverage.append(cover)
                    start_point_right += 1
                else:
                    indicator_right = False
        return query, np.asarray(coverage)

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
            if n != len(query) - 1:
                query_str += " and "
        card = len(gen.query(query_str))
        return card / sample_size

    def query_decoding(self, query, coverage=None, epsilon=0.5):
        """
        Convert the query to the encodings BN recognize
        """
        n_distinct = dict()
        for attr in query:
            if self.attr_type[attr] == 'continuous':
                if coverage is None:
                    n_d_temp = None
                    if type(query[attr]) == tuple:
                        l = max(self.domain[attr][0], query[attr][0])
                        r = min(self.domain[attr][1], query[attr][1])
                    else:
                        l = query[attr]-epsilon
                        r = query[attr]+epsilon
                        if attr in self.n_distinct_mapping:
                            if query[attr] in self.n_distinct_mapping[attr]:
                                n_d_temp = self.n_distinct_mapping[attr][query[attr]]
                    if l > r:
                        return None, None
                    query[attr], n_distinct[attr] = self.continuous_range_map(attr, (l, r))
                    if query[attr] is None:
                        return None, None
                    if n_d_temp is not None:
                        n_distinct[attr] *= n_d_temp
                else:
                    n_distinct[attr] = coverage[attr]
            elif type(query[attr]) == tuple:
                query_list = []
                if self.null_values is None or len(self.null_values) == 0 or attr not in self.null_values:
                    for val in self.encoding[attr]:
                        if query[attr][0] <= val <= query[attr][1]:
                            query_list.append(val)
                else:
                    for val in self.encoding[attr]:
                        if val != self.null_values[attr] and query[attr][0] <= val <= query[attr][1]:
                            query_list.append(val)
                encode_value = self.apply_encoding_to_value(query_list, attr)
                if encode_value is None or (encode_value == []):
                    return None, None
                n_distinct[attr] = self.apply_ndistinct_to_value(encode_value, query_list, attr)
                query[attr], n_distinct[attr] = self.realign(encode_value, n_distinct[attr])
            else:
                encode_value = self.apply_encoding_to_value(query[attr], attr)
                if encode_value is None or (encode_value == []):
                    return None, None
                n_distinct[attr] = self.apply_ndistinct_to_value(encode_value, query[attr], attr)
                query[attr], n_distinct[attr] = self.realign(encode_value, n_distinct[attr])
        return query, n_distinct

    def get_fanout_values(self, fanout_attrs):
        if len(fanout_attrs) == 1:
            return self.fanouts[fanout_attrs[0]]
        else:
            fanout_attrs_shape = tuple([len(self.fanouts[i]) for i in fanout_attrs])
            res = None
            for i in fanout_attrs:
                if res is None:
                    res = self.fanouts[i]
                else:
                    res = np.outer(res, self.fanouts[i]).reshape(-1)
            return res.reshape(fanout_attrs_shape)

    def align_cpds_in_topological(self):
        cpds = self.model.cpds
        sampling_order = []
        while len(sampling_order) < len(self.structure):
            for i, deps in enumerate(self.structure):
                if i in sampling_order:
                    continue  # already ordered
                if all(d in sampling_order for d in deps):
                    sampling_order.append(i)
        topological_order = sampling_order
        topological_order_node = [self.node_names[i] for i in sampling_order]
        new_cpds = []
        for n in topological_order_node:
            for cpd in cpds:
                if cpd.variable == n:
                    new_cpds.append(cpd)
                    break
        assert len(cpds) == len(new_cpds)
        return new_cpds, topological_order, topological_order_node


    def get_condition(self, evidence, cpd, topological_order_node, var_evidence, n_distinct=None, hard_sample=False):
        values = cpd.values
        if evidence[0][0] == -1:
            assert len(values.shape) == 1
            if n_distinct:
                probs = values[var_evidence] * n_distinct
            else:
                probs = values[var_evidence]
            return_prob = np.sum(probs)
            probs = probs / return_prob  # re-normalize
            new_evidence = np.random.choice(var_evidence, p=probs, size=evidence.shape[-1])
        else:
            scope = cpd.variable
            condition = cpd.variables[1:]
            condition_ind = [topological_order_node.index(c) for c in condition]
            condition_evidence = evidence[condition_ind]
            # the following is hardcoded for fast computation
            if len(condition) == 1:
                probs = values[:, condition_evidence[0]]
            elif len(condition) == 2:
                probs = values[:, condition_evidence[0], condition_evidence[1]]
            elif len(condition) == 3:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2]]
            elif len(condition) == 4:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3]]
            elif len(condition) == 5:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4]]
            elif len(condition) == 6:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4], condition_evidence[5]]
            elif len(condition) == 7:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4], condition_evidence[5],
                        condition_evidence[6]]
            else:
                # no more efficient tricks
                probs = np.zeros((values.shape[0], evidence.shape[-1]))
                for j in range(values.shape[0]):
                    probs[j, :] = values[j]
            #print(len(var_evidence))
            #print(probs.shape)
            if n_distinct:
                probs = (probs[var_evidence, :].transpose() * n_distinct).transpose()
            else:
                probs = probs[var_evidence, :]
            #print(probs.shape)
            return_prob = np.sum(probs, axis=0)
            #print(return_prob.shape)
            probs = (probs / return_prob)
            probs[np.isnan(probs)] = 0
            if hard_sample:
                probs += 1e-7
                probs = probs/np.sum(probs, axis=0)
                new_evidence = np.asarray([np.random.choice(var_evidence, p=probs[:,i]) for i in range(evidence.shape[-1])])
                #print(probs.shape)
            else:
                generate_probs = probs.mean(axis=1)
                if np.sum(generate_probs) == 0:
                    return 0, None
                generate_probs = generate_probs / np.sum(generate_probs)
                new_evidence = np.random.choice(var_evidence, p=generate_probs, size=evidence.shape[-1])
        return return_prob, new_evidence



    def progressive_sampling(self, query, sample_size, n_distinct=None, hard_sample=False):
        """Using progressive sampling method as described in Naru paper"""
        if self.cpds is None:
            cpds, topological_order, topological_order_node = self.align_cpds_in_topological()
            self.cpds = cpds
            self.topological_order_node = topological_order_node

        evidence = np.zeros((len(self.topological_order_node), sample_size), dtype=int) - 1
        probs = np.ones(sample_size)
        for i, node in enumerate(self.topological_order_node):
            if node in query:
                var_evidence = query[node]
                if n_distinct:
                    n_distinct_value = n_distinct[node]
                else:
                    n_distinct_value = None
            else:
                var_evidence = np.arange(self.cpds[i].values.shape[0])
                n_distinct_value = None
            if type(var_evidence) == int:
                var_evidence = [var_evidence]
            new_probs, new_evidence = self.get_condition(evidence, self.cpds[i],
                                                         self.topological_order_node, var_evidence, 
                                                         n_distinct_value, hard_sample=hard_sample)
            if new_evidence is None:
                return 0
            evidence[i, :] = new_evidence
            probs *= new_probs
        return np.sum(probs) / evidence.shape[-1]

    def progressive_sampling_expectation(self, query, fanout_attrs, sample_size, n_distinct=None, hard_sample=False):
        """Using progressive sampling to do expectation"""
        if self.cpds is None:
            cpds, topological_order, topological_order_node = self.align_cpds_in_topological()
            self.cpds = cpds
            self.topological_order_node = topological_order_node

        evidence = np.zeros((len(self.topological_order_node), sample_size), dtype=int) - 1
        exps = np.ones(sample_size)
        for i, node in enumerate(self.topological_order_node):
            is_fanout = False
            if node in query:
                var_evidence = query[node]
                if n_distinct:
                    n_distinct_value = n_distinct[node]
                else:
                    n_distinct_value = None
            else:
                var_evidence = np.arange(self.cpds[i].values.shape[0])
                n_distinct_value = None
                if node in fanout_attrs:
                    # fanout attr for expectation computing
                    is_fanout = True
            #print(is_fanout)
            if type(var_evidence) == int:
                var_evidence = [var_evidence]
            new_probs, new_evidence = self.get_condition(evidence, self.cpds[i],
                                                         self.topological_order_node, var_evidence, 
                                                         n_distinct_value, hard_sample=hard_sample)
            if new_evidence is None:
                return 0
            evidence[i, :] = new_evidence
            exps *= new_probs
            if is_fanout:
                exps *= self.fanouts[node][new_evidence]
        return np.sum(exps) / evidence.shape[-1]


    def query(self, query, num_samples=1, n_distinct=None, coverage=None, return_prob=False, sample_size=1000, hard_sample=False):
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
        
        if len(query) == 0:
            if return_prob:
                return 1, self.nrows
            else:
                return self.nrows

        nrows = self.nrows
        if n_distinct is None:
            query, n_distinct = self.query_decoding(query, coverage)
        #print(f"decoded query is {query}")
        if query is None:
            if return_prob:
                return 0, nrows
            else:
                return 0

        if self.infer_algo == "progressive_sampling":
            p_estimate = self.progressive_sampling(query, sample_size, n_distinct, hard_sample=hard_sample)
            if return_prob:
                return (p_estimate, self.nrows)
            return p_estimate * self.nrows

        elif self.infer_algo == "exact-jit" or self.infer_algo == "exact-jit-torch":
            p_estimate = self.infer_machine.query(query, n_distinct)
            if return_prob:
                return (p_estimate, self.nrows)
            return p_estimate * self.nrows

        elif self.infer_algo == "exact" or num_samples == 1:
            sampling_order = list(query.keys())
            p_estimate = 1
            for attr in sampling_order:
                if attr in query:
                    val = query.pop(attr)
                    probs = self.infer_machine.query([attr], evidence=query).values
                    if np.any(np.isnan(probs)):
                        p_estimate = 0
                        break
                    p = np.sum(probs[val] * n_distinct[attr])
                    p_estimate *= p

        else:
            p_estimates = []
            for i in range(num_samples):
                p_estimates.append(self.one_iter_of_infer(query, n_distinct))
            p_estimate = sum(p_estimates) / num_samples

        if return_prob:
            return (p_estimate, nrows)
        return round(p_estimate * nrows)


    def expectation(self, query, fanout_attrs, num_samples=1, n_distinct=None, coverage=None,
                    return_prob=False, sample_size=1000, hard_sample=False):
        """
        Calculating the expected value E[P(Q|F)*F]
        Parameters
        ----------
        fanout_attrs: a list of fanout variables F, where we would like to compute the expectation
        Rest parameters: the same as previous function .query().
        """
        if fanout_attrs is None or len(fanout_attrs) == 0:
            return self.query(query, num_samples, n_distinct, coverage, return_prob, sample_size)

        elif self.infer_algo == "progressive_sampling":
            if n_distinct is None:
                query, n_distinct = self.query_decoding(query, coverage)
            exp = self.progressive_sampling_expectation(query, fanout_attrs, sample_size, n_distinct, hard_sample)
            if return_prob:
                return exp, self.nrows
            else:
                return exp * self.nrows

        elif self.infer_algo == "exact-jit" or self.infer_algo == "exact-jit-torch":
            if n_distinct is None:
                query, n_distinct = self.query_decoding(query, coverage)
            exp = self.infer_machine.expectation(query, fanout_attrs, n_distinct)
            if return_prob:
                return exp, self.nrows
            else:
                return exp * self.nrows

        else:
            query_prob = copy.deepcopy(query)
            probsQ, _ = self.query(query_prob, num_samples, n_distinct, coverage, True)
            if probsQ == 0:
                if return_prob:
                    return 0, self.nrows
                else:
                    return 0
            if n_distinct is None:
                query, n_distinct = self.query_decoding(query, coverage)
            if query is None:
                if return_prob:
                    return 0, self.nrows
                else:
                    return 0
            
            probsQF = self.infer_machine.query(fanout_attrs, evidence=query).values
            
            if np.any(np.isnan(probsQF)):
                if return_prob:
                    return 0, self.nrows
                else:
                    return 0
            else:
                probsQF = probsQF / (np.sum(probsQF))
                
            fanout_attrs_shape = tuple([len(self.fanouts[i]) for i in fanout_attrs])
            probsQF = probsQF.reshape(fanout_attrs_shape)
            exp = np.sum(probsQF * self.get_fanout_values(fanout_attrs)) * probsQ
            if return_prob:
                return exp, self.nrows
            else:
                return exp * self.nrows


    def legitimacy_check(self):
        """
        Checking whether a BN is legitimate
        """
        # Step 1: checking the attrs
        attr_names = list(self.attr_type.keys())
        for col in attr_names:
            if self.attr_type[col] == "boolean":
                assert self.mapping[col] is None or len(
                    self.mapping[col]) == 0, f"mapping is for continuous values only"
                assert self.n_in_bin[col] is None or len(
                    self.n_in_bin[col]) == 0, f"n_in_bin is for categorical values only"
            elif self.attr_type[col] == "categorical":
                assert self.mapping[col] is None or len(
                    self.mapping[col]) == 0, f"mapping is for continuous values only"
                reverse_encoding = dict()
                for k in self.encoding[col]:
                    enc = self.encoding[col][k]
                    if enc in reverse_encoding:
                        reverse_encoding[enc].append(k)
                    else:
                        reverse_encoding[enc] = [k]
                for enc in self.n_in_bin[col]:
                    assert enc in reverse_encoding, f"{enc} in {col} in n_in_bin is not a valid encoding"
                    n_in_bin_keys = set(list(self.n_in_bin[col][enc].keys()))
                    reverse_keys = set(reverse_encoding[enc])
                    assert n_in_bin_keys == reverse_keys, f"{col} has n_in_bin and encoding mismatch"
            elif self.attr_type[col] == "continuous":
                assert self.encoding[col] is None or len(
                    self.encoding[col] == 0), f"encoding is for categorical values only"
                assert self.n_in_bin[col] is None or len(
                    self.n_in_bin[col] == 0), f"n_in_bin is for categorical values only"
                prev = None
                for enc in self.mapping[col]:
                    interval = self.mapping[col][enc]
                    if prev:
                        assert interval.right > prev, f"{col} has unordered intervals for continuous variable"
                    else:
                        prev = interval.right
            else:
                assert False, f"Unknown column type {self.attr_type[col]}"

        # Step 2: checking the CPDs
        for cpd in self.model.cpds:
            for col in cpd.state_names:
                assert col in self.attr_type, f"column {col} not found"
                if self.attr_type[col] == "continuous":
                    mapping = set(list(self.mapping[col].keys()))
                    assert mapping == set(cpd.state_names[col]), f"{col} does not have correct mapping"
                else:
                    encoding = set(list(self.encoding[col].values()))
                    assert encoding == set(cpd.state_names[col]), f"{col} does not have correct encoding"

        # Step 3: checking fanout values
        for col in self.fanout_attr:
            assert col in self.attr_type, f"fanout column {col} not found"
            assert col in self.fanouts, f"fanout column {col} does not have saved values"
            if self.attr_type[col] == "continuous":
                mapping = set(list(self.mapping[col].keys()))
                assert len(self.fanouts[col]) == len(mapping), f"fanout column {col} has fanout values length mismatch"
            else:
                encoding = set(list(self.encoding[col].values()))
                assert len(self.fanouts[col]) == len(encoding), f"fanout column {col} has fanout values length mismatch"
            if col in self.fanout_attr_inverse:
                assert np.max(self.fanouts[col]) <= 1, f"inverse fanout value in {col} greater than 1"
            else:
                assert col in self.fanout_attr_positive, f"Unknown fanout type for {col}"

