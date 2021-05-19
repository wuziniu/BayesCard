import itertools
import numpy as np
import bz2
import pickle
import logging
import ast

from DeepDBUtils.ensemble_compilation.graph_representation import Query
from DeepDBUtils.ensemble_compilation.probabilistic_query import IndicatorExpectation, Expectation
from DataPrepare.join_data_preparation import JoinDataPreparator
from Models.Bayescard_BN import Bayescard_BN

logger = logging.getLogger(__name__)

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}

def evaluate_factors(dry_run, factors_full, cached_expecation_vals, confidence_intervals=False,
                     confidence_interval_samples=None, gen_code_stats=None):
    # This function is derived from FLAT and DeepDB
    # see if we can exclude inverted factors, quadratic complexity but usually only very few factors
    factors_to_be_deleted = set()
    for left_factor in factors_full:
        if not isinstance(left_factor, IndicatorExpectation):
            continue
        for right_factor in factors_full:
            if not isinstance(right_factor, IndicatorExpectation):
                continue
            if left_factor.is_inverse(right_factor):
                factors_to_be_deleted.add(left_factor)
                factors_to_be_deleted.add(right_factor)
    factors = [factor for factor in factors_full if factor not in factors_to_be_deleted]

    # evaluate factors to obtain cardinality and formula
    formula = None
    cardinality = None
    values = []
    non_constant_factors = []
    if confidence_intervals:
        # idea: compute indicator stds with bernoulli approach
        # store all indicator exps in [:,1]
        no_exp = len([factor for factor in factors if isinstance(factor, Expectation)])
        factor_exps = np.ones((1, no_exp + 2))
        factor_exps[:, 0] = factors[0]
        factor_stds = np.zeros((1, no_exp + 2))

    exps_counter = 0
    for i, factor in enumerate(factors):
        if formula is None:
            assert i == 0
            formula = str(factor)
            cardinality = factor
            values.append(factor)
            continue

        formula += " * " + str(factor)
        if not dry_run:
            if isinstance(factor, IndicatorExpectation):
                if cached_expecation_vals.get(hash(factor)) is not None:
                    exp = cached_expecation_vals[hash(factor)]
                else:
                    _, exp = factor.spn.evaluate_indicator_expectation(factor, gen_code_stats=gen_code_stats,
                                                                       standard_deviations=False)
                    cached_expecation_vals[hash(factor)] = exp

                if confidence_intervals:
                    factor_exps[:, 1] *= exp
                non_constant_factors.append(i)
                values.append(exp)
                cardinality *= exp

            elif isinstance(factor, Expectation):
                if not confidence_intervals and cached_expecation_vals.get(hash(factor)) is not None:
                    std, exp = cached_expecation_vals[hash(factor)]
                else:
                    std, exp = factor.spn.evaluate_expectation(factor, standard_deviations=confidence_intervals,
                                                               gen_code_stats=gen_code_stats)
                    if confidence_intervals:
                        ci_index = exps_counter + 2
                        factor_exps[:, ci_index] = exp
                        factor_stds[:, ci_index] = std
                    cached_expecation_vals[hash(factor)] = std, exp
                non_constant_factors.append(i)
                values.append(exp)
                cardinality *= exp

                exps_counter += 1

            else:
                cardinality *= factor
                values.append(factor)
                if confidence_intervals:
                    factor_exps[:, 0] *= factor

    if confidence_intervals:
        assert confidence_interval_samples is not None, \
            "confidence_interval_samples is required for confidence interval calculation"

        bernoulli_p = factor_exps[:, 1]
        factor_stds[:, 1] = np.sqrt(bernoulli_p * (1 - bernoulli_p) / confidence_interval_samples)
        cardinality_stds = std_of_products(factor_exps, factor_stds)
        return cardinality_stds, values, cardinality, formula
    else:
        return values, cardinality, formula

def std_of_products(exps, stds):
    """
    Computes the std of independent random variables.
    :param exps:
    :param stds:
    :param non_constant_factors:
    :return:
    """
    # product(var(X_i) + E(X_i)^2)-product(E(X_i))^2
    std_shape = (exps.shape[0], 1)

    product_left = np.ones(std_shape)
    product_right = np.ones(std_shape)

    for i in range(exps.shape[1]):
        # var(X_i) + E(X_i)^2
        product_left *= np.reshape(np.square(stds[:, i]) + np.square(exps[:, i]), std_shape)
        # E(X_i)^2
        product_right *= np.reshape(np.square(exps[:, i]), std_shape)

    return np.sqrt(product_left - product_right)


def infer_column(condition):
    column = None
    if '<=' in condition:
        column, _ = condition.split('<=', 1)
    elif '>=' in condition:
        column, _ = condition.split('>=', 1)
    elif '>' in condition:
        column, _ = condition.split('>', 1)
    elif '<' in condition:
        column, _ = condition.split('<', 1)
    elif '=' in condition:
        column, _ = condition.split('=', 1)
    elif 'NOT IN' in condition:
        column, _ = condition.split('NOT IN', 1)
    elif 'IN' in condition:
        column, _ = condition.split('IN', 1)

    assert column is not None, "Condition not recognized"
    return column.strip()

class BN_ensemble():
    """
    Several BNs combined one for each table.
    """

    def __init__(self, schema_graph, bns=None):
        self.schema_graph = schema_graph
        if bns is None:
            self.bns = []
        else:
            self.bns = bns
        self.cached_expectation_vals = dict()
        self.join_size = dict()
        self.join_prepare = None

    def add_BN(self, bn):
        self.bns.append(bn)

    def _cardinality_greedy(self, query, rdc_spn_selection=False, rdc_attribute_dict=None, dry_run=False,
                            merge_indicator_exp=True, exploit_overlapping=False, return_factor_values=False,
                            exploit_incoming_multipliers=True, prefer_disjunct=False, gen_code_stats=None):
        """
        Find first BN for cardinality estimate.
        """
        # Greedily select first BN
        first_bn, next_mergeable_relationships, next_mergeable_tables = self._greedily_select_first_cardinality_bn(
            query, rdc_spn_selection=rdc_spn_selection, rdc_attribute_dict=rdc_attribute_dict)

        return self._cardinality_with_injected_start(query, first_bn, next_mergeable_relationships,
                                                     next_mergeable_tables, rdc_spn_selection=rdc_spn_selection,
                                                     rdc_attribute_dict=rdc_attribute_dict,
                                                     dry_run=dry_run,
                                                     merge_indicator_exp=merge_indicator_exp,
                                                     exploit_overlapping=exploit_overlapping,
                                                     return_factor_values=return_factor_values,
                                                     exploit_incoming_multipliers=exploit_incoming_multipliers,
                                                     prefer_disjunct=prefer_disjunct,
                                                     gen_code_stats=gen_code_stats)

    def _cardinality_with_injected_start(self, query, first_bn, next_mergeable_relationships, next_mergeable_tables,
                                         rdc_spn_selection=False, rdc_attribute_dict=None, dry_run=False,
                                         merge_indicator_exp=True, exploit_overlapping=False,
                                         return_factor_values=False, exploit_incoming_multipliers=True,
                                         prefer_disjunct=False, gen_code_stats=None):
        """
        Always use BN that matches most where conditions.
        """
        factors = []

        # only operate on copy so that query object is not changed
        # for greedy strategy it does not matter whether query is changed
        # optimized version of:
        # original_query = copy.deepcopy(query)
        # query = copy.deepcopy(query)
        original_query = query.copy_cardinality_query()
        query = query.copy_cardinality_query()

        # First BN: Full_join_size*E(outgoing_mult * 1/multiplier * 1_{c_1 Λ… Λc_n})
        # Again create auxilary query because intersection of query relationships and bn relationships
        # is not necessarily a tree.
        auxilary_query = Query(self.schema_graph)
        for relationship in next_mergeable_relationships:
            auxilary_query.add_join_condition(relationship)
        auxilary_query.table_set.update(next_mergeable_tables)
        auxilary_query.table_where_condition_dict = query.table_where_condition_dict

        factors.append(first_bn.full_join_size)
        conditions = first_bn.relevant_conditions(auxilary_query)
        multipliers = first_bn.compute_multipliers(auxilary_query)

        # E(1/multipliers * 1_{c_1 Λ… Λc_n})
        expectation = IndicatorExpectation(multipliers, conditions, spn=first_bn, table_set=auxilary_query.table_set)
        factors.append(expectation)

        # mark tables as merged, remove merged relationships
        merged_tables = next_mergeable_tables
        query.relationship_set -= set(next_mergeable_relationships)

        # remember which BN was used to merge tables
        corresponding_exp_dict = {}
        for table in merged_tables:
            corresponding_exp_dict[table] = expectation
        extra_multplier_dict = {}

        # merge subsequent relationships
        while len(query.relationship_set) > 0:

            # for next joins:
            # if not exploit_overlapping: cardinality next subquery / next_neighbour.table_size

            # compute set of next joinable neighbours
            next_neighbours, neighbours_relationship_dict = self._next_neighbours(query, merged_tables)

            # compute possible next merges and select greedily
            next_bn, next_neighbour, next_mergeable_relationships = self._greedily_select_next_table(original_query,
                                                                                                      query,
                                                                                                      next_neighbours,
                                                                                                      exploit_overlapping,
                                                                                                      merged_tables,
                                                                                                      prefer_disjunct=prefer_disjunct,
                                                                                                      rdc_spn_selection=rdc_spn_selection,
                                                                                                      rdc_attribute_dict=rdc_attribute_dict)

            # if outgoing: outgoing_mult appended to multipliers
            relationship_to_neighbour = neighbours_relationship_dict[next_neighbour]
            relationship_obj = self.schema_graph.relationship_dictionary[relationship_to_neighbour]

            incoming_relationship = True
            if relationship_obj.start == next_neighbour:
                incoming_relationship = False
                # outgoing relationship. Has to be included by E(outgoing_mult | C...)
                if merge_indicator_exp:
                    # For this computation we simply add the multiplier to the respective indicator expectation.
                    end_table = relationship_obj.end
                    indicator_expectation_outgoing_bn = corresponding_exp_dict[end_table]
                    indicator_expectation_outgoing_bn.nominator_multipliers.append(
                        (end_table, relationship_obj.multiplier_attribute_name))
                else:
                    # E(outgoing_mult | C...) weighted by normalizing_multipliers
                    end_table = relationship_obj.end
                    feature = (end_table, relationship_obj.multiplier_attribute_name)

                    # Search BN with maximal considered conditions
                    max_considered_where_conditions = -1
                    bn_for_exp_computation = None

                    for bn in self.bns:
                        # attribute not even available
                        if hasattr(bn, 'column_names'):
                            if end_table + '.' + relationship_obj.multiplier_attribute_name not in bn.column_names:
                                continue
                        conditions = bn.relevant_conditions(original_query)
                        if len(conditions) > max_considered_where_conditions:
                            max_considered_where_conditions = len(conditions)
                            bn_for_exp_computation = bn

                    assert bn_for_exp_computation is not None, "No BN found for expectation computation"

                    # if bn_for_exp_computation is already used for outgoing multiplier computation it should be used
                    # again. This captures correlations of multipliers better.
                    if extra_multplier_dict.get(bn_for_exp_computation) is not None:
                        expectation = extra_multplier_dict.get(bn_for_exp_computation)
                        expectation.features.append(feature)
                    else:
                        normalizing_multipliers = bn_for_exp_computation.compute_multipliers(original_query)
                        conditions = bn_for_exp_computation.relevant_conditions(original_query)

                        expectation = Expectation([feature], normalizing_multipliers, conditions,
                                                  spn=bn_for_exp_computation)
                        extra_multplier_dict[bn_for_exp_computation] = expectation
                        factors.append(expectation)

            # remove relationship_to_neighbour from query
            if relationship_to_neighbour in next_mergeable_relationships:
                next_mergeable_relationships.remove(relationship_to_neighbour)
            query.relationship_set.remove(relationship_to_neighbour)
            merged_tables.add(next_neighbour)

            # tables which are merged in the next step
            next_merged_tables = self._merged_tables(next_mergeable_relationships)
            next_merged_tables.add(next_neighbour)

            # find overlapping relationships (relationships already merged that also appear in next_f)
            overlapping_relationships, overlapping_tables, no_overlapping_conditions = self._compute_overlap(
                next_neighbour, query, original_query,
                next_mergeable_relationships,
                next_merged_tables,
                next_bn)
            # remove neighbour
            overlapping_tables.remove(next_neighbour)

            # do not ignore overlap. Exploit knowledge of overlap.
            # in the computation use:
            # correct_indicator_expectation_with_overlap/ indicator_expectation_of_overlap

            # nominator query: indicator expectation of overlap + mergeable relationships
            nominator_query = Query(self.schema_graph)
            for relationship in overlapping_relationships:
                nominator_query.add_join_condition(relationship)
            for relationship in next_mergeable_relationships:
                nominator_query.add_join_condition(relationship)
            nominator_query.table_set.update(next_merged_tables)
            nominator_query.table_where_condition_dict = query.table_where_condition_dict
            conditions = next_bn.relevant_conditions(nominator_query,
                                                      merged_tables=next_merged_tables.union(overlapping_tables))
            multipliers = next_bn.compute_multipliers(nominator_query)

            nominator_expectation = IndicatorExpectation(multipliers, conditions, spn=next_bn,
                                                         table_set=next_merged_tables.union(overlapping_tables))

            # we can still exploit the outgoing multiplier if the multiplier is present
            if incoming_relationship and exploit_incoming_multipliers and len(overlapping_tables) == 0:
                nominator_expectation.nominator_multipliers \
                    .append((next_neighbour, relationship_obj.multiplier_attribute_name))

            factors.append(nominator_expectation)

            # denominator: indicator expectation of overlap
            denominator_query = Query(self.schema_graph)
            for relationship in overlapping_relationships:
                denominator_query.add_join_condition(relationship)
            denominator_query.table_set.update(next_merged_tables)
            denominator_query.table_where_condition_dict = query.table_where_condition_dict

            # constraints for next neighbor would not have any impact otherwise
            conditions = next_bn.relevant_conditions(denominator_query, merged_tables=overlapping_tables)

            next_neighbour_obj = self.schema_graph.table_dictionary[next_neighbour]
            # add not null condition for next neighbor
            conditions.append((next_neighbour, next_neighbour_obj.table_nn_attribute + " IS NOT NULL"))
            multipliers = next_bn.compute_multipliers(denominator_query)
            denominator_exp = IndicatorExpectation(multipliers, conditions, spn=next_bn, inverse=True,
                                                   table_set=overlapping_tables)

            # we can still exploit the outgoing multiplier if the multiplier is present
            if incoming_relationship and exploit_incoming_multipliers and len(overlapping_tables) == 0:
                denominator_exp.nominator_multipliers \
                    .append((next_neighbour, relationship_obj.multiplier_attribute_name))
            factors.append(denominator_exp)

            # mark tables as merged, remove merged relationships
            for table in next_merged_tables:
                merged_tables.add(table)
                corresponding_exp_dict[table] = nominator_expectation

            query.relationship_set -= set(next_mergeable_relationships)

        values, cardinality, formula = evaluate_factors(dry_run, factors, self.cached_expecation_vals,
                                                        gen_code_stats=gen_code_stats)

        if not return_factor_values:
            return formula, factors, cardinality
        else:
            return formula, factors, cardinality, values


    def _greedily_select_first_cardinality_bn(self, query, rdc_spn_selection=False, rdc_attribute_dict=None):
        """
        Select first F by maximization of applicable where selections.
        """
        first_bn = None
        next_mergeable_relationships = None
        next_mergeable_tables = None
        current_best_candidate_vector = None

        for bn in self.bns:
            # to get mergeable relationships we could use
            # intersection_relationships = query.relationship_set.intersection(bn.relationship_set)
            # However, this does not work if mergeable relationships are not connected
            for start_table in bn.table_set:
                if start_table not in query.table_set:
                    continue

                mergeable_relationships = bn.compute_mergeable_relationships(query, start_table)
                no_mergeable_relationships = len(mergeable_relationships) + 1

                mergeable_tables = self._merged_tables(mergeable_relationships)
                mergeable_tables.add(start_table)

                where_conditions = set(query.table_where_condition_dict.keys()).intersection(mergeable_tables)
                unnecessary_tables = len(bn.table_set.difference(query.table_set))

                current_candidate_vector = (len(where_conditions), no_mergeable_relationships, -unnecessary_tables)

                if rdc_spn_selection:
                    rdc_sum = self.merged_rdc_sum(mergeable_tables, query, rdc_attribute_dict)
                    current_candidate_vector = (rdc_sum,) + current_candidate_vector

                if current_best_candidate_vector is None or current_candidate_vector > current_best_candidate_vector:
                    current_best_candidate_vector = current_candidate_vector
                    first_bn = bn
                    next_mergeable_relationships = mergeable_relationships
                    next_mergeable_tables = mergeable_tables

        return first_bn, next_mergeable_relationships, next_mergeable_tables

    def _merged_tables(self, mergeable_relationships):
        """
        Compute merged tables if different relationships are merged.
        """

        merged_tables = set()

        for relationship in mergeable_relationships:
            relationship_obj = self.schema_graph.relationship_dictionary[relationship]
            merged_tables.add(relationship_obj.start)
            merged_tables.add(relationship_obj.end)

        return merged_tables

    def merged_rdc_sum(self, mergeable_tables, query, rdc_attribute_dict):
        merged_where_columns = set()
        for table, conditions in query.table_where_condition_dict.items():
            if table not in mergeable_tables:
                continue
            for condition in conditions:
                column = infer_column(condition)
                merged_where_columns.add(table + '.' + column)
        rdc_sum = sum([rdc_attribute_dict[column_combination]
                       for column_combination in itertools.combinations(list(merged_where_columns), 2)
                       if rdc_attribute_dict.get(column_combination) is not None])
        return rdc_sum

    def _greedily_select_next_table(self, original_query, query, next_neighbours, exploit_overlapping, merged_tables,
                                    rdc_spn_selection=False, rdc_attribute_dict=None, prefer_disjunct=False):
        """
        Compute possible next merges and select greedily.
        """
        next_bn = None
        next_neighbour = None
        next_mergeable_relationships = None
        current_best_candidate_vector = None

        for bn in self.bns:

            if len(bn.table_set.intersection(merged_tables)) > 0 and prefer_disjunct:
                continue

            possible_neighbours = bn.table_set.intersection(next_neighbours)

            # for one BN we can have several starting points
            for neighbour in possible_neighbours:

                # plus 1 because we can also merge edge directing to neighbour
                mergeable_relationships = bn.compute_mergeable_relationships(query, neighbour)
                no_mergeable_relationships = len(mergeable_relationships) + 1

                mergeable_tables = self._merged_tables(mergeable_relationships)
                mergeable_tables.add(neighbour)

                where_condition_tables = set(query.table_where_condition_dict.keys()).intersection(mergeable_tables)
                unnecessary_tables = len(bn.table_set) - len(mergeable_tables)

                if not exploit_overlapping:
                    current_candidate_vector = (len(where_condition_tables), no_mergeable_relationships,
                                                -unnecessary_tables)

                else:
                    # find overlapping relationships (relationships already merged that also appear in next_bn)
                    _, overlapping_tables, no_overlapping_conditions = self._compute_overlap(
                        next_neighbour, query, original_query, mergeable_relationships, mergeable_tables, bn)

                    unnecessary_tables = len(bn.table_set.difference(mergeable_tables).difference(overlapping_tables))
                    current_candidate_vector = (len(where_condition_tables), no_mergeable_relationships,
                                                no_overlapping_conditions, -unnecessary_tables)

                # if rdc based selection is active we should this should be the first part of the candidate vector
                if rdc_spn_selection:
                    # find attributes with where conditions
                    rdc_sum = self.merged_rdc_sum(mergeable_tables, query, rdc_attribute_dict)
                    current_candidate_vector = (rdc_sum,) + current_candidate_vector

                if current_best_candidate_vector is None or \
                        current_candidate_vector > current_best_candidate_vector:
                    next_bn = bn
                    next_neighbour = neighbour
                    next_mergeable_relationships = mergeable_relationships
                    current_best_candidate_vector = current_candidate_vector

        if next_bn is None:
            # recursive call with prefer false because there is no disjunct candidate
            return self._greedily_select_next_table(original_query, query, next_neighbours, exploit_overlapping,
                                                    merged_tables, prefer_disjunct=False)

        return next_bn, next_neighbour, next_mergeable_relationships

    def _next_neighbours(self, query, merged_tables):
        """
        List tables which have direct edge to already merged tables. Should be merged in next step.
        """

        next_neighbours = set()
        neighbours_relationship_dict = {}

        for relationship in query.relationship_set:
            relationship_obj = self.schema_graph.relationship_dictionary[relationship]
            if relationship_obj.start in merged_tables and \
                    relationship_obj.end not in merged_tables:

                neighbour = relationship_obj.end
                next_neighbours.add(neighbour)
                neighbours_relationship_dict[neighbour] = relationship

            elif relationship_obj.end in merged_tables and \
                    relationship_obj.start not in merged_tables:

                neighbour = relationship_obj.start
                next_neighbours.add(neighbour)
                neighbours_relationship_dict[neighbour] = relationship

        return next_neighbours, neighbours_relationship_dict

    def _compute_overlap(self, next_neighbour, query, original_query, next_mergeable_relationships, next_merged_tables,
                         next_bn):
        """
        Find overlapping relationships (relationships already merged that also appear in next_bn)

        :param next_neighbour:
        :param original_query:
        :param next_mergeable_relationships:
        :param next_bn:
        :return:
        """
        overlapping_relationships = set()
        overlapping_tables = {next_neighbour}
        new_overlapping_table = True
        while new_overlapping_table:
            new_overlapping_table = False
            for relationship_obj in self.schema_graph.relationships:
                if relationship_obj.identifier in original_query.relationship_set \
                        and relationship_obj.identifier not in overlapping_relationships \
                        and relationship_obj.identifier not in next_mergeable_relationships \
                        and relationship_obj.identifier in next_bn.relationship_set:
                    if relationship_obj.start in overlapping_tables \
                            and relationship_obj.end not in overlapping_tables:
                        new_overlapping_table = True
                        overlapping_tables.add(relationship_obj.end)
                        overlapping_relationships.add(relationship_obj.identifier)
                    elif relationship_obj.start not in overlapping_tables \
                            and relationship_obj.end in overlapping_tables:
                        new_overlapping_table = True
                        overlapping_tables.add(relationship_obj.start)
                        overlapping_relationships.add(relationship_obj.identifier)

        # overlapping conditions
        no_overlapping_conditions = len(set(query.table_where_condition_dict.keys())
                                        .intersection(overlapping_tables.difference(next_merged_tables)))

        return overlapping_relationships, overlapping_tables, no_overlapping_conditions


    def save(self, path, compress=False):
        if compress:
            with bz2.BZ2File(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def build_from_data(self, hdf_path, ensemble_path, max_table_data, sample_size=None, schema=None):
        """
        We will learn one BN each for each table in the schema graph
        Parameters
        ----------
        hdf_path: path to the folder that contain all pandas dataframe
        ensemble_path: path to save the learned BN ensemble
        max_table_data: max row per hdf file
        sample_size: How many sample to learning BN, if None then use the full data
        schema: containing all information about the graph and attributes
        """
        meta_data_path = hdf_path + '/meta_data.pkl'
        prep = JoinDataPreparator(meta_data_path, schema, max_table_data=max_table_data)
        self.join_prepare = prep

        if schema is not None:
            self.schema_graph = schema

        logger.info(f"Creating a BN for every relationship.")
        for relationship_obj in schema.relationships:
            logger.info(f"Learning BN for {relationship_obj.identifier}.")
            df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier])
            logger.debug(f"Requested {sample_size} samples and got {len(df_samples)}")
            bn = Bayescard_BN()
            bn.build_from_data(df_samples)
            self.add_BN(bn)

        ensemble_path += '/ensemble_relationships_' + str(sample_size) + '.pkl'
        logger.info(f"Saving ensemble to {ensemble_path}")
        self.save(ensemble_path)

    def str_pattern_matching(self, s):
        # split the string "attr==value" to ('attr', '=', 'value')
        op_start = 0
        if len(s.split(' IN ')) != 1:
            s = s.split(' IN ')
            attr = s[0].strip()
            try:
                value = list(ast.literal_eval(s[1].strip()))
            except:
                value = s[1].strip()[1:][:-1].split(',')
            return attr, 'in', value

        for i in range(len(s)):
            if s[i] in OPS:
                op_start = i
                if i + 1 < len(s) and s[i + 1] in OPS:
                    op_end = i + 1
                else:
                    op_end = i
                break
        attr = s[:op_start]
        value = s[(op_end+1):]
        ops = s[op_start:(op_end+1)]
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            try:
                value = float(value)
            except:
                value = value
        
        return attr, ops, value
    
    def construct_table_query(self, table_name, table_query, attr, ops, val, epsilon=1e-6):
        query_domain = []
        if self.bns[table_name] is None or attr not in self.bns[table_name].attr_type:
            return None
        if self.bns[table_name].attr_type[attr] == 'continuous':
            if ops == ">=": query_domain = [val, np.infty]
            elif ops == ">": query_domain = [val+epsilon, np.infty]
            elif ops == "<=": query_domain = [-np.infty, val]
            elif ops == "<": query_domain = [-np.infty, val-epsilon]
            elif ops == "=" or ops == "==": query_domain = [val, val]
            else:
                assert False, f"operation {ops} is invalid for continous domain"
                
            if attr not in table_query:
                table_query[attr] = query_domain
            else:
                prev_l = table_query[attr][0]
                prev_r = table_query[attr][1]
                query_domain = [max(prev_l, query_domain[0]), min(prev_r, query_domain[1])]
                table_query[attr] = query_domain
            
        else:
            attr_domain = self.bns[table_name].domain[attr]
            if type(attr_domain[0]) != str:
                attr_domain = np.asarray(attr_domain)
            if ops == "in": 
                assert type(val) == list, "use list for in query"
                query_domain = val
            elif ops == "=" or ops == "==":
                if type(val) == list:  query_domain = val
                else: query_domain = [val]
            else:
                if type(val) == list:
                    assert len(val) == 1
                    val = val[0]
                    assert (type(val) == int or type(val) == float)
                operater = OPS[ops]
                query_domain = list(attr_domain[operater(attr_domain, val)])
                
            if attr not in table_query:
                table_query[attr] = query_domain
            else:
                query_domain = [i for i in query_domain if i in table_query[attr]]
                table_query[attr] = query_domain
                
        return table_query
            

    def store_join_size(self):
        # Store the table size for each relation in the schema
        for relationship_obj in self.schema_graph.relationships:
            rel = relationship_obj.identifier
            self.join_size[rel] = self.join_prepare._size_estimate(relationship_list=[rel])[1]

    def get_full_join_size(self, rel):
        if type(rel) == list and len(rel) == 1:
            rel = rel[0]
        if type(rel) == str:
            return self.join_size[rel]
        else:
            return self.join_prepare._size_estimate(relationship_list=rel)[1]

    def naive_cardinality(self, query):
        # estimate the cardinality of given query using the join uniformity assumption
        select_tables = query.table_set
        conditions = query.table_where_condition_dict
        relations = query.relationship_set
        # reformulate them to Single_BN executable form
        p_estimate = 1
        for table_name in select_tables:
            if table_name in conditions:
                table_query = dict()
                for condition in conditions[table_name]:
                    attr, ops, val = self.str_pattern_matching(condition)
                    #attr = table_name + '.' + attr
                    table_query = self.construct_table_query(table_name, table_query, attr, ops, val)
                    if table_query is None:
                        return None
                p = self.bns[table_name].query(table_query, return_prob=True)
                p_estimate *= p[0]
        return p_estimate * self.get_full_join_size(list(relations))

    def parse_query_all(self, table_queries):
        res_table_queries = []
        for table_query in table_queries:
            res_table_query = []
            res_table_query.append(table_query[0])
            for i, query in enumerate(table_query[1:]):
                new_query = dict()
                ind = i+1
                if ind+1 < len(table_query):
                    if query["bn_index"] == table_query[ind+1]["bn_index"] and \
                        query["query"] == table_query[ind+1]["query"] and \
                            query["expectation"] == table_query[ind+1]["expectation"]:
                        continue
                if i > 0:
                    if query["bn_index"] == table_query[i]["bn_index"] and \
                        query["query"] == table_query[i]["query"] and \
                            query["expectation"] == table_query[i]["expectation"]:
                        continue
                new_query["bn_index"] = query["bn_index"]
                new_query["inverse"] = query["inverse"]
                new_query["expectation"] = query["expectation"]
                bn = self.bns[query["bn_index"]]
                new_query["query"], new_query["n_distinct"] = bn.query_decoding(query["query"])
                res_table_query.append(new_query)
            res_table_queries.append(res_table_query)
        return res_table_queries


    def cardinality(self, table_query, sample_size=1000, hard_sample=False):
        card = table_query[0]
        for query in table_query[1:]:
            bn = self.bns[query["bn_index"]]
            #print("=============================================")
            #print("bn_index:", query["bn_index"])
            #print(bn.relationship_set)
            if len(query["expectation"]) == 0:
                q = query["query"]
                #print(f"query: {q}")
                p, _ = bn.query(query["query"], n_distinct=query["n_distinct"],
                                  return_prob=True, sample_size=sample_size, hard_sample=hard_sample)
            else:
                q = query["query"]
                e = query["expectation"]
                #print(f"expectation: {q}, fanout_attr: {e}")
                p, _ = bn.expectation(query["query"], query["expectation"], n_distinct=query["n_distinct"],
                                      return_prob=True, sample_size=sample_size, hard_sample=hard_sample)

            if p == 0:
                return 1
            elif query["inverse"]:
                #print("inverse p:", 1/p)
                card *= (1/p)
            else:
                #print("p", p)
                card *= p
        if card <= 1:
            card = 1
        return card

