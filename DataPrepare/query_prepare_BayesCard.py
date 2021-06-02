from DeepDBUtils.evaluation.utils import parse_query
from DeepDBUtils.ensemble_compilation.graph_representation import Query
from DeepDBUtils.ensemble_compilation.probabilistic_query import IndicatorExpectation, Expectation
from DeepDBUtils.rspn.algorithms.ranges import NominalRange, NumericRange
from Models.BN_ensemble_model import BN_ensemble
import pickle
import os
import numpy as np

import sys
from DeepDBUtils import ensemble_compilation, aqp_spn, rspn

sys.modules['ensemble_compilation'] = ensemble_compilation
sys.modules['aqp_spn'] = aqp_spn
sys.modules['rspn'] = rspn


def prepare_single_query(range_conditions, factor, epsilon=0.1):
    query = dict()
    fanout = []
    col_name = list(factor.spn.column_names)
    assert len(range_conditions) == 1
    table_range = range_conditions[0]
    assert len(table_range) == len(col_name)
    for i, col in enumerate(table_range):
        if isinstance(col, NumericRange):
            if col.ranges[0][0] == col.ranges[0][1]:
                query[col_name[i]] = col.ranges[0][0]
            else:
                inclusive = col.inclusive_intervals[0]
                interval = []
                if col.ranges[0][0] < -10000:
                    #This is hard coded because all null values are less than -10000
                    col.ranges[0][0] = -9999
                if inclusive[0]:
                    interval.append(col.ranges[0][0])
                else:
                    interval.append(col.ranges[0][0] + epsilon)
                if inclusive[1]:
                    interval.append(col.ranges[0][1])
                else:
                    interval.append(col.ranges[0][1] - epsilon)
                query[col_name[i]] = tuple(interval)
        elif isinstance(col, NominalRange):
            assert col.possible_values.size == 1
            query[col_name[i]] = col.possible_values[0]

    for table, f in factor.nominator_multipliers:
        fanout.append(table + "." + f)
    
    for table, f in factor.denominator_multipliers:
        fanout.append(table + "." + f)

    return query, fanout


def generate_factors(bn_ensemble, query, first_bn, next_mergeable_relationships, next_mergeable_tables,
                     rdc_bn_selection=False, rdc_attribute_dict=None,
                     merge_indicator_exp=True, exploit_overlapping=False,
                     return_factor_values=False, exploit_incoming_multipliers=True,
                     prefer_disjunct=False):
    factors = []

    # only operate on copy so that query object is not changed
    # for greedy strategy it does not matter whether query is changed
    # optimized version of:
    # original_query = copy.deepcopy(query)
    # query = copy.deepcopy(query)
    original_query = query.copy_cardinality_query()
    query = query.copy_cardinality_query()

    # First BN: Full_join_size*E(outgoing_mult * 1/multiplier * 1_{c_1 Λ… Λc_n})
    # Again create auxilary query because intersection of query relationships and spn relationships
    # is not necessarily a tree.
    auxilary_query = Query(bn_ensemble.schema_graph)
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
        next_neighbours, neighbours_relationship_dict = bn_ensemble._next_neighbours(query, merged_tables)

        # compute possible next merges and select greedily
        next_bn, next_neighbour, next_mergeable_relationships = bn_ensemble._greedily_select_next_table(
            original_query,
            query,
            next_neighbours,
            exploit_overlapping,
            merged_tables,
            prefer_disjunct=prefer_disjunct,
            rdc_spn_selection=rdc_bn_selection,
            rdc_attribute_dict=rdc_attribute_dict)

        # if outgoing: outgoing_mult appended to multipliers
        relationship_to_neighbour = neighbours_relationship_dict[next_neighbour]
        relationship_obj = bn_ensemble.schema_graph.relationship_dictionary[relationship_to_neighbour]

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

                for bn in bn_ensemble.bns:
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
        next_merged_tables = bn_ensemble._merged_tables(next_mergeable_relationships)
        next_merged_tables.add(next_neighbour)

        # find overlapping relationships (relationships already merged that also appear in next_spn)
        overlapping_relationships, overlapping_tables, no_overlapping_conditions = bn_ensemble._compute_overlap(
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
        nominator_query = Query(bn_ensemble.schema_graph)
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
        denominator_query = Query(bn_ensemble.schema_graph)
        for relationship in overlapping_relationships:
            denominator_query.add_join_condition(relationship)
        denominator_query.table_set.update(next_merged_tables)
        denominator_query.table_where_condition_dict = query.table_where_condition_dict

        # constraints for next neighbor would not have any impact otherwise
        conditions = next_bn.relevant_conditions(denominator_query, merged_tables=overlapping_tables)

        next_neighbour_obj = bn_ensemble.schema_graph.table_dictionary[next_neighbour]
        # add not null condition for next neighbor
        conditions.append((next_neighbour, next_neighbour_obj.table_nn_attribute + " IS NOT NULL"))
        multipliers = next_bn.compute_multipliers(denominator_query)
        #print("denominator_exp:", multipliers, denominator_query.table_set, denominator_query.relationship_set)
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

    return factors


def factor_refine(factors_full):
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
    return [factor for factor in factors_full if factor not in factors_to_be_deleted]


def further_refine(results, bn_ensemble, query):
    seen = []
    seen_tables = set()
    relationship_set = query.relationship_set
    print(relationship_set)
    for i in range(1, len(results)):
        query1 = results[i]
        bn_index1 = query1["bn_index"]
        if i in seen:
            continue
        curr_tables = bn_ensemble.bns[bn_index1].table_set
        for j in range(i+1, len(results)):
            query2 = results[j]
            bn_index2 = query2["bn_index"]
            if j in seen:
                continue
            if bn_index1 == bn_index2:
                f_attrs = list(bn_ensemble.bns[bn_index1].fanouts.keys())
                f_inners = [x for x in f_attrs if "mul_" in x and "_nn" in x]
                if query2["inverse"]:
                    for f in f_inners:
                        left = f.split("mul_")[0] + "Id"
                        right = f.split("mul_")[1].split("_nn")[0]
                        print(bn_index1, ":", f)
                        if left+" = "+right in relationship_set or right+" = "+left in relationship_set:
                            print(len(curr_tables.intersection(seen_tables)) != 0 and f not in query1["expectation"])
                            print(curr_tables, seen_tables)
                            print(query1["expectation"])
                            if len(curr_tables.intersection(seen_tables)) != 0 and f not in query1["expectation"]:
                                print("adding:", bn_index1, ":", f)
                                query1["expectation"].append(f)
             
                elif query1["inverse"]:
                    for f in f_inners:
                        left = f.split("mul_")[0] + "Id"
                        right = f.split("mul_")[1].split("_nn")[0]
                        print(bn_index1, ":", f)
                        if left+" = "+right in relationship_set or right+" = "+left in relationship_set:
                            print(len(curr_tables.intersection(seen_tables)) != 0 and f not in query2["expectation"])
                            if len(curr_tables.intersection(seen_tables)) != 0 and f not in query2["expectation"]:
                                print("adding:", bn_index1, ":", f)
                                query2["expectation"].append(f)
                seen.append(i)
                seen.append(j)
            seen_tables |= curr_tables
    return results
                    
    

def load_ensemble(schema, model_path="/home/ziniu.wzn/stats/BN_ensemble/"):
    bn_ensemble = BN_ensemble(schema)
    for file in os.listdir(model_path):
        if file.endswith(".pkl"):
            with open(model_path + file, "rb") as f:
                try:
                    bn = pickle.load(f)
                    bn.infer_algo = "exact-jit"
                    bn.init_inference_method()
                except:
                    continue
            bn_ensemble.bns.append(bn)
    return bn_ensemble


def prepare_join_queries(schema, ensemble_location, pairwise_rdc_path, query_filename,
                         join_3_rdc_based=False, true_card_exist=False):
    bn_ensemble = load_ensemble(schema, ensemble_location)
    parsed_queries = []
    
    if pairwise_rdc_path:
        with open(pairwise_rdc_path, 'rb') as handle:
            rdc_attribute_dict = pickle.load(handle)
    else:
        rdc_attribute_dict = dict()

    true_card = []
    with open(query_filename) as f:
        queries = f.readlines()
        for query_no, query_str in enumerate(queries):
            #print("=====================================================")
            #print(query_no, query_str)
            if true_card_exist:
                try:
                    true_card.append(int(query_str.split("||")[-1]))
                    query_str = query_str.split("||")[0]
                except:
                    true_card.append(int(query_str.split("||")[0]))
                    query_str = query_str.split("||")[-1]
            query_str = query_str.strip()

            query = parse_query(query_str.strip(), schema)

            first_bn, next_mergeable_relationships, next_mergeable_tables = \
                bn_ensemble._greedily_select_first_cardinality_bn(
                    query, rdc_spn_selection=True, rdc_attribute_dict=rdc_attribute_dict)

            factors = generate_factors(bn_ensemble, query, first_bn, next_mergeable_relationships,
                                       next_mergeable_tables, rdc_bn_selection=True,
                                       rdc_attribute_dict=rdc_attribute_dict, merge_indicator_exp=True,
                                       exploit_incoming_multipliers=True, prefer_disjunct=False)

            factors = factor_refine(factors)

            parse_result = []
            for i, factor in enumerate(factors):
                if isinstance(factor, IndicatorExpectation):
                    range_conditions = factor.spn._parse_conditions(factor.conditions, group_by_columns=None,
                                                                    group_by_tuples=None)

                    actual_query, fanout = prepare_single_query(range_conditions, factor)
                    
                    parse_result.append({"bn_index": bn_ensemble.bns.index(factor.spn),
                                         "inverse": factor.inverse,
                                         "query": actual_query,
                                         "expectation": fanout,
                                         })

                elif isinstance(factor, Expectation):
                    raise NotImplementedError
                else:
                    parse_result.append(factor)
                    
            #parse_result = further_refine(parse_result, bn_ensemble, query)

            parsed_queries.append(parse_result)

        return parsed_queries, true_card
