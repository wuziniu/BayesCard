import logging
from time import perf_counter

import numpy as np
import pandas as pd

from DeepDBUtils.ensemble_compilation.graph_representation import QueryType
from DeepDBUtils.ensemble_compilation.physical_db import DBConnection, TrueCardinalityEstimator
from DeepDBUtils.ensemble_compilation.spn_ensemble import read_ensemble
from DeepDBUtils.evaluation.utils import parse_query, save_csv

logger = logging.getLogger(__name__)

def compute_ground_truth(query_filename, target_path, physical_db_name):
    """
    Queries database for each query and stores result rows in csv file.
    :param query_filename: where to take queries from
    :param target_path: where to store dictionary
    :param physical_db_name: name of the database
    :return:
    """

    db_connection = DBConnection(db=physical_db_name)

    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()

    csv_rows = []
    for query_no, query_str in enumerate(queries):
        logger.debug(f"Computing ground truth for cardinality query {query_no}: {query_str}")
        query_str = query_str.strip()
        cardinality_true = db_connection.get_result(query_str)

        csv_rows.append({'query_no': query_no,
                         'query': query_str,
                         'cardinality_true': cardinality_true})

    save_csv(csv_rows, target_path)


class GenCodeStats:

    def __init__(self):
        self.calls = 0
        self.total_time = 0.0
        
        
def estimate_cardinalities(ensemble_location, query_filename, schema,
                           rdc_spn_selection, pairwise_rdc_path, use_generated_code=False,
                           max_variants=1, merge_indicator_exp=False, exploit_overlapping=False,
                           min_sample_ratio=0, return_res=True):
    """
    Loads ensemble and evaluates cardinality for every query in query_filename
    :param exploit_overlapping:
    :param min_sample_ratio:
    :param max_variants:
    :param merge_indicator_exp:
    :param target_csv_path:
    :param query_filename:
    :param true_cardinalities_path:
    :param ensemble_location:
    :param physical_db_name:
    :param schema:
    :return:
    """
    # load ensemble
    spn_ensemble = read_ensemble(ensemble_location, build_reverse_dict=True)

    with open(query_filename) as f:
        queries = f.readlines()
        
    if use_generated_code:
        spn_ensemble.use_generated_code()

    latencies = []
    predict = []
    invalid_query = set()
    for query_no, query_str in enumerate(queries):
        if "||" in query_str:
            try:
                true_card = int(query_str.split("||")[0])
                query_str = query_str.split("||")[-1]
            except:
                true_card = int(query_str.split("||")[-1])
                query_str = query_str.split("||")[0]
        else:
            true_card = query_no
        query_str = query_str.strip()
        print(f"Predicting cardinality for query {query_no}: {query_str}")
        
        try:
            query = parse_query(query_str.strip(), schema)
        except:
            cardinality_predict = -1
            print(f"FK-FK join not supported, query no: {true_card}")
            latencies.append(1)
            predict.append(cardinality_predict)
            invalid_query.add(true_card)
            continue
        assert query.query_type == QueryType.CARDINALITY

        # only relevant for generated code
        gen_code_stats = GenCodeStats()

        card_start_t = perf_counter()
        try:
            _, factors, cardinality_predict, factor_values = spn_ensemble \
                .cardinality(query, rdc_spn_selection=rdc_spn_selection, pairwise_rdc_path=pairwise_rdc_path,
                             merge_indicator_exp=merge_indicator_exp, max_variants=max_variants,
                             exploit_overlapping=exploit_overlapping, return_factor_values=True,
                             gen_code_stats=gen_code_stats)
        except:
            cardinality_predict = -1
            invalid_query.add(true_card)
            print(f"cyclic join, query no: {true_card}")
        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000
        print(f"\t\tLatency: {latency_ms:.2f}ms")
        print(f"\t\tPredicted: {cardinality_predict}")
        latencies.append(latency_ms)
        predict.append(cardinality_predict)
    return predict, latencies, invalid_query
        


def evaluate_cardinalities(ensemble_location, physical_db_name, query_filename, target_csv_path, schema,
                           rdc_spn_selection, pairwise_rdc_path, use_generated_code=False,
                           true_cardinalities_path='./benchmarks/job-light/sql/job_light_true_cardinalities.csv',
                           max_variants=1, merge_indicator_exp=False, exploit_overlapping=False,
                           min_sample_ratio=0, return_res=True):
    """
    Loads ensemble and evaluates cardinality for every query in query_filename
    :param exploit_overlapping:
    :param min_sample_ratio:
    :param max_variants:
    :param merge_indicator_exp:
    :param target_csv_path:
    :param query_filename:
    :param true_cardinalities_path:
    :param ensemble_location:
    :param physical_db_name:
    :param schema:
    :return:
    """
    if true_cardinalities_path is not None:
        df_true_card = pd.read_csv(true_cardinalities_path)
    else:
        df_true_card = None

    # load ensemble
    spn_ensemble = read_ensemble(ensemble_location, build_reverse_dict=True)

    csv_rows = []
    q_errors = []

    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()
        
    if use_generated_code:
        spn_ensemble.use_generated_code()

    latencies = []
    for query_no, query_str in enumerate(queries):
        if "||" in query_str:
            try:
                true_card = int(query_str.split("||")[0])
                query_str = query_str.split("||")[-1]
            except:
                true_card = int(query_str.split("||")[-1])
                query_str = query_str.split("||")[0]
        query_str = query_str.strip()
        print(f"Predicting cardinality for query {query_no}: {query_str}")
        
        try:
            query = parse_query(query_str.strip(), schema)
        except:
            print("FK-FK join not supported")
            continue
        assert query.query_type == QueryType.CARDINALITY

        if df_true_card is None:
            cardinality_true = true_card
        else:
            cardinality_true = df_true_card.loc[df_true_card['query_no'] == query_no, ['cardinality_true']].values[0][0]

        # only relevant for generated code
        gen_code_stats = GenCodeStats()

        card_start_t = perf_counter()
        _, factors, cardinality_predict, factor_values = spn_ensemble \
            .cardinality(query, rdc_spn_selection=rdc_spn_selection, pairwise_rdc_path=pairwise_rdc_path,
                         merge_indicator_exp=merge_indicator_exp, max_variants=max_variants,
                         exploit_overlapping=exploit_overlapping, return_factor_values=True,
                         gen_code_stats=gen_code_stats)
        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000

        print(f"\t\tLatency: {latency_ms:.2f}ms")
        print(f"\t\tTrue: {cardinality_true}")
        print(f"\t\tPredicted: {cardinality_predict}")

        q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
        if cardinality_predict == 0 and cardinality_true == 0:
            q_error = 1.0

        print(f"Q-Error was: {q_error}")
        q_errors.append(q_error)
        csv_rows.append({'query_no': query_no,
                         'query': query_str,
                         'cardinality_predict': cardinality_predict,
                         'cardinality_true': cardinality_true,
                         'latency_ms': latency_ms,
                         'generated_spn_calls': gen_code_stats.calls,
                         'latency_generated_code': gen_code_stats.total_time * 1000})
        latencies.append(latency_ms)

    # print percentiles of published JOB-light
    q_errors = np.array(q_errors)
    #q_errors.sort()
    logger.info(f"{q_errors[-10:]}")
    # https://arxiv.org/pdf/1809.00677.pdf
    #ibjs_vals = [1.59, 150, 3198, 14309, 590]
    #mcsn_vals = [3.82, 78.4, 362, 927, 57.9]
    for i, percentile in enumerate([50, 90, 95, 99]):
        print(f"Q-Error {percentile}%-Percentile: {np.percentile(q_errors, percentile)}")

    print(f"Q-Mean wo inf {np.mean(q_errors[np.isfinite(q_errors)])}")
    print(f"Latency avg: {np.mean(latencies):.2f}ms")

    # write to csv
    if return_res:
        return (q_errors, latencies)
    #save_csv(csv_rows, target_csv_path)
