import ast
import time
import pandas as pd
import numpy as np
import logging
from time import perf_counter
from Evaluation.utils import parse_query, save_csv
from Structure.BN_ensemble_model import load_BN_ensemble

logger = logging.getLogger(__name__)

def parse_query_single_table(query):
    result = dict()
    useful = query.split(' WHERE ')[-1].strip()
    for sub_query in useful.split(' AND '):
        if len(sub_query.split(' in ')) != 1:
            sub_query = sub_query.split(' in ')
        elif len(sub_query.split(' == ')) != 1:
            sub_query = sub_query.split(' == ')
        else:
            raise NotImplemented
        col_name = sub_query[0].strip().split('.')[-1]
        try:
            value = list(ast.literal_eval(sub_query[1].strip()))
        except:
            value = sub_query[1].strip()[1:][:-1].split(',')
        result[col_name] = value
    return result


def evaluate_card(bn, query_filename='/home/ziniu.wzn/deepdb-public/benchmarks/imdb_single/query.sql'):
    with open(query_filename) as f:
        queries = f.readlines()
    latencies = []
    error = []
    for query_no, query in enumerate(queries):
        query = query.strip().split('||')
        query_str = query[0]
        true_card = int(query[1])
        tic = time.time()
        est = bn.infer_query(parse_query_single_table(query_str))
        latencies.append(time.time() - tic)
        error = max(est / true_card, true_card / est)
        print(true_card, est)
    return latencies, error

def single_table_experiment():
    from Structure.pgmpy_BN import Pgmpy_BN
    df = pd.read_hdf("/home/ziniu.wzn/imdb-benchmark/gen_single_light/title.hdf")
    new_cols = []
    for col in df.columns:
        new_cols.append(col.replace('.', '__'))
    df.columns = new_cols
    BN = Pgmpy_BN('title')
    BN.build_from_data(df, algorithm="greedy", max_parents=1, n_mcv=30, n_bins=30, ignore_cols=['title_id'],
                       sample_size=500000)
    gd_latency, gd_error = evaluate_card(BN)
    np.save('gd_latency', np.asarray(gd_latency))
    np.save('gd_error', np.asarray(gd_error))

    BN = Pgmpy_BN('title')
    BN.build_from_data(df, algorithm="chow-liu", max_parents=1, n_mcv=30, n_bins=30, ignore_cols=['title_id'],
                       sample_size=500000)
    cl_latency, cl_error = evaluate_card(BN)
    np.save('cl_latency', np.asarray(cl_latency))
    np.save('cl_error', np.asarray(cl_error))

    BN.model = BN.model.to_junction_tree()
    BN.algorithm = "junction"
    BN.init_inference_method()
    jt_latency, jt_error = evaluate_card(BN)
    np.save('jt_latency', np.asarray(jt_latency))
    np.save('jt_error', np.asarray(jt_error))

def evaluate_cardinalities(ensemble_location,  query_filename, target_csv_path, schema,
                           rdc_spn_selection, pairwise_rdc_path,
                           true_cardinalities_path='./benchmarks/job-light/sql/job_light_true_cardinalities.csv',
                           max_variants=1, merge_indicator_exp=False, exploit_overlapping=False, min_sample_ratio=0):
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

    # load ensemble
    bn_ensemble = load_BN_ensemble(ensemble_location, build_reverse_dict=True)

    csv_rows = []
    q_errors = []

    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()

    latencies = []
    for query_no, query_str in enumerate(queries):

        query_str = query_str.strip()
        logger.debug(f"Predicting cardinality for query {query_no}: {query_str}")

        query = parse_query(query_str.strip(), schema)

        cardinality_true = df_true_card.loc[df_true_card['query_no'] == query_no, ['cardinality_true']].values[0][0]

        card_start_t = perf_counter()
        _, factors, cardinality_predict, factor_values = bn_ensemble \
            .cardinality(query, rdc_spn_selection=rdc_spn_selection, pairwise_rdc_path=pairwise_rdc_path,
                         merge_indicator_exp=merge_indicator_exp, max_variants=max_variants,
                         exploit_overlapping=exploit_overlapping, return_factor_values=True)
        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000

        logger.debug(f"\t\tLatency: {latency_ms:.2f}ms")
        logger.debug(f"\t\tTrue: {cardinality_true}")
        logger.debug(f"\t\tPredicted: {cardinality_predict}")

        q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
        if cardinality_predict == 0 and cardinality_true == 0:
            q_error = 1.0

        logger.debug(f"Q-Error was: {q_error}")
        q_errors.append(q_error)
        csv_rows.append({'query_no': query_no,
                         'query': query_str,
                         'cardinality_predict': cardinality_predict,
                         'cardinality_true': cardinality_true,
                         'latency_ms': latency_ms
                         })
        latencies.append(latency_ms)

    # print percentiles of published JOB-light
    q_errors = np.array(q_errors)
    q_errors.sort()
    logger.info(f"{q_errors[-10:]}")
    # https://arxiv.org/pdf/1809.00677.pdf
    ibjs_vals = [1.59, 150, 3198, 14309, 590]
    mcsn_vals = [3.82, 78.4, 362, 927, 57.9]
    for i, percentile in enumerate([50, 90, 95, 99]):
        logger.info(f"Q-Error {percentile}%-Percentile: {np.percentile(q_errors, percentile)} (vs. "
                    f"MCSN: {mcsn_vals[i]} and IBJS: {ibjs_vals[i]})")

    logger.info(f"Q-Mean wo inf {np.mean(q_errors[np.isfinite(q_errors)])} (vs. "
                f"MCSN: {mcsn_vals[-1]} and IBJS: {ibjs_vals[-1]})")
    logger.info(f"Latency avg: {np.mean(latencies):.2f}ms")

    # write to csv
    save_csv(csv_rows, target_csv_path)



if __name__ == "main":
    single_table_experiment()
