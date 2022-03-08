from Evaluation.cardinality_estimation import parse_query_single_table
from DataPrepare.query_prepare_BayesCard import prepare_join_queries
from Models.BN_ensemble_model import BN_ensemble
from time import perf_counter
import numpy as np
import pickle
import time
import os

def evaluate_cardinality_single_table(model_path, query_path, infer_algo, sample_size=1000):
    # load BN
    with open(model_path, 'rb') as f:
        BN = pickle.load(f)
    if BN.infer_machine is None:
        BN.infer_algo = infer_algo
        BN.init_inference_method()
    # read all queries
    with open(query_path) as f:
        queries = f.readlines()
    latencies = []
    q_errors = []
    for query_no, query_str in enumerate(queries):
        cardinality_true = int(query_str.split("||")[-1])
        query_str = query_str.split("||")[0]
        try:
            print(f"Predicting cardinality for query {query_no}: {query_str}")
            query = parse_query_single_table(query_str.strip(), BN)
            card_start_t = perf_counter()
            cardinality_predict = BN.query(query, sample_size=sample_size)
        except:
            #In the case, that the query itself is invalid or contains some values that are not recognizable by BN
            continue
        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000
        if cardinality_predict == 0 and cardinality_true == 0:
            q_error = 1.0
        elif np.isnan(cardinality_predict) or cardinality_predict == 0:
            cardinality_predict = 1
            q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
        elif cardinality_true == 0:
            cardinality_true = 1
            q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
        else:
            q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
        print(f"latency: {latency_ms} and error: {q_error}")
        latencies.append(latency_ms)
        q_errors.append(q_error)

    print("=====================================================================================")
    for i in [50, 90, 95, 99, 100]:
        print(f"q-error {i}% percentile is {np.percentile(q_errors, i)}")
    print(f"average latency is {np.mean(latencies)} ms")

    return latencies, q_errors


def load_ensemble(schema, model_path):
    bn_ensemble = BN_ensemble(schema)
    for file in os.listdir(model_path):
        if file.endswith(".pkl"):
            with open(model_path+file, "rb") as f:
                bn = pickle.load(f)
                bn.infer_algo = "exact-jit"
                bn.init_inference_method()
            bn_ensemble.bns.append(bn)
    return bn_ensemble


def evaluate_cardinality_imdb(schema, ensemble_location, query_filename, infer_algo, learning_algo, max_parents):
    query_filename = "/home/ubuntu/BayesCard/Benchmark/IMDB/job-light.sql"
    parsed_queries, true = prepare_join_queries(schema, ensemble_location, pairwise_rdc_path=None, 
                                                query_filename=query_filename, true_card_exist=True)
    
    with open(query_filename, "rb") as f:
        real_query = f.readlines()
    
    bn_ensemble = load_ensemble(schema, ensemble_location)

    queries = bn_ensemble.parse_query_all(parsed_queries)

    latency = []
    q_errors = []
    for i, q in enumerate(queries):
        tic = time.time()
        try:
            pred = bn_ensemble.cardinality(q)
        except:
            print(f"query {i} is invalid!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # this query itself is invalid or it is not recognizable by the learnt BN
            continue
        latency.append(time.time() - tic)
        if pred is None or pred <= 1:
            pred = 1
        error = max(pred / true[i], true[i] / pred)
        print(f"predicting query no {i}: {real_query[i]} \n")
        print(f"true cardinality {true[i]}, predicted {pred} with q-error {error}")
        q_errors.append(error)
    print("=====================================================================================")
    for i in [50, 90, 95, 99, 100]:
        print(f"q-error {i}% percentile is {np.percentile(q_errors, i)}")
    print(f"average latency is {np.mean(latency)*1000} ms")

    return latency, q_errors
