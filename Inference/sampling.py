from DataPrepare.join_data_preparation import JoinDataPreparator
from Inference.utils import str_pattern_matching, convert_to_pandas_query
from Evaluation.utils import parse_query
from time import perf_counter
import numpy as np

class Sampling:
    def __init__(self, meta_data_path, schema, max_table_data=100000000):
        self.schema = schema
        prep = JoinDataPreparator(meta_data_path, schema, max_table_data=max_table_data)
        self.join_prepare = prep
        self.join_size = dict()
        self.store_join_size()

    def store_join_size(self):
        # Store the table size for each relation in the schema
        for relationship_obj in self.schema.relationships:
            rel = relationship_obj.identifier
            self.join_size[rel] = self.join_prepare._size_estimate(relationship_list=[rel])[1]

    def get_full_join_size(self, rel):
        if type(rel) == list and len(rel) == 1:
            rel = rel[0]
        if rel in self.join_size:
            return self.join_size[rel]
        else:
            return self.join_prepare._size_estimate(relationship_list=rel)[1]

    def single_table_query(self, query, data_sample, return_prob=True):
        query = convert_to_pandas_query(query)
        if return_prob:
            return len(data_sample.query(query))/len(data_sample), len(data_sample)
        else:
            return len(data_sample.query(query))

    def naive_cardinality(self, query, data_samples):
        # estimate the cardinality of given query
        conditions = query.table_where_condition_dict
        relations = query.relationship_set
        # reformulate them to Single_BN executable form
        p_estimate = 1
        for table_name in conditions:
            print(table_name)
            table_query = dict()
            for condition in conditions[table_name]:
                attr, val = str_pattern_matching(condition)
                attr = table_name + '__' + attr
                if attr in table_query:
                    table_query[attr].append(val)
                else:
                    table_query[attr] = [val]
            print(table_query)
            p, _ = self.single_table_query(table_query, data_samples[table_name], return_prob=True)
            print(p)
            p_estimate *= p
        print(p_estimate, self.get_full_join_size(list(relations)))
        return p_estimate * self.get_full_join_size(list(relations))


def evaluate_cardinalities(schema, query_file, sample_df,
                           meta_data_path="/home/ziniu.wzn/imdb-benchmark/gen_single_light/" + '/meta_data.pkl'):
    samp = Sampling(meta_data_path, schema)
    with open(query_file) as f:
        queries = f.readlines()

    q_errors = []
    latencies = []
    true_cards = []
    for query_no, query_str in enumerate(queries):

        query_str = query_str.strip().split("||")

        cardinality_true = int(query_str[1])
        true_cards.append(cardinality_true)
        query_str = query_str[0]

        print(f"Predicting cardinality for query {query_no}: {query_str}")

        query = parse_query(query_str.strip(), schema)

        card_start_t = perf_counter()
        cardinality_predict = samp.naive_cardinality(query, sample_df)
        print(cardinality_predict)
        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000
        if cardinality_predict == 0:
            # avoid division by zero
            cardinality_predict = 1
        print(f"\t\tLatency: {latency_ms:.2f}ms")
        print(f"\t\tTrue: {cardinality_true}")
        print(f"\t\tPredicted: {cardinality_predict}")

        q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
        if cardinality_predict == 0 and cardinality_true == 0:
            q_error = 1.0

        print(f"Q-Error was: {q_error}")
        q_errors.append(q_error)
        latencies.append(latency_ms)

    # print percentiles of published JOB-light
    q_errors = np.array(q_errors)
    q_errors.sort()

    for i, percentile in enumerate([50, 90, 95, 99]):
        print(f"Q-Error {percentile}%-Percentile")

    print(f"Q-Mean wo inf {np.mean(q_errors[np.isfinite(q_errors)])}")
    print(f"Latency avg: {np.mean(latencies):.2f}ms")
    return q_errors, latencies
