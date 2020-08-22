import time
import numpy as np
import bz2
import pickle
import logging
import ast

from DataPrepare.join_data_preparation import JoinDataPreparator
from Models.pgmpy_BN import Pgmpy_BN

logger = logging.getLogger(__name__)

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}

class BN_ensemble():
    """
    Several BNs combined one for each table.
    """

    def __init__(self, schema_graph, bns=dict()):
        self.schema_graph = schema_graph
        self.bns = bns
        self.cached_expectation_vals = dict()
        self.join_size = dict()
        self.join_prepare = None

    def add_BN(self, bn):
        self.bns[bn.table_name] = bn

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
            bn = Pgmpy_BN()
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
        # estimate the cardinality of given query
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
    