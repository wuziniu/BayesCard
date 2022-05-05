import argparse
import logging
import os
import time
import shutil
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/ubuntu/BayesCard')
from DataPrepare.join_data_preparation import prepare_sample_hdf
from DataPrepare.prepare_single_tables import prepare_all_tables
from Schemas.imdb.schema import gen_job_light_imdb_schema
from Schemas.stats.schema import gen_stats_light_schema
from Testing.BN_training import train_DMV, train_Census, train_imdb
from Testing.BN_testing import evaluate_cardinality_single_table, evaluate_cardinality_imdb


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dmv', help='Which dataset to be used')

    # generate hdf, this part is only related to imdb job
    # the preprocessing uses the code from deepdb-public
    parser.add_argument('--generate_hdf', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--generate_sampled_hdfs', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--csv_seperator', default=',')
    parser.add_argument('--csv_path', default='../imdb-benchmark')
    parser.add_argument('--hdf_path', default='../imdb-benchmark/gen_hdf')
    parser.add_argument('--max_rows_per_hdf_file', type=int, default=20000000)
    parser.add_argument('--hdf_sample_size', type=int, default=1000000)

    # generate models/ensembles
    parser.add_argument('--generate_models', help='Trains BNs on dataset', action='store_true')
    parser.add_argument('--model_path', default='../imdb-benchmark/bn_ensembles')
    parser.add_argument('--learning_algo', default='chow-liu', help="BN's structure learning algorithm")
    parser.add_argument('--max_parents', type=int, default=1, help="BN's constrain on max number of parents")
    parser.add_argument('--sample_size', type=int, default=200000, help="use a subsample for structure learning")
    parser.add_argument('--n_mcv', type=int, default=30, help="Related to binning large or continuous domain")
    parser.add_argument('--n_bins', type=int, default=60, help="Related to binning large or continuous domain")

    # evaluation
    parser.add_argument('--evaluate_cardinalities', help='Evaluates SPN ensemble to compute cardinalities',
                        action='store_true')
    parser.add_argument('--model_location', nargs='+', default='../imdb-benchmark/bn_ensembles/clt.pkl')
    parser.add_argument('--infer_algo', default='exact', help="BN's structure learning algorithm")
    parser.add_argument('--query_file_location', default='./benchmarks/dmv/sql/cardinality_queries.sql')
    parser.add_argument('--ground_truth_file_location', default=None)



    # log level
    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    #dealing with imdb job
    print(args.dataset)
    if args.dataset == 'imdb' or args.dataset == 'stats':
        print(args.dataset)
        table_csv_path = args.csv_path + '/{}.csv'
        if args.dataset == 'imdb':
            schema = gen_job_light_imdb_schema(table_csv_path)
        elif args.dataset == "stats":
            schema = gen_stats_light_schema(table_csv_path)
        else:
            schema = gen_DB0_schema(table_csv_path)
        # generate hdf file the way deepdb does
        if args.generate_hdf:
            logger.info(f"Generating HDF files for tables in {args.csv_path} and store to path {args.hdf_path}")

            if os.path.exists(args.hdf_path):
                logger.info(f"Removing target path {args.hdf_path}")
                shutil.rmtree(args.hdf_path)

            logger.info(f"Making target path {args.hdf_path}")
            os.makedirs(args.hdf_path)

            prepare_all_tables(schema, args.hdf_path, csv_seperator=args.csv_seperator,
                               max_table_data=args.max_rows_per_hdf_file)
            logger.info(f"Files successfully created")

        # Generate sampled HDF files for fast join calculations
        elif args.generate_sampled_hdfs:
            logger.info(
                f"Generating sampled HDF files for tables in {args.csv_path} and store to path {args.hdf_path}")
            prepare_sample_hdf(schema, args.hdf_path, args.max_rows_per_hdf_file, args.hdf_sample_size)
            logger.info(f"Files successfully created")

        elif args.generate_models:
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            if args.dataset == 'imdb':
                train_imdb(schema, args.hdf_path, args.model_path, args.learning_algo, args.max_parents, args.sample_size)
            

        elif args.evaluate_cardinalities:
            if args.dataset == 'imdb':
                evaluate_cardinality_imdb(schema, args.model_path, args.query_file_location, args.infer_algo,
                                          args.learning_algo, args.max_parents)

    elif args.dataset == 'dmv':
        if args.generate_models:
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            train_DMV(args.csv_path, args.model_path, args.learning_algo, args.max_parents, args.sample_size)

        elif args.evaluate_cardinalities:
            evaluate_cardinality_single_table(args.model_path, args.query_file_location, args.infer_algo)

    elif args.dataset == 'census':
        if args.generate_models:
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            train_Census(args.csv_path, args.model_path, args.learning_algo, args.max_parents, args.sample_size)

        elif args.evaluate_cardinalities:
            evaluate_cardinality_single_table(args.model_path, args.query_file_location, args.infer_algo)

