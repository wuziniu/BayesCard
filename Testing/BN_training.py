import pandas as pd
import pickle
from DataPrepare.join_data_preparation import JoinDataPreparator
from Models.Bayescard_BN import Bayescard_BN, build_meta_info

def train_DMV(csv_path, model_path, algorithm, max_parents, sample_size):
    data = pd.read_csv(csv_path)
    new_cols = []
    #removing unuseful columns
    for col in data.columns:
        if col in ['VIN', 'Zip', 'City', 'Make', 'Unladen Weight', 'Maximum Gross Weight', 'Passengers',
                   'Reg Valid Date', 'Reg Expiration Date', 'Color']:
            data = data.drop(col, axis=1)
        else:
            new_cols.append(col.replace(" ", "_"))
    data.columns = new_cols
    BN = Bayescard_BN('dmv')
    BN.build_from_data(data, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'], sample_size=sample_size)
    model_path += f"/{algorithm}_{max_parents}.pkl"
    pickle.dump(BN, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"model saved at {model_path}")
    return None

def train_Census(csv_path, model_path, algorithm, max_parents, sample_size):
    df = pd.read_csv(csv_path, header=0, sep=",")
    df = df.drop("caseid", axis=1)
    df = df.dropna(axis=0)
    BN = Bayescard_BN('Census')
    BN.build_from_data(df, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'], sample_size=sample_size)
    model_path += f"/{algorithm}_{max_parents}.pkl"
    pickle.dump(BN, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"model saved at {model_path}")
    return None

def train_imdb(schema, hdf_path, model_folder, algorithm, max_parents, sample_size):
    meta_data_path = hdf_path + '/meta_data.pkl'
    prep = JoinDataPreparator(meta_data_path, schema, max_table_data=20000000)
    print(f"BN will be trained on the full outer join of following relations")
    for relationship_obj in schema.relationships:
        print(relationship_obj.identifier)

    for i, relationship_obj in enumerate(schema.relationships):
        print("training on relationship_obj.identifier")
        df_sample_size = 10000000
        relation = [relationship_obj.identifier]
        df, meta_types, null_values, full_join_est = prep.generate_n_samples(
            df_sample_size, relationship_list=relation, post_sampling_factor=10)
        columns = list(df.columns)
        assert len(columns) == len(meta_types) == len(null_values)
        meta_info = build_meta_info(df.columns, null_values)
        bn = Bayescard_BN(schema, relation, column_names=columns, full_join_size=full_join_est,
                      table_meta_data=prep.table_meta_data, meta_types=meta_types, null_values=null_values,
                      meta_info=meta_info)
        model_path = model_folder + f"/{i}_{algorithm}_{max_parents}.pkl"
        bn.build_from_data(df, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'],
                           sample_size=sample_size)
        pickle.dump(bn, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"model saved at {model_path}")
    return None

