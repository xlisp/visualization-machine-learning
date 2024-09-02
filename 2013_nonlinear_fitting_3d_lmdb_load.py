
def load_from_lmdb(model, lmdb_path):
    # Open the LMDB environment
    env = lmdb.open(lmdb_path)
    with env.begin() as txn:
        # Retrieve the serialized model
        model_bytes = txn.get(b'model_parameters')
        
        # Deserialize the model parameters
        model_parameters = pickle.loads(model_bytes)
        
        # Load the parameters into the model
        model.load_state_dict(model_parameters)
    env.close()
    print(f'Model parameters loaded from {lmdb_path}')

# Load the model parameters from the LMDB database
load_from_lmdb(model, lmdb_path)

# $ tree model_parameters.lmdb/
#  88K	model_parameters.lmdb/
#model_parameters.lmdb/
#├── data.mdb
#└── lock.mdb

