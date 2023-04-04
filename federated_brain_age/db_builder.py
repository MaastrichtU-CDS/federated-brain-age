from federated_brain_age.constants import *
from federated_brain_age.utils import run_sql

def get_model_by_id(model_id, pg, table=MODELS_TABLE):
    """ Retrieve the person id from the source id.
    """
    return run_sql(
        pg,
        f"SELECT * FROM {table} WHERE model_id='{model_id}' LIMIT 1",
        fetch_one=True,
    )

def get_last_run_by_id(model_id, pg, table=RUNS_TABLE):
    """ Retrieve the person id from the source id.
    """
    return run_sql(
        pg,
        f"SELECT * FROM {table} WHERE model_id='{model_id}' ORDER BY date DESC LIMIT 1",
        fetch_one=True,
    )

def get_run_by_id_round(model_id, round_number, pg, table=RUNS_TABLE):
    """ Retrieve the person id from the source id.
    """
    return run_sql(
        pg,
        f"SELECT * FROM {table} WHERE model_id='{model_id}' AND round='{round_number}' ORDER BY date DESC LIMIT 1",
        fetch_one=True,
    )

def insert_model(model_id, seed, data_split, pg, table=MODELS_TABLE):
    """ Insert a new model.
    """
    return run_sql(
        pg,
        f"INSERT INTO {table} (model_id, date, seed, data_split) VALUES (%s, current_timestamp, %s, %s)",
        parameters=(model_id, seed, data_split),
    )

def insert_run(model_id, round, weights, mae, mse, metrics, pg, table=RUNS_TABLE):
    """ Insert a new model.
    """
    return run_sql(
        pg,
        f"INSERT INTO {table} (model_id, date, round, weights, mae, mse, metrics) " + 
            "VALUES (%s, current_timestamp, %s, %s, %s, %s, %s) RETURNING run_id",
        parameters=(model_id, round, weights, mae, mse, metrics),
        fetch_one=True,
    )

def update_run(model_id, run_id, mae, mse, val_mae, val_mse, metrics, predictions, age_gap, pg, table=RUNS_TABLE):
    """ Insert a new model.
    """
    return run_sql(
        pg,
        f"UPDATE {table} SET mae = %s, mse = %s, val_mae = %s, val_mse = %s, metrics = %s, predictions = %s, age_gap = %s " +
            f"WHERE model_id = '{model_id}' AND run_id = {run_id}",
        parameters=(mae, mse, val_mae, val_mse, metrics, predictions, age_gap),
    )
