# Local run to test the algorithm using a local RDB to store the models
# and results.

import os
import importlib
from unittest.mock import Mock, patch

import psycopg2

from federated_brain_age.constants import *

DATA_PATH = "/mnt"

PARAMETERS = {
    TASK: TRAIN,
    MODEL_ID: "test21",
    SAVE_MODEL: True,
    DB_TYPE: DB_CSV,
    MAX_NUMBER_TRIES: 10,
    SLEEP_TIME: 5,
    HISTORY: True,
    MODEL: {
        MASTER: {
            ROUNDS: 4
        },
        NODE: {
            USE_MASK: True,
            EPOCHS: 3,
            PATIENTS_PER_EPOCH: 4,
            BATCH_SIZE: 4,
        },
        DATA_SPLIT: 0.7,
    }
}
round_count = 0
input = None

lib = importlib.import_module("federated_brain_age")

def mock_execute_task(client, parameters, org_ids, algorithm_image = None):
    print(f"Mock: execute task for the organizations {org_ids}")
    global input
    input = parameters["kwargs"]
    return org_ids[0]

def mock_get_orgarnization(client, org_ids):
    print("Mock: get organizations")
    return [
        {
            'id': 1,
        },
        # {
        #     'id': 2,
        # }
    ]

def mock_get_result(client, tasks, max_number_tries, sleep_time):
    print(f"Mock: get result for tasks {tasks.keys()} within {max_number_tries} tries")
    brain_age_method = getattr(lib, "RPC_brain_age")
    output = {}
    global round_count
    round_count += 1
    global input
    for task_id in tasks.keys():
        output[task_id] = brain_age_method(
            client,
            input["parameters"],
            input[WEIGHTS],
            input[DATA_SEED],
            input[SEED],
            input[DATA_SPLIT],
        )
    return output

@patch("federated_brain_age.execute_task", wraps=mock_execute_task)
@patch("federated_brain_age.get_orgarnization", wraps=mock_get_orgarnization)
@patch("federated_brain_age.get_result", wraps=mock_get_result)
@patch.dict(os.environ, {DATA_FOLDER: DATA_PATH})
def test_master_train(mock_bar, mock_bar2, mock_bar3):
    # with patch("federated_brain_age.execute_task", wraps=mock_f) as mock_bar:
    master_method = getattr(lib, "master")
    # Connection to the local postgres database running on your laptop
    # (not in a docker container, in that case, replace host.docker.internal with the
    # container's address)
    # DB - database name
    connection = psycopg2.connect("postgresql://user:password@host.docker.internal:5432/DB")
    db_client = connection.cursor()
    #try:
    result = master_method(None, db_client, PARAMETERS)
    for key in result.keys():
        if key != WEIGHTS:
            print(key)
            print(result[key])
    #except Exception as error:
    #    print("Database unavailable")
    #    print(error)
    db_client.close()
    connection.close()

if __name__ == '__main__':
    test_master_train()
