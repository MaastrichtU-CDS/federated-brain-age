import os
import importlib

from unittest.mock import Mock, patch

from federated_brain_age.constants import *

DATA_PATH = "/mnt"

PARAMETERS = {
    TASK: TRAIN,
    MODEL_ID: "test",
    SAVE_MODEL: False,
    DB_TYPE: DB_CSV,
    MAX_NUMBER_TRIES: 10,
    SLEEP_TIME: 5,
    HISTORY: False,
    MODEL: {
        MASTER: {
            ROUNDS: 2
        },
        NODE: {
            USE_MASK: True,
            EPOCHS: 2,
            # TRAINING_IDS: {
            #     1: ["2a", "3a"],
            #     # 2: ["2a", "3a"]
            # },
            # VALIDATION_IDS: {
            #     1: ["4a"],
            #     # 2: ["4a"]
            # },
            # #"TESTING_IDS": [[]]
        },
        DATA_SPLIT: 0.8,
    }
}
round_count = 0

lib = importlib.import_module("federated_brain_age")

def mock_execute_task(client, parameters, org_ids, algorithm_image = None):
    print(f"Mock: execute task for the organizations {org_ids}")
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
    for task_id in tasks.keys():
        # db_client, parameters, weights, data_seed, seed, data_split
        output[task_id] = brain_age_method(
            None,
            {
                **PARAMETERS[MODEL][NODE],
                MODEL_ID: PARAMETERS[MODEL_ID],
                DB_TYPE: PARAMETERS[DB_TYPE],
                ROUNDS: round_count,
                HISTORY: PARAMETERS[HISTORY],
                DATA_SPLIT: PARAMETERS[MODEL][DATA_SPLIT]
            },
            None,
            1,
            1,
            0.7,
        )
    return output

@patch("federated_brain_age.execute_task", wraps=mock_execute_task)
@patch("federated_brain_age.get_orgarnization", wraps=mock_get_orgarnization)
@patch("federated_brain_age.get_result", wraps=mock_get_result)
@patch.dict(os.environ, {DATA_FOLDER: DATA_PATH})
def test_master_train(mock_bar, mock_bar2, mock_bar3):
    # with patch("federated_brain_age.execute_task", wraps=mock_f) as mock_bar:
    master_method = getattr(lib, "master")
    result = master_method(None, None, PARAMETERS)
    for key in result.keys():
        if key != WEIGHTS:
            print(key)
            print(result[key])

if __name__ == '__main__':
    test_master_train()
