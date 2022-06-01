import os
import importlib

from unittest.mock import Mock, patch

from federated_brain_age.constants import *

DATA_PATH = "/mnt"

PARAMETERS = {
    TASK: TRAIN,
    TASK_ID: "test",
    DB_TYPE: DB_CSV,
    MAX_NUMBER_TRIES: 10,
    MODEL: {
        MASTER: {
            ROUNDS: 2
        },
        NODE: {
            USE_MASK: True,
            EPOCHS: 1,
            TRAINING_IDS: {
                1: ["2a", "3a"],
                # 2: ["2a", "3a"]
            },
            VALIDATION_IDS: {
                1: ["4a"],
                # 2: ["4a"]
            },
            #"TESTING_IDS": [[]]
        }
    }
}

lib = importlib.import_module("federated_brain_age")

def mock_execute_task(client, parameters, org_ids):
    print(f"Mock: execute task for the organizations {org_ids}")
    return org_ids[0]

def mock_get_orgarnization(client):
    print("Mock: get organizations")
    return [
        {
            'id': 1,
        },
        # {
        #     'id': 2,
        # }
    ]

def mock_get_result(client, task_id, max_number_tries):
    print(f"Mock: get result for task {task_id} within {max_number_tries} tries")
    brain_age_method = getattr(lib, "RPC_brain_age")
    return [brain_age_method(
        None,
        {
            **PARAMETERS[MODEL][NODE],
            TRAINING_IDS: PARAMETERS[MODEL][NODE][TRAINING_IDS][1],
            VALIDATION_IDS: PARAMETERS[MODEL][NODE][VALIDATION_IDS][1],
            TASK_ID: PARAMETERS[TASK_ID],
            DB_TYPE: PARAMETERS[DB_TYPE]
        },
        None
    )]

@patch("federated_brain_age.execute_task", wraps=mock_execute_task)
@patch("federated_brain_age.get_orgarnization", wraps=mock_get_orgarnization)
@patch("federated_brain_age.get_result", wraps=mock_get_result)
@patch.dict(os.environ, {DATA_FOLDER: DATA_PATH})
def test_master_train(mock_bar, mock_bar2, mock_bar3):
    # with patch("federated_brain_age.execute_task", wraps=mock_f) as mock_bar:
    master_method = getattr(lib, "master")
    result = master_method(None, None, PARAMETERS)

if __name__ == '__main__':
    test_master_train()
