""" Handle the master and node tasks to predict the brain age
"""
import json
import random

from vantage6.tools.util import warn, info

from federated_brain_age.constants import *
from federated_brain_age.db_builder import get_last_run_by_id, get_run_by_id_round, get_model_by_id
from federated_brain_age.server_handler import execute_task, get_result
from federated_brain_age.utils import np_array_to_list

def predict(parameters, ids, algorithm_image, db_client, client):
    """ Send the task to predict the brain age
    """
    # Retrieve the necessary data from the database
    model_info = {
        ID: None,
        SEED: None,
        ROUND: 0,
        WEIGHTS: None,
        DATA_SPLIT: None,
    }
    info("Request to save the model")
    model_info[ID] = parameters[MODEL_ID]
    try:
        result = get_model_by_id(parameters[MODEL_ID], db_client)
        if result:
            info("Get existing model")
            model_info[SEED] = result[2]
            model_info[DATA_SPLIT] = result[3]
            last_run = get_run_by_id_round(parameters[MODEL_ID], parameters[ROUND], db_client) if \
                ROUND in parameters else get_last_run_by_id(parameters[MODEL_ID], db_client)
            if last_run:
                model_info[ROUND] = last_run[3]
                model_info[WEIGHTS] = last_run[4]
        else:
            error_message = f"Unable to find the model with ID: {str(parameters[MODEL_ID])}"
            warn(error_message)
            return {
                ERROR: error_message
            }
    except Exception as error:
        error_message = f"Unable to connect to the database and retrieve the model: {str(error)}"
        warn(error_message)
        return {
            ERROR: error_message
        }
    # Set the seed value
    info(f"Using {model_info[SEED]} as the seed")
    random.seed(model_info[SEED])
    # Set 3 constant seeds for the training/validation split
    data_seeds = [random.randint(0, 1000000) for j in range(len(ids))]
    # Send the tasks
    tasks = {}
    for org_num, id in enumerate(ids):
        input = {
            "method": PREDICT,
            "args": [],
                "kwargs": {
                    "parameters": {
                        ROUNDS: model_info[ROUND],
                        HISTORY: parameters.get(HISTORY),
                        MODEL_ID: parameters.get(MODEL_ID),
                        DB_TYPE: parameters.get(DB_TYPE),
                        IS_TRAINING_DATA: parameters.get(IS_TRAINING_DATA),
                    },
                    WEIGHTS: json.dumps(np_array_to_list(model_info[WEIGHTS])),
                    DATA_SEED: data_seeds[org_num],
                    SEED: data_seeds[org_num],
                    DATA_SPLIT: 1 if int(parameters.get(DATA_SPLIT)) == 1 else model_info[DATA_SPLIT],
                }
        }
        task_id = execute_task(client, input, [id], algorithm_image)
        tasks[task_id] = {
            ORGANIZATION_ID: id
        }
    output = get_result(
        client,
        tasks,
        max_number_tries=parameters.get(MAX_NUMBER_TRIES) or DEFAULT_MAX_NUMBER_TRIES,
        sleep_time=parameters.get(SLEEP_TIME) or DEFAULT_SLEEP_TIME,
    )
    return {
        PREDICT: [{
            ORGANIZATION_ID: tasks[key],
            RESULT: result,
            } for key, result in output.items()]
        }
