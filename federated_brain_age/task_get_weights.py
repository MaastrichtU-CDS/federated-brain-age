""" Handle the task to get model weights by id and round (master only)
"""
import json

from vantage6.tools.util import warn, info

from federated_brain_age.constants import *
from federated_brain_age.db_builder import get_model_by_id, get_last_run_by_id, get_run_by_id_round
from federated_brain_age.utils import np_array_to_list

def get_weights(parameters, db_client):
    """ Get model weights by id and round
    """
    # Retrieve the necessary data from the database
    model_info = {
        ID: None,
        SEED: None,
        ROUND: 0,
        WEIGHTS: None,
        DATA_SPLIT: None,
    }
    info("Get model")
    model_info[ID] = parameters[MODEL_ID]
    try:
        result = get_model_by_id(parameters[MODEL_ID], db_client)
        if result:
            info("Model found")
            model_info[SEED] = result[2]
            model_info[DATA_SPLIT] = result[3]
            last_run = get_run_by_id_round(parameters[MODEL_ID], parameters[ROUND], db_client) if \
                ROUND in parameters else get_last_run_by_id(parameters[MODEL_ID], db_client)
            if last_run:
                info("Run found: parsing the weights")
                model_info[ROUND] = last_run[3]
                model_info[WEIGHTS] = json.dumps(np_array_to_list(last_run[4]))
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
    return model_info[WEIGHTS]
