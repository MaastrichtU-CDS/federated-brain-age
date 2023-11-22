""" Handle the master and node tasks to check the settings
"""
import os

import tensorflow as tf
from vantage6.tools.util import warn

from federated_brain_age.constants import *
from federated_brain_age.db_builder import get_model_by_id
from federated_brain_age.server_handler import execute_task, get_result

def check_centers(parameters, ids, algorithm_image, client):
    # Send the tasks
    tasks = {}
    for id in ids:
        input = {
            "method": CHECK,
            "args": [],
            "kwargs": {
                "parameters": {
                    **parameters,
                },
            },
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
        CHECK: [{
            ORGANIZATION_ID: tasks[key],
            RESULT: result,
            } for key, result in output.items()]
        }

def check_center(parameters, db_client):
    output = {}
    # Check the number of scans actually available
    output[IMAGES_FOLDER] = {}
    images_path = os.getenv(IMAGES_FOLDER)
    if images_path and os.path.isdir(images_path):
        scans = [f for f in os.path.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        output[IMAGES_FOLDER][NUMBER_SCANS] = len(scans)
        # missing = []
        # if TRAINING_IDS in parameters or VALIDATION_IDS in parameters:
        #     for id in (parameters.get(TRAINING_IDS) or []) + (parameters.get(VALIDATION_IDS) or []):
        #         if id not in scans:
        #             missing.append[id]
            
    else:
        output[IMAGES_FOLDER][MESSAGE] = "Image folder not found"

    # Check if the clinical data is available
    if parameters[DB_TYPE] == DB_CSV:
        output[DB_CSV] = os.path.isfile(os.getenv(DATA_FOLDER) or "" + "/dataset.csv")
    elif parameters[DB_TYPE] == DB_POSTGRES:
        output[DB_POSTGRES] = False
        #if os.getenv(DB_CNN_DATABASE):
        #    db_cnn_client = PostgresManager(default_db=False, db_env_var=DB_CNN_DATABASE)
        #else:
        if db_client:    
            output[DB_POSTGRES] = True
        else:
            warn("Error connecting to the database")        

    # Check if task ID already exists
    if MODEL_ID in parameters and db_client:
        result = get_model_by_id(parameters[MODEL_ID], MODELS_TABLE, db_client)
        output[MODEL_ID] = result is not None

    # Check the connection to the XNAT
    # if os.getenv(XNAT_URL):
    #     pass

    # Check the GPU availability
    output[GPU_COUNT] = len(tf.config.list_physical_devices('GPU'))
    return output
