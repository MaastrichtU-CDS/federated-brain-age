""" Handle the master and node tasks to predict the brain age
"""
import json
import numpy as np
import os
import random

from vantage6.tools.util import warn, info

from federated_brain_age.brain_age import BrainAge
from federated_brain_age.constants import *
from federated_brain_age.data_loader import DataLoader
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

def task_predict_locally(parameters, weights, data_seed, seed, data_split, db_client):
    output = {
        PREDICTIONS: {},
        METRICS: {},
        AGE_GAP: {},
    }
    try:
        # Initialize the model
        brain_age = BrainAge(
            parameters,
            parameters[MODEL_ID],
            os.getenv(IMAGES_FOLDER),
            parameters[DB_TYPE],
            db_client if parameters[DB_TYPE] != DB_CSV else os.getenv(DATA_FOLDER) + "/dataset.csv",
            # parameters[TRAINING_IDS],
            # parameters[VALIDATION_IDS],
            seed=data_seed,
            split=data_split,
        )
        participants_by_subset = {
            TRAIN: brain_age.train_loader.participant_list,
            VALIDATION: brain_age.train_loader.participant_list,
        }
        for subset, subset_participants in participants_by_subset.items():
            if len(subset_participants[1]) > 0:
                warn(f"{len(subset_participants[1])} of {subset} participants with incomplete \
                    information: {', '.join(subset_participants[1])}")
            if len(subset_participants[2]) > 0:
                warn(f"{len(subset_participants[2])} of {subset} participants without imaging '\
                    data available: {', '.join(subset_participants[2])}")
        if weights:
            # Set the initial weights if available
            parsed_weights = [
                np.array(weights_by_layer, dtype=np.double) for weights_by_layer in json.loads(weights)
            ]
            brain_age.model.set_weights(parsed_weights)
        # Predict
        info("Predict")
        data_loader = None
        # More generically it could be a "Dataset" field
        if IS_TRAINING_DATA in parameters and not parameters[IS_TRAINING_DATA]:
            data_loader = DataLoader(
                brain_age.images_path,
                parameters[DB_TYPE],
                db_client if parameters[DB_TYPE] != DB_CSV else os.getenv(DATA_FOLDER) + "/dataset.csv",
                training=False,
                validation=False,
                seed=data_seed,
                split=1,
            )
            output[DATASET] = [TEST]
            output[SAMPLE_SIZE] = len(data_loader.participants)
            if data_loader and len(data_loader.participants) > 0:
                output[PREDICTIONS] = brain_age.predict({
                    TEST: data_loader
                })
                metrics = [brain_age.get_metrics(
                    data_loader,
                    list(output[PREDICTIONS][TEST].values()),
                )]
                output[METRICS] = [{
                    key: [metric[key] for metric in metrics if key in metric] for key in [
                        MAE, MSE, SDAE, SDSE
                    ]
                }]
                output[AGE_GAP] = metrics[0].get(AGE_GAP, [])
            else:
                raise Exception("No participants found for the prediction dataset requested")
        else:
            output[DATASET] = [TRAIN, VALIDATION]
            output[SAMPLE_SIZE] = [
                len(brain_age.train_loader.participants), len(brain_age.validation_loader.participants)
            ]
            output[PREDICTIONS] = brain_age.predict()
            metrics = [
                brain_age.get_metrics(
                    brain_age.train_loader,
                    list(output[PREDICTIONS][TRAIN].values()),
                ),
                brain_age.get_metrics(
                    brain_age.validation_loader,
                    list(output[PREDICTIONS][VALIDATION].values()),
                    prefix="val_",
                ),
            ]
            output[METRICS] = [{
                key: [metric[key] for metric in metrics if key in metric] for key in [
                    MAE, MSE, SDAE, SDSE, VAL_MAE, VAL_MSE, VAL_SDAE, VAL_SDSE,
                ]
            }]
            output[AGE_GAP] = metrics[0].get(AGE_GAP, {})
            output[VAL_AGE_GAP] = metrics[1].get(VAL_AGE_GAP, {}),
    except Exception as error:
        message = f"Error while predicting: {str(error)}"
        warn(message)
        output[ERROR] = message
    return output
