import json
import time
import os
import random

import numpy as np
import tensorflow as tf
from vantage6.tools.util import warn, info

from federated_brain_age.brain_age import BrainAge
from federated_brain_age.constants import *
from federated_brain_age.utils import *
from federated_brain_age.data_loader import DataLoader
from federated_brain_age.db_builder import *
from federated_brain_age.task_get_weights import get_weights
from federated_brain_age.task_check import check_centers
from federated_brain_age.server_handler import get_orgarnization
from federated_brain_age.task_predict import predict
from federated_brain_age.task_train import task_train

def master(client, db_client, parameters = None, org_ids = None, algorithm_image = None):
    """
    Master algorithm to compute the brain age CNN of the federated datasets.

    Parameters
    ----------
    client : ContainerClient
        Interface to the central server. This is supplied by the wrapper.
    db_client : DBClient
        The database client.
    parameters: Dict
        Explicitly provide the parameters to be used in the CNN.

    Returns
    -------
    Dict
        A dictionary containing the model weights.
    """
    # Validating the input
    info("Validating the input arguments")
    if TASK not in parameters:
        return {
            ERROR: f"Missing the following parameter: {TASK}"
        }

    # Get the organizations in the collaboration
    orgs = get_orgarnization(client, org_ids)
    ids = [org.get("id") for org in orgs]
    info(f"Sending the algorithm to the following organizations: {', '.join(str(id) for id in ids)}")

    # Check which task has been requested
    info(f"Task requested: {parameters[TASK]}")
    if parameters[TASK] == GET_MODEL:
        missing_parameters = validate_parameters(parameters, {
            DB_TYPE: {},
            MODEL_ID: {}
        })
        if len(missing_parameters) > 0:
            return parse_error(
                f"Missing the following parameters: {', '.join(missing_parameters)}"
            )
        return get_weights(parameters, db_client)
    elif parameters[TASK] == CHECK:
        missing_parameters = validate_parameters(parameters, {DB_TYPE: {}})
        if len(missing_parameters) > 0:
            return parse_error(
                f"Missing the following parameters: {', '.join(missing_parameters)}"
            )
        return check_centers(parameters,ids, algorithm_image, client)
    elif parameters[TASK] == PREDICT:
        missing_parameters = validate_parameters(parameters, {
            DB_TYPE: {},
            MODEL_ID: {}
        })
        if len(missing_parameters) > 0:
            return parse_error(
                f"Missing the following parameters: {', '.join(missing_parameters)}"
            )
        return predict(parameters, ids, algorithm_image, db_client, client)
    elif parameters[TASK] == TRAIN:
        missing_parameters = validate_parameters(parameters, {
            MODEL: {
                MASTER: {
                    ROUNDS: {},
                },
                NODE: {
                    EPOCHS: {},
                    USE_MASK: {},
                },
                DATA_SPLIT: {},
            },
            DB_TYPE: {},
        })
        if len(missing_parameters) > 0:
            return parse_error(
                f"Missing the following parameters: {', '.join(missing_parameters)}"
            )
        return task_train(parameters, ids, algorithm_image, db_client, client)

    info("Task not recognized")
    return {
        ERROR: "Task not recognized"
    }

def RPC_check(db_client, parameters, org_ids = None):
    """
    Check the status of the different components:
    - Cluster by successfully running this task;
    - XNAT connection;

    Parameters
    ----------
    parameters : Dict
        Task parameters.

    Returns
    -------
    Dict
        Information regarding the connection to the XNAT.
    """
    info("Check components - Node method")
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

def RPC_brain_age(db_client, parameters, weights, data_seed, seed, data_split):
    """
    Run the CNN to compute the brain age

    Parameters
    ----------
    db_client : DBClient
        The database client.
    parameters : Dict
        Explicitly provide the parameters to be used in the CNN.

    Returns
    -------
    Dict
        A Dict containing the CNN weights.
    """
    info("Brain age CNN - Node method - Training")
    output = {
        METRICS: {},
        HISTORY: {},
        PREDICTIONS: {},
        AGE_GAP: {},
    }
    # Retrieve the data from XNAT if necessary
    # data_path = os.path.join(os.getenv(DATA_FOLDER), parameters[MODEL_ID])
    # if ...:
    #     # Check if the folder exists and if the data is already there
    #     # folder_exists(data_path)
    # else:
    #     if os.getenv(XNAT_URL):
    #         retrieve_data(data_path)
    #     else:
    #         return {
    #             ERROR: ""
    #         }
    try:
        # Initialize the model
        info("Initialize")
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
        info("Check participants")
        participants_by_subset = {
            TRAIN: brain_age.train_loader.participant_list,
            VALIDATION: brain_age.validation_loader.participant_list,
        }
        for subset, subset_participants in participants_by_subset.items():
            if len(subset_participants[1]) > 0:
                warn(f"{str(len(subset_participants[1]))} of {subset} participants with incomplete " +
                    f"information: {', '.join([str(participant) for participant in subset_participants[1]])}")
            if len(subset_participants[2]) > 0:
                warn(f"{str(len(subset_participants[2]))} of {subset} participants without imaging " +
                    f"data available: {', '.join([str(participant) for participant in subset_participants[2]])}")
            if len(subset_participants[3]) > 0:
                warn(f"{str(len(subset_participants[3]))} of {subset} participants with duplicate information " +
                    f"data available: {', '.join([str(participant) for participant in subset_participants[3]])}")
        output[SAMPLE_SIZE] = [
            len(brain_age.train_loader.participants), len(brain_age.validation_loader.participants)
        ]
        if len(brain_age.train_loader.participants) > 0:
            if weights:
                # Set the initial weights if available
                parsed_weights = [
                    np.array(weights_by_layer, dtype=np.double) for weights_by_layer in json.loads(weights)
                ]
                brain_age.model.set_weights(parsed_weights)
            info("Predictions with the aggregated network")
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
            output[AGE_GAP] = {
                AGE_GAP: metrics[0].get(AGE_GAP, []),
                VAL_AGE_GAP: metrics[1].get(VAL_AGE_GAP, []),
            }
            info("Training the network")
            # Set the random seed
            random.seed(seed)
            # Train the model - history is necessary for model selection
            history = parameters.get(HISTORY)
            model_selection = parameters.get(MODEL_SELECTION)
            result = brain_age.train(
                history=history or model_selection,
                class_weight=parameters.get(CLASS_WEIGHTS),
                save_model=parameters.get(SAVE_MODEL),
                complete_metrics=parameters.get(COMPLETE_METRICS, True),
            )
            # Retrieve the weights, metrics for the first and last epoch, and the 
            # history if requested
            info("Retrieve the results")
            if model_selection:
                info("Model selection requested")
                output[WEIGHTS] = json.dumps(np_array_to_list(brain_age.history.best_model))
            else:
                output[WEIGHTS] = json.dumps(np_array_to_list(brain_age.model.get_weights()))
            # Calculate the metrics
            if history:
                epoch = brain_age.history.best_epoch if model_selection else -1
                metrics.extend([
                    {
                        MAE: brain_age.history.train_metrics[MAE][epoch],
                        MSE: brain_age.history.train_metrics[MSE][epoch],
                        SDAE: brain_age.history.train_metrics[SDAE][epoch],
                        SDSE: brain_age.history.train_metrics[SDSE][epoch],
                    },
                    {
                        VAL_MAE: brain_age.history.val_metrics[MAE][epoch],
                        VAL_MSE: brain_age.history.val_metrics[MSE][epoch],
                        VAL_SDAE: brain_age.history.val_metrics[SDAE][epoch],
                        VAL_SDSE: brain_age.history.val_metrics[SDSE][epoch],
                    },
                ])
            else:
                local_predictions = brain_age.predict()
                metrics.extend([
                    brain_age.get_metrics(
                        brain_age.train_loader,
                        list(local_predictions[TRAIN].values()),
                    ),
                    brain_age.get_metrics(
                        brain_age.validation_loader,
                        list(local_predictions[VALIDATION].values()),
                        prefix="val_",
                    ),
                ])
            output[METRICS] = {
                key: [metric[key] for metric in metrics if key in metric] for key in [
                    MAE, MSE, SDAE, SDSE, VAL_MAE, VAL_MSE, VAL_SDAE, VAL_SDSE,
                ]
            }
            # Metrics from the augmented data
            # for metric in result.history.keys():
            #     output[METRICS][metric] = [result.history[metric][0], result.history[metric][-1]]
            if history:
                # Tensorflow history is an average of the results by batch
                # except for the validation metrics (result.history[VAL_MSE])
                # output[HISTORY] = result.history
                output[HISTORY] = {
                    MAE: brain_age.history.train_metrics[MAE],
                    MSE: brain_age.history.train_metrics[MSE],
                    VAL_MAE: brain_age.history.val_metrics[MAE],
                    VAL_MSE: brain_age.history.val_metrics[MSE],
                }
        else:
            raise Exception("No participants found for the training set")
    except Exception as error:
       message = f"Error while training the model: {str(error)}"
       warn(message)
       output[ERROR] = message
    return output

def RPC_predict(db_client, parameters, weights, data_seed, seed, data_split):
    """ Predict the brain age
    """
    info("Brain age CNN - Node method - Prediction")
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
