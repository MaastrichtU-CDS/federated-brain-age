import time
import os

import tensorflow as tf
from vantage6.tools.util import warn, info

from federated_brain_age.brain_age import BrainAge, get_parameter, DEFAULT_HYPERPARAMETERS
from federated_brain_age.constants import *
from federated_brain_age.utils import *
from federated_brain_age.xnat_client import retrieve_data
from federated_brain_age.postgres_manager import PostgresManager
from federated_brain_age.db_builder import get_task_by_id

def get_orgarnization(client):
    # obtain organizations that are within the collaboration
    info("Obtaining the organizations in the collaboration")
    organizations = client.get_organizations_in_my_collaboration()
    # ids = [organization.get("id") for organization in organizations]
    return organizations 

def execute_task(client, input, org_ids):
    # collaboration and image are stored in the key, so we do not need
    # to specify these
    info("Creating node tasks")
    task = client.create_new_task(
        input,
        organization_ids=org_ids
    )
    return task.get("id")

def get_result(client, tasks, max_number_tries=DEFAULT_MAX_NUMBER_TRIES):
    # Get the task's result
    results = {}
    tries = 0
    while len(results.keys()) != len(tasks) or tries > max_number_tries:
        # TODO: Set a better value for the timer
        time.sleep(60)
        for task_id in tasks.keys():
            if task_id not in results:
                info("Waiting for results")
                task = client.request(f"task/{task_id}")
                if task.get("complete"):
                    info("Obtaining results")
                    results[task_id] = client.get_results(task_id=task.get("id"))
    return results

def master(client, db_client, parameters = None):
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
    # TODO
    # DB_CSV

    # Get the organizations in the collaboration
    orgs = get_orgarnization(client)
    ids = [org.get("id") for org in orgs]

    # Check which task has been requested
    info(f"Task requested: {parameters[TASK]}")
    if parameters[TASK] == CHECK:
        tasks = {}
        for id in ids:
            input = {
                "method": CHECK,
                "args": [],
                "kwargs": {
                    "parameters": {
                        **parameters,
                        TRAINING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                        VALIDATION_IDS: parameters[MODEL][NODE][VALIDATION_IDS][id],
                        # TESTING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                    },
                },
            }
            task_id = execute_task(client, input, [id])
            tasks[task_id] = {
                ORGANIZATION_ID: id
            }
        output = get_result(
            client, tasks, max_number_tries=parameters.get(MAX_NUMBER_TRIES) or DEFAULT_MAX_NUMBER_TRIES
        )
        return [{
            ORGANIZATION_ID: tasks[key],
            RESULT: result,
        } for key, result in output.items()]
    elif parameters[TASK] == TRAIN:
        # Intialize the model
        learning_rate = parameters.get(LEARNING_RATE, 1)
        model_parameters = dict(DEFAULT_HYPERPARAMETERS)
        #model_parameters[INPUT_SHAPE] = parameters[INPUT_SHAPE]
        brain_age_model = BrainAge.cnn_model(model_parameters.get)
        brain_age_weights = brain_age_model.get_weights()
        results = {
            METRICS: {}
        }
        # Execute the training
        for i in range(0, parameters[MODEL][MASTER][ROUNDS]):
            info(f"Round {i}/{parameters[MODEL][MASTER][ROUNDS]}")
            tasks = {}
            for id in ids:
                input = {
                    "method": ALGORITHM,
                    "args": [],
                    "kwargs": {
                        "parameters": {
                            **parameters[MODEL][NODE],
                            TRAINING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                            VALIDATION_IDS: parameters[MODEL][NODE][VALIDATION_IDS][id],
                            # TESTING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                        },
                        "weights": brain_age_weights,
                    }
                }
                task_id = execute_task(client, input, [id])
                tasks[task_id] = {
                    ORGANIZATION_ID: id
                }
            output = get_result(
                client, tasks, max_number_tries=parameters.get(MAX_NUMBER_TRIES) or DEFAULT_MAX_NUMBER_TRIES
            )
            # TODO: Validate if there are errors in the results:
            # Log the information and terminate the training
            info("Check if any exception occurred")
            output_data = list(output.values())
            errors = check_errors(output_data)
            if errors:
                warn("Encountered an error, please review the parameters")
                return errors

            info("Aggregating the results")
            metrics = {}
            # Collect the metrics by organization
            for task_id, result in output.items():
                metrics[task_id] = {
                    ORGANIZATION_ID: tasks[task_id][ORGANIZATION_ID],
                    HISTORY: result[HISTORY],
                    METRICS: result[METRICS],
                }
            results[METRICS][i] = metrics
            # Update the model weights
            total_samples = sum([result[SAMPLES] for result in output_data])
            brain_age_weights_update = []
            for i in range(0, len(output_data[0][WEIGHTS])):
                    brain_age_weights_update.append(
                        tf.math.reduce_sum([
                            result[WEIGHTS][i] * result[SAMPLES] / total_samples for result in output_data
                        ], axis=0)
                    )
            brain_age_weights = brain_age_weights * (1 - learning_rate) + brain_age_weights_update * learning_rate
        results[WEIGHTS] = brain_age_weights
        return results
    elif parameters[TASK] == PREDICT:
        input_ = {
            "method": "predict",
            "args": [],
            "kwargs": {
                "parameters": parameters,
            }
        }

    # info("Check if any exception occurred")
    # if any([ERROR in result for result in results]):
    #     warn("Encountered an error, please review the parameters")
    #     return [result[ERROR] for result in results if ERROR in result]

    # process the output
    info("Process the node results")
    output = {}

    return output

def RPC_check(db_client, parameters):
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
    if os.getenv(images_path) and os.path.isdir(images_path):
        scans = [f for f in os.path.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        output[IMAGES_FOLDER][NUMBER_SCANS] = len(scans)
        missing = []
        if TRAINING_IDS in parameters or VALIDATION_IDS in parameters:
            for id in (parameters.get(TRAINING_IDS) or []) + (parameters.get(VALIDATION_IDS) or []):
                if id not in scans:
                    missing.append[id]
            
    else:
        output[IMAGES_FOLDER][MESSAGE] = "Image folder not found"

    # Check if the clinical data is available
    db_cnn_client = None
    if parameters[DB_TYPE] == DB_CSV:
        output[DB_CSV] = os.path.isfile(os.getenv(DATA_FOLDER) + "/dataset.csv")
    elif parameters[DB_TYPE] == DB_POSTGRES:
        output[DB_POSTGRES] = False
        if os.getenv(DB_CNN_DATABASE):
            db_cnn_client = PostgresManager(default_db=False, db_env_var=DB_CNN_DATABASE)
        else:
            warn("CNN database not provided")            
        output[DB_POSTGRES] = db_cnn_client.isConnected

    # Check if task ID already exists
    if TASK_ID in parameters and db_cnn_client:
        result = get_task_by_id(parameters[TASK_ID])
        output[TASK_ID] = result is not None

    # Check the connection to the XNAT
    # if os.getenv(XNAT_URL):
    #     pass

    # Check the GPU availability
    output[GPU_COUNT] = len(tf.config.list_physical_devices('GPU'))
    return output

def RPC_brain_age(db_client, parameters, weights):
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
    info("Brain age CNN - Node method")
    output = {
        METRICS: {},
        HISTORY: {},
    }
    # Retrieve the data from XNAT if necessary
    # data_path = os.path.join(os.getenv(DATA_FOLDER), parameters[TASK_ID])
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
        brain_age = BrainAge(
            parameters,
            parameters[TASK_ID],
            os.getenv(IMAGES_FOLDER),
            parameters[DB_TYPE],
            db_client if parameters[DB_TYPE] != DB_CSV else os.getenv(DATA_FOLDER) + "/dataset.csv",
            parameters[TRAINING_IDS],
            parameters[VALIDATION_IDS],
        )
        if weights:
            # Set the initial weights if available
            brain_age.model.set_weights(weights)
        output[SAMPLES] = 1
        # Train the model
        result = brain_age.train()
        # Retrieve the weights, metrics for the first and last epoch, and the 
        # history if requested
        output[WEIGHTS] = brain_age.model.get_weights()
        for metric in result.history.keys():
            output[METRICS][metric] = [result.history[metric][0], result.history[metric][-1]]
        if parameters.get(HISTORY):
            output[HISTORY] = result.history
    except Exception as error:
        output[ERROR] = f"Error while training the model: {str(error)}"
    return output
