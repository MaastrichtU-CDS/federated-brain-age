import json
import time
import os
import random

import numpy as np
import tensorflow as tf
from vantage6.tools.util import warn, info

from federated_brain_age.brain_age import BrainAge, DEFAULT_HYPERPARAMETERS
from federated_brain_age.constants import *
from federated_brain_age.utils import *
from federated_brain_age.xnat_client import retrieve_data
from federated_brain_age.postgres_manager import PostgresManager
from federated_brain_age.db_builder import get_model_by_id, get_last_run_by_id, insert_model, insert_run, update_run

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

def get_result(client, tasks, max_number_tries=DEFAULT_MAX_NUMBER_TRIES, sleep_time=60):
    # Get the task's result
    results = {}
    tries = 0
    while len(results.keys()) != len(tasks) or tries > max_number_tries:
        # TODO: Set a better value for the timer
        time.sleep(sleep_time)
        tries += 1
        for task_id in tasks.keys():
            if task_id not in results:
                info("Waiting for results")
                task = client.request(f"task/{task_id}")
                if task.get("complete"):
                    info("Obtaining results")
                    results[task_id] = client.get_results(task_id=task.get("id"))
    # Check the tasks that didn't complete in time
    for task_id in tasks.keys():
        if task_id not in results:
            results[task_id] = {
                ERROR: "Task did not complete in time"
            }
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
    if TASK not in parameters:
        return {
            ERROR: f"Missing the following parameter: {TASK}"
        }

    # Get the organizations in the collaboration
    orgs = get_orgarnization(client)
    ids = [org.get("id") for org in orgs]

    # Check which task has been requested
    info(f"Task requested: {parameters[TASK]}")
    if parameters[TASK] == CHECK:
        # Validate the input
        missing_parameters = validate_parameters(parameters, {DB_TYPE: {}})
        if missing_parameters:
            parse_error(
                f"Missing the following parameters: {', '.join(missing_parameters)}"
            )
        # Send the tasks
        tasks = {}
        for id in ids:
            input = {
                "method": CHECK,
                "args": [],
                "kwargs": {
                    "parameters": {
                        **parameters,
                        # TRAINING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                        # VALIDATION_IDS: parameters[MODEL][NODE][VALIDATION_IDS][id],
                        # TESTING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                    },
                },
            }
            task_id = execute_task(client, input, [id])
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
    elif parameters[TASK] == TRAIN:
        # Validate the input
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

        # Recover the previous state if requested
        model_info = {
            ID: None,
            SEED: None,
            ROUND: 0,
            WEIGHTS: None,
        }
        store_model = SAVE_MODEL in parameters and parameters[SAVE_MODEL]
        run_id = None
        if store_model:
            info("Request to save the model")
            if MODEL_ID in parameters:
                model_info[ID] = parameters[MODEL_ID]
                try:
                    result = get_model_by_id(parameters[MODEL_ID], db_client)
                    if result:
                        info("Get existing model")
                        model_info[SEED] = result[2]
                        last_run = get_last_run_by_id(parameters[MODEL_ID], db_client)
                        if last_run:
                            run_id = last_run[0]
                            model_info[ROUND] = last_run[3]
                            model_info[WEIGHTS] = last_run[4]
                    else:
                        # Insert a new entry for the model
                        info("Insert the new model")
                        model_info[SEED] = parameters.get(SEED) if \
                            parameters.get(SEED) is not None else random.randint(0, 10000)
                        insert_model(model_info[ID], model_info[SEED], db_client)
                except Exception as error:
                    error_message = f"Unable to connect to the database and retrieve the model: {str(error)}"
                    warn(error_message)
                    return {
                        ERROR: error_message
                    }
            else:
                error_message = "In order to save the model an ID must be provided."
                warn(error_message)
                return {
                    ERROR: error_message
                }

        # Set the seed value
        info(f"Using {model_info[SEED]} as the seed")
        random.seed(model_info[SEED])

        # Intialize the model
        learning_rate = parameters.get(LEARNING_RATE, 1)
        model_parameters = dict(DEFAULT_HYPERPARAMETERS)
        # model_parameters[INPUT_SHAPE] = parameters[INPUT_SHAPE]
        brain_age_model = BrainAge.cnn_model(model_parameters.get)
        brain_age_weights = model_info[WEIGHTS] or brain_age_model.get_weights()
        # Output
        results = {
            METRICS: {},
            SEED: model_info[SEED],
        }
        # Execute the training
        for i in range(int(model_info[ROUND]), parameters[MODEL][MASTER][ROUNDS]):
            info(f"Round {i}/{parameters[MODEL][MASTER][ROUNDS]}")
            seeds = [random.randint(0, 10000) for j in range(len(ids))]
            tasks = {}
            # Store the model if requested. If the weigths were previously store,
            # it'll skip for the first round
            if store_model and (model_info[WEIGHTS] is None or i != model_info[ROUND]):
                result = insert_run(
                    model_info[ID],
                    i,
                    json.dumps(np_array_to_list(brain_age_weights)),
                    None,
                    None,
                    None,
                    db_client
                )
                if result:
                    run_id = result[0]
            for org_num, org_id in enumerate(ids):
                input = {
                    "method": ALGORITHM,
                    "args": [],
                    "kwargs": {
                        "parameters": {
                            **parameters[MODEL][NODE],
                            # TRAINING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                            # VALIDATION_IDS: parameters[MODEL][NODE][VALIDATION_IDS][id],
                            # TESTING_IDS: parameters[MODEL][NODE][TRAINING_IDS][id],
                            ROUNDS: i,
                            HISTORY: parameters.get(HISTORY),
                            MODEL_ID: parameters.get(MODEL_ID),
                            DB_TYPE: parameters.get(DB_TYPE)
                        },
                        WEIGHTS: json.dumps(np_array_to_list(brain_age_weights)),
                        SEED: seeds[org_num],
                        DATA_SPLIT: parameters[MODEL][DATA_SPLIT],
                    }
                }
                task_id = execute_task(client, input, [org_id])
                tasks[task_id] = {
                    ORGANIZATION_ID: org_id
                }
            output = get_result(
                client,
                tasks,
                max_number_tries=parameters.get(MAX_NUMBER_TRIES) or DEFAULT_MAX_NUMBER_TRIES,
                sleep_time=parameters.get(SLEEP_TIME) or DEFAULT_SLEEP_TIME
            )
            # TODO: Validate if there are errors in the results:
            # Log the information and terminate the training
            info("Check if any exception occurred")
            output_data = list(output.values())
            errors = check_errors(output_data)
            if errors:
                warn("Encountered an error, please review the parameters")
                return { ERROR: errors }

            info("Aggregating the results")
            metrics = {
                GLOBAL: {}
            }
            metrics_aggregator = {
                MAE: [],
                MSE: [],
            }
            sample_size = [result[SAMPLES] for result in output_data]
            # Collect the metrics by organization
            for task_id, result in output.items():
                metrics[task_id] = {
                    ORGANIZATION_ID: tasks[task_id][ORGANIZATION_ID],
                    HISTORY: result[HISTORY],
                    METRICS: result[METRICS],
                    SAMPLE_SIZE: result[SAMPLE_SIZE]
                }
                for metric in metrics_aggregator.keys():
                    metrics_aggregator[metric].append(result[METRICS][metric][0])
            for metric in metrics_aggregator.keys():
                metrics[GLOBAL][metric] = np.average(metrics_aggregator[metric], weights=sample_size)
            results[METRICS][i] = metrics
            # Store the results
            if store_model:
                if not run_id:
                    warn("Error: missing run_id!")
                    return {
                        ERROR: "Error: missing run_id!"
                    }
                # Update the model with the metrics
                update_run(
                    model_info[ID],
                    run_id,
                    metrics[GLOBAL][MAE],
                    metrics[GLOBAL][MSE],
                    json.dumps(np_array_to_list(metrics)),
                    db_client
                )
            # Update the model weights
            total_samples = sum(sample_size)
            brain_age_weights_update = []
            models_parsed = []
            for result in output_data:
                models_parsed.append({
                    WEIGHTS: json.loads(result[WEIGHTS]),
                    SAMPLES: result[SAMPLES]
                })
            for j in range(0, len(models_parsed[0][WEIGHTS])):
                brain_age_weights_update.append(
                    tf.math.reduce_sum([
                        np.array(result[WEIGHTS][j], dtype=np.double) * result[SAMPLES] / total_samples for result in models_parsed
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
    output = {
        # SEED: seed
    }

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
        # missing = []
        # if TRAINING_IDS in parameters or VALIDATION_IDS in parameters:
        #     for id in (parameters.get(TRAINING_IDS) or []) + (parameters.get(VALIDATION_IDS) or []):
        #         if id not in scans:
        #             missing.append[id]
            
    else:
        output[IMAGES_FOLDER][MESSAGE] = "Image folder not found"

    # Check if the clinical data is available
    if parameters[DB_TYPE] == DB_CSV:
        output[DB_CSV] = os.path.isfile(os.getenv(DATA_FOLDER) + "/dataset.csv")
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

def RPC_brain_age(db_client, parameters, weights, seed, data_split):
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
        brain_age = BrainAge(
            parameters,
            parameters[MODEL_ID],
            os.getenv(IMAGES_FOLDER),
            parameters[DB_TYPE],
            db_client if parameters[DB_TYPE] != DB_CSV else os.getenv(DATA_FOLDER) + "/dataset.csv",
            # parameters[TRAINING_IDS],
            # parameters[VALIDATION_IDS],
            seed=seed,
            split=data_split,
        )
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
            output[SAMPLES] = 1
            # Train the model
            result = brain_age.train()
            # Retrieve the weights, metrics for the first and last epoch, and the 
            # history if requested
            output[WEIGHTS] = json.dumps(np_array_to_list(brain_age.model.get_weights()))
            for metric in result.history.keys():
                output[METRICS][metric] = [result.history[metric][0], result.history[metric][-1]]
            if parameters.get(HISTORY):
                output[HISTORY] = result.history
    except Exception as error:
        output[ERROR] = f"Error while training the model: {str(error)}"
    return output
