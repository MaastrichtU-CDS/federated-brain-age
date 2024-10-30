""" Handle the master and node tasks to train the brain age model
"""
import json
import os
import random

import numpy as np
import tensorflow as tf
from vantage6.tools.util import warn, info

from federated_brain_age.brain_age import BrainAge, DEFAULT_HYPERPARAMETERS
from federated_brain_age.constants import *
from federated_brain_age.utils import np_array_to_list, check_errors
from federated_brain_age.db_builder import *
from federated_brain_age import server_handler

def retrieve_model(parameters, db_client):
    """ If the model is not yet stored in the database, it stores the model
        according to the parameters provided.
        If the model already exists, it retrieves the information (seed, data
        split, and weights from the last run) to continue the training.
    """
    model_info = {
        ID: parameters[MODEL_ID],
        SEED: parameters.get(SEED),
        ROUND: 0,
        WEIGHTS: None,
        DATA_SPLIT: parameters[MODEL][DATA_SPLIT],
    }
    run_id = None
    result = get_model_by_id(parameters[MODEL_ID], db_client)
    if result:
        info("Get existing model")
        model_info[SEED] = result[2]
        model_info[DATA_SPLIT] = result[3]
        last_run = get_last_run_by_id(parameters[MODEL_ID], db_client)
        if last_run:
            run_id = last_run[0]
            model_info[ROUND] = last_run[3]
            model_info[WEIGHTS] = last_run[4]
    else:
        # Insert a new entry for the model
        info("Insert the new model")
        model_info[SEED] = parameters.get(SEED) if \
            parameters.get(SEED) is not None else random.randint(0, 1000000)
        insert_model(model_info[ID], model_info[SEED], model_info[DATA_SPLIT], db_client)
    return model_info, run_id


def aggregate_metrics(output_data, output, tasks):
    """ Aggregate the metrics from the training.
    """
    metrics = {
        GLOBAL: {}
    }
    metrics_aggregator = {
        MAE: [],
        MSE: [],
        VAL_MAE: [],
        VAL_MSE: [],
    }
    predictions = {}
    age_gap = {}
    sample_size_training = [result[SAMPLE_SIZE][0] for result in output_data]
    sample_size_validation = [result[SAMPLE_SIZE][1] for result in output_data]
    info(
        "Total number of samples for training and validation: " + 
            f"{sum(sample_size_training)}, {sum(sample_size_validation)}"
    )
    # Collect the metrics by organization
    for task_id, result in output.items():
        metrics[task_id] = {
            ORGANIZATION_ID: tasks[task_id][ORGANIZATION_ID],
            HISTORY: result[HISTORY],
            METRICS: result[METRICS],
            SAMPLE_SIZE: result[SAMPLE_SIZE]
        }
        predictions[tasks[task_id][ORGANIZATION_ID]] = result.get(PREDICTIONS, [])
        age_gap[tasks[task_id][ORGANIZATION_ID]] = result.get(AGE_GAP, [])
        for metric in metrics_aggregator.keys():
            if metric in result[METRICS] and len(result[METRICS][metric]) > 0:
                metrics_aggregator[metric].append(result[METRICS][metric][0])
            else:
                warn(
                    f"Metric {metric} not found in the results from the node with " +
                        f"id {metrics[task_id][ORGANIZATION_ID]}"
                )
    for metric in metrics_aggregator.keys():
        weights = sample_size_training
        if 'val' in metric:
            weights = sample_size_validation
        if len(metrics_aggregator[metric]) > 0 and sum(weights) > 0:
            metrics[GLOBAL][metric] = np.average(metrics_aggregator[metric], weights=weights)
        else:
            metrics[GLOBAL][metric] = -1
    return metrics, predictions, age_gap, sample_size_training, sample_size_validation

def task_train(parameters, ids, algorithm_image, db_client, client):
    """ Prepare the parameters and send the task to the cohorts. Recover the previous state 
        if the model is already stored in the database.
    """
    model_info = {
        ID: None,
        SEED: parameters.get(SEED),
        ROUND: 0,
        WEIGHTS: None,
        DATA_SPLIT: parameters[MODEL][DATA_SPLIT],
    }
    store_model = SAVE_MODEL in parameters and parameters[SAVE_MODEL]
    run_id = None
    if store_model:
        info("Request to save the model")
        if MODEL_ID in parameters:
            model_info[ID] = parameters[MODEL_ID]
            try:
                model_info, run_id = retrieve_model(parameters, db_client)
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
    # Set 3 constant seeds for the training/validation split
    data_seeds = [random.randint(0, 1000000) for j in range(len(ids))]
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
        # Set 3 random seeds each round to have a different random shuffle 
        # in each round for the batch split
        seeds = [random.randint(0, 1000000) for j in range(len(ids))]
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
                    DATA_SEED: data_seeds[org_num],
                    SEED: seeds[org_num],
                    DATA_SPLIT: model_info[DATA_SPLIT],
                }
            }
            task_id = None
            if RESTART_TRAINING in parameters and org_id in parameters[RESTART_TRAINING] \
                and i == model_info[ROUND]:
                # Allows to reuse of a previous task avoiding a new training in case of failing
                # to merge the results or a failure in one of the organizations
                task_id = parameters[RESTART_TRAINING][org_id]
                info(f"Restarting the training for organizaion {str(org_id)} using task id {str(task_id)}")
            else:
                task_id = server_handler.execute_task(client, input, [org_id], algorithm_image)
            if task_id:
                tasks[task_id] = {
                    ORGANIZATION_ID: org_id
                }
        output = server_handler.get_result(
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
        metrics, predictions, age_gap, n_training, n_validation = aggregate_metrics(output_data, output, tasks)
        results[METRICS][i] = metrics
        # Store the results
        if store_model:
            if not run_id:
                warn("Error: missing run_id!")
                return {
                    ERROR: "Error: missing run_id!"
                }
            # Update the model with the metrics
            info("Update the model")
            update_run(
                model_info[ID],
                run_id,
                metrics[GLOBAL][MAE],
                metrics[GLOBAL][MSE],
                metrics[GLOBAL][VAL_MAE],
                metrics[GLOBAL][VAL_MSE],
                json.dumps(metrics),
                json.dumps(predictions),
                json.dumps(age_gap),
                db_client
            )
        # Update the model weights
        total_training_samples = sum(n_training)
        brain_age_weights_update = []
        models_parsed = []
        # Perform a weighted average of give the same weight to all participants
        info("Aggregating the weights")
        weighted_average = True
        if AVERAGING_WEIGHTS in parameters[MODEL][MASTER]:
            weighted_average = parameters[MODEL][MASTER][AVERAGING_WEIGHTS]
            if not weighted_average:
                info("Model averaging without weights")
                total_training_samples = len(output_data)
        for result in output_data:
            models_parsed.append({
                WEIGHTS: json.loads(result[WEIGHTS]),
                SAMPLE_SIZE: result[SAMPLE_SIZE][0] if weighted_average else 1
            })
        info(f"{total_training_samples} samples: {', '.join([str(model[SAMPLE_SIZE]) for model in models_parsed])}")
        for j in range(0, len(models_parsed[0][WEIGHTS])):
            brain_age_weights_update.append(
                tf.math.reduce_sum([
                    np.array(result[WEIGHTS][j], dtype=np.double) * result[SAMPLE_SIZE] / total_training_samples for result in models_parsed
                ], axis=0)
            )
        brain_age_weights = brain_age_weights * (1 - learning_rate) + brain_age_weights_update * learning_rate
    # Store the model from the last round
    if store_model and int(model_info[ROUND]) < parameters[MODEL][MASTER][ROUNDS]:
        info("Save last merged model")
        result = insert_run(
            model_info[ID],
            parameters[MODEL][MASTER][ROUNDS],
            json.dumps(np_array_to_list(brain_age_weights)),
            None,
            None,
            None,
            db_client
        )
    if parameters.get(RETURN_WEIGTHS):
        results[WEIGHTS] = brain_age_weights
    info("Send the output")
    return results

def task_train_locally(parameters, weights, data_seed, data_split, seed, db_client):
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
