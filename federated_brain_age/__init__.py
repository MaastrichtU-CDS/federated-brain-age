import time

import tensorflow as tf
from vantage6.tools.util import warn, info

from federated_brain_age.brain_age import BrainAge, get_parameter, DEFAULT_HYPERPARAMETERS
from federated_brain_age.constants import *
from federated_brain_age.utils import *
from federated_brain_age.xnat_client import retrieve_data

def execute_task(client, input):
    # obtain organizations that are within the collaboration
    info("Obtaining the organizations in the collaboration")
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get("id") for organization in organizations]

    # collaboration and image are stored in the key, so we do not need
    # to specify these
    info("Creating node tasks")
    task = client.create_new_task(
        input,
        organization_ids=ids
    )

    # wait for all results
    task_id = task.get("id")
    task = client.request(f"task/{task_id}")
    while not task.get("complete"):
        task = client.request(f"task/{task_id}")
        info("Waiting for results")
        # TODO: Time should be a higher value based on the expected time
        # for each task
        time.sleep(5)

    info("Obtaining results")
    return client.get_results(task_id=task.get("id"))

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

    # Check which task has been requested
    info(f"Task requested: {parameters[TASK]}")
    if parameters[TASK] == CHECK:
        input_ = {
            "method": "check",
            "args": [],
            "kwargs": {
                "parameters": parameters,
            }
        }
    elif parameters[TASK] == TRAIN:
        # Intialize the model
        model_parameters = dict(DEFAULT_HYPERPARAMETERS)
        #model_parameters[INPUT_SHAPE] = parameters[INPUT_SHAPE]
        brain_age_model = BrainAge.cnn_model(model_parameters.get)
        brain_age_weights = brain_age_model.get_weights()
        results = None
        # Execute the training
        for i in range(0, parameters[MODEL][MASTER][ROUNDS]):
            info(f"Round {i}")
            input = {
                "method": "brain_age",
                "args": [],
                "kwargs": {
                    "parameters": parameters[MODEL][NODE],
                    "weights": brain_age_weights,
                }
            }
            results = execute_task(client, input)
            # FedAvg
            info("Aggregating the results")
            brain_age_weights = tf.reduce_mean([result[WEIGHTS] for result in results], 2)

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

def RPC_check():
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
    # Check the connection to the XNAT
    if os.getenv(XNAT_URL):
        pass
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
    output = {}
    # Retrieve the data from XNAT if necessary
    data_path = os.path.join(os.getenv(DATA_FOLDER), parameters[TASK_ID])
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
    brain_age = BrainAge(
        parameters,
        parameters[TASK_ID],
        data_path,
        parameters[DB_TYPE],
        db_client if parameters[DB_TYPE] != DB_CSV else data_path + "/dataset.csv",
        parameters[TRAINING_IDS],
        parameters[VALIDATION_IDS],
    )
    if weights:
        brain_age.model.set_weights(weights)
    brain_age.train()
    output[WEIGHTS] = brain_age.model.get_weights()
    return output
