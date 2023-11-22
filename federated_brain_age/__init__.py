""" Federated Brain Age

    Handler for the server (master) and local methods (RPC_).
"""

from vantage6.tools.util import warn, info

from federated_brain_age.constants import *
from federated_brain_age.utils import *
from federated_brain_age.task_get_weights import get_weights
from federated_brain_age.task_check import check_centers, check_center
from federated_brain_age.server_handler import get_orgarnization
from federated_brain_age.task_predict import predict, task_predict_locally
from federated_brain_age.task_train import task_train, task_train_locally

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
    info("Check components - Node method - Check")
    return check_center(parameters, db_client)

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
    return task_train_locally(parameters, weights, data_seed, data_split, seed, db_client)

def RPC_predict(db_client, parameters, weights, data_seed, seed, data_split):
    """ Predict the brain age
    """
    info("Brain age CNN - Node method - Prediction")
    return task_predict_locally(parameters, weights, data_seed, seed, data_split, db_client)
