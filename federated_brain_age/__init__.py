import time

from vantage6.tools.util import warn, info

from federated_brain_age.constants import *
from federated_brain_age.utils import *

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

    # define the input for the brain age algorithm
    info("Defining input parameters")
    input_ = {
        "method": "brain_age",
        "args": [],
        "kwargs": {
            "parameters": parameters,
        }
    }

    # obtain organizations that are within the collaboration
    info("Obtaining the organizations in the collaboration")
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get("id") for organization in organizations]

    # collaboration and image are stored in the key, so we do not need
    # to specify these
    info("Creating node tasks")
    task = client.create_new_task(
        input_,
        organization_ids=ids
    )

    # wait for all results
    task_id = task.get("id")
    task = client.request(f"task/{task_id}")
    while not task.get("complete"):
        task = client.request(f"task/{task_id}")
        info("Waiting for results")
        time.sleep(5)

    info("Obtaining results")
    results = client.get_results(task_id=task.get("id"))

    info("Check if any exception occurred")
    if any([ERROR in result for result in results]):
        warn("Encountered an error, please review the parameters")
        return [result[ERROR] for result in results if ERROR in result]

    # process the output
    info("Process the node results")
    output = {}

    return output

def RPC_brain_age(db_client, parameters):
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
    # Process the cohort if included in the request
    sql_condition = None

    return output
