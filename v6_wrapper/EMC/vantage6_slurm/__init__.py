# modified from https://gitlab.com/UM-CDS/projects/memorabel

import time

from vantage6.tools.util import info

try:
    import federated_brain_age as brain_age
except ImportError:
    def master(client, db_client, parameters=None, org_ids=None, algorithm_image=None):
        # Calling the master and providing the algorithm image to run in the GPU
        return_dict = {
                    "ERROR": f"Master function is not available with this image."
        }
        return return_dict
else:
    def master(client, db_client, parameters = None, org_ids = None, algorithm_image = None):
        # Calling the master and providing the algorithm image to run in the GPU
        return brain_age.master(client, db_client, parameters, org_ids, algorithm_image)


def relay(client, _, algorithm_image, algorithm_method, **kwargs):
    """Combine partials to global model

    First we collect the parties that participate in the collaboration.
    Then we send a task to all the parties to compute their partial (the
    row count and the column sum). Then we wait for the results to be
    ready. Finally when the results are ready, we combine them to a
    global average.

    Note that the master method also receives the (local) data of the
    node. In most usecases this data argument is not used.

    The client, provided in the first argument, gives an interface to
    the central server. This is needed to create tasks (for the partial
    results) and collect their results later on. Note that this client
    is a different client than the client you use as a user.
    """

    # Info messages can help you when an algorithm crashes. These info
    # messages are stored in a log file which is send to the server when
    # either a task finished or crashes.
    info('Collecting participating organizations')

    # Collect all organization that participate in this collaboration.
    # These organizations will receive the task to compute the partial.
    node_org_mapping = kwargs['node_org_mapping']
    ids = [int(node_org_mapping[str(client.host_node_id)])]
    del kwargs['node_org_mapping']

    # Request all participating parties to compute their partial. This
    # will create a new task at the central server for them to pick up.
    # We've used a kwarg but is is also possible to use `args`. Although
    # we prefer kwargs as it is clearer.
    info('Requesting partial computation')
    task = client.create_new_task(
        input_={
            'algorithm_image': algorithm_image,
            'method': algorithm_method,
            'kwargs': kwargs
        },
        organization_ids=ids
    )

    # Now we need to wait untill all organizations(/nodes) finished
    # their partial. We do this by polling the server for results. It is
    # also possible to subscribe to a websocket channel to get status
    # updates.
    info("Waiting for resuls")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(10)

    # Once we now the partials are complete, we can collect them.
    info("Obtaining results")
    return client.get_results(task_id=task.get("id"))
