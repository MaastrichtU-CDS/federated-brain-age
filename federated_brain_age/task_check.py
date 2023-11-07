""" Handle the master and node tasks to check the settings
"""

from federated_brain_age.constants import *
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
