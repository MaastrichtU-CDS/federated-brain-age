""" Functions to interact with the vantage6 API
"""
import time

from vantage6.tools.util import warn, info

from federated_brain_age.constants import *

def execute_task(client, input, org_ids, algorithm_image = None):
    # collaboration and image are stored in the key, so we do not need
    # to specify these
    info(f"Creating node tasks for organization(s): {', '.join([str(org_id) for org_id in org_ids])}")
    if algorithm_image is not None:
        input["algorithm_image"] = algorithm_image
    task = client.create_new_task(
        input,
        organization_ids=org_ids
    )
    task_id = task.get("id")
    print(f"Task id: {str(task_id)}")
    return task_id

def get_result(client, tasks, max_number_tries=DEFAULT_MAX_NUMBER_TRIES, sleep_time=60):
    # Get the task's result
    results = {}
    tries = 0
    while len(results.keys()) < len(tasks) and tries <= max_number_tries:
        # TODO: Set a better value for the timerz
        time.sleep(sleep_time)
        tries += 1
        for task_id in tasks.keys():
            if task_id not in results.keys():
                info(f"Waiting for results: {task_id}")
                task = client.request(f"task/{task_id}")
                if task.get("complete"):
                    info("Obtaining results")
                    task_results = client.get_results(task_id=task.get("id"))
                    if task_results and len(task_results) == 1 and \
                        type(task_results[0]) is dict and len(task_results[0].keys()) > 0:
                        results[task_id] = task_results[0]
                    else:
                        results[task_id] = {
                            ERROR: "Task didn't return a result"
                        }   
    # Check the tasks that didn't complete in time
    for task_id in tasks.keys():
        if task_id not in results:
            results[task_id] = {
                ERROR: "Task did not complete in time"
            }
    return results

def get_orgarnization(client, org_ids):
    # obtain organizations that are within the collaboration
    info("Obtaining the organizations in the collaboration")
    organizations = [organization for organization in client.get_organizations_in_my_collaboration() if \
        not org_ids or organization.get("id") in org_ids]
    return organizations
