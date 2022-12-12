""" Manage all operations related to openshift.
"""

import json
import openshift as oc
import time
import os
import subprocess

from vantage6.tools.util import info

from ncdc_maastricht_wrapper.utils import run_command

def login(token, server):
    """ Login into openshift using the oc CLI.
    """
    run_command(
        [
            'oc',
            'login',
            f'--token={token}',
            f'--server={server}',
            '--insecure-skip-tls-verify'
        ],
        "Logged in",
        info
    )


def create_tasks(task_folder, input_file, output_file, volume, task_id, algorithm_image):
    """ Creates the tasks that should get executed in the cluster.
    """
    return [
        {
            "task": "start-up-app",
            "description": "Start up and wait for the input",
            "file": f"{os.getenv('TEMPLATES_FOLDER_PATH')}/template-start-up.json",
            "sleep": 20,
            "volume": volume,
            "task_command": [
                "sh",
                "-c",
                f"until [[ -d {task_folder} ]]; do echo waiting for the input; sleep 10; done; sleep 60;"
            ],
            "commands": [
                {
                    "message": "Copying the input data from the wrapper to the pod",
                    "command": ['oc', 'cp', task_folder, f'{task_id}:{task_folder}']
                }
            ]
        },
        {
            "task": "run-algorithm-app",
            "description": "Run the main algorithm",
            "file": f"{os.getenv('TEMPLATES_FOLDER_PATH')}/template-run-algorithm.json",
            "sleep": 120,
            "volume": volume,
            "env": ["INPUT_FILE", "OUTPUT_FILE"],
            "algorithm_image": algorithm_image
        },
        {
            "task": "clear-up-app",
            "description": "Clear up and finish",
            "file": f"{os.getenv('TEMPLATES_FOLDER_PATH')}/template-clear-up.json",
            "sleep": 20,
            "volume": volume,
            "task_command": [
                "sh",
                "-c",
                f"until [[ ! -d {task_folder} ]]; do echo waiting to clear up; sleep 10; done;"
            ],
            "commands": [
                {
                    "message": "Copying the output data from the pod to the wrapper",
                    "command": ['oc', 'cp', f'{task_id}:{output_file}', output_file],
                },
                {
                    "message": "Deleting the task's data from the pod",
                    "command": ['kubectl', 'exec', task_id, '--', 'rm', '-rf', task_folder],
                }
            ]
        },
    ]

def check_task_status(container_info, status):
    """ Check if a task status shows that it
    """
    return 'status' in container_info and 'phase' in container_info['status'] \
        and container_info['status']['phase'] == status

def run_task(task_id, task_definition):
    """ Run the task using the openshift client.
    """
    info(f"Creating new task {task_definition['task']} with id {task_id}")
    # Load the task's template, fill the necessary information, and create the pod
    with open(task_definition['file']) as json_file:
        template = json.load(json_file)
        template["metadata"]["name"] = task_id
        template["metadata"]["labels"]["task"] = task_id
        if "task_command" in task_definition:
            template["spec"]["containers"][0]["command"] = task_definition["task_command"]
        if "algorithm_image" in task_definition and task_definition["algorithm_image"]:
            template["spec"]["containers"][0]["image"] = task_definition["algorithm_image"]
        if "env" in task_definition:
            template["spec"]["containers"][0]["env"] = []
            for env in task_definition["env"]:
                template["spec"]["containers"][0]["env"].append({
                    "name": env,
                    "value": os.environ[env]
                })
        if "volume" in task_definition and task_definition["volume"] is not None:
            template["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] = task_definition["volume"]

        output = oc.create(template)


    info("Getting the information on the newly created pod")
    c = oc.selector('pods', labels={"task": task_id, "app": task_definition["task"]})
    obj = c.objects()

    # Error in 'actions' (list) - 'err' for status 'status'
    if len(obj) == 0:
        raise Exception("Error - no pod found")
    elif len(obj) > 1:
        raise Exception("Error - multiple pods found")
    info("Pod created succesfully")
    container_info = obj[0].as_dict()

    #while container_info["status"]["phase"] != "Running" or container_info["status"]["phase"] != "Completed":
    while container_info["status"]["phase"] == "Pending":
        info("Waiting for the pod to be ready")
        time.sleep(30)
        obj[0].refresh()
        container_info = obj[0].as_dict()

    # Running the specified commands
    # TODO: Validating the exit code
    if "commands" in task_definition:
        for command in task_definition["commands"]:
            info(command["message"])
            run_command(command["command"], "Success", info)

    n_tries = 0
    complete = False
    # TODO: Task may end in error but in some cases the error may be temporary...
    while n_tries < 200 and not check_task_status(container_info, 'Succeeded'):
        info(f"Task still running, waiting {task_definition['sleep']} seconds")
        time.sleep(task_definition["sleep"])
        obj[0].refresh()
        container_info = obj[0].as_dict()
        n_tries += 1

    # TODO: Validate the exit code
    # container_info['status']['containerStatuses'][0]['state']['terminated']['exitCode']
    #  if container_info['status']['exitCode'] != 0:
    #      raise Exception("test")
    info(f"Successfully executed task {task_definition['task']}")

    info("Deleting the pod")
    c.delete()
    info("Completed")
    time.sleep(5)
