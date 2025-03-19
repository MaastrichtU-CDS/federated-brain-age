# Run script for testing purposes of the vantage6-whitelisting
# version 2. Submiting an algorithm to a cluster.
from vantage6.client import Client
import time
import random

print("Attempt login to Vantage6 API")
client = Client("https://v6-server-domain.nl", 443, "/api")
client.authenticate("username", "password")

client.setup_encryption(None)

input_ = {
    "master": "true",
    "method":"master",
    'algorithm_image': "pmateus/brain-age-gpu:1.0.0",
    "input_format": "json",
    "output_format": "json",
    "kwargs": {
        "parameters": {
            "TASK": "check",
            "DB_TYPE": "CSV",
            "MAX_NUMBER_TRIES": 50,
            "SLEEP_TIME": 60,
        },
        # Organizations that take part in this task
        "org_ids": [2]
    }
}

print("Requesting to execute summary algorithm")
task = client.post_task(
    name="task-check",
    # Execute the task in the VM where the node is running
    image="pmateus/brain-age-gpu:1.0.0",
    # Placeholder so it runs on the GPU cluster:
    # image="gpu_image",
    collaboration_id=1,
    input_= input_,
    # Organization that will act as the aggregation node
    organization_ids=[4]
)

print("Wait and fetch results")
res = client.get_results(task_id=task.get("id"))
attempts=1
print(task.get("id"))
while((res[0]["result"] == None) and attempts < 300):
    print("waiting...")
    time.sleep(120)
    res = client.get_results(task_id=task.get("id"))
    attempts += 1
print(res[0]["result"])
