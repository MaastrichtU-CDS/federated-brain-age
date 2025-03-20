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
            "TASK": "predict",
            "MODEL_ID": "model_id",
            "SAVE_MODEL": True,
            "DB_TYPE": "CSV",
            "MAX_NUMBER_TRIES": 1000,
            "SLEEP_TIME": 30,
            "HISTORY": True,
            "is_training_data": True,
            # Select the round
            "ROUND": 5,
            "MODEL": {
                "NODE": {
                    "USE_MASK": True,
                    "BATCH_SIZE": 4,
                    "PATIENTS_PER_EPOCH": 1000,
                },
            }
        },
        "org_ids": [2]
    }
}

print("Requesting to execute summary algorithm")
task = client.post_task(
    name="testing-all",
    image="gpu_image",
    collaboration_id=1,
    input_= input_,
    organization_ids=[4]
)

print("Wait and fetch results")
res = client.get_results(task_id=task.get("id"))
attempts=1
print(task.get("id"))
while((res[0]["result"] == None) and attempts < 300):
    print("waiting...")
    time.sleep(30)
    res = client.get_results(task_id=task.get("id"))
    attempts += 1
print(res[0]["result"])
