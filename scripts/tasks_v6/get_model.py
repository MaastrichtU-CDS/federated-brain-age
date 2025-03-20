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
            "TASK": "get_model",
            "MODEL_ID": "model-id",
            "SLEEP_TIME": 30,
            "DB_TYPE": "CSV",
            "MAX_NUMBER_TRIES": 1000,
            "SLEEP_TIME": 60,
            "ROUND": 5,
        },
        "org_ids": [4]
    }
}

print("Requesting to execute summary algorithm")
task = client.post_task(
    name="task-check",
    image="pmateus/brain-age-gpu:1.0.0",
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
    time.sleep(10)
    res = client.get_results(task_id=task.get("id"))
    attempts += 1
print(res[0]["result"])
