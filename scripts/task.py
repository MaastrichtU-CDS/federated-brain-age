from vantage6.client import Client
import time
import matplotlib.pyplot as plt
import json

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")

client.setup_encryption(None)

# Parameters
# - task: Which task to execute, can be either "CHECK", "TRAIN", "PREDICT", ...
input_ = {
    "master": "true",
    "method":"master", 
    "args": [], 
    "kwargs": {
        "task": "CHECK"
    }
}

print("Requesting to execute summary algorithm")
task = client.post_task(
    name="brain-age-testing",
    image="pmateus/federated-brain-age:1.0.0",
    collaboration_id=1,
    input_= input_,
    organization_ids=[2, 3]
)

print("Wait and fetch results")
response = client.get_results(task_id=task.get("id"))
max_attempts=20
while((response[0]["result"] == None) and max_attempts < max_attempts):
    print("waiting...")
    time.sleep(20)
    response = client.get_results(task_id=task.get("id"))
    max_attempts += 1

# Checking the results
# response[0]["result"]
