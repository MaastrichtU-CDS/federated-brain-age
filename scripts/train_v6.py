# Run script for testing purposes of the vantage6-whitelisting
# version 2. Submiting an algorithm to a cluster.
from vantage6.client import Client
import time

print("Attempt login to Vantage6 API")
client = Client("https://server.nl", 443, "/api")
client.authenticate("user", "password")

client.setup_encryption(None)

input_ = {
    "master": "true",
    "method":"master",
    # CPU image:
    # 'algorithm_image': "pmateus/brain-age:0.0.41",
    # GPU image:
    'algorithm_image': "pmateus/brain-age-gpu:0.0.21",
    "input_format": "json",
    "output_format": "json",
    "kwargs": {
        "parameters": {
            # Task: training the model
            "TASK": "train",
            "MODEL_ID": "model-id",
            # Save the model in the postgres database?
            "SAVE_MODEL": True,
            "DB_TYPE": "CSV",
            # Evaluate the necessary time for the algorithm to run
            # according to the hyperparameters
            "MAX_NUMBER_TRIES": 1000,
            "SLEEP_TIME": 120,
            # History: the metrics for each epoch by cohort by round
            # There can be some discrepancies between the metrics that
            # tensorflow provide by epoch during training and the 
            # initial/final metrics calculated
            "HISTORY": False,
            "seed": 1,
            "MODEL": {
                "MASTER": {
                    "ROUNDS": 10
                },
                "NODE": {
                    "USE_MASK": True,
                    "EPOCHS": 50,
                    "BATCH_SIZE": 4,
                    "PATIENTS_PER_EPOCH": 400,
                    "EARLY_STOPPING": False,
                    "DECAY": 0.01,
                },
                "data_split": 0.7,
            }
        },
        # Organizations that will take part in the training/validation process
        "org_ids": [2, 3, 4]
    }
}

print("Requesting to execute federated brain age algorithm")
task = client.post_task(
    name="task-name",
    image="gpu_image",
    collaboration_id=1,
    input_= input_,
    # Master node
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
