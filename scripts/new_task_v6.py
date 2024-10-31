from vantage6.client import Client
import time
import random

print("Attempt login to Vantage6 API")
port = 443
client = Client("https://server-domain.com", port, "/path_to_api")
client.authenticate("user", "password")

client.setup_encryption(None)

seed = random.randint(0, 100000)

input_ = {
    "master": "true",
    "method":"master",
    # In case you're using the cluster wrapper image, you need
    # to specify the algorithm image here:
    # 'algorithm_image': "pmateus/brain-age-gpu:1.0.88",
    "input_format": "json",
    "output_format": "json",
    "kwargs": {
        "parameters": {
            "TASK": "train",
            "MODEL_ID": "cnn-brain-age-1",
            "SAVE_MODEL": True,
            "DB_TYPE": "CSV",
            "MAX_NUMBER_TRIES": 1000,
            "SLEEP_TIME": 60,
            "HISTORY": True,
            "seed": seed,
            # In case of a problem, it's possible to restart the training from a 
            # previous task instead of re-running the round.
            # "RESTART_TRAINING": {
            #     1: task_id_org_1,
            #     2: task_id_org_2,
            # },
            "MODEL": {
                "MASTER": {
                    "ROUNDS": 20,
                    "WEIGHTED_AVERAGE": False,
                },
                "NODE": {
                    "USE_MASK": True,
                    "EPOCHS": 3,
                    # Check the memory available
                    "BATCH_SIZE": 8,
                    "PATIENTS_PER_EPOCH": 1000,
                    "EARLY_STOPPING": False,
                    "LEARNING_RATE": 0.001,
                    "DECAY": 0.01,
                    "MODEL_SELECTION": True,
                    "DROPOUT": 0.25,
                    "SINGLE_SCAN_BY_PATIENT": True,
                    "COMPLETE_METRICS": False
                },
                "data_split": 0.8,
            }
        },
        "org_ids": [1, 2]
    }
}

print("Requesting to execute summary algorithm")
task = client.post_task(
    name="task_1",
    # Using the HPC/GPU cluster wrapper:
    # image="gpu_image",
    # Directly running the docker image:
    image="pmateus/brain-age:latest",
    collaboration_id=1,
    input_= input_,
    # the master organization (where the results are merged and
    # stored in the RDB)
    organization_ids=[1]
)

# Retrieve the results:
task_id=task.get("id")
print(f"Wait and fetch results for task {task_id}")
res = client.get_results(task_id=task_id)
attempts=1
while((res[0]["result"] == None) and attempts < 300):
    print("waiting...")
    time.sleep(120)
    res = client.get_results(task_id=task_id)
    attempts += 1
print(res[0]["result"])
