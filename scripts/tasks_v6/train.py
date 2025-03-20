from vantage6.client import Client
import time
import random

print("Attempt login to Vantage6 API")
client = Client("https://v6-server-domain.nl", 443, "/api")
client.authenticate("username", "password")

client.setup_encryption(None)

seed = random.randint(0, 1000000)
print(f"Seed: {seed}")
input_ = {
    "master": "true",
    "method":"master",
    'algorithm_image': "pmateus/brain-age-gpu:1.0.0",
    "input_format": "json",
    "output_format": "json",
    #"args": ["pmateus/brain-age-gpu:0.0.16", "brain_age"],
    "kwargs": {
        "parameters": {
            "TASK": "train",
            # The model id is only necessary if you choose to save the model
            "MODEL_ID": "model-id",
            "SAVE_MODEL": True,
            "DB_TYPE": "CSV",
            "MAX_NUMBER_TRIES": 1000,
            "SLEEP_TIME": 60,
            "HISTORY": True,
            # If it's not provided, a seed will be generated and stored in the database to
            # guarantee the correct data split.
            "seed": seed,
            # If an error occurs when retrieving the results/merging the weights/etc (there are several reasons
            # that can cause this - e.g., connection timeout) but the cohort finalized the training, it's possible
            # to restart the training by first collecting the local models.
            # However, it's necessary to provide the task_id for each cohort (can be found in the logs)
            # "RESTART_TRAINING": {
            #     3: 3030, # For cohort 3, collect the results from task id 3030
            #     4: 3026  # For cohort 4, collect the results from task id 3026
            # },
            "MODEL": {
                "MASTER": {
                    "ROUNDS": 15,
                    # The weighted average takes into consideration the number of samples in each cohort
                    "WEIGHTED_AVERAGE": False,
                },
                "NODE": {
                    "USE_MASK": True,
                    "EPOCHS": 5,
                    "BATCH_SIZE": 8,
                    "PATIENTS_PER_EPOCH": 2000,
                    "EARLY_STOPPING": False,
                    "LEARNING_RATE": 0.001,
                    "DECAY": 0.01,
                    "MODEL_SELECTION": "val_mae",
                    "DROPOUT": 0.5,
                    "SINGLE_SCAN_BY_PATIENT": True,
                },
                "data_split": 0.8,
            }
        },
        "org_ids": [3, 4]
    }
}

print("Requesting to execute summary algorithm")
task = client.post_task(
    name="train",
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
    time.sleep(120)
    res = client.get_results(task_id=task.get("id"))
    attempts += 1
print(res[0]["result"])
