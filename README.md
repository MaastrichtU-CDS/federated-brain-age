# Federated Brain Age

Project for the federated implementation of the gray matter age prediction as a biomarker for risk of dementia.
Based on the implementation from the following [repository](https://gitlab.com/radiology/neuro/brain-age/brain-age) and 
[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6800321/).

## Description

The base for this work consists of an algorithm, a Convolutional Neural Network (CNN), that predicts a person's age based on imaging data and clinical variables.

## Architecture

The federated architecture for this project follows the Personal Health Train (PHT) concept. Each participating organization keeps the data locally, only sharing aggregated information that doesn't disclose individual-level data.

### Data

Facilitating the process of training an algorithm using a federated approach requires a certain level of harmony between the data in each center.
To accomplish this, the data is expected to follow a similar structure:
* Imaging data: Stored in XNAT and retrieved only once before starting the training at each center.
* Clinical data: Stored in a relational database harmonised according to a Common Data Model (CDM).

## Build Image

To build the docker image, run the following command: `docker image build -t brain-age:latest .`

## Running

Vantage6 is used to implement the PHT infrastructure, implying dependencies on this library (e.g., communication between nodes and server) in the algorithm implementation. However, as demonstrated by the examples, it's simple to decouple the algorithm core and adapt to a different infrastructure.

Running the algorithm as a task using the vantage6 client:
```
```

Parameter description:
- TASK: task to execute (TRAIN, CHECK, PREDICT);
  - TRAIN: train the model;
  - CHECK: evaluate if the necessary components are available (GPU cluster, data, XNAT, ...)
- TASK_ID: identifies the task to later resume if necessary;
- DB_TYPE: database type (POSTGRES, CSV);
- MAX_NUMBER_TRIES: number of tries to retrieve the result from the node;
- MODEL: parameters for the model
  - MASTER: parameters for the optimization process in the master node
    - ROUNDS: number of rounds to iterate the model training across the cohorts;
  - NODE: parameters for the model in the cohorts' node
    - USE_MASK: use the mask to initialize the parameters for the image pre-processing;
    - EPOCHS: number of epochs to train the model;
    - TRAINING_IDS: the ids used for the training, provided by each organization;
    - VALIDATION_IDS: the ids used for the validation, provided by each organization;

```python
PARAMETERS = {
    TASK: "TRAIN",
    TASK_ID: "test",
    DB_TYPE: "CSV,
    MAX_NUMBER_TRIES: 10,
    MODEL: {
        MASTER: {
            ROUNDS: 2
        },
        NODE: {
            USE_MASK: True,
            EPOCHS: 2,
            TRAINING_IDS: {
                1: ["2a", "3a"]
            },
            VALIDATION_IDS: {
                1: ["4a"]
            },
        }
    }
}
```

### Locally

There are two possible approaches to test the algorithm locally:
- Running the vantage6 infrastructure locally and sending a task to execute the algorithm;
- Running the algorithm by mocking the calls to the vantage6 API:
An example using docker is provided (`/federated_brain_age/test/docker-compose.yaml`) allowing to run the algorithm simulating the federated architecture.
To be able to run the example, the path to the imaging data, clinical data, and brain mask should be provided in the docker compose file.
Running `docker-compose brain-age` will create a container with the necessary dependencies. Once within the container, the command `python3 run_master.py` will start the training with the data provided.
