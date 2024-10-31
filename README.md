# Federated Brain Age

Federated implementation to train a CNN model for gray matter age prediction (biomarker for risk of dementia).
Developed in the context of NCDC (Netherlands Consortium of Dementia Cohorts) combining 3 cohorts with large population-based studies.

- [Federated Brain Age](#federated-brain-age)
  - [Description](#description)
  - [Architecture](#architecture)
  - [Data](#data)
  - [Build Docker Image](#build-docker-image)
  - [Running](#running)
    - [Vantage6](#vantage6)
    - [Locally](#locally)
  - [Publication](#publication)

## Description

The base for this work consists of an algorithm, a Convolutional Neural Network (CNN), that predicts a person's age based on imaging data and clinical variables. Here, we adopted a CNN to the federated learning context to train, validate, and test it across multiple cohorts.

The CNN was based on the implementation from the following [repository](https://gitlab.com/radiology/neuro/brain-age/brain-age) and 
[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6800321/).

## Architecture

The federated architecture for this project follows the Personal Health Train (PHT) concept. Each participating organization keeps the data locally, only sharing aggregated information that doesn't disclose individual-level data.

In practice, we installed [Vantage6](https://distributedlearning.ai) in each node (one for each cohort) which provided the necessary tools to establish the connection (a server controls the communication with each cohort via https). Since the CNN training required more intensive computation and GPU resources are not continuously available for a single project, we developed an extension of the Vantage6 node Docker image ([/v6_wrapper](/v6_wrapper)) to use high performance computing (HPC)/GPU clusters in each cohort. In this version, a wrapper is specifically created for each cohort, which comunicates with the HPC/GPU cluster to perform the training when necessary. Such solution was necessary since each cluster had a different platform available (e.g., openshift cluster, singularity based cluster).

We performed this work using Vantage6 version 2.1.0 for the node (python libraries: `vantage6==2.1.1`, `vantage6-node==2.1.0`, `vantage6-client==2.1.0`, `vantage6-common==2.1.0`).

## Data

Facilitating the process of training an algorithm using a federated approach requires a certain level of harmonization between the data in each center.
To accomplish this, the data is expected to follow a similar structure:
* Imaging data: Results from the pre-processing pipeline should be available directly in the cluster.
* Clinical data: Available in a CSV file with the following columns: `id,clinical_id,imaging_id,age,sex,is_training_data` (example provided in [/test/example/clinical_data.csv](/test/example/clinical_data.csv)).

## Build Docker Image

To build the docker image for CPU, run the following command: `docker image build -t brain-age:latest .`
To build the docker image for GPU (based on the tensorflow docker image), run the following command: `docker image build -f Dockerfile.gpu -t brain-age-gpu:latest .`

## Running

Vantage6 is used to implement the PHT infrastructure, implying dependencies on this library (e.g., communication between nodes and server) in the algorithm implementation. However, as demonstrated by the examples ([/test/run_cnn.py](/test/run_cnn.py)), it's simple to decouple the algorithm core and adapt to a different infrastructure.

### Vantage6

An example on how to send a new task using Vantage6 is provided in [/scripts/new_task_v6.py](/scripts/new_task_v6.py)

Parameter description:
- TASK: task to execute (TRAIN, CHECK, PREDICT);
  - TRAIN: train the model;
  - CHECK: evaluate if the necessary components are available (GPU cluster, data, XNAT, ...)
- MODEL_ID: identifies the task to later resume if necessary;
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
    MODEL_ID: "test",
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
An example using docker is provided ([/federated_brain_age/test/docker-compose.yaml](/federated_brain_age/test/docker-compose.yaml)) allowing to run the algorithm simulating the federated architecture.
To be able to run the example, the path to the imaging data, clinical data, and brain mask should be provided in the docker compose file.
Running `docker-compose brain-age` will create a container with the necessary dependencies. Once within the container, the command `python3 run_master.py` will start the training with the data provided.

## Publication

If you're interested in knowing more details, our work is currenctly available as a [pre-print](https://doi.org/10.48550/arXiv.2409.01235).
