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

## Running

Vantage6 is used to implement the PHT infrastructure, implying dependencies on this library (e.g., communication between nodes and server) in the algorithm implementation. However, as demonstrated by the examples, it's simple to decouple the algorithm core and adapt to a different infrastructure.

Running the algorithm as a task using the vantage6 client:
```
```

### Locally
