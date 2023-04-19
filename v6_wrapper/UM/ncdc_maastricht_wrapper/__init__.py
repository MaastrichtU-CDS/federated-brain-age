import time

import federated_brain_age as brain_age

def master(client, db_client, parameters = None, org_ids = None, algorithm_image = None):
    # Calling the master and providing the algorithm image to run in the GPU
    return brain_age.master(client, db_client, parameters, org_ids, algorithm_image)

def RPC_node(client, *args, **kwargs):
    # Finalize any last procedures
    return {}