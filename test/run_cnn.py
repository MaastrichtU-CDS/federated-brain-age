import os

import tensorflow as tf

from federated_brain_age.brain_age import BrainAge
from federated_brain_age.data_loader import DataLoader

data_path = "/mnt"

# training = DataLoader(
#     data_path,
#     "CSV",
#     data_path,
#     ["1002822", "1001009"]
# )

# validation =  DataLoader(
#     data_path,
#     "CSV",
#     data_path,
#     ["1000049"]
# )

parameters = {
    "USE_MASK": True,
    "EPOCHS": 1,
}
#model = BrainAge(parameters, "test", data_path + "/data/", "CSV", data_path + "/clinical_data.csv", ["1002822", "1001009"], ["1000049"])
model = BrainAge(parameters, "test", data_path + "/data/", "CSV", data_path + "/dataset.csv", ["2a", "3a"], ["4a"])
model.train()

file = model.model.to_json()


# Rplacing the weights
# weights = model.model.get_weights()
# weights_copy = model.model.get_weights()
# ww = []
# for i in range(0, len(weights)):
#     ww.append(tf.reduce_mean([w[i] for w in [weights, weights_copy]], axis=0))
# model2 = BrainAge(parameters, "test", data_path + "/data/", "CSV", data_path + "/dataset.csv", ["2a", "3a"], ["4a"])
# model2.model.set_weights(ww)
# model2.train()
