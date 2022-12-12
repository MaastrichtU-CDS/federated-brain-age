# Test the brain age algorithm
import os
import random

import tensorflow as tf

from federated_brain_age.brain_age import BrainAge
from federated_brain_age.data_loader import DataLoader

data_path = "/mnt"

parameters = {
    "USE_MASK": True,
    "EPOCHS": 2,
    "PATIENTS_PER_EPOCH": 2,
    "ROUNDS": 1,
}

seed = 1

model = BrainAge(parameters, "test", data_path + "/data/", "CSV", data_path + "/dataset.csv", seed, 0.7)

result = model.train()


# img_size = model.initialize()
# batch_size = len(model.train_loader.participants)
# img_scale = model.get_parameter("IMG_SCALE")
# predictions = model.model.predict(
#     model.train_loader.data_generator(
#         img_size, 2, img_scale, mask=model.mask, augment=False, mode=[], shuffle=False, crop=model.crop
#     ),
#     max_queue_size = 1,
#     batch_size=2,
#     steps=1
# )
# print("Predictions")
# print(predictions)

# dict_keys(['loss', 'mae', 'mse', 'val_loss', 'val_mae', 'val_mse'])
output = {}
for metric in result.history.keys():
    output[metric] = result.history[metric][-1]
print(output)
