import os

from tensorflow import keras

from federated_brain_age.constants import *
 
class LossHistory(keras.callbacks.Callback):
    def __init__(self, model, store_models=False):
        # self.ne = epochs
        # self.mv = modelversion 
        self.model_class = model
        self.best_model = None
        self.best_mae = None
        self.best_epoch = -1
        self.store_models = store_models

    def on_train_begin(self, logs={}):
        self.train_metrics = {
            MAE: [],
            MSE: [],
            SDAE: [],
            SDSE: [],
        }
        self.val_metrics = {
            MAE: [],
            MSE: [],
            SDAE: [],
            SDSE: [],
        }
        print('Start training ...')

    def on_epoch_end(self, epoch, logs={}):
        # Predictions for both sets
        local_predictions = self.model_class.predict()
        # Calculate the metrics for the training set
        # (the ones provided by tensor are an average of the metrics by batch)
        metrics_epoch = self.model_class.get_metrics(
            self.model_class.train_loader,
            list(local_predictions[TRAIN].values()),
        )
        for metric, value in metrics_epoch.items():
            self.train_metrics[metric].append(value)
        print(f"Training MAE {metrics_epoch['mae']} MSE {metrics_epoch['mse']}")
        print(metrics_epoch)
        # Calculate the metrics for the validation set
        metrics_epoch = self.model_class.get_metrics(
            self.model_class.validation_loader,
            list(local_predictions[VALIDATION].values()),
        )
        for metric, value in metrics_epoch.items():
            self.val_metrics[metric].append(value)
        print(f"Validation MAE {metrics_epoch['mae']} MSE {metrics_epoch['mse']}")
        print(metrics_epoch)
        # self.val_epoch_mae.append(logs.get('val_mae'))
        # self.val_epoch_mse.append(logs.get('val_mse'))

        # Model Selection
        if self.best_mae is None or self.best_mae > logs.get('val_mae'):
            self.best_mae = logs.get('val_mae')
            self.best_epoch = epoch
            self.best_model = self.model.get_weights()
            if self.store_models:
                self.model_class.save_model(suffix=f"-{str(epoch)}")

    def on_train_end(self, logs=None):
        print(self.train_metrics)
        print(self.val_metrics)
        print(f"Best model at epoch {self.best_epoch} with a MAE of {self.best_mae}")
        print("Stop training")
