import json
import math
import nibabel as nib
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Conv3D, MaxPooling3D, BatchNormalization, Dropout, GlobalAveragePooling3D, Input, concatenate
from tensorflow.keras import metrics
import keras.backend as K

from federated_brain_age.callbacks import DecayingLRSchedule
from federated_brain_age.constants import *
from federated_brain_age.data_loader import DataLoader
from federated_brain_age.image_processing import imgZeropad
from federated_brain_age.loss_history import LossHistory
from federated_brain_age.utils import np_array_to_list

DEFAULT_HYPERPARAMETERS = {
    INPUT_SHAPE: (160, 192, 144, 1),
    LEARNING_RATE: 0.001,
    BETA1: 0.9,
    BETA2: 0.999,
    EPSILON: 1e-08,
    DECAY: 1e-4,
    USE_PADDING: True,
    CROP_INDEXES: None,
    AUGMENT_TRAIN: False,
    IMG_SCALE: 1.0,
    BATCH_SIZE: 4,
    PATIENTS_PER_EPOCH: 4, # steps_per_epoch = patients_per_epoch / batch_size
    EPOCHS: 4,
    DROPOUT: 0.2,
    STARTING_STEP: 0,
    DECAY_STEPS: 1,
    ROUNDS: 0,
    EARLY_STOPPING: True,
    USE_MASK: True,
}

DEFAULT_MASK_NAME = "Brain_GM_mask_1mm_MNI_kNN_conservative.nii.gz"

def loss_fun(y_true, y_pred, sample_weight=None):
    weights = 1
    axis = -1
    sample_weight = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 3, 39: 3, 40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 2, 47: 2, 48: 2, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1, 61: 1, 62: 2, 63: 2, 64: 2, 65: 2, 66: 2, 67: 2, 68: 2, 69: 2, 70: 2, 71: 2, 72: 2, 73: 2, 74: 2, 75: 3, 76: 3, 77: 3, 78: 3, 79: 3, 80: 3, 81: 3, 82: 3, 83: 3, 84: 3}
    #sample_weight = [i/len(sample_weight.values()) if i == 1 else i for i in list(sample_weight.values())]
    sample_weight = list(sample_weight.values())
    # classSelectors = K.argmax(y_true, axis=axis)
    classSelectors = [K.equal(K.cast(i, K.floatx()), y_true) for i in range(len(sample_weight))]
    classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]
    weights = [sel * w for sel,w in zip(classSelectors, sample_weight)]
    weightMultiplier = weights[0]
    for i in range(1, len(weights)):
        weightMultiplier = weightMultiplier + weights[i] 
    # if sample_weight:
    #     weights = K.sum(
    #         [K.cast(K.equal(y_true, list(sample_weight.keys())[i]), K.floatx()) * list(sample_weight.items())[i] \
    #             for i in range(0, len(sample_weight.items()))]
    # )
    return K.mean(K.square((y_true - y_pred) * weightMultiplier))

class BrainAge:
    def __init__(self, parameters, id, images_path, db_type, db_client, seed=None, split=None):
        """ Initialize the CNN model.
        """
        self.parameters = parameters
        self.id = id
        self.mask = None
        self.crop = None
        self.history = None
        self.images_path = images_path
        self.train_loader = DataLoader(
            images_path,
            db_type,
            db_client,
            training=True,
            seed=seed,
            split=split,
        )
        self.validation_loader = DataLoader(
            images_path,
            db_type,
            db_client,
            training=False,
            validation=True,
            seed=seed,
            split=split,
            exclude=self.train_loader.participants,
        )
        steps_per_epoch = int(math.ceil(float(min(
            self.get_parameter(PATIENTS_PER_EPOCH), len(self.train_loader.participants)
        ) / self.get_parameter(BATCH_SIZE))))
        self.model = self.cnn_model(self.get_parameter, steps_per_epoch)
        # Load the mask if required and available
        mask_path = f"{os.getenv(MODEL_FOLDER)}/{os.getenv(MASK_FILENAME, DEFAULT_MASK_NAME)}"
        if self.get_parameter(USE_MASK) and os.path.exists(mask_path):
            self.mask = nib.load(mask_path).get_data()
    
    @staticmethod
    def get_metrics(loader, y_pred, prefix=''):
        """ Calculate the metrics.
        """
        metrics = {
            f"{prefix}{MAE}": -1,
            f"{prefix}{MSE}": -1,
        }
        if len(loader.participants) > 0:
            ages = loader.clinical_data.loc[loader.participants, AGE].values
            y_true = list(ages.astype(float))
            metrics = {
                f"{prefix}{MAE}": float(keras.metrics.mean_absolute_error(y_true, y_pred)),
                f"{prefix}{SDAE}": float(np.std(np.absolute(np.subtract(y_true, y_pred)))),
                f"{prefix}{MSE}": float(keras.metrics.mean_squared_error(y_true, y_pred)),
                f"{prefix}{SDSE}": float(np.std(np.square(np.subtract(y_true, y_pred)))),
            }
        return metrics

    @staticmethod
    def cnn_model(parameters, steps_per_epoch = 1):
        """ Define the CNN mode.

            parameters: callback to retrieve the parameters.
        """
        input1 = Input(parameters(INPUT_SHAPE))

        c1 = Conv3D(32, kernel_size=(5,5,5), strides=(2,2,2), padding=PADDING_SAME)(input1)
        c1 = BatchNormalization()(c1)
        c1 = Activation(RELU)(c1)
        
        c2 = Conv3D(32, (3,3,3), strides=(1,1,1), padding=PADDING_SAME)(c1)
        c2 = BatchNormalization()(c2)
        c2 = Activation(RELU)(c2)
        p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)
        
        c3 = Conv3D(48, (3,3,3), strides=(1,1,1), padding=PADDING_SAME)(p2)
        c3 = BatchNormalization()(c3)
        c3 = Activation(RELU)(c3)
        
        c4 = Conv3D(48, (3,3,3), strides=(1,1,1), padding=PADDING_SAME)(c3)
        c4 = BatchNormalization()(c4)
        c4 = Activation(RELU)(c4)
        p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
        
        c5 = Conv3D(64, (3,3,3), strides=(1,1,1), padding=PADDING_SAME)(p4)
        c5 = BatchNormalization()(c5)
        c5 = Activation(RELU)(c5)
        
        c6 = Conv3D(64, (3,3,3), strides=(1,1,1), padding=PADDING_SAME)(c5)
        c6 = BatchNormalization()(c6)
        c6 = Activation(RELU)(c6)
        p6 = MaxPooling3D(pool_size=(2, 2, 2))(c6)

        c7 = Conv3D(80, (3,3,3), strides=(1,1,1), padding=PADDING_SAME)(p6)
        c7 = BatchNormalization()(c7)
        c7 = Activation(RELU)(c7)
        
        c8 = Conv3D(80, (3,3,3), strides=(1,1,1), padding=PADDING_SAME)(c7)
        c8 = BatchNormalization()(c8)
        c8 = Activation(RELU)(c8)

        x1 = GlobalAveragePooling3D()(c8)

        #right input branch
        input2 = Input((1,))

        # merging braches into final model
        y1 = concatenate([x1, input2]) # other modes: multiply, concatenate, dot
        
        y2 = Dropout(parameters(DROPOUT))(y1)
        y2 = Dense(32, activation=RELU)(y2)

        final = Dense(1, activation='linear')(y2)

        model = Model(inputs=[input1, input2], outputs=final)

        # Adam optimizer with an extended learning rate sceduler
        # to allow restarting the training from a previous point
        # in the same conditions.
        adam_opt = keras.optimizers.Adam(
            learning_rate = DecayingLRSchedule(
                parameters(LEARNING_RATE),
                parameters(DECAY),
                parameters(STARTING_STEP) or steps_per_epoch * parameters(EPOCHS) * parameters(ROUNDS),
                parameters(DECAY_STEPS),
            ),
            beta_1 = parameters(BETA1),
            beta_2 = parameters(BETA2),
            epsilon = parameters(EPSILON), 
            #decay = parameters(DECAY),
        )
        model.compile(
            loss=loss_fun,
            #loss="mean_squared_error",
            optimizer=adam_opt,
            metrics=['mae', 'mse']
        )
        
        return model

    def get_parameter(self, parameter):
        """ Get parameter from the parameters provided, otherwise use the
            default value.
        """
        return self.parameters[parameter] if parameter in self.parameters \
            else DEFAULT_HYPERPARAMETERS.get(parameter)

    def initialize(self):
        """ Initialize and load the necessary information.
        """
        if self.mask is not None:
            # when applying a mask, initialize zerocropping
            self.crop = imgZeropad(self.mask, use_padding=self.get_parameter(USE_PADDING))
            img_size = np.array(np.array(self.crop.zerocrop_img(self.mask, augment=self.get_parameter(AUGMENT_TRAIN))).shape)
            # img_size = np.array(np.array(zerocrop_img(self.mask, padding=self.get_parameter(USE_PADDING))).shape)
        else:
            # TODO: Getting the first scan may require some changes in the data folder path
            img_size = np.array(np.array(nib.load(self.images_path + os.listdir(self.images_path)[0]).get_data()).shape)
        return [int(math.ceil(img_d)) for img_d in img_size * self.get_parameter(IMG_SCALE)]
    
    def save_model(self, suffix=""):
        """ Save the CNN model.
        """
        #with open(f"{os.getenv(MODEL_FOLDER)}/{self.id}/model{suffix}.json", 'w') as json_file:
        #    json_file.write(self.model.to_json())
        with open(f"{os.getenv(MODEL_FOLDER)}/{self.id}/model{suffix}.json", 'w') as json_file:
            json_file.write(json.dumps(np_array_to_list(self.model.get_weights())))
        return ModelCheckpoint(
            f"{os.getenv(MODEL_FOLDER)}/{self.id}/model.h5",
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='auto'
        )

    def load_model(self):
        """ Load the CNN model from a previous session.
        """
        self.model = keras.models.load_model(f"{os.getenv(MODEL_FOLDER)}/{self.id}/model.h5")
        self.model.summary()

    def train(self, history=False, class_weight=None):
        """ Train the CNN model.
        """
        model_version=f"BrainAge_{self.id}"
        batch_size = self.get_parameter(BATCH_SIZE)
        img_size = self.initialize()
        img_scale = self.get_parameter(IMG_SCALE)
        # history = LossHistory(epochs, modelversion)
        # checkpoint = self.save_model()
        # Early stopping
        callbacks = []
        if self.get_parameter(EARLY_STOPPING):
            stoptraining = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
            callbacks.append(stoptraining)
        if history:
            self.history = LossHistory(self)
            callbacks.append(self.history)
        # Calculate the training steps
        patients_per_epoch = min(self.get_parameter(PATIENTS_PER_EPOCH), len(self.train_loader.participants))
        steps_per_epoch = int(math.ceil(float(patients_per_epoch)/batch_size))
        # Check if there is data for the validation
        validation_data = None
        validation_steps = None
        if len(self.validation_loader.participants) > 0:
            validation_steps = int(math.ceil(float(len(self.validation_loader.participants))/batch_size))
            validation_data = self.validation_loader.data_generator(
                img_size, batch_size, img_scale, mask=self.mask, mode=[], shuffle=False, crop=self.crop
            )
        # Train the model
        return self.model.fit(
            self.train_loader.data_generator(
                img_size, batch_size, img_scale, mask=self.mask, mode=[], shuffle=True, crop=self.crop
            ),
            steps_per_epoch = steps_per_epoch,
            epochs=self.get_parameter(EPOCHS),
            validation_data = validation_data,
            validation_steps = validation_steps,
            max_queue_size = 1,
            callbacks = callbacks,
            # class_weight = class_weight,
        )

    def predict(self, data_loader = None):
        """ Make the predictions for the data
        """
        predictions_by_participant = {}
        # Load the parameters
        batch_size = self.get_parameter(BATCH_SIZE)
        img_size = self.initialize()
        img_scale = self.get_parameter(IMG_SCALE)
        # Prepare the data loader
        if not data_loader:
            data_loader = {
                TRAIN: self.train_loader,
                VALIDATION: self.validation_loader
            }
        for key, loader in data_loader.items():
            predictions_by_participant[key] = {}
            if len(loader.participants) > 0:
                predictions = self.model.predict(
                    loader.data_generator(
                        img_size,
                        batch_size,
                        img_scale,
                        mask=self.mask,
                        mode=[],
                        shuffle=False,
                        crop=self.crop
                    ),
                    max_queue_size = 1,
                    batch_size = self.get_parameter(BATCH_SIZE),
                    steps=int(math.ceil(float(len(loader.participants))/batch_size))
                )
                predictions_by_participant[key] = dict(
                    zip(loader.participants, [float(p) for p in predictions.flatten()])
                )
        return predictions_by_participant
