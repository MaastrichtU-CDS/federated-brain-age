import math
import nibabel as nib
import numpy as np
import os
from federated_brain_age.utils import get_parameter

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Conv3D, MaxPooling3D, BatchNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import Input, concatenate

from federated_brain_age.callbacks import DecayingLRSchedule
from federated_brain_age.constants import *
from federated_brain_age.data_loader import DataLoader
from federated_brain_age.image_processing import zerocrop_img, imgZeropad

DEFAULT_HYPERPARAMETERS = {
    INPUT_SHAPE: (160, 192, 144, 1),
    LEARNING_RATE: 0.001,
    BETA1: 0.9,
    BETA2: 0.999,
    EPSILON: 1e-08,
    DECAY: 1e-4,
    USE_PADDING: True,
    CROP_INDEXES: None,
    AUGMENT_TRAIN: True,
    IMG_SCALE: 1.0,
    BATCH_SIZE: 1,
    PATIENTS_PER_EPOCH: 4, # steps_per_epoch = patients_per_epoch / batch_size
    EPOCHS: 4,
    DROPOUT: 0.2,
    STARTING_STEP: 0,
    DECAY_STEPS: 1,
    ROUNDS: 0,
}

DEFAULT_MASK_NAME = "Brain_GM_mask_1mm_MNI_kNN_conservative.nii.gz"

class BrainAge:
    def __init__(self, parameters, id, images_path, db_type, db_client, train_participants, val_participants):
        """ Initialize the CNN model.
        """
        self.parameters = parameters
        self.id = id
        self.mask = None
        self.crop = None
        self.images_path = images_path
        self.train_loader = DataLoader(images_path, db_type, db_client, train_participants)
        self.validation_loader = DataLoader(images_path, db_type, db_client, val_participants)
        self.model = self.cnn_model(self.get_parameter)
        # Load the mask if required and available
        mask_path = f"{os.getenv(MODEL_FOLDER)}/{os.getenv(MASK_FILENAME, DEFAULT_MASK_NAME)}"
        if USE_MASK in parameters and parameters[USE_MASK] and os.path.exists(mask_path):
            self.mask = nib.load(mask_path).get_data()

    @staticmethod
    def cnn_model(parameters):
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
        
        y2 = Dense(32, activation=RELU)(y1)
        y2 = Dropout(parameters(DROPOUT))(y2)

        final = Dense(1, activation='linear')(y2)

        model = Model(inputs=[input1, input2], outputs=final)

        # Adam optimizer with an extended learning rate sceduler
        # to allow restarting the training from a previous point
        # in the same conditions.
        steps_per_epoch = parameters(PATIENTS_PER_EPOCH) / parameters(BATCH_SIZE)
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
            loss='mean_squared_error',
            optimizer=adam_opt,
            metrics=['mae', 'mse']
        )
        
        return model

    def get_parameter(self, parameter):
        """ Get parameter from the parameters provided, otherwise use the
            default value.
        """
        return self.parameters[parameter] if parameter in self.parameters \
            else DEFAULT_HYPERPARAMETERS[parameter]

    def initialize(self):
        """ Initialize and load the necessary information.
        """
        if self.mask is not None:
            # when applying a mask, initialize zerocropping
            self.crop = imgZeropad(self.mask, use_padding=self.get_parameter(USE_PADDING))
            img_size = np.array(np.array(self.crop.zerocrop_img(self.mask)).shape)
            # img_size = np.array(np.array(zerocrop_img(self.mask, padding=self.get_parameter(USE_PADDING))).shape)
        else:
            # TODO: Getting the first scan may require some changes in the data folder path
            img_size = np.array(np.array(nib.load(self.images_path + os.listdir(self.images_path)[0]).get_data()).shape)
        return [int(math.ceil(img_d)) for img_d in img_size * self.get_parameter(IMG_SCALE)]
    
    def save_model(self):
        """ Save the CNN model.
        """
        with open(f"{os.getenv(MODEL_FOLDER)}/{self.id}/model.json", 'w') as json_file:
            json_file.write(self.model.to_json())
        ModelCheckpoint(f"{os.getenv(MODEL_FOLDER)}/{self.id}/model.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    def load_model(self):
        """ Load the CNN model from a previous session.
        """
        self.model = keras.models.load_model(f"{os.getenv(MODEL_FOLDER)}/{self.id}/model.h5")
        self.model.summary()

    def train(self):
        """ Train the CNN model.
        """
        model_version=f"BrainAge_{self.id}"
        batch_size = self.get_parameter(BATCH_SIZE)
        img_size = self.initialize()
        img_scale = self.get_parameter(IMG_SCALE)
        # history = LossHistory(epochs, modelversion)
        #checkpoint = self.save_model()
        stoptraining = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')

        patients_per_epoch = min(self.get_parameter(PATIENTS_PER_EPOCH), len(self.train_loader.participants))
        steps_per_epoch = int(math.ceil(float(patients_per_epoch)/batch_size))
        validation_steps = int(math.ceil(float(len(self.validation_loader.participants))/batch_size))

        return self.model.fit(
            self.train_loader.data_generator(
                img_size, batch_size, img_scale, mask=self.mask, augment=False, mode=[], shuffle=True, crop=self.crop
            ),
            steps_per_epoch=steps_per_epoch,
            epochs=self.get_parameter(EPOCHS),
            validation_data=self.validation_loader.data_generator(
                img_size, batch_size, img_scale, mask=self.mask, augment=False, mode=[], shuffle=False, crop=self.crop
            ),
            validation_steps=validation_steps,
            max_queue_size=1,
            callbacks=[stoptraining])
            #callbacks=[history, checkpoint, stoptraining])
