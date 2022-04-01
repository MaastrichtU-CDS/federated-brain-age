import math
import nibabel as nib
import numpy as np
import os
from federated_brain_age.utils import get_parameter

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import Input, concatenate, multiply, add, Reshape, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from federated_brain_age.constants import *
from federated_brain_age.image_processing import zerocrop_img

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
    PATIENTS_PER_EPOCH: 4, # steps_per_epoch = patients_per_epoch/batch_size
    EPOCHS: 4,
}

DEFAULT_MASK_NAME = "Brain_GM_mask_1mm_MNI_kNN_conservative.nii.gz"

class BrainAge:
    def __init__(self, parameters, id):
        """ Initialize the CNN model.
        """
        self.parameters = parameters
        self.id = id
        self.model = BrainAge.cnn_model()
        self.model.summary()
        self.mask = None
        
        # Load the mask if available and necessary
        mask_path = f"{os.getenv(MODEL_FOLDER)}/{self.id}/{os.getenv(MASK_FILENAME, DEFAULT_MASK_NAME)}"
        if USE_MASK in self.parameters and self.parameters[USE_MASK] and os.path.exists(mask_path):
            self.mask = nib.load().get_data(mask_path)

    def get_parameter(self, parameter):
        """ Get parameter from the parameters provided, otherwise use the
            default value.
        """
        return self.parameters[parameter] if parameter in self.parameters \
            else DEFAULT_HYPERPARAMETERS[parameter]

    def initialize(self):
        """ Initialize and load the necessary information.
        """
        if self.mask:
            # when applying a mask, initialize zerocropping
            img_size = np.array(np.array(zerocrop_img(self.mask, True, padding=self.get_parameter(USE_PADDING))).shape)
        else:
            # TODO: Getting the first scan may require some changes in the data folder path
            img_size = np.array(np.array(nib.load(os.getenv(DATA_FOLDER) + os.listdir(os.getenv(DATA_FOLDER))[0]).get_data()).shape)
        return [int(math.ceil(img_d)) for img_d in img_size * self.get_parameter(IMG_SCALE)]

    def cnn_model(self):
        """ Define the CNN mode.
        """
        input1 = Input(self.get_parameter(INPUT_SHAPE))

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
        y2 = Dropout(0.2)(y2)

        final = Dense(1, activation='linear')(y2)

        model = Model(inputs=[input1, input2], outputs=final)

        adam_opt = keras.optimizers.Adam(
            lr = self.get_parameter(LEARNING_RATE),
            beta_1 = self.get_parameter(BETA1),
            beta_2 = self.get_parameter(BETA2),
            epsilon = self.get_parameter(EPSILON), 
            decay = self.get_parameter(DECAY),
        )
        model.compile(
            loss='mean_squared_error',
            optimizer=adam_opt,
            metrics=['mae', 'mse']
        )
        
        return model
    
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
        model_version=f"BrainAge_{id}"
        batch_size = self.get_parameter(BATCH_SIZE)
        img_size = self.initialize()
        # history = LossHistory(epochs, modelversion)
        checkpoint = self.save_model(model_version, self.model)
        stoptraining = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')

        patients_per_epoch = min(self.get_parameter(PATIENTS_PER_EPOCH), train_size)
        steps_per_epoch = int(math.ceil(float(patients_per_epoch)/batch_size))
        validation_steps = int(math.ceil(float(validation_size)/batch_size))

        self.model.fit_generator(
            data_generator(
                list(train_set),
                img_size,
                batch_size,
                img_scale,
                mask,
                augment=augment_train,
                mode='train'
            ),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=data_generator(list(validation_set), img_size, batch_size, img_scale, mask),
            validation_steps=validation_steps,
            max_queue_size=1,
            callbacks=[history, checkpoint, stoptraining])

        print('Succesfully trained the model.')
