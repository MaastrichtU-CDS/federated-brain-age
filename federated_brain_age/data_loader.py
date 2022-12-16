import pandas as pd
import numpy as np
import nibabel as nib
import random
import os

from federated_brain_age.constants import *
from federated_brain_age.image_processing import zerocrop_img, resize_img
from federated_brain_age.utils import read_csv

class DataLoader:
    def __init__(self, images_path, db_type, db_client, training=True, validation=False, seed=None, split=None, exclude=[]):
        self.images_path = images_path
        self.seed = seed
        if db_type == DB_CSV:
            #data = read_csv(db_client, column_names=[ID, AGE, SEX], filter=participants, filterKey=ID)
            data = read_csv(db_client, column_names=[ID, CLINICAL_ID, IMAGING_ID, AGE, SEX, IS_TRAINING_DATA])
        elif db_type == DB_POSTGRES:
            # TODO: Query the database to obtain the data.
            pass
        else:
            raise Exception("Missing the database type variable.")

        self.clinical_data = pd.DataFrame(
            data,
            columns = [ID, CLINICAL_ID, IMAGING_ID, AGE, SEX, IS_TRAINING_DATA],
            dtype=str
        )
        self.clinical_data = self.clinical_data.set_index(ID)
        self.participant_list = self.validate_participants(
            list([str(id) for id in data[ID]]),
            training,
            validation
        )
        # Set the seed
        random.seed(self.seed)
        # Make the data split
        self.participants = []
        if len(self.participant_list[0]) > 0:
            if training:
                self.participants = [
                    self.participant_list[0][i] for i in sorted(
                        random.sample(
                            list(range(0, len(self.participant_list[0]))),
                            int(len(self.participant_list[0]) * (split or DEFAULT_SPLIT)),
                        )
                    )
                ]
            else:
                self.participants = [
                    participant for participant in self.participant_list[0] if participant not in exclude
                ]

    def validate_participants(self, participants, training, validation):
        """ Validate if the data from the participants
            is available.
        """
        participants_list = [[], [], []]
        for participant in participants:
            try:
                patient_info = self.clinical_data.loc[str(participant)]
                # The flag 'IS_TRAINING_DATA' identifies the data that will be used for
                # training and thus splitted between training and validation.
                if (training or validation) == int(patient_info[IS_TRAINING_DATA]):
                    patient_filename = str(patient_info[IMAGING_ID]).strip() + (os.getenv(IMAGE_SUFFIX) or DEFAULT_IMAGE_NAME)
                    if os.path.exists(os.path.join(self.images_path, patient_filename)):
                        participants_list[0].append(participant)
                    else:
                        participants_list[2].append(participant)
            except KeyError as error:
                participants_list[1].append(participant)
        return participants_list
        

    def retrieve_data(self, patient_index, img_size, img_scale=1.0, mask=None, mode=[], crop=None):
        """
        Function to retrieve data from a single patient
        
        Inputs:
        - patient_index = list of bigrfullnames identifying scans
        - mask = mask image if necessary [default = None]
        - mode = train, validate or test (used to find appropriate data)
        
        Outputs:
        - img_data = MRI data
        - input2 = sex
        - label = age

        """
        # Retrieve patient info and label(=SNP) of the patient
        # if mode == 'train':
        #     patient_info = train_label_set.loc[patient_index]
        # elif mode == 'validate':
        #     patient_info = validation_label_set.loc[patient_index]
        # elif mode == 'test':
        #     patient_info = test_label_set.loc[patient_index]
        # else: # validation set might not use validation flag
        #     patient_info = validation_label_set.loc[patient_index]
        patient_info = self.clinical_data.loc[patient_index]
        # Get patient label (incident dementia or not)
        label = patient_info.get(AGE)

        # Get second input (sex)
        input2 = patient_info.get(SEX)    

        # Get image
        patient_filename = patient_index.strip() + (os.getenv(IMAGE_SUFFIX) or DEFAULT_IMAGE_NAME)
        # TODO: Check if file exists
        # Probably better to be done prior to this stage and get the "real"
        # number of participants
        img = nib.load(os.path.join(self.images_path, patient_filename))
        img_data = img.get_data()

        # Apply mask to imagedata (if requested)
        if mask is not None:
            img_data = img_data * mask
            if crop is not None:
                img_data = crop.zerocrop_img(img_data)
            else:
                img_data = zerocrop_img(img_data)

        # Rescale imagedata (if requested)
        if img_scale < 1.0:
            img_data = resize_img(img_data, img_size)
        
        return np.array(img_data), np.array(int(input2)), float(label)

    def generate_batch(self, patients, img_size, img_scale=1.0, mask=None, mode=[], crop=None):
        """
        iterate through a batch of patients and get the corresponding data
        
        Input: 
        - patients = list of bigrfullnames identifying scans
        - img_size = size of MRI images
        - img_scale = scale of the MRI scans [default = 1]
        - mask = mask image if necessary [default = None]
        - mode
        
        Outputs:
        - [input data] = sex
        - [label data] = age

        """    
        # Get data of each patient
        img_data = []
        label_data = []
        sex = []
        for patient in patients:
            try:
                x, x2, y = self.retrieve_data(patient, img_size, img_scale, mask, mode, crop)
                img_data.append(x)
                sex.append(x2)
                label_data.append(y)
            except KeyError as e:
                print('\nERROR: No label found for file {}'.format(patient))
            except IOError as e:            
                print('\nERROR: Problem loading file {}. File probably corrupted.'.format(patient))
                print(e)
                

        # Convert to correct input format for network
        img_data = np.array(img_data)
        img_data = np.reshape(img_data,(-1, 160, 192, 144, 1))

        sex_data = np.array(sex)
        
        label_data = np.array(label_data)

        return ([img_data, sex_data], [label_data])

    def data_generator(self, img_size, batch_size, img_scale=1.0, mask=None, mode=[], shuffle=True, crop=None):
        """
        Provides the inputs and the label to the convolutional network during training
        
        Input:
        - img_size = size of MRI images
        - batch_size = size of batch used in training
        - img_scale = scale of the MRI scans [default = 1]
        - mask = mask image if necessary [default = None]
        
        Output:
        - Data = continous data output for batches used in training the network

        """
        while 1:
            if shuffle:
                #shuffle list/order of patients
                pl_shuffled = random.sample(self.participants, len(self.participants))
                #divide list of patients into batches
                batch_size = int(batch_size)
                patient_sublist = [pl_shuffled[p:p+batch_size] for p in range(0, len(pl_shuffled), batch_size)]
            else:
                batch_size = int(batch_size)
                patient_sublist = [self.participants[p:p+batch_size] for p in range(0, len(self.participants), batch_size)]
            count = 0
            data = []
            for batch in range(0, len(patient_sublist)):
                #get the data of a batch samples/patients
                data.append(self.generate_batch(patient_sublist[batch], img_size, img_scale, mask, mode, crop))
                count = count + len(patient_sublist[batch])
                #yield the data and pop for memory clearing
                yield data.pop()
