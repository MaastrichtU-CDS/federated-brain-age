import pandas as pd
import os
import numpy as np
import nibabel as nib
import sys
import math
import random
import csv
import nipy
import seaborn as sns
from datetime import datetime
from dateutil import relativedelta
import gc

from scipy import ndimage as nd
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

import tensorflow as tf
from tensorflow.python.framework import ops

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import Input, concatenate, multiply, add, Reshape, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

import h5py

IMAGE_DIR = 'path to VBM image directory'
MASK_DIR = 'path to Mask image directory'
MODEL_DIR = 'path to load and save model/results'

class LossHistory(keras.callbacks.Callback):
    def __init__(self, epochs, modelversion):
        self.ne = epochs
        self.mv = modelversion        
    
    def on_train_begin(self, logs={}):
        self.batch_num = 0
        self.batch_losses = []
        self.epoch_losses = []

        print('Start training ...')
        
        self.stats = ['loss'] #TODO: check
        self.logs = [{} for _ in range(self.ne)]

        self.evolution_file = 'evolution_'+self.mv+'.csv'
        with open(MODEL_DIR+self.evolution_file, "w") as f:
            f.write(';'.join(self.stats + ['val_'+s for s in self.stats]) + "\n")
        
        self.progress_file = 'training_progress_'+self.mv+'.out'
        with open(MODEL_DIR+self.progress_file, "w") as f:
            f.write('Start training ...\n')
            
    def on_batch_end(self, epoch, logs={}):
        self.batch_losses.append(logs.get('loss'))

        
        with open(MODEL_DIR+self.progress_file, "a") as f:
            f.write('  >> batch {} >> loss:{} \r'.format(self.batch_num, self.batch_losses[-1]))
        
        self.batch_num += 1
        
    def on_epoch_end(self, epoch, logs={}):
        self.batch_num = 0
        self.epoch_losses.append(logs.get('loss'))
        
        
#        print('\n    >>> logs:', logs)
        self.logs[epoch] = logs
#        evolution_file = 'evolution_'+self.mv+'.csv'
        loss_fig = 'loss_'+self.mv+'.png'
        
        with open(MODEL_DIR+self.evolution_file, "a") as myfile:
            num_stats = len(self.stats)
            
            plt.figure(figsize=(40, num_stats*15))
            plt.suptitle(loss_fig, fontsize=34, fontweight='bold')

            gs = gridspec.GridSpec(len(self.stats), 2) 

            last_losses = []
            last_val_losses = []
            for idx, stat in enumerate(self.stats):
                losses = [self.logs[e][stat] for e in range(epoch+1)]
                last_losses.append('{}'.format(losses[-1]))
                val_losses = [self.logs[e]['val_'+stat] for e in range(epoch+1)]
                last_val_losses.append('{}'.format(val_losses[-1]))

                plt.subplot(gs[idx,0])
                plt.ylabel(stat, fontsize=34)
                plt.plot(range(0, epoch+1), losses, '-', color = 'b')
                plt.plot(range(0, epoch+1), val_losses, '-', color = 'r')
                plt.tick_params(axis='x', labelsize=30)
                plt.tick_params(axis='y', labelsize=30)
                plt.grid(True)

                recent_n = 10
                recent_losses = losses[-recent_n:]
                recent_val_losses = val_losses[-recent_n:]
                miny_range = 5
                lowery = min([min(losses), recent_losses[-1]-miny_range, min(val_losses), recent_val_losses[-1]-miny_range])
                uppery = max([max(recent_losses), recent_losses[-1]+miny_range, max(recent_val_losses), recent_val_losses[-1]+miny_range])
                plt.subplot(gs[idx,1])
                plt.ylabel(stat, fontsize=34)
                plt.plot(range(0, epoch+1), losses, '-', color = 'b')
                plt.plot(range(0, epoch+1), val_losses, '-', color = 'r')
                plt.ylim(lowery, uppery)
                plt.tick_params(axis='x', labelsize=30)
                plt.tick_params(axis='y', labelsize=30)
                plt.grid(True)
                
            myfile.write(';'.join(last_losses + last_val_losses) + '\n')
            try:                
                plt.savefig(MODEL_DIR+loss_fig)
            except Exception as inst:
                print(type(inst))
                print(inst)
            plt.close()
        

        with open(MODEL_DIR+self.progress_file, "a") as f:
            f.write('epoch {}/{}:\n'.format(epoch, self.ne))
            for idx, stat in enumerate(self.stats):
                f.write('  {} = {}\n  val_{} = {}\n'.format(stat, last_losses[idx], stat, last_val_losses[idx]))

        gc.collect()

def save_model(name, model):
    model_file = 'model_'+name+'.json'
    # serialize model to JSON
    with open(MODEL_DIR+model_file, 'w') as json_file:
        json_file.write(model.to_json())
    print('Saved model to '+MODEL_DIR+model_file)

#save the best model on validation set
def save_checkpoint(name, model):
    save_model(name, model)
    weights_file = 'model_'+name+'.h5'
    return ModelCheckpoint(MODEL_DIR+weights_file, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

def save_history(name, history, score, sets, distrs):
    history_file = 'history_'+name+'.h5'

    f = h5py.File(MODEL_DIR+history_file, 'w')

    f.create_dataset('batch_losses', data=history.batch_losses)
    f.create_dataset('epoch_losses', data=history.epoch_losses)
    f.create_dataset("score", data=score)

    f.close()

    print('Saved history to '+MODEL_DIR+history_file)

class imgZeropad:

    def __init__(self, img, use_padding=False):
        self.set_crop(img, use_padding)
    
    #set crop locations
    def set_crop(self, img, use_padding=False):
        # argwhere will give you the coordinates of every non-zero point
        true_data = np.argwhere(img)
        # take the smallest points and use them as the top left of your crop
        top_left = true_data.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_data.max(axis=0)
        crop_indeces = [top_left, bottom_right+1]  # plus 1 because slice isn't inclusive

        print('crop set to x[{}:{}], y[{}:{}], z[{}:{}]'.format(crop_indeces[0][0], crop_indeces[1][0], 
                                                                crop_indeces[0][1], crop_indeces[1][1], 
                                                                crop_indeces[0][2], crop_indeces[1][2]))

        if use_padding == True:
            shape = crop_indeces[1]-crop_indeces[0]
            bottom_net = shape.astype(float)/2/2**3
            top_net = np.ceil(bottom_net)*2*2**3
            padding = (top_net-shape)/2
            print('applying [{},{},{}] padding to image..'.format(padding[0], padding[1], padding[2]))
            padding_l = padding.astype(int)
            padding_r = np.ceil(padding).astype(int)
            crop_indeces[0] -= padding_l
            crop_indeces[1] += padding_r

            print('crop set to x[{}:{}], y[{}:{}], z[{}:{}]'.format(crop_indeces[0][0], crop_indeces[1][0], 
                                                                    crop_indeces[0][1], crop_indeces[1][1], 
                                                                    crop_indeces[0][2], crop_indeces[1][2]))
        else:
            padding = np.zeros(3)
        self.crop_indeces = crop_indeces
        self.padding = padding
        
        shape = crop_indeces[1]-crop_indeces[0]
        self.img_size = (shape[0], shape[1], shape[2])

    #crop according to crop_indeces
    def zerocrop_img(self, img, augment=False):
        if augment:
            randx = np.random.rand(3)*2-1
            new_crop = self.crop_indeces+(self.padding*randx).astype(int)

            cropped_img = img[new_crop[0][0]:new_crop[1][0],  
                              new_crop[0][1]:new_crop[1][1],
                              new_crop[0][2]:new_crop[1][2]]

            flip_axis = np.random.rand(3)
            if round(flip_axis[0]):
                cropped_img = cropped_img[::-1,:,:]
            if round(flip_axis[1]):
                cropped_img = cropped_img[:,::-1,:]
            if round(flip_axis[2]):
                cropped_img = cropped_img[:,:,::-1]
                
        else:
            cropped_img = img[self.crop_indeces[0][0]:self.crop_indeces[1][0],  
                              self.crop_indeces[0][1]:self.crop_indeces[1][1],
                              self.crop_indeces[0][2]:self.crop_indeces[1][2]]
            
        return cropped_img


#crops the zero-margin of a 3D image (based on mask)
def zerocrop_img(img, set_crop=False, padding=False):
    global crop_indeces
    
    #set crop locations if there are none yet or if requested
    if (crop_indeces is None) or (set_crop):
        # argwhere will give you the coordinates of every non-zero point
        true_data = np.argwhere(img)
        # take the smallest points and use them as the top left of your crop
        top_left = true_data.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_data.max(axis=0)
        crop_indeces = [top_left, bottom_right+1]  # plus 1 because slice isn't inclusive
        
        print('crop set to x[{}:{}], y[{}:{}], z[{}:{}]'.format(crop_indeces[0][0], crop_indeces[1][0], 
                                                                crop_indeces[0][1], crop_indeces[1][1], 
                                                                crop_indeces[0][2], crop_indeces[1][2]))

        if padding == True:
            shape = crop_indeces[1]-crop_indeces[0]
            bottom_unet = shape.astype(float)/2/2**3
            top_unet = np.ceil(bottom_unet)*2*2**3
            padding = (top_unet-shape)/2
            print('applying [{},{},{}] padding to image..'.format(padding[0], padding[1], padding[2]))
            padding_l = padding.astype(int)
            padding_r = np.ceil(padding).astype(int)
            crop_indeces[0] -= padding_l
            crop_indeces[1] += padding_r

            print('crop set to x[{}:{}], y[{}:{}], z[{}:{}]'.format(crop_indeces[0][0], crop_indeces[1][0], 
                                                                    crop_indeces[0][1], crop_indeces[1][1], 
                                                                    crop_indeces[0][2], crop_indeces[1][2]))
    
    try:
        cropped_img = img[crop_indeces[0][0]:crop_indeces[1][0],  
                          crop_indeces[0][1]:crop_indeces[1][1],
                          crop_indeces[0][2]:crop_indeces[1][2]]
        return cropped_img
    except ValueError:
        print('ERROR: No crop_indeces defined for zerocrop. Returning full image...')
        return img

def retrieve_data(patient_index, img_size, img_scale=1.0, mask=None, augment=False, mode=[]):
    """
    Function to retrieve data from a single patient
    
    Inputs:
    - patient_index = list of bigrfullnames identifying scans
    - img_size = size of MRI images
    - img_scale = scale of the MRI scans [default = 1]
    - mask = mask image if necessary [default = None]
    - augment = Boolean if data augmentation should be used [default = False]
    - mode = train, validate or test (used to find appropriate data)
    
    Outputs:
    - img_data = MRI data
    - input2 = sex
    - label = age

    """
    # Retrieve patient info and label(=SNP) of the patient
    if mode == 'train':
        patient_info = train_label_set.loc[patient_index]
    elif mode == 'validate':
        patient_info = validation_label_set.loc[patient_index]
    elif mode == 'test':
        patient_info = test_label_set.loc[patient_index]
    else: # validation set might not use validation flag
        patient_info = validation_label_set.loc[patient_index]
    
    # Get patient label (incident dementia or not)
    label = patient_info.get('age')
    
    # Get second input (sex)
    input2 = patient_info.get('sex')    
    

    # Get image
    patient_filename = patient_index.strip()+'_GM_to_template_GM_mod.nii.gz'
    img = nib.load(IMAGE_DIR+patient_filename)  
    img_data = img.get_data()
    
    # Apply mask to imagedata (if requested)
    if mask is not None:
        img_data = img_data*mask
        img_data = zerocrop_img(img_data)

    # Rescale imagedata (if requested)
    if img_scale < 1.0:
        img_data = resize_img(img_data, img_size)
    
    return np.array(img_data), np.array(int(input2)), label


def generate_batch(patients, img_size, img_scale=1.0, mask=None, augment=False, mode=[]):
    """
    iterate through a batch of patients and get the corresponding data
    
    Input: 
    - patients = list of bigrfullnames identifying scans
    - img_size = size of MRI images
    - img_scale = scale of the MRI scans [default = 1]
    - mask = mask image if necessary [default = None]
    - augment = Boolean if data augmentation should be used [default = False]
    - mode
    
    Outputs:
    - [input data] = sex
    - [label data] = age

    """    
    #get data of each patient
    img_data = []
    label_data = []
    sex = []
    for patient in patients:
        try:
            x, x2, y = retrieve_data(patient, img_size, img_scale, mask, augment, mode)
            img_data.append(x)
            sex.append(x2)
            label_data.append(y)
        except KeyError as e:
            print('\nERROR: No label found for file {}'.format(patient))
        except IOError as e:            
            print('\nERROR: Problem loading file {}. File probably corrupted.'.format(patient))
            

    #convert to correct input format for network
    img_data = np.array(img_data)
    img_data = np.reshape(img_data,(-1, 160, 192, 144, 1))

    sex_data = np.array(sex)
    
    label_data = np.array([label_data])


    return ([img_data, sex_data], [label_data])

def data_generator(patient_list, img_size, batch_size, img_scale=1.0, mask=None, augment=False, mode=[], shuffle=True):
    """
    Provides the inputs and the label to the convolutional network during training
    
    Input:
    - patient_list = list of bigrfullnames identifying scans
    - img_size = size of MRI images
    - batch_size = size of batch used in training
    - img_scale = scale of the MRI scans [default = 1]
    - mask = mask image if necessary [default = None]
    - augment = Boolean if data augmentation should be used [default = False]
    
    Output:
    - Data = continous data output for batches used in training the network

    """
    while 1:
        if shuffle:
            #shuffle list/order of patients
            pl_shuffled = random.sample(patient_list, len(patient_list))
            #divide list of patients into batches
            batch_size = int(batch_size)
            patient_sublist = [pl_shuffled[p:p+batch_size] for p in range(0, len(pl_shuffled), batch_size)]
        else:
            batch_size = int(batch_size)
            patient_sublist = [patient_list[p:p+batch_size] for p in range(0, len(patient_list), batch_size)]
        count = 0
        data = []
        for batch in range(0, len(patient_sublist)):         
            #get the data of a batch samples/patients
            data.append(generate_batch(patient_sublist[batch], img_size, img_scale, mask, augment, mode))
            count = count + len(patient_sublist[batch])
            #yield the data and pop for memory clearing
            yield data.pop()

