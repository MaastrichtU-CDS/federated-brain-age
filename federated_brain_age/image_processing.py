import numpy as np

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
def zerocrop_img(img, padding=False, crop_indeces=None):
    #set crop locations if there are none yet or if requested
    if crop_indeces is None:
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
