'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the custom data augmentation functions.
'''

import numpy as np

# Wrapper for Keras to use these functions
def custom_train_data_augmentation(x):
    if np.random.randint(2):
        x = salt_pepper_noise(x, salt=0.2, amount=0.1)
    return x


def salt_pepper_noise(image, salt=0.2, amount=0.004):
    row, col, chan = image.shape
    num_salt = np.ceil(amount * row * salt)
    num_pepper = np.ceil(amount * row * (1.0 - salt))

    for n in range(chan-1): # -1 so we don't augment the mask
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[0:2]]
        image[coords[0], coords[1], n] = 3072

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[0:2]]
        image[coords[0], coords[1], n] = -1024

    return image
