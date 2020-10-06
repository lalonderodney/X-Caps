'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains a few utility functions for safely making directories and plotting training.
'''

import os
import errno

import numpy as np
from matplotlib import pyplot as plt

def safe_mkdir(dir_to_make: str) -> None:
    '''
    Attempts to make a directory following the Pythonic EAFP strategy which prevents race conditions.

    :param dir_to_make: The directory path to attempt to make.
    :return: None
    '''
    try:
        os.makedirs(dir_to_make)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('ERROR: Unable to create directory: {}'.format(dir_to_make), e)
            raise


def plot_training(training_history, arguments):

    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    if arguments.net.find('caps') != -1:
        ax1.plot(training_history.history['out_mal_accuracy'])
        ax1.plot(training_history.history['val_out_mal_accuracy'])
    else:
        ax1.plot(training_history.history['accuracy'])
        ax1.plot(training_history.history['val_accuracy'])

    ax1.set_title('Accuracy')
    ax1.legend(['Train_Accuracy', 'Val_Accuracy'],
               loc='lower right')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    if arguments.net.find('caps') != -1:
        ax1.set_xticks(np.arange(0, len(training_history.history['out_mal_accuracy'])))
    else:
        ax1.set_xticks(np.arange(0, len(training_history.history['accuracy'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    if arguments.net.find('caps') != -1:
        ax2.plot(training_history.history['out_mal_loss'])
        ax2.plot(training_history.history['val_out_mal_loss'])
    else:
        ax2.plot(training_history.history['loss'])
        ax2.plot(training_history.history['val_loss'])

    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    if arguments.net.find('caps') != -1:
        ax1.set_xticks(np.arange(0, len(training_history.history['out_mal_loss'])))
    else:
        ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(os.path.join(arguments.output_dir, arguments.output_name + '_plots_' + arguments.time + '.png'))
    plt.close()
