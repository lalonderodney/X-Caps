'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

from __future__ import print_function

import numpy as np
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from custom_data_aug import custom_train_data_augmentation
from model_helper import compile_model, get_callbacks
from load_nodule_data import get_pseudo_label, normalize_img
from utils import plot_training

# debug is for visualizing the created images
debug = False

def train(args, u_model, train_samples, val_samples):
    # Compile the loaded model
    model = compile_model(args=args, uncomp_model=u_model)

    # Load pre-trained weights
    if args.finetune_weights_path != '':
        try:
            model.load_weights(args.finetune_weights_path)
        except Exception as e:
            print(e)
            print('!!! Failed to load custom weights file. Training without pre-trained weights. !!!')

    # Set the callbacks
    callbacks = get_callbacks(args)

    if args.aug_data:
        train_datagen = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False,
            rotation_range=45,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            rescale=None,
            preprocessing_function=custom_train_data_augmentation)

        val_datagen = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False,
            rescale=None)
    else:
        train_datagen = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False,
            rotation_range=0,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            fill_mode='nearest',
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None)

        val_datagen = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False,
            rescale=None)

    if debug:
        save_dir = args.img_aug_dir
    else:
        save_dir = None

    def xcaps_data_gen(gen):
        while True:
            x, y = gen.next()
            if args.num_classes == 1:
                mal = np.array([y[i][0][6, 0] for i in range(y.shape[0])])
            else:
                mal = np.array([y[i][0][6, 1:] for i in range(y.shape[0])])
            yield x, [mal,
                      np.array([y[i][0][0, 0] for i in range(y.shape[0])]),
                      np.array([y[i][0][1, 0] for i in range(y.shape[0])]),
                      np.array([y[i][0][2, 0] for i in range(y.shape[0])]),
                      np.array([y[i][0][3, 0] for i in range(y.shape[0])]),
                      np.array([y[i][0][4, 0] for i in range(y.shape[0])]),
                      np.array([y[i][0][5, 0] for i in range(y.shape[0])]),
                      x * np.expand_dims(np.array([y[i][1] for i in range(y.shape[0])]), axis=-1)]

    def capsnet_data_gen(gen):
        while True:
            x, y = gen.next()
            if args.num_classes == 1:
                y = np.array([y[i][0][6,0] for i in range(y.shape[0])])
            else:
                y = np.array([y[i][0][6,1:] for i in range(y.shape[0])])
            yield [x, y], [y, x]

    # Prepare images and labels for training
    train_imgs = normalize_img(np.expand_dims(train_samples[0], axis=-1).astype(np.float32))
    val_imgs = normalize_img(np.expand_dims(val_samples[0], axis=-1).astype(np.float32))

    train_labels = []; val_labels = []; n_attr = 9 # 8 attr + mal score
    skip_attr_list = [1,2]
    for i in range(n_attr):
        skip = False
        if skip_attr_list:
            for j in skip_attr_list:
                if i == j: #indexing from negative side
                    skip_attr_list.remove(j)
                    skip = True
        if args.num_classes == 1 and i == n_attr-1:
            tlab = np.repeat(np.expand_dims(train_samples[2][:, 2*i + n_attr], axis=-1), 6, axis=1)
            tlab[tlab < 3.] = 0.
            tlab[tlab >= 3.] = 1.
            train_labels.append(tlab)
            vlab = np.repeat(np.expand_dims(val_samples[2][:, 2*i + n_attr], axis=-1), 6, axis=1)
            vlab[vlab < 3.] = 0.
            vlab[vlab >= 3.] = 1.
            val_labels.append(vlab)
            skip = True
        if not skip:
            train_labels.append(
                np.hstack((np.expand_dims((train_samples[2][:, 2 * i + n_attr]-1)/ 4., axis=-1),
                           get_pseudo_label([1., 2., 3., 4., 5.], train_samples[2][:, 2*i + n_attr],
                                          train_samples[2][:, 2*i+1 + n_attr]))))
            val_labels.append(
                np.hstack((np.expand_dims((val_samples[2][:, 2 * i + n_attr] - 1) / 4., axis=-1),
                           get_pseudo_label([1., 2., 3., 4., 5.], val_samples[2][:, 2 * i + n_attr],
                                          val_samples[2][:, 2 * i + 1 + n_attr]))))

    train_labels = np.rollaxis(np.asarray(train_labels), 0, 2)
    val_labels = np.rollaxis(np.asarray(val_labels), 0, 2)

    new_labels = np.empty((len(train_labels),2), dtype=np.object)
    for i in range(len(train_labels)):
        new_labels[i, 0] = train_labels[i]
        if args.masked_recon:
            new_labels[i, 1] = train_samples[1][i]
        else:
            new_labels[i, 1] = np.ones_like(train_samples[1][i])
    train_labels = new_labels

    new_labels = np.empty((len(val_labels), 2), dtype=np.object)
    for i in range(len(val_labels)):
        new_labels[i, 0] = val_labels[i]
        if args.masked_recon:
            new_labels[i, 1] = val_samples[1][i]
        else:
            new_labels[i, 1] = np.ones_like(val_samples[1][i])
    val_labels = new_labels

    train_flow_gen = train_datagen.flow(x=train_imgs,
                                        y=train_labels,
                                        batch_size=args.batch_size, shuffle=True, seed=12, save_to_dir=save_dir)

    val_flow_gen = val_datagen.flow(x=val_imgs,
                                    y=val_labels,
                                    batch_size=args.batch_size, shuffle=True, seed=12, save_to_dir=save_dir)

    if args.net.find('xcaps') != -1:
        train_gen = xcaps_data_gen(train_flow_gen)
        val_gen = xcaps_data_gen(val_flow_gen)
    elif args.net.find('capsnet') != -1:
        train_gen = capsnet_data_gen(train_flow_gen)
        val_gen = capsnet_data_gen(val_flow_gen)
    else:
        raise NotImplementedError('Data generator not found for specified network. Please check train.py file.')

    # Settings
    train_steps = len(train_samples[0])//args.batch_size
    val_steps = len(val_samples[0])//args.batch_size
    workers = 4
    multiproc = True

    # Run training
    history = model.fit_generator(train_gen,
                                  max_queue_size=40, workers=workers, use_multiprocessing=multiproc,
                                  steps_per_epoch=train_steps,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  epochs=args.epochs,
                                  class_weight=None,
                                  callbacks=callbacks,
                                  verbose=args.verbose,
                                  shuffle=True)

    # Plot the training data collected
    plot_training(history, args)
