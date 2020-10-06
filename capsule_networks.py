'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of the capsule networks used (i.e. X-Caps and CapsNet).
'''

import numpy as np
from keras import layers, models
from keras import backend as K
K.set_image_data_format('channels_last')

from capsule_layers import ConvCapsuleLayer, FullCapsuleLayer, Mask, Length, ExpandDim

def XCaps(input_shape, n_class=5, routings=3, n_attr=6, caps_activ='sigmoid', order=0):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    conv1_reshaped = ExpandDim(name='expand_dim')(conv1)

    if order == 0:
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primary_caps = ConvCapsuleLayer(kernel_size=9, num_capsule=32, num_atoms=8, strides=2, padding='same',
                                        routings=1, name='primary_caps')(conv1_reshaped)
    else:
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primary_caps = ConvCapsuleLayer(kernel_size=9, num_capsule=8, num_atoms=32, strides=2, padding='same',
                                        routings=1, name='primary_caps')(conv1_reshaped)

    # Layer 3: Capsule layer. Routing algorithm works here.
    attr_caps = FullCapsuleLayer(num_capsule=n_attr, num_atoms=16, routings=routings, activation=caps_activ,
                                 name='attr_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_attr_concat = Length(num_classes=n_attr, name='out_attr_concat')(attr_caps)

    out_attr_caps_list = []
    for i in range(n_attr):
        out_attr_caps_list.append(layers.Lambda(lambda x: x[:, i], output_shape=(1,),
                                                name='out_attr_{}'.format(i))(out_attr_concat))

    flat_attr = layers.Flatten()(attr_caps)
    if n_class == 1:
        out_mal = layers.Dense(n_class, activation='sigmoid', name='out_mal')(flat_attr)
    else:
        out_mal = layers.Dense(n_class, activation='softmax', name='out_mal')(flat_attr)

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='out_recon')
    decoder.add(layers.Flatten(input_shape=(n_attr, 16)))
    decoder.add(layers.Dense(512, activation='relu'))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model(x, [out_mal] + out_attr_caps_list + [decoder(attr_caps)])
    eval_model = models.Model(x, [out_mal] + out_attr_caps_list + [decoder(attr_caps)])

    # manipulate model
    noise = layers.Input(shape=(n_attr, 16))
    noised_malcaps = layers.Add()([attr_caps, noise])
    manipulate_model = models.Model([x, noise], [out_mal] + out_attr_caps_list + [decoder(noised_malcaps)])

    return train_model, eval_model, manipulate_model


def CapsNet(input_shape, n_class=5, routings=3, noactiv=False):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    conv1_reshaped = ExpandDim(name='expand_dim')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primary_caps = ConvCapsuleLayer(kernel_size=9, num_capsule=32, num_atoms=8, strides=2, padding='same',
                                    routings=1, name='primary_caps')(conv1_reshaped)

    # Layer 3: Capsule layer. Routing algorithm works here.
    malcaps = FullCapsuleLayer(num_capsule=n_class, num_atoms=16, routings=routings, name='malcaps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    if noactiv:
        out_mal = Length(num_classes=n_class, name='out_mal')(malcaps)
    else:
        mal_mag = Length(num_classes=n_class, name='mal_mag')(malcaps)
        out_mal = layers.Activation('softmax', name='out_mal')(mal_mag)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask(n_class)([malcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask(n_class)(malcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='out_recon')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_mal, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_mal, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_malcaps = layers.Add()([malcaps, noise])
    masked_noised_y = Mask(n_class)([noised_malcaps, y])
    manipulate_model = models.Model([x, y, noise], [out_mal, decoder(masked_noised_y)])

    return train_model, eval_model, manipulate_model
