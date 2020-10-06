'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is a helper file for choosing which model to create.
'''

import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

from custom_losses import margin_loss

def create_model(args, input_shape):
    if args.net.find('xcaps') != -1:
        from capsule_networks import XCaps
        if args.net.find('_sm') != -1:
            return XCaps(input_shape=input_shape, n_class=args.num_classes, routings=args.routings1,
                         caps_activ='softmax')
        else:
            return XCaps(input_shape=input_shape, n_class=args.num_classes, routings=args.routings1)

    elif args.net.find('capsnet') != -1:
        from capsule_networks import CapsNet
        if args.net.find('_nsm') != -1:
            return CapsNet(input_shape, args.num_classes, args.routings1, noactiv=True)
        else:
            return CapsNet(input_shape, args.num_classes, args.routings1)

    else:
        raise Exception('Unknown network type specified: {}'.format(args.net))


def get_loss(net, recon_wei, attr_wei, choice, classes):
    if choice == 'ce':
        if classes == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        attr_loss = 'binary_crossentropy'
    elif choice == 'mar':
        loss, attr_loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0), margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    elif choice == 'mse':
        loss, attr_loss = 'mse', 'mse'
    elif choice == 'kl':
        if classes == 1:
            raise Exception("Cannot compute KL divergence with scalar regression output.")
        loss = tf.keras.losses.KLDivergence()
        attr_loss = 'mse'
    else:
        raise Exception("Unknown loss_type")

    if net.find('capsnet') != -1:
        return {'out_mal': loss, 'out_recon': 'mse'}, {'out_mal': 1., 'out_recon': recon_wei}
    elif net.find('xcaps') != -1:
        return {'out_mal': loss, 'out_recon': 'mse', 'out_attr_0': attr_loss, 'out_attr_1': attr_loss,
                'out_attr_2': attr_loss, 'out_attr_3': attr_loss, 'out_attr_4': attr_loss, 'out_attr_5': attr_loss}, \
               {'out_mal': 1., 'out_attr_0': attr_wei, 'out_attr_1': attr_wei,  'out_attr_2': attr_wei,
                'out_attr_3': attr_wei, 'out_attr_4': attr_wei, 'out_attr_5': attr_wei, 'out_recon': recon_wei}
    else:
        return loss, None

def get_callbacks(arguments):
    monitor_name = 'val_loss'

    csv_logger = CSVLogger(os.path.join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(os.path.join(arguments.check_dir, arguments.output_name + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=True, save_weights_only=True,
                                       verbose=1, mode='min')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=12,verbose=1, mode='min')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='min')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]

def compile_model(args, uncomp_model):
    try:
        opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6, amsgrad=True)
    except:
        opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)

    metrics = {'out_mal': 'accuracy'}

    loss, loss_weighting = get_loss(net=args.net, recon_wei=args.recon_wei, attr_wei=args.attr_wei, choice=args.loss,
                                    classes=args.num_classes)

    uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
    return uncomp_model

