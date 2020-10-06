'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is the main file for the project. From here you can train, test, and manipulate the X-Caps of models.
Please see the README for detailed instructions for this project.
'''

from __future__ import print_function

import os
import argparse
from time import gmtime, strftime
time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

from load_nodule_data import load_data, resize_data
from model_helper import create_model
from utils import safe_mkdir

def main(args):
    # Directory to save images for using flow_from_directory
    args.output_name = 'split-' + str(args.split_num) + '_nclass-' + str(args.num_classes) + \
                       '_batch-' + str(args.batch_size) +  '_shuff-' + str(args.shuffle_data) + \
                       '_aug-' + str(args.aug_data) + '_loss-' + str(args.loss) + '_lr-' + str(args.initial_lr) + \
                       '_reconwei-' + str(args.recon_wei) + '_attrwei-' + str(args.attr_wei) + \
                       '_r1-' + str(args.routings1) + '_r2-' + str(args.routings2)
    args.time = time

    # Create all the output directories
    args.check_dir = os.path.join(args.data_root_dir, 'saved_models', args.exp_name, args.net)
    safe_mkdir(args.check_dir)

    args.log_dir = os.path.join(args.data_root_dir, 'logs', args.exp_name, args.net)
    safe_mkdir(args.log_dir)

    args.tf_log_dir = os.path.join(args.log_dir, 'tf_logs', args.time)
    safe_mkdir(args.tf_log_dir)

    args.output_dir = os.path.join(args.data_root_dir, 'plots', args.exp_name, args.net)
    safe_mkdir(args.output_dir)

    args.img_aug_dir = os.path.join(args.data_root_dir, 'logs', 'aug_imgs')
    safe_mkdir(args.img_aug_dir)

    # Load the training, validation, and testing data
    train_imgs, train_masks, train_labels, val_imgs, val_masks, val_labels, test_imgs, test_masks, test_labels = \
        load_data(root=args.data_root_dir, split=args.split_num,
                  k_folds=args.k_fold_splits, val_split=args.val_split)
    print('Found {} 3D nodule images for training, {} for validation, and {} for testing.'
          ''.format(len(train_imgs), len(val_imgs), len(test_imgs)))

    # Resize images to args.resize_shape
    print('Resizing training images to {}.'.format(args.resize_shape))
    train_imgs, train_masks, train_labels = resize_data(train_imgs, train_masks, train_labels, args.resize_shape)
    print('Resizing validation images to {}.'.format(args.resize_shape))
    val_imgs, val_masks, val_labels = resize_data(val_imgs, val_masks, val_labels, args.resize_shape)
    print('Resizing testing images to {}.'.format(args.resize_shape))
    test_imgs, test_masks, test_labels = resize_data(test_imgs, test_masks, test_labels, args.resize_shape)

    # Create the model
    model_list = create_model(args=args, input_shape=args.resize_shape + [1])
    model_list[0].summary()

    # Run the chosen functions
    if args.train:
        from train import train
        print('-'*98,'\nRunning Training\n','-'*98)
        train(args=args, u_model=model_list[0], train_samples=(train_imgs, train_masks, train_labels),
              val_samples=(val_imgs, val_masks, val_labels))

    if args.test:
        from test import test
        print('-'*98,'\nRunning Testing\n','-'*98)
        if args.net.find('caps') != -1:
            test(args=args, u_model=model_list[1], test_samples=(test_imgs, test_masks, test_labels))
        else:
            test(args=args, u_model=model_list[0], test_samples=(test_imgs, test_masks, test_labels))

    if args.manip and args.net.find('caps') != -1:
        from manip import manip
        print('-'*98,'\nRunning Manipulate\n','-'*98)
        manip(args=args, u_model=model_list[2], test_samples=(test_imgs, test_masks, test_labels))

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable Lung Nodule Diagnosis')

    # MANDATORY ARGUMENTS: Where the data is stored, what to call the experiment, and binary/multi-class setting.
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory where your datasets are stored.')
    parser.add_argument('--exp_name', type=str, default="Default",
                        help='Name of the current experiment (easy way to keep results/configs separate.')
    parser.add_argument('--num_classes', type=int, default=5, choices=[1, 5],
                        help='1: Binary classification - Benign or Malignant.'
                             '5: Classification - Malignancy scores 1, 2, 3, 4, or 5.')

    # What functions to performing (i.e. training, testing, and/or manipulation of the capsule vectors)
    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    parser.add_argument('--manip', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable manipulation.')

    # Network architecture and objective function
    parser.add_argument('--net', type=str.lower, default='xcaps',
                        choices=['xcaps', 'xcaps_sm', 'capsnet', 'capsnet_nsm'],
                        help='Choose your network.')
    parser.add_argument('--loss', type=str.lower, default='kl', choices=['mse', 'ce', 'mar', 'kl'],
                        help='Which loss to use: mean-squared error, cross-entropy, margin loss, or KL-divergence.')

    # Weight paths for testing or pre-training
    parser.add_argument('--finetune_weights_path', type=str, default='',
                        help='Set to /path/to/trained_model.hdf5 from root for loading pre-trained weights for '
                             'training. Set to "" for none.')
    parser.add_argument('--test_weights_path', type=str, default='',
                        help='Set to /path/to/trained_model.hdf5 from root for testing with a specific weights file. '
                             'If continuing from training, the best saved weights will be used automatically if '
                             'set to "".')

    # Cross-Validation Settings
    parser.add_argument('--k_fold_splits', type=int, default=5,
                        help='Number of training splits to create for k-fold cross-validation.')
    parser.add_argument('--split_num', type=int, default=0,
                        help='Which training split to train/test on.')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Percentage between 0 and 1 of training split to use as validation.')

    # Data shuffling and augmentation
    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to shuffle the training data (both per epoch and in slice order.')
    parser.add_argument('--aug_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to use data augmentation during training.')

    # Resize the data to a standard size (necessary for forming batches)
    parser.add_argument('--resize_hei', type=int, default=32,
                        help="Image resize height for forming equal size batches")
    parser.add_argument('--resize_wid', type=int, default=32,
                        help="Image resize width for forming equal size batches")

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training/testing.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train for.')
    parser.add_argument('--initial_lr', type=float, default=0.02,
                        help='Initial learning rate for Adam.')

    # Network Settings (Especially for Capsule Networks)
    parser.add_argument('--recon_wei', type=float, default=0.512,
                        help="If using a Capsule Network: The coefficient (weighting) for the loss of decoder")
    parser.add_argument('--masked_recon', type=int, default=1, choices=[0,1],
                        help="If using X-Caps: Set to 1 to reconstruct a segmented version of the nodules.")
    parser.add_argument('--attr_wei', type=float, default=1.0,
                        help="If using a Capsule Network: The coefficient (weighting) for the attributes")
    parser.add_argument('--k_size', type=int, default=9,
                        help='Kernel size for capsnet.')
    parser.add_argument('--output_atoms', type=int, default=16,
                        help='Number of output atoms for capsnet.')
    parser.add_argument('--routings1', type=int, default=3,
                        help="If using capsnet: The number of iterations used in routing algorithm for layers which "
                             "maintain spatial resolution. should > 0")
    parser.add_argument('--routings2', type=int, default=3,
                        help="If using capsnet: The number of iterations used in routing algorithm for layers which "
                             "change spatial resolution. should > 0")

    # Output verbosity
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')

    # GPU settings
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')

    arguments = parser.parse_args()

    # Ensure training, testing, and manip are not all turned off
    assert (arguments.train or arguments.test or arguments.manip or arguments.pred), \
        'Cannot have train, test, pred, and manip all set to 0, Nothing to do.'

    arguments.resize_shape = [arguments.resize_hei, arguments.resize_wid]

    # Mask the GPUs for TensorFlow
    if arguments.which_gpus == -2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                  'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'

    main(arguments)
