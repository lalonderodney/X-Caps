'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains all the functions related to loading the nodule data.
'''

import os
from glob import glob
import csv

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm, trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from utils import safe_mkdir

debug = False

def load_data(root, split=0, k_folds=5, val_split=0.2):
    fig_path = os.path.join(root, 'figs')
    safe_mkdir(fig_path)

    def _load_helper(f_list):
        imgs = np.empty(len(f_list), dtype=np.object)
        masks = np.empty(len(f_list), dtype=np.object)
        labels = np.zeros((len(f_list), 27), dtype=np.float32)
        for i, nod in enumerate(tqdm(f_list)):
            img_path_list = sorted(glob(os.path.join(root, nod[0], '???.tif')))
            mask_path_list = sorted(glob(os.path.join(root, nod[0], 'gt_???.tif')))
            assert len(img_path_list) == len(mask_path_list), 'Found different numbers of images and masks at ' \
                                                              '{}'.format(os.path.join(root, nod[0]))
            img_list = []; mask_list = []
            for j in range(len(img_path_list)):
                img_list.append(np.asarray(Image.open(img_path_list[j]), dtype=np.int16))
                mask_list.append(np.asarray(Image.open(mask_path_list[j]), dtype=np.uint8))
            imgs[i] = np.rollaxis(np.asarray(img_list), 0, 3)
            masks[i] = np.rollaxis(np.asarray(mask_list), 0, 3)
            labels[i, :] = np.asarray(nod[1:], dtype=np.float32)

            try:
                img = imgs[i]
                mask = masks[i]
                label = 'Sub: {} ISt: {} Cal: {} Sph: {} Mar: {} Lob: {} Spi: {} Tex: {} Mal: {}' \
                        ''.format(labels[i][0], labels[i][1], labels[i][2], labels[i][3], labels[i][4],
                                  labels[i][5], labels[i][6], labels[i][7], labels[i][8])
                f, ax = plt.subplots(1, 1, figsize=(15, 15))
                ax.imshow(img[:, :, img.shape[2] // 2], cmap='gray')
                ax.imshow(mask[:, :, img.shape[2] // 2], alpha=0.2, cmap='Reds')
                ax.set_title('{}'.format(label), fontsize=20)
                ax.axis('off')
                plt.savefig(os.path.join(fig_path, nod[0][8:].replace("/", "_").replace("\\", "_") + '_m.png'),
                            format='png', bbox_inches='tight')
                plt.close()
                f, ax = plt.subplots(1, 1, figsize=(15, 15))
                ax.imshow(img[:, :, img.shape[2] // 2], cmap='gray')
                ax.set_title('{}'.format(label), fontsize=20)
                ax.axis('off')
                plt.savefig(os.path.join(fig_path, nod[0][8:].replace("/", "_").replace("\\", "_") + '_i.png'),
                            format='png', bbox_inches='tight')
                plt.close()
            except Exception as e:
                print('\n' + '-' * 100)
                print('Error creating qualitative figure for {}'.format(nod[0]))
                print(e)
                print('-' * 100 + '\n')

        return imgs, masks, labels

    # Main functionality of loading and spliting the data
    def _load_data():
        outfile = os.path.join(root, 'np_data', 'split_{:02d}.npz'.format(split))
        try:
            print('Loading np_files for split {} of LIDC-IDRI dataset.'.format(split))
            npzfiles = np.load(outfile, allow_pickle=True, encoding='bytes')
            return npzfiles['train_imgs'], npzfiles['train_masks'], npzfiles['train_labels'], \
                   npzfiles['val_imgs'], npzfiles['val_masks'], npzfiles['val_labels'], \
                   npzfiles['test_imgs'], npzfiles['test_masks'], npzfiles['test_labels']
        except Exception as e:
            print('Unable to load numpy files. Loading from scratch instead.', e)
            with open(os.path.join(root, 'file_lists', 'train_split_{:02d}.csv'.format(split)), 'r') as f:
                reader = csv.reader(f)
                training_list = list(reader)
            with open(os.path.join(root, 'file_lists', 'test_split_{:02d}.csv'.format(split)), 'r') as f:
                reader = csv.reader(f)
                test_list = list(reader)

            print('Found training file lists, loading images and labels as numpy arrays.')
            print('Loading training images/labels.')
            train_val_imgs, train_val_masks, train_val_labels = _load_helper(training_list)
            print('Loading testing images/labels.')
            test_imgs, test_masks, test_labels = _load_helper(test_list)

            X_train, X_val, y_train, y_val = train_test_split(np.stack((train_val_imgs, train_val_masks), axis=-1),
                                                              train_val_labels, test_size=val_split,
                                                              random_state=12, stratify=train_val_labels[:,8])
            train_imgs, train_masks = np.split(X_train, 2, axis=-1)
            val_imgs, val_masks = np.split(X_val, 2, axis=-1)

            print('Finished loading files as numpy arrays. Saving arrays to avoid this in the future.')
            safe_mkdir(os.path.dirname(outfile))
            np.savez(outfile, train_imgs=np.squeeze(train_imgs), train_masks=np.squeeze(train_masks), train_labels=y_train,
                     val_imgs=np.squeeze(val_imgs), val_masks=np.squeeze(val_masks), val_labels=y_val,
                     test_imgs=test_imgs, test_masks=test_masks, test_labels=test_labels)

            return np.squeeze(train_imgs), np.squeeze(train_masks), y_train, \
                   np.squeeze(val_imgs), np.squeeze(val_masks), y_val, test_imgs, test_masks, test_labels

    # Try-catch to handle calling split data before load only if files are not found.
    try:
        Tr_i, Tr_m, Tr_l, Va_i, Va_m, Va_l, Te_i, Te_m, Te_l = _load_data()
        return Tr_i, Tr_m, Tr_l, Va_i, Va_m, Va_l, Te_i, Te_m, Te_l
    except Exception as e:
        # Create the training and test splits if not found
        print('Training lists not found. Creating {}-fold cross-validation training/testing lists'.format(k_folds))
        split_data(root, num_splits=k_folds)
        try:
            Tr_i, Tr_m, Tr_l, Va_i, Va_m, Va_l, Te_i, Te_m, Te_l = _load_data()
            return Tr_i, Tr_m, Tr_l, Va_i, Va_m, Va_l, Te_i, Te_m, Te_l
        except Exception as e:
            print(e)
            print('Failed to load data, see load_data in load_nodule_data.py')
            exit(1)

def split_data(root_path, num_splits=4):
    with open(os.path.join(root_path, 'file_lists', 'master_nodule_list.csv'), 'r') as f:
        reader = csv.reader(f)
        img_list = np.asarray(list(reader))

    labels_list = []
    indices = [0]
    nodule_list = []
    mal_score_list = []
    mal_scores = []
    curr_nodule = os.path.dirname(img_list[0][0])
    for i, img_label in enumerate(img_list):
        if os.path.dirname(img_label[0]) != curr_nodule:
            nodule_list.append(curr_nodule)
            mal_score_list.append(np.rint(np.mean(mal_scores)))
            indices.append(i)
            mal_scores = []
            curr_nodule = os.path.dirname(img_label[0])

        split_name = os.path.basename(img_label[0]).split('_')
        mal_scores.append(int(split_name[-1][-1]))
        labels_list.append([int(n[-1]) for n in split_name[1:]])

    outdir = os.path.join(root_path, 'file_lists')
    safe_mkdir(outdir)

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=12)
    n = 0
    for train_index, test_index in skf.split(nodule_list, mal_score_list):
        with open(os.path.join(outdir,'train_split_{:02d}.csv'.format(n)), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in train_index:
                for j in range(indices[i], indices[i+1]):
                    writer.writerow([img_list[j][0].split(root_path)[1][1:]] + labels_list[j] + list(img_list[j][1:]))
        with open(os.path.join(outdir,'test_split_{:02d}.csv'.format(n)), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in test_index:
                for j in range(indices[i], indices[i+1]):
                    writer.writerow([img_list[j][0].split(root_path)[1][1:]] + labels_list[j] + list(img_list[j][1:]))
        n += 1

def resize_data(imgs, masks, labels, out_dims):
    img_list = []; mask_list = []; label_list = []
    for i in trange(len(imgs)):
        for j in range(imgs[i].shape[2]):
            img = Image.fromarray(imgs[i][:, :, j])
            mask = Image.fromarray(masks[i][:, :, j])
            out_img = img.resize(out_dims)
            out_mask = mask.resize(out_dims)
            img_list.append(np.asarray(out_img, dtype=np.int16))
            mask_list.append(np.asarray(out_mask, dtype=np.uint8))
            label_list.append(labels[i])
    return np.asarray(img_list), np.asarray(mask_list), np.asarray(label_list)

def get_pseudo_label(x, mu, sig):
    sig[sig < .05] = .05
    g = (1. / (np.sqrt(2. * np.pi) * sig) * \
        np.exp(-np.power((np.repeat(np.expand_dims(np.asarray(x), axis=-1), mu.shape[0], axis=-1) - mu)
                         / sig, 2.) / 2))
    return (g / np.sum(g, axis=0)).T

def normalize_img(x):
    CT_MIN = -1024; CT_MAX = 3072
    x[x < CT_MIN] = CT_MIN
    x[x > CT_MAX] = CT_MAX
    x -= CT_MIN
    x /= (CT_MAX - CT_MIN)
    return x

def recover_img(x):
    CT_MIN = -1024; CT_MAX = 3072
    x *= (CT_MAX - CT_MIN)
    x += CT_MIN
    x[x < CT_MIN] = CT_MIN
    x[x > CT_MAX] = CT_MAX
    return x
