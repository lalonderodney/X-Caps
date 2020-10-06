'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about training.
'''

import csv
import os

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from model_helper import compile_model
from load_nodule_data import get_pseudo_label, normalize_img
from capsule_layers import combine_images
from utils import safe_mkdir

def compute_within_one_acc(cmat):
    eps = 1e-7
    class_acc = []
    for i, row in enumerate(cmat):
        curr_acc = 0
        try:
            curr_acc += row[i-1]
        except:
            pass
        try:
            curr_acc += row[i]
        except:
            pass
        try:
            curr_acc += row[i+1]
        except:
            pass
        class_acc.append(curr_acc)

    class_acc = np.asarray(class_acc)

    return class_acc / (cmat.sum(axis=1)+eps), np.sum(class_acc) / cmat.sum()

def test(args, u_model, test_samples):
    out_dir = os.path.join(args.data_root_dir, 'results', args.exp_name, args.net)
    safe_mkdir(out_dir
               )
    out_img_dir = os.path.join(out_dir, 'recons')
    safe_mkdir(out_img_dir)

    # Compile the loaded model
    model = compile_model(args=args, uncomp_model=u_model)

    # Load testing weights
    if args.test_weights_path != '':
        output_filename = os.path.join(out_dir, 'results_' + os.path.basename(args.test_weights_path)[:-5] + '.csv')
        try:
            model.load_weights(args.test_weights_path)
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')
    else:
        output_filename = os.path.join(out_dir, 'results_' + args.output_name + '_model_' + args.time + '.csv')
        try:
            model.load_weights(os.path.join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5'))
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')

    test_datagen = ImageDataGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rescale=None)

    # TESTING SECTION
    def data_gen(gen):
        while True:
            x, y = gen.next()
            yield x, y

    x_test = normalize_img(np.expand_dims(test_samples[0], axis=-1).astype(np.float32))

    if args.num_classes == 1:
        y_test = np.rint(test_samples[2][:,-2])
    else:
        y_test = get_pseudo_label([1.,2.,3.,4.,5.], test_samples[2][:,-2], test_samples[2][:,-1])

    test_gen = data_gen(test_datagen.flow(x=x_test, y=y_test, batch_size=1, shuffle=False, seed=12))

    results = model.predict_generator(test_gen, max_queue_size=1, workers=1, use_multiprocessing=False,
                                      steps=len(x_test), verbose=1)

    if args.net.find('caps') != -1:
        y_pred = results[0]
        x_recon = results[-1]

        img = combine_images(np.concatenate([x_test[:250:5], x_recon[:250:5]]))
        pil_img = Image.fromarray(255*img).convert('L')
        if args.test_weights_path != '':
            img_filename = os.path.join(out_img_dir,  os.path.basename(args.test_weights_path)[:-5] + '_real_and_recon.png')
        else:
            img_filename = os.path.join(out_img_dir, args.output_name + '_model_' + args.time + '_real_and_recon.png')
        pil_img.save(os.path.join(out_img_dir, img_filename))
    else:
        y_pred = results

    if args.num_classes == 1:
        gt = y_test; pred = np.squeeze(np.rint(y_pred * 4 + 1))
    else:
        gt = np.argmax(y_test, axis=1) + 1; pred = np.argmax(y_pred, axis=1) + 1

    if args.num_classes == 1:
        cmat = confusion_matrix(gt, pred, labels=[1, 2, 3, 4, 5])
        test_acc_cat, test_acc_all = compute_within_one_acc(cmat)
        test_acc_cat_weighted, test_acc_all_weighted = np.zeros_like(test_acc_cat), np.zeros_like(test_acc_all)
    else:
        cmat = confusion_matrix(gt, pred, labels=[1, 2, 3, 4, 5])
        test_acc_cat, test_acc_all = compute_within_one_acc(cmat)
        cmat_weighted = confusion_matrix(gt, pred, labels=[1, 2, 3, 4, 5], sample_weight=1./np.var([1.,2.,3.,4.,5.] * y_pred, axis=1))
        test_acc_cat_weighted, test_acc_all_weighted = compute_within_one_acc(cmat_weighted)

    with open(output_filename, 'w', newline='') as f:
        fw = csv.writer(f, delimiter=',')
        fw.writerow(['Malignancy Accuracy', 'Malignancy Accuracy Confidence Weighted'])
        fw.writerow(['{:05f}'.format(test_acc_all), '{:05f}'.format(test_acc_all_weighted)])
        fw.writerow(['Malignancy Accuracy by Score:'])
        fw.writerow(['{:05f}'.format(num) for num in test_acc_cat])
        fw.writerow(['Malignancy Accuracy by Score Confidence Weighted:'])
        fw.writerow(['{:05f}'.format(num) for num in test_acc_cat_weighted])
        fw.writerow(['Malignancy Confusion Matrix:'])
        for row in cmat:
            fw.writerow(['{:05f}'.format(num) for num in row])
        if args.net.find('dcaps') != -1 or args.net == 'xcapsnet':
            if args.net.find('simple') != -1 or args.net == 'xcapsnet':
                attr_pred = np.rint(np.swapaxes(np.asarray(results[1:-1]), 0, -1) * 4 + 1)
                y_attr = np.rint(np.concatenate((np.expand_dims(test_samples[2][:, -18], axis=-1),
                                                 np.expand_dims(test_samples[2][:, -12], axis=-1),
                                                 np.expand_dims(test_samples[2][:, -10], axis=-1),
                                                 np.expand_dims(test_samples[2][:, -8], axis=-1),
                                                 np.expand_dims(test_samples[2][:, -6], axis=-1),
                                                 np.expand_dims(test_samples[2][:, -4], axis=-1)),
                                                axis=1)).astype(np.int64)
                for i in range(y_attr.shape[1]):
                    attr_cmat = confusion_matrix(y_attr[:, i], attr_pred[:, i], labels=[1, 2, 3, 4, 5])
                    class_acc, total_acc = compute_within_one_acc(attr_cmat)
                    fw.writerow(['Attribute {} Accuracy'.format(i)])
                    fw.writerow(['{:05f}'.format(total_acc)])
                    fw.writerow(['Attribute {} Accuracy by Score:'.format(i)])
                    fw.writerow(['{:05f}'.format(num) for num in class_acc])
                    fw.writerow(['Attribute {} Confusion Matrix:'.format(i)])
                    for row in attr_cmat:
                        fw.writerow(['{:05f}'.format(num) for num in row])
            else:
                attr_pred = np.argmax(np.rollaxis(np.asarray(results[1:-1]), 0, -1), axis=-1) + 1
                y_attr = np.concatenate(
                    (np.expand_dims(get_pseudo_label([1., 2., 3., 4., 5.], test_samples[2][:, -18], test_samples[2][:, -17]), axis=1),
                     np.expand_dims(get_pseudo_label([1., 2., 3., 4., 5.], test_samples[2][:, -12], test_samples[2][:, -11]), axis=1),
                     np.expand_dims(get_pseudo_label([1., 2., 3., 4., 5.], test_samples[2][:, -10], test_samples[2][:, -9]), axis=1),
                     np.expand_dims(get_pseudo_label([1., 2., 3., 4., 5.], test_samples[2][:, -8], test_samples[2][:, -7]), axis=1),
                     np.expand_dims(get_pseudo_label([1., 2., 3., 4., 5.], test_samples[2][:, -6], test_samples[2][:, -5]), axis=1),
                     np.expand_dims(get_pseudo_label([1., 2., 3., 4., 5.], test_samples[2][:, -4], test_samples[2][:, -3]), axis=1)),
                    axis=1)
                gt_attr = np.argmax(y_attr, axis=2) + 1

                for i in range(gt_attr.shape[1]):
                    attr_cmat = confusion_matrix(gt_attr[:, i], attr_pred[:, i], labels=[1, 2, 3, 4, 5])
                    attr_cmat_weighted = confusion_matrix(gt_attr[:, i], attr_pred[:, i], labels=[1, 2, 3, 4, 5],
                                                          sample_weight=1./np.var([1.,2.,3.,4.,5.] * np.rollaxis(np.asarray(results[1:-1]), 0, -1)[:, i], axis=1))
                    class_acc, total_acc = compute_within_one_acc(attr_cmat)
                    class_acc_weighted, total_acc_weighted = compute_within_one_acc(attr_cmat_weighted)
                    fw.writerow(['Attribute {} Accuracy'.format(i), 'Attribute {} Accuracy Confidence Weighted'.format(i)])
                    fw.writerow(['{:05f}'.format(total_acc), '{:05f}'.format(total_acc_weighted)])
                    fw.writerow(['Attribute {} Accuracy by Score:'.format(i)])
                    fw.writerow(['{:05f}'.format(num) for num in class_acc])
                    fw.writerow(['Attribute {} Accuracy by Score Confidence Weighted:'.format(i)])
                    fw.writerow(['{:05f}'.format(num) for num in class_acc_weighted])
                    fw.writerow(['Attribute {} Confusion Matrix:'.format(i)])
                    for row in attr_cmat:
                        fw.writerow(['{:05f}'.format(num) for num in row])
