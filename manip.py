'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for manipulating the vectors of the final layer of capsules.
This manipulation attempts to show what each dimension of these final vectors are storing (paying attention to).
Please see the README for further details about how to use this file.
'''

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join, basename
from os import makedirs
from tqdm import trange
import numpy as np
from PIL import Image

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import to_categorical

from model_helper import compile_model
from load_nodule_data import normalize_img
from capsule_layers import combine_images


def manip(args, u_model, test_samples):
    out_dir = join(args.data_root_dir, 'results', args.exp_name, args.net)
    try:
        makedirs(out_dir)
    except:
        pass
    out_img_dir = join(out_dir, 'manip_output')
    try:
        makedirs(out_img_dir)
    except:
        pass

    # Compile the loaded model
    model = compile_model(args=args, uncomp_model=u_model)

    # Load testing weights
    if args.test_weights_path != '':
        try:
            model.load_weights(args.test_weights_path)
            out_name = basename(args.test_weights_path)[:-5]
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')
    else:
        try:
            model.load_weights(join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5'))
            out_name = args.output_name + '_model_' + args.time
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')

    x_test = normalize_img(np.expand_dims(test_samples[0], axis=-1).astype(np.float32))
    if args.num_classes == 1:
        y_test = np.expand_dims(test_samples[2][:,25], axis=-1) # 25 should be avg mal score
        y_test[y_test < 3.] = 0.
        y_test[y_test >= 3.] = 1.
    else:
        y_test = to_categorical(np.rint(test_samples[2][:,-2]) - 1)

    print('Creating manipulated outputs.')
    for mal_val in trange(y_test.shape[1]):
        index = np.argmax(y_test, 1) == mal_val
        number = np.random.randint(low=0, high=sum(index) - 1)
        x, y = x_test[index][number], y_test[index][number]
        x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
        if args.net.find('xcaps') != -1:
            noise = np.zeros([1, 6, 16])
        elif args.net == 'capsnet':
            noise = np.zeros([1, y_test.shape[1], 16])
        else:
            raise NotImplementedError('Specified Network does not have proper implementation in manip.py')
        x_recons = []
        for attr in range(noise.shape[0]):
            for dim in range(16):
                for r in [-0.5, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]:
                    tmp = np.copy(noise)
                    tmp[attr, :, dim] = r
                    if args.net.find('xcaps') != -1:
                        x_recon = model.predict([x, tmp])
                    elif args.net == 'capsnet':
                        x_recon = model.predict([x, y, tmp])
                    else:
                        raise NotImplementedError('Specified Network does not have proper implementation in manip.py')
                    x_recons.append(x_recon[-1])

            x_recons = np.concatenate(x_recons)

            img = combine_images(x_recons, height=16)
            pil_img = Image.fromarray(255*img).convert('L')
            pil_img.save(join(out_img_dir, out_name + '_{}_{}.png'.format(attr, mal_val+1)))
