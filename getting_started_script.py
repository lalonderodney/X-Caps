'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This script will call the necessary functions for setting up the LIDC_IDRI dataset.
'''


import argparse
import os

from convert_lidc_format import create_cropped_nodules
from convert_lidc_format import create_master_list

def getting_started(ROOT):
    IMG_ROOT = os.path.join(ROOT, 'DOI')
    OUT_ROOT = os.path.join(ROOT, 'nodules')
    FILE_ROOT = os.path.join(ROOT, 'files_lists')

    create_cropped_nodules(IMG_ROOT, OUT_ROOT)
    create_master_list(OUT_ROOT, FILE_ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lung Nodule Diagnosis')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory where your LIDC-IDRI dataset is stored.')
    args = parser.parse_args()

    getting_started(args.data_root_dir)