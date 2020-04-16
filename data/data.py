import numpy as np
import pandas as pd

from glob import glob
import pydicom

import cv2
from skimage.transform import resize

import os
from tqdm import tqdm
import shutil

from joblib import Parallel, delayed

import argparse



def argparser():
    parser = argparse.ArgumentParser(description='Prepare png dataset for pneumatorax')
    parser.add_argument('-train_path', default='../input/train/', type=str, nargs='?', help='directory with train')
    parser.add_argument('-test_path', default='../input/test/', type=str, nargs='?', help='directory with test')
    parser.add_argument('-out_path', default='../dataset', type=str, nargs='?', help='path for saving dataset')
    parser.add_argument('-n_train', default=-1, type=int, nargs='?', help='size of train dataset')
    parser.add_argument('-img_size', default=1024, type=int, nargs='?', help='image size')
    parser.add_argument('-n_threads', default=4, type=int, nargs='?', help='number of using threads')
    return parser.parse_args()




def save_train_file(f, out_path, img_size):
    img = pydicom.read_file(f).pixel_array
    name = f.split('/')[-1][:-4]
    img = resize(img, (img_size, img_size))
    cv2.imwrite('{}/train_png/{}.png'.format(out_path, name), img * 255)



def save_test_file(f, out_path, img_size):
    img = pydicom.read_file(f).pixel_array
    name = f.split('/')[-1][:-4]
    img = resize(img, (img_size, img_size)) * 255
    cv2.imwrite('{}/test_png/{}.png'.format(out_path, name), img)


def save_train(train_images_names, out_path='../dataset128',
               img_size=128, n_train=-1, n_threads=1):
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path + '/train_png', exist_ok=True)
    if n_train < 0:
        n_train = len(train_images_names)
    try:
        Parallel(n_jobs=n_threads, backend='threading')(delayed(save_train_file)(
            f, out_path, img_size) for f in tqdm(train_images_names[:n_train]))
    except pydicom.errors.InvalidDicomError:
        print('InvalidDicomError')


def save_test(test_images_names, out_path='../dataset128', img_size=128, n_threads=1):
    os.makedirs(out_path + '/test_png', exist_ok=True)
    try:
        Parallel(n_jobs=n_threads, backend='threading')(delayed(save_test_file)(
            f, out_path, img_size) for f in tqdm(test_images_names))
    except pydicom.errors.InvalidDicomError:
        print('InvalidDicomError')


def main():
    args = argparser()
    train_fns = sorted(glob('{}/*/*/*.dcm'.format(args.train_path)))
    test_fns = sorted(glob('{}/*/*/*.dcm'.format(args.test_path)))
    out_path = args.out_path
    img_size = args.img_size
    n_train = args.n_train
    n_threads = args.n_threads

    save_train(train_fns, out_path, img_size, n_train, n_threads)
    save_test(test_fns, out_path, img_size, n_threads)


if __name__ == '__main__':
    main()