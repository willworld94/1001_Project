import os
import sys
import scipy.misc
import numpy as np
import tensorflow as tf
import requests
import zipfile
from io import StringIO
from PIL import Image
from scipy.misc import imresize
from scipy import interpolate
import cv2


def _download_images(dir):
    z = zipfile.ZipFile('/Users/will/Desktop/DS1001-Intro-to-data-science /project/project_material/SelfExSR-master.zip', 'r')
    z.extractall('/Users/will/Desktop/DS1001-Intro-to-data-science /project/project_material')
    for name in z.namelist():
        if '.png' in name:
            to_file = os.path.join(dir, name)
            if not os.path.exists(os.path.dirname(to_file)):
                os.makedirs(os.path.dirname(to_file))
            img = Image.open(StringIO.StringIO(z.read(name)))
            img.save(os.path.join(dir, name))


class SuperResData:
    def __init__(self, upscale_factor=None, imageset='Set5'):
        self.upscale_factor = upscale_factor
        self.imageset = imageset
        self._base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # , 'data')
        if not os.path.exists(self._base_dir):
            _download_images(self._base_dir)
        self.data_dir = os.path.join(self._base_dir, 'SelfExSR-master/data/', imageset,
                                     'image_SRF_%i' % self.upscale_factor)

    def read(self):
        hr_images = {}
        lr_images = {}
        for i, f in enumerate(os.listdir(self.data_dir)):
            img = scipy.misc.imread(os.path.join(self.data_dir, f))
            if "HR" in f:
                hr_images["".join(f.split("_")[:3])] = img
            elif "LR" in f:
                lr_images["".join(f.split("_")[:3])] = img
        lr_keys = sorted(lr_images.keys())
        hr_keys = sorted(hr_images.keys())
        assert lr_keys == hr_keys
        for k in hr_keys:
            yield lr_images[k], hr_images[k]

    def make_patches(self, patch_size=None, stride=None):
        X_sub = []
        Y_sub = []
        for x, y in self.read():
            if len(x.shape) != 3:
                continue
            x = imresize(x, size=(y.shape[0], y.shape[1]), interp='bicubic')
            h, w, _ = x.shape
            for i in np.arange(0, h, stride):
                for j in np.arange(0, w, stride):
                    hi_low, hi_high = i, i + patch_size
                    wi_low, wi_high = j, j + patch_size
                    if (hi_high > h) or (wi_high > w):
                        continue
                    X_sub.append(x[np.newaxis, hi_low:hi_high, wi_low:wi_high] / 255)
                    Y_sub.append(y[np.newaxis, hi_low:hi_high, wi_low:wi_high] / 255)
        X_sub = np.concatenate(X_sub, axis=0)
        Y_sub = np.concatenate(Y_sub, axis=0)
        return X_sub, Y_sub

    def get_images(self):
        X, Y = [], []
        for x, y in self.read():
            if len(x.shape) != 3:
                continue
            X.append(x[np.newaxis].astype(np.float32))
            Y.append(y[np.newaxis].astype(np.float32))
        return X, Y

    def get_data_length(self):
        train_img, label_img = self.get_images()
        lenth_data = len(train_img)
        return lenth_data
