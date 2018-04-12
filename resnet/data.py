import tensorflow as tf
import numpy as np
import os
import re
from random import shuffle

def read_filenames_and_labels(root):
    labeled_filenames = []
    all_labels = []
    for dir in os.listdir(root):
        cur_dir = os.path.join(root, dir)
        if os.path.isdir(cur_dir):
            label = dir
            all_labels.append(label)
            for subdir,_,files in os.walk(cur_dir):
                labeled_filenames.extend([ (os.path.join(subdir, file), len(all_labels)-1)
                                           for file in files if re.search('.jpg', file)])
    files = [ tmp[0] for tmp in labeled_filenames ]
    labels = [ tmp[1] for tmp in labeled_filenames ]
    return files, labels, all_labels

def train_test_split(files, labels):
    frac = 0.1
    zipped = list(zip(files, labels))
    shuffle(zipped)
    testsize = int(len(zipped)*frac)
    test = zipped[:testsize]
    train = zipped[testsize:]
    train_files = [ e[0] for e in train ]
    train_labels = [ e[1] for e in train ]
    test_files = [ e[0] for e in test ]
    test_labels = [ e[1] for e in test ]

    return train_files, train_labels, test_files, test_labels

    
def parse_image(filename):
    filecontents = tf.read_file(filename)
    jpeg = tf.image.decode_jpeg(filecontents)
    jpeg.set_shape([256, 256, 3])
    return jpeg
def normalize_image(img):
    img = tf.cast(img, dtype=tf.float32)/255.0
    img = img - tf.constant([ 0.485, 0.456, 0.406 ], shape=[1,1,3])
    img = img/tf.constant([0.229,0.224,0.225], shape=[1,1,3])
    return img


def _color_jitter(img, b, c, s):
    img = tf.image.random_brightness(img, max_delta = b)
    img = tf.image.random_contrast(img, 1-c, 1+c)
    img = tf.image.random_saturation(img, 1-s, 1+s)
    return img

def _lighting_noise(img, alphastd):
    eigvec = tf.constant([[-0.5675,0.7192,0.4009 ],
                          [-0.5808,-0.0045,-0.8140 ],
                          [-0.5836,-0.6948,0.4203]])
    eigval = tf.constant([ 0.2175, 0.0188, 0.0045 ])
    alpha = tf.random_normal((), mean=0, stddev=alphastd)
    imgnoise = tf.reshape(tf.reduce_sum(eigvec*eigval*alpha, 1), [1, 1, 3])
    return img + imgnoise


def streetview_dataset(files, labels, batch_size, augment = True):
    def augment_image(img):
        flip = tf.greater(tf.random_uniform((), 0, 1), 0.5)

        img = tf.cond(flip, lambda: img, lambda: img[:,::-1,:])
        img = _color_jitter(img, 0.4, 0.4, 0.4)
        img = _lighting_noise(img, 0.1)
        return img

    d = tf.data.Dataset.from_tensor_slices(files).map(parse_image)
    daug = d.map(normalize_image).map(augment_image)
    d = tf.data.Dataset.zip((d, daug if augment else d.map(normalize_image),
                             tf.data.Dataset.from_tensor_slices(labels)))
    d = d.shuffle(2000).batch(batch_size)
    return d
