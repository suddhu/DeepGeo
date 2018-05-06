import tensorflow as tf
import numpy as np
import os
import re
from random import shuffle
import pdb


# Utility function for getting all of the training examples in a directory
# datasets have the structure of root/<Sate Name>/<image ID>_<heading>.png

def get_files(cur_dir, regex):
    out_files = []
    for subdir,_,files in os.walk(cur_dir):
        out_files.extend([ os.path.join(subdir, file)
                           for file in files if re.search(regex, file)])
    return out_files

def get_file_prefix(filename):
    return re.search('(.*/[a-z0-9A-Z_-]*)_[0-9]*.jpg', filename).group(1)

def all_directions_exist(prefix):
    return (os.path.isfile(prefix + '_0.jpg') and
            os.path.isfile(prefix + '_90.jpg') and
            os.path.isfile(prefix + '_180.jpg') and
            os.path.isfile(prefix + '_270.jpg'))

def prefix_to_filenames(prefix):
    return (prefix + '_0.jpg', prefix + '_90.jpg', prefix + '_180.jpg',
            prefix + '_270.jpg')

def read_grouped_filenames_and_labels(root):
    labeled_filenames = []
    all_labels = []
    for dir in os.listdir(root):
        cur_dir = os.path.join(root, dir)
        if os.path.isdir(cur_dir):
            label = dir
            all_labels.append(label)
            file_prefixes = [ get_file_prefix(file) for file in get_files(cur_dir, '_0.jpg') ]
            cur_files = [ prefix_to_filenames(prefix) for prefix in file_prefixes
                          if all_directions_exist(prefix) ]
            labeled_filenames.extend([ (filenames, len(all_labels)-1)
                                       for filenames in cur_files ])
    files = [ tmp[0] for tmp in labeled_filenames ]
    labels = [ tmp[1] for tmp in labeled_filenames ]

    return files,labels, all_labels, 
    
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


# Split a dataset into a random train/val split

def train_val_split(files, labels):
    frac = 0.1
    zipped = list(zip(files, labels))
    shuffle(zipped)
    valsize = int(len(zipped)*frac)
    val = zipped[:valsize]
    train = zipped[valsize:]
    train_files = [ e[0] for e in train ]
    train_labels = [ e[1] for e in train ]
    val_files = [ e[0] for e in val ]
    val_labels = [ e[1] for e in val ]

    return train_files, train_labels, val_files, val_labels



# Tensorflow functions for mapping a dataset of filenames to the actual images they reference

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


def grouped_streetview_dataset(files, labels, batch_size, augment = True, shuffle = True):
    def augment_image(img):
        flip = tf.greater(tf.random_uniform((), 0, 1), 0.5)

        img = tf.cond(flip, lambda: img, lambda: img[:,::-1,:])
        img = _color_jitter(img, 0.4, 0.4, 0.4)
        img = _lighting_noise(img, 0.1)
        return img
    def augment_images(n,e,s,w):
        return augment_image(n), augment_image(e), augment_image(s), augment_image(w)
    def parse_images(files):
        return (parse_image(files[0]), parse_image(files[1]),
                parse_image(files[2]), parse_image(files[3]))
    def normalize_images(n,e,s,w):
        return (normalize_image(n), normalize_image(e),
                normalize_image(s), normalize_image(w))
    filename_dataset = tf.data.Dataset.from_tensor_slices(files)
    d = filename_dataset.map(parse_images).prefetch(100)
    daug = d.map(normalize_images).map(augment_images)
    d = tf.data.Dataset.zip((filename_dataset, d, daug if augment else d.map(normalize_images),
                             tf.data.Dataset.from_tensor_slices(labels))).prefetch(100)
    if shuffle:
        d = d.shuffle(4000)
    d = d.batch(batch_size)
    return d


def streetview_dataset(files, labels, batch_size, augment = True, shuffle = True):
    def augment_image(img):
        flip = tf.greater(tf.random_uniform((), 0, 1), 0.5)

        img = tf.cond(flip, lambda: img, lambda: img[:,::-1,:])
        img = _color_jitter(img, 0.4, 0.4, 0.4)
        img = _lighting_noise(img, 0.1)
        return img

    filename_dataset = tf.data.Dataset.from_tensor_slices(files)
    d = filename_dataset.map(parse_image).prefetch(100)
    daug = d.map(normalize_image).map(augment_image)
    d = tf.data.Dataset.zip((d, daug if augment else d.map(normalize_image),
                             tf.data.Dataset.from_tensor_slices(labels))).prefetch(100)
    if shuffle:
        d = d.shuffle(4000)
    d = d.batch(batch_size)
    return d
