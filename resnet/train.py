import tensorflow as tf
import numpy as np
import data
import resnet
import argparse
import pickle
import os
import pdb

def num_correct(pred, labels):
    predicted_labels = np.argmax(pred,axis=1)
    return np.sum(predicted_labels == labels)

def accuracy(pred, labels):
    return float(sum([p == l for p, l in zip(pred,labels)]))/len(labels)

def in_class_accuracy(pred, labels, nLabels):
    acc = [ 0 for _ in range(nLabels) ]
    for l in range(nLabels):
        nCorrect = sum([ l == p for (p, t) in zip(pred, labels) if t == l ])
        n = sum([ l == t for t in labels ])
        acc[l] = float(nCorrect)/n
    return acc
def predictions(pred):
    predictions = np.argmax(pred, axis=1)
    return predictions.tolist()

parser = argparse.ArgumentParser()

parser.add_argument('dir', help = 'the directory to store all of the output')
parser.add_argument('phase', choices = [ 'train', 'test' ], help = 'train or test')
parser.add_argument('--data_dir',
                    default = '/dataset/streetview/pop',
                    help = 'The root directory containing the dataset')
parser.add_argument('--dont_use_batchnorm',
                    help = 'If present the  model will not use batch normalization',
                    default = True, action = 'store_false')
parser.add_argument('--block_sizes',
                    help = 'The sizes of each block for the network',
                    default = [2, 2, 2, 2], type = int, nargs = '+')
parser.add_argument('--block_filters',
                    help = 'The number of filters in each block for the network',
                    default = [32, 64, 128, 256], type = int, nargs = '+')
parser.add_argument('--block_strides',
                    help = 'The strides of each block for the network',
                    default = [1, 2, 2, 2], type = int, nargs = '+')
parser.add_argument('--block_type',
                    help = 'The type of each block for the network',
                    default = 'basicblock')
parser.add_argument('--preset',
                    help = 'use preset for the various network structure options',
                    default = None)
parser.add_argument('--resume_file',
                    help = 'a checkpoint to resume from',
                    default = None) 

opt = parser.parse_args()
opt.use_batchnorm = not opt.dont_use_batchnorm


pickle_path = os.path.join(opt.data_dir, 'data.pickle')
if os.path.isfile(pickle_path):
    print('Loading premade train test split')
    with open(pickle_path, 'rb') as f:
        pickle_dict = pickle.load(f)
        label_names = pickle_dict['label_names']
        train_files = pickle_dict['train_files']
        train_labels = pickle_dict['train_labels']
        test_files = pickle_dict['test_files']
        test_labels = pickle_dict['test_labels']
else:
    print('Making train test split')
    files, labels, label_names = data.read_filenames_and_labels(opt.data_dir)
    train_files, train_labels, test_files, test_labels = data.train_test_split(files, labels)
    pickle_dict = { 'label_names' : label_names,
                    'train_files' : train_files,
                    'train_labels' : train_labels,
                    'test_files' :  test_files,
                    'test_labels' : test_labels }
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_dict, f)

opt.num_classes = len(label_names)    
print(opt.num_classes)




config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=config) as sess:
    nepochs = 20
    is_training = tf.placeholder(tf.bool, name='is_training')

    train_data = data.streetview_dataset(train_files, train_labels, 32, augment = True)
    test_data = data.streetview_dataset(test_files, test_labels, 32,
                                        augment = False, shuffle = False)
    it = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    rgb, images, labels = it.get_next()
    train_init_op = it.make_initializer(train_data)
    test_init_op = it.make_initializer(test_data)    
    
    network = resnet.Resnet(opt)
    pred, loss = network.build_net(images, labels, is_training)

    optimizer = tf.train.AdamOptimizer()
    extra_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_train_op):
        train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if opt.resume_file:
        saver.restore(sess, opt.resume_file)
        
    for i in range(nepochs):
        sess.run(train_init_op)
        j = 0
        while opt.phase == 'train':
            try:
                [ _, loss_arr ] = sess.run([train_op, loss], feed_dict = { is_training : True })
                if j % 20 == 19:
                    print('{} Loss: {}'.format(j, loss_arr))
                j = j + 1
            except tf.errors.OutOfRangeError:
                saver.save(sess, 'output/model_{}'.format(i))
                break
        
        sess.run(test_init_op)
        acc = 0
        num_test_examples = 0
        j = 0
        test_predictions = []
        while True:
            try:
                [ pred_arr, labels_arr ] = sess.run([pred, labels],
                                                    feed_dict = { is_training :  False })
                test_predictions.extend(predictions(pred_arr))
                num_test_examples += pred_arr.shape[0]
                acc += num_correct(pred_arr, labels_arr)
                if j % 4 == 3:
                    print('{} acc: {}'.format(j, acc/num_test_examples))
                j = j + 1
            except tf.errors.OutOfRangeError:
                break

        class_acc = in_class_accuracy(test_predictions, test_labels, opt.num_classes)
        
        print('Accuracy {}: {}'.format(accuracy(test_predictions, test_labels),
                                       [(name, acc) for name, acc in zip(label_names, class_acc)]))
                
