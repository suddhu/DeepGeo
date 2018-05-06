import tensorflow as tf
import numpy as np
import data
import resnet
import argparse
import pickle
import os
import pdb

# Utilities for evalutating accuracy

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
def predictions_to_labels(pred):
    predictions = np.argmax(pred, axis=1)
    return predictions.tolist()

# Given operations that when run successively iterate the network over a dastaset,
# this function collectes performance statistics over the whole set
def eval_dataset(pred_op, labels_op, filenames_op, nLabels):
    nCorrect = 0
    num_val_examples = 0
    j = 0
    predictions = []
    labels = []
    output = None
    filenames = None
    while True:
        try:
            [ pred_arr, labels_arr, filenames_arr ] = sess.run([pred_op, labels_op,
                                                            filenames_op],
                                                feed_dict = { is_training :  False })
            
            if output is None:
                output = pred_arr
                filenames = filenames_arr
            else:
                output = np.concatenate((output, pred_arr), axis = 0)
                filenames = np.concatenate((filenames, filenames_arr), axis = 0)
            predictions.extend(predictions_to_labels(pred_arr))
            labels.extend(labels_arr)
            num_val_examples += pred_arr.shape[0]
            nCorrect += num_correct(pred_arr, labels_arr)
            if j % 4 == 3:
                print('{} acc: {}'.format(j, nCorrect/num_val_examples))
            j = j + 1
        except tf.errors.OutOfRangeError:
            break

    class_acc = in_class_accuracy(predictions, labels, nLabels)
    return nCorrect/num_val_examples, class_acc, { 'output' : output,
                                                   'labels' : labels,
                                                   'filenames' : filenames }

# Command line arugment parsing
parser = argparse.ArgumentParser()

parser.add_argument('name', help = 'the name of the model and diretory to store all of the output')
parser.add_argument('phase', choices = [ 'train', 'test' ], help = 'train or test')
parser.add_argument('--data_dir',
                    default = '/dataset/streetview/full',
                    help = 'The root directory containing the dataset')
parser.add_argument('--test_dir',
                    default = '/dataset/streetview/test/test_data',
                    help = 'The directory containing the test data')
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
parser.add_argument('--type',
                    help = 'what type of model to train',
                    default = 'resnet',
                    choices = ['resnet', 'early-int', 'late-int',
                               'med-int'])

parser.add_argument('--batch_size',
                    help = 'the batch size',
                    default = 32,
                    type = int)
parser.add_argument('--resume_epoch',
                    help = 'epoch number to start at, useful for resuming training',
                    default = 0,
                    type = int)

parser.add_argument('--summary_dir',
                    help = 'the directory in which to store summaries',
                    default = 'summaries')

opt = parser.parse_args()
opt.use_batchnorm = not opt.dont_use_batchnorm
if opt.phase == 'test':
    opt.nepochs = 1
if not os.path.isdir(opt.name):
    os.mkdir(opt.name)
opt.dir = opt.name

summary_subdir = os.path.join(opt.summary_dir, opt.name)
if not os.path.isdir(summary_subdir):
    os.mkdir(summary_subdir)
    

# Make traing val split, but reuse one if it's already been made
pickle_path = os.path.join(opt.data_dir, 'data.pickle')
if os.path.isfile(pickle_path):
    print('Loading premade train val split')
    with open(pickle_path, 'rb') as f:
        pickle_dict = pickle.load(f)
        label_names = pickle_dict['label_names']
        train_files = pickle_dict['train_files']
        train_labels = pickle_dict['train_labels']
        val_files = (pickle_dict['val_files'] if 'val_files' in pickle_dict
                     else pickle_dict['test_files'])
        val_labels = (pickle_dict['val_labels'] if 'val_labels' in pickle_dict
                      else pickle_dict['test_labels'])
else:
    print('Making train val split')
    files, labels, label_names = data.read_grouped_filenames_and_labels(opt.data_dir)
    train_files, train_labels, val_files, val_labels = data.train_val_split(files, labels)
    pickle_dict = { 'label_names' : label_names,
                    'train_files' : train_files,
                    'train_labels' : train_labels,
                    'val_files' :  val_files,
                    'val_labels' : val_labels }
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_dict, f)

nepochs = 50
if opt.phase == 'test':
    nepochs = 1
    test_files, test_labels, label_names = data.read_grouped_filenames_and_labels(opt.test_dir)


opt.num_classes = 50 #len(label_names)    
print(opt.num_classes)




config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=config) as sess:

    # Make dataset and graph
    is_training = tf.placeholder(tf.bool, name='is_training')

    train_data = data.grouped_streetview_dataset(train_files, train_labels, opt.batch_size,
                                                 augment = True, shuffle = True)
    val_data = data.grouped_streetview_dataset(val_files, val_labels, opt.batch_size,
                                               augment = False, shuffle = False)

    it = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    filenames_op, rgb, images, labels = it.get_next()
    train_init_op = it.make_initializer(train_data)
    val_init_op = it.make_initializer(val_data)    
    if opt.phase == 'test':
        test_data = data.grouped_streetview_dataset(test_files, test_labels, 100,
                                                    augment = False, shuffle = False)
        test_init_op = it.make_initializer(test_data)

    if opt.type == 'resnet':
        network = resnet.Resnet(opt)
    elif opt.type == 'early-int':
        network = resnet.EarlyIntResnet(opt)
    elif opt.type == 'late-int':
        network = resnet.LateIntResnet(opt)        
    elif opt.type == 'med-int':
        network = resnet.MedIntResnet(opt)        

    pred, loss = network.build_net(images, labels, is_training)
    
    
    optimizer = tf.train.AdamOptimizer()
    extra_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_train_op):
        train_op = optimizer.minimize(loss)

    tf.summary.scalar('training loss', loss)
    accuracy_tensor = tf.placeholder(tf.float32, name='accuracy')
    tf.summary.scalar('validation accuracy', accuracy_tensor)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(summary_subdir, sess.graph)
    
    saver = tf.train.Saver(max_to_keep = 20)
    sess.run(tf.global_variables_initializer())
    if opt.resume_file:
        saver.restore(sess, opt.resume_file)

    # Training
    k = 0
    acc = 0
    for i in range(opt.resume_epoch, nepochs):
        sess.run(train_init_op)
        j = 0
        while opt.phase == 'train':
            try:
                [ _, summary, loss_arr ] = sess.run([train_op, merged, loss],
                                                    feed_dict = { is_training : True ,
                                                                  accuracy_tensor : acc})
                if k % 10 == 9:
                    writer.add_summary(summary, k)
                if j % 20 == 19:
                    print('{} Loss: {}'.format(j, loss_arr))
                j = j + 1
                k = k + 1
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(opt.dir, 'model_{}'.format(i)))
                break

        if opt.phase == 'test':
            sess.run(test_init_op)
        else:
            sess.run(val_init_op)
        acc, class_acc, inout_data = eval_dataset(pred, labels, filenames_op, opt.num_classes)
        with open(os.path.join(opt.dir, 'inout_data.pickle'), 'wb') as f:
            pickle.dump(inout_data, f)
        print('Accuracy {}: {}'.format(acc,
                                       [(name, acc) for name, acc in zip(label_names, class_acc)]))
                
