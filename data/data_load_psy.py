
# coding: utf-8

'''
#usage example
train_seq_folder = 'c:\\TFSegmentation\\data\\synthetic_seq\\train_seq'
dataset = load_datase(train_seq_folder, 5)
iterator = dataset.make_one_shot_iterator()
next_example = iterator.get_next()
sess = tf.Session()
sess.run(next_example)
'''

import os
import h5py
import warnings
import tensorflow as tf
import numpy as np
import skimage.transform as st
from train.basic_train import BasicTrain
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit
from utils.average_meter import FPSMeter

MIN_DEPTH = 300
MAX_DEPTH = 1500

#Reduces the number of different labels to 8 + background
#label names (in same order as cutoffs): [torso,neck,r_arm, l_arm, head, waist, r_leg, l_leg, background]
def reduce_labels(in_label):
    correspondance_dict = {0:0,
                           1:1,
                           2:2,
                           3:3, 4:3, 5:3,
                           6:4, 7:4,
                           8:5,
                           9:6, 10:6, 11:6,
                           12:7, 13:7,
                           14:8,
                           15:9,
                           16:10,
                           17:11, 18:11,
                           19:12, 20:12, 21:12,
                           22:13, 23:13,
                           24:14, 25:14, 26:14,
                           27: 15}
    return correspondance_dict[in_label]

def get_dims(img, bg_val = 0):
    y_dim, x_dim = img.shape[:2]
    new_shape = [0,y_dim,0,x_dim]
    for i in range(y_dim):
        if np.sum(img[i,:] != bg_val) > 0:
            new_shape[0] = i
            break
    for j in range(y_dim):
        i = y_dim - j - 1
        if np.sum(img[i,:] != bg_val) > 0:
            new_shape[1] = i
            break
    for i in range(x_dim):
        if np.sum(img[:,i] != bg_val) > 0:
            new_shape[2] = i
            break
    for j in range(x_dim):
        i = x_dim - j - 1
        if np.sum(img[:,i] != bg_val) > 0:
            new_shape[3] = i
            break

    return new_shape

def crop_pad(img, x_fin, y_fin, bg_val = 0):
    cur_limits = get_dims(img, bg_val = bg_val)

    y_dim = cur_limits[1] - cur_limits[0]
    x_dim = cur_limits[3] - cur_limits[2]
    img_cropped = img[cur_limits[0]:cur_limits[1],cur_limits[2]:cur_limits[3]]

    if y_fin/x_fin >  y_dim/x_dim:  #need to pad in y_dim
        pad = int((y_fin*x_dim/x_fin - y_dim)/2)
        pad_shape = [(pad,pad),(0,0)]

    else:  #need to pad in x_dim or do no padding
        pad = int((x_fin*y_dim/y_fin - x_dim)/2)
        pad_shape = [(0,0),(pad,pad)]
    for i in range(len(img.shape)-len(pad_shape)):
        pad_shape = pad_shape + [(0,0)]

    pad_val = []
    if hasattr(bg_val, '__iter__'):
        for val in bg_val:
            pad_val = pad_val + [(val,val)]
    else:
        pad_val = bg_val
    padded = np.pad(img_cropped, pad_shape,'constant',constant_values=pad_val)

    #resize
    resized = st.resize(padded, (y_fin,x_fin),preserve_range=True)

    return resized.astype('uint8')

def _read_hdf5_func(filename, label, h, w, num_channels, num_classes):
    out_size = (h, w);
    #print('file: ' + str(filename))
    filename_decoded = filename.decode("utf-8")
    #print(filename_decoded)
    h5_file_name, group_name = filename_decoded.split('__')
    h5_file = h5py.File(h5_file_name, "r")

    if num_classes != 28 and num_classes != 16:
        print("invalid number of classes, please choose 28 or 16")
        num_classes = 28

    # Read depth image
    depth_image_path = group_name + 'Z'
    depth_image = h5_file[depth_image_path].value

    depth_image_scaled = np.array(depth_image, copy=False)
    depth_image_scaled.clip(MIN_DEPTH, MAX_DEPTH, out=depth_image_scaled)
    depth_image_scaled -= MIN_DEPTH
    np.floor_divide(depth_image_scaled, (MAX_DEPTH - MIN_DEPTH + 1) / 256,
                    out=depth_image_scaled, casting='unsafe')
    final_image = crop_pad(depth_image_scaled, w, h, 0)
    final_image = np.expand_dims(final_image, 2)

    # Read labels
    label_image_path = group_name + 'LABEL'
    if num_classes != 28:
        label_image_path = label_image_path + '_' + str(num_classes)
    label_image = h5_file[label_image_path].value

    label_resize = crop_pad(label_image, w, h, np.max(label_image))
    return final_image, label_resize


def load_dataset(seq_file, batch_size, h, w, num_channels, num_classes):
    print("USING DATAFILE")
    with open(seq_file, 'r') as infile:
        filenames = infile.read().splitlines()

    num_images = len(filenames)
    int_num_images = int(np.floor(num_images / batch_size) * batch_size)
    if num_images != int_num_images :
        del filenames[int_num_images:]

    labels = [0]*len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_hdf5_func, [filename, labels, h, w, num_channels, num_classes], [tf.uint8, tf.uint8])), num_parallel_calls=1)


    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset

def _load_dataset(train_seq_folder, batch_size, h, w, num_channels, num_classes):
    train_seq_files = []
    for (dirpath, dirnames, filenames) in os.walk(train_seq_folder):
        train_seq_files.extend(os.path.join(dirpath, x) for x in filenames)
    filenames = []
    for train_seq_name in train_seq_files:
        if not train_seq_name.endswith(".h5"):
             continue
        train_seq = h5py.File(train_seq_name, "r")
        num_cameras = train_seq['INFO']['NUM_CAMERAS'].value[0]
        key_list = list(train_seq.keys())
        train_seq.close()
        for key in key_list:
            if "FRAME" in key:
                for cam_idx in range(num_cameras):
                    filename_str = train_seq_name + '__' + '{:s}/RAW/CAM{:d}/'.format(key, cam_idx)
                    filenames.append(filename_str)

    num_images = len(filenames)
    int_num_images = int(np.floor(num_images / batch_size) * batch_size)
    if num_images != int_num_images :
        del filenames[int_num_images:]

    labels = [0]*len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_hdf5_func, [filename, labels, h, w, num_channels, num_classes], [tf.uint8, tf.uint8])), num_parallel_calls=1)


    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset
