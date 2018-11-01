
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
    cutoffs = [2,3,9,15,16,17,22,27,10000]
    for i, cutoff in enumerate(cutoffs):
        if in_label < cutoff:
            return i

def _read_hdf5_func(filename, label, h, w, num_channels, num_classes):
    out_size = (h, w);
    #print('file: ' + str(filename))
    filename_decoded = filename.decode("utf-8")
    #print(filename_decoded)
    h5_file_name, group_name = filename_decoded.split('__')
    h5_file = h5py.File(h5_file_name, "r")

    if num_channels != 1 and num_channels != 4:
        print("invalid number of channels, please choose 4 or 1. Proceeding on just depth information")
        num_channels = 1
    if num_classes != 28 and num_classes != 9:
        print("invalid number of classes, please choose 28 or 9")
        num_classes = 28

    # Read depth image
    depth_image_path = group_name + 'Z'
    depth_image = h5_file[depth_image_path].value
    depth_image_scaled = np.array(depth_image, copy=False)
    depth_image_scaled.clip(MIN_DEPTH, MAX_DEPTH, out=depth_image_scaled)
    depth_image_scaled -= MIN_DEPTH
    np.floor_divide(depth_image_scaled, (MAX_DEPTH - MIN_DEPTH + 1) / 256,
                    out=depth_image_scaled, casting='unsafe')

    # Read labels
    label_image_path = group_name + 'LABEL'
    label_image = h5_file[label_image_path].value
    h5_file.close()
    #resize output
    depth_resize = depth_image_scaled
    if num_classes == 9: #move after resize??
        label_image = np.vectorize(reduce_labels)(label_image)

    if(depth_image_scaled.shape != out_size):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            depth_resize = st.resize(depth_resize, out_size,
                                     order = 0, preserve_range = True)
            label_image = st.resize(label_image, out_size,
                                     order = 0, preserve_range = True)
            label_image = label_image.astype('uint8')
    depth_resize = np.expand_dims(depth_resize,2)
    if num_channels == 4:  #assumed to be 3-channel bgr image with values in the range 0 to 255
        bgr_image_path = group_name + 'BGR'
        bgr_image = h5_file[depth_image_path].value
        if(depth_image_scaled.shape != out_size):
            with warnings.catch_warnings():
                brg_resized= st.resize(bgr_image, (h,w,num_channels),
                                         order = 0, preserve_range = True)
        final_image = np.concatenate((depth_resize,brg_resize), axis=2)

    else:
        final_image = depth_resize
    final_image = final_image.astype(np.uint8)

    #import ipdb; ipdb.set_trace()

    return final_image, label_image


def load_dataset(train_seq_folder, batch_size, h, w, num_channels, num_classes):
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
