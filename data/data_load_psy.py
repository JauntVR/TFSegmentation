
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
import matplotlib.pyplot as plt
import skimage.transform as st
from train.basic_train import BasicTrain
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit
from utils.average_meter import FPSMeter
from tensorflow import errors
import data.add_depth_noise as noise
MIN_DEPTH = 300
MAX_DEPTH = 1500

#Reduces the number of different labels to 8 + background
#label names (in same order as cutoffs): [torso,neck,r_arm, l_arm, head, waist, r_leg, l_leg, background]

#takes a depth image measured in mm and subtracts out the mean depth value (not counting the background) and then
#clips the image according to width and transforms all of the values to lie within out range
def preprocess_depth(img, bg_val, bg_val_out, width, out_range = [0,255], out_type = None):
    assert width >= 0
    mask = (img != bg_val)
    avg = np.sum(img*mask)/np.sum(mask)

    out = (img-(avg - width/2) + out_range[0])*(out_range[1]-out_range[0])/width

    mask_2 = (out >= out_range[0])*(out <= out_range[1])
    out = out*mask_2 + bg_val_out*(1-mask_2)

    if out_type is None:
        return out
    else:
        return out.astype(out_type)


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

def crop_pad(img, x_fin, y_fin, bg_val = 0, ref_img = None):
    if ref_img is None:
        cur_limits = get_dims(img, bg_val = bg_val)
    else:
        cur_limits = get_dims(ref_img, bg_val = bg_val)
    y_dim = cur_limits[1] - cur_limits[0]
    x_dim = cur_limits[3] - cur_limits[2]
    img_cropped = img[cur_limits[0]:cur_limits[1],cur_limits[2]:cur_limits[3]]
    if x_dim == 0 or y_dim == 0:
        return np.zeros((y_fin, x_fin)).astype('uint8'), 0
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

    return resized

def _read_hdf5_func(filename, label, h, w, num_channels, num_classes):
    try:
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

        label_resize= crop_pad(label_image, w, h, np.max(label_image))

        return final_image#, label_resize
    except (ZeroDivisionError, Exception, errors.UnknownError):
        import ipdb; ipdb.set_trace()

def _read_hdf5_func_npy_inference(filename, label, h, w, num_channels, num_classes):
    out_size = (h, w);

    filename_decoded = filename.decode("utf-8")

    depth_image = np.load(filename_decoded)

    depth_resize = crop_pad(depth_image, w, h, 0)
    if 'basic_fixed' in filename_decoded:
        depth_image_scaled = depth_resize.astype('uint8')
    else:
        depth_image_scaled = preprocess_depth(depth_resize, 0, 0, 1000, out_range=[0,255], out_type='uint8')
    final_depth = np.expand_dims(depth_image_scaled, 2)
    return final_depth, filename

def _read_npy(filename, label, h, w, num_channels, num_classes, add_noise=True):
    out_size = (h, w)
    #print('file: ' + str(filename))
    filename_decoded = filename.decode("utf-8")
    setup_path, frame = filename_decoded.split('__')

    #Read depth image
    depth_image_path = setup_path + '_depth_' + frame
    if add_noise:
        depth_image_path = setup_path + '_noise_depth_' + frame
        assert '8_cam_npy' in depth_image_path
        depth_image_path = depth_image_path.replace('8_cam_npy', '8_cam_npy_w_noise')
    if num_classes != 29:
        label_image_path = setup_path + '_label_' + str(num_classes) + '_' + frame
    else:
        label_image_path = setup_path + '_label_' + frame
    depth_image = np.load(depth_image_path)

    label_image = np.load(label_image_path)

    label_resize = crop_pad(label_image, w, h, 0,ref_img = depth_image).astype('uint8')

    depth_resize = crop_pad(depth_image, w, h, 0)
    depth_image_scaled = preprocess_depth(depth_resize, 0, 0, 1000, out_range=[0,255], out_type='uint8')
    final_depth = np.expand_dims(depth_image_scaled, 2)
    return final_depth, label_resize

def load_dataset_npy_file(seq_file, batch_size, h, w, num_channels, num_classes):
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
            _read_npy, [filename, labels, h, w, num_channels, num_classes], [tf.uint8, tf.uint8])), num_parallel_calls=1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset

    with open(seq_file, 'r') as infile:
        filenames = infile.read().splitlines()

def load_dataset_npy_dir_inference(seq_file, batch_size, h, w, num_channels, num_classes):
    all_files = os.listdir(seq_file)
    filenames = []
    for file in all_files:
        if file.endswith('.npy') and 'label' not in file:
            filenames.append(seq_file+file)

    num_images = len(filenames)
    int_num_images = int(np.floor(num_images / batch_size) * batch_size)
    if num_images != int_num_images :
        del filenames[int_num_images:]
    filenames = sorted(filenames)
    labels = [0]*len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_hdf5_func_npy_inference, [filename, labels, h, w, num_channels, num_classes], [tf.uint8, tf.string])), num_parallel_calls=1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset

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
            _read_hdf5_func_png, [filename, labels, h, w, num_channels, num_classes], [tf.uint8])), num_parallel_calls=1)


    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset



def load_dataset(seq_file, batch_size, h, w, num_channels, num_classes):
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

def load_dataset_file(file, batch_size, h, w, num_channels, num_classes):
    train_seq = h5py.File(file, "r")
    filenames = []
    num_cameras = train_seq['INFO']['NUM_CAMERAS'].value[0]
    key_list = list(train_seq.keys())
    train_seq.close()
    for key in key_list:
        if "FRAME" in key:
            for cam_idx in range(num_cameras):
                filename_str = file + '__' + '{:s}/RAW/CAM{:d}/'.format(key, cam_idx)
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
