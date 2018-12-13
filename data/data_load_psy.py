
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
from data.reduce_labels import reduce_labels
import time
MIN_DEPTH = 300
MAX_DEPTH = 1500

#Reduces the number of different labels to 8 + background
#label names (in same order as cutoffs): [torso,neck,r_arm, l_arm, head, waist, r_leg, l_leg, background]

#takes a depth image measured in mm and subtracts out the mean depth value (not counting the background) and then
#clips the image according to width and transforms all of the values to lie within out range
def disp_kps(img, kps):
    fig, ax = plt.subplots()
    ax.imshow(img)
    for kp in kps:
        x, y, _  = kp
        circ = plt.Circle((x,y), 10)
        ax.add_artist(circ)

    plt.show()
keypoint_names = [#Left side
                'LHip',
                'LKnee',
                'LAnkle',
                'LToeJoint',  #the joint connecting the toe to the foot
                'LToeTip',
                'LShoulder',
                'LElbow',
                'LWrist',
                'LFingerJoint', #the joint connecting fingers to hand
                'LFingerTip',
                'LThumbTip',
                #Right side
                'RHip',
                'RKnee',
                'RAnkle',
                'RToeJoint',  #the joint connecting the toe to the foot
                'RToeTip',
                'RShoulder',
                'RElbow',
                'RWrist',
                'RFingerJoint', #the joint connecting fingers to hand
                'RFingerTip',
                'RThumbTip',
                #Center
                'HeadHigh',
                'HeadMiddle',
                'NeckHigh',
                'NeckMiddle',
                'NeckLow',
                'SpineHigh',
                'SpineMiddle',
                'SpineLow',
                'CenterHipsHigh',
                'CenterHips']
keypoints_to_use = [1,1,1,0,1,1,1,1,0,1,0,  #left_side
                    1,1,1,0,1,1,1,1,0,1,0,  #right_side
                    1,0,0,0,1,0,1,0,0,1]    #middle
def gen_kp_heatmaps(kps, kp_width, img_dims, out_size, num_keypoints):
    assert num_keypoints == sum(keypoints_to_use)
    y0, y1, x0, x1 = img_dims
    h0 = y1-y0
    w0 = x1-x0
    hf, wf = out_size

    if hf/wf >  h0/w0:  #need to pad in y_dim
        pad = int((hf*w0/wf - h0)/2)
        y0 = y0-pad
        h0 = h0 + 2*pad

    else:  #need to pad in x_dim or do no padding
        pad = int((wf*h0/hf - w0)/2)
        x0 = x0-pad
        w0 = w0 + 2*pad

    xs, ys = np.meshgrid(range(hf),range(wf))

    rough_kp = np.zeros((hf,wf,num_keypoints))
    fine_kp = np.zeros((hf,wf,num_keypoints))
    j = 0
    for i, kp in enumerate(kps):
        if keypoints_to_use[i]:
            x,y, _ = kp
            xf = (x-x0)*wf/w0
            yf = (y-y0)*hf/h0

            rough_kp[:,:,j] = 1*(((xs-xf)**2 + (ys-yf)**2) <= kp_width**2)  #solid circles
            fine_kp[:,:,j] = np.exp(-((xs-xf)**2+(ys-yf)**2)/(2*kp_width**2))
            j += 1
    return rough_kp.astype('uint8'), fine_kp.astype('float32')



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

def crop_pad(img, x_fin, y_fin, cur_limits, bg_val = 0):
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

def read_h5_w_labels(filename, label, h, w, clip_width, depth_range, num_classes, add_g_noise):
    out_size = (h, w);
    #print('file: ' + str(filename))
    filename_decoded = filename.decode("utf-8")
    #print(filename_decoded)
    h5_file_name, group_name = filename_decoded.split('__')
    h5_file = h5py.File(h5_file_name, "r")

    # Read depth image
    depth_image_path = group_name + 'Z'
    depth_image = h5_file[depth_image_path].value

    # Read labels
    label_image_path = group_name + 'LABEL'
    label_image = h5_file[label_image_path].value

    label_resize = crop_pad(label_image, w, h, 0,ref_img = depth_image).astype('uint8')
    depth_resize = crop_pad(depth_image, w, h, 0)
    depth_image_scaled = preprocess_depth(depth_resize, 0, 0, clip_width, out_range=depth_range, out_type='uint8')
    final_depth = np.expand_dims(depth_image_scaled, 2)
    label_reduced = reduce_labels(label_resize, num_classes).astype('uint8')


    return final_depth, label_reduced

def read_npy_no_labels(filename, label, h, w, clip_width, depth_range):
    out_size = (h, w);

    filename_decoded = filename.decode("utf-8")

    depth_image = np.load(filename_decoded)
    img_dims = get_dims(depth_image)
    depth_resize = crop_pad(depth_image, w, h, img_dims, 0)
    if 'basic_fixed' in filename_decoded:
        depth_image_scaled = depth_resize.astype('uint8')
    else:
        depth_image_scaled = preprocess_depth(depth_resize, 0, 0, clip_width, out_range=depth_range, out_type='uint8')
    final_depth = np.expand_dims(depth_image_scaled, 2)

    return final_depth, filename

def read_npy_w_labels(filename, label, h, w, clip_width, depth_range, num_classes, add_g_noise = False, add_artifacts = False):
    out_size = (h, w)
    filename_decoded = filename.decode("utf-8")
    setup_path, frame = filename_decoded.split('__')

    #Read depth image
    depth_image_path = setup_path + '_depth_' + frame
    if add_artifacts:
        depth_image_path = setup_path + '_noise_depth_' + frame
        assert '8_cam_npy' in depth_image_path
        depth_image_path = depth_image_path.replace('8_cam_npy', '8_cam_npy_w_noise')

    include_label_num = True
    if include_label_num:
        label_image_path = setup_path + '_label_29_' + frame
    else:
        label_image_path = setup_path + '_label_' + frame

    depth_image = np.load(depth_image_path)
    label_image = np.load(label_image_path)

    img_dims = get_dims(depth_image)
    label_resize = crop_pad(label_image, w, h, img_dims, 0).astype('uint8')
    depth_resize = crop_pad(depth_image, w, h, img_dims, 0)
    depth_image_scaled = preprocess_depth(depth_resize, 0, 0, clip_width, out_range=depth_range, out_type='uint8')
    final_depth = np.expand_dims(depth_image_scaled, 2)
    label_reduced = reduce_labels(label_resize, num_classes).astype('uint8')

    return final_depth, label_reduced

def read_npy_w_labels_kps(filename, label, h, w, clip_width, depth_range, num_classes, num_keypoints, kp_width, add_g_noise = False, add_artifacts = False):
    out_size = (h, w)
    filename_decoded = filename.decode("utf-8")
    setup_path, frame = filename_decoded.split('__')

    #Read depth image
    depth_image_path = setup_path + '_depth_' + frame
    if add_artifacts:
        depth_image_path = setup_path + '_noise_depth_' + frame
        assert '8_cam_npy' in depth_image_path
        depth_image_path = depth_image_path.replace('8_cam_npy', '8_cam_npy_w_noise')

    include_label_num = True
    if include_label_num:
        label_image_path = setup_path + '_label_29_' + frame
    else:
        label_image_path = setup_path + '_label_' + frame

    kp_file_path = setup_path +'_keypoints_' + frame

    depth_image = np.load(depth_image_path)
    label_image = np.load(label_image_path)
    kps = np.load(kp_file_path)

    img_dims = get_dims(depth_image)
    label_resize = crop_pad(label_image, w, h, img_dims, 0).astype('uint8')

    depth_resize = crop_pad(depth_image, w, h, img_dims, 0)
    depth_image_scaled = preprocess_depth(depth_resize, 0, 0, clip_width, out_range=depth_range, out_type='uint8')
    final_depth = np.expand_dims(depth_image_scaled, 2)

    rough_kp, fine_kp = gen_kp_heatmaps(kps, kp_width, img_dims, out_size, num_keypoints)

    label_reduced = reduce_labels(label_resize, num_classes).astype('uint8')
    return final_depth, label_reduced, rough_kp, fine_kp

def load_dataset_no_labels(seq_file, file_type, args):
    h = args.img_height
    w = args.img_width
    clip_width = args.clip_width
    depth_range = (args.depth_range_min, args.depth_range_max)
    batch_size = args.batch_size
    if file_type == 'folder.npy': #folder with numpy files
        all_files = os.listdir(seq_file)
        filenames = []
        for file in all_files:
            if file.endswith('.npy') and 'label' not in file:
                filenames.append(args.data_dir + seq_file+file)

        num_images = len(filenames)
        int_num_images = int(np.floor(num_images / batch_size) * batch_size)
        if num_images != int_num_images :
            del filenames[int_num_images:]
        filenames = sorted(filenames)
        labels = [0]*len(filenames)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                read_npy_no_labels, [filename, labels, h, w, clip_width, depth_range], [tf.uint8, tf.string])), num_parallel_calls=1)
    else:
        assert 0==1 #invalid file type

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset

def load_dataset_w_labels(seq_file, file_type, args):
    h = args.img_height
    w = args.img_width
    num_classes = args.num_classes
    add_g_noise = args.add_gaussian_noise
    add_artifacts = args.add_artifacts
    batch_size = args.batch_size
    clip_width = args.clip_width
    depth_range = (args.depth_range_min, args.depth_range_max)
    num_keypoints = args.num_keypoints

    if num_keypoints > 0: #We are training on segmentation and keypoints
        assert file_type == '.npy'
        kp_label_width = args.kp_label_width
        with open(seq_file, 'r') as infile:
            filenames = infile.read().splitlines()
        if args.base_data_dir != '':
            filenames = [args.base_data_dir + x for x in filenames]
        num_images = len(filenames)
        int_num_images = int(np.floor(num_images / batch_size) * batch_size)
        if num_images != int_num_images :
            del filenames[int_num_images:]

        labels = [0]*len(filenames)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                read_npy_w_labels_kps, [filename, labels, h, w, clip_width, depth_range, num_classes, num_keypoints, kp_label_width, add_g_noise, add_artifacts], [tf.uint8, tf.uint8, tf.uint8, tf.float32])), num_parallel_calls=1)

    elif file_type == '.npy': #seq file contains a list of .npy file paths
        with open(seq_file, 'r') as infile:
            filenames = infile.read().splitlines()
        if args.base_data_dir != '/':
            filenames = [args.base_data_dir + x for x in filenames]
        num_images = len(filenames)
        int_num_images = int(np.floor(num_images / batch_size) * batch_size)
        if num_images != int_num_images :
            del filenames[int_num_images:]

        labels = [0]*len(filenames)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                read_npy_w_labels, [filename, labels, h, w, clip_width, depth_range, num_classes, add_g_noise, add_artifacts], [tf.uint8, tf.uint8])), num_parallel_calls=1)

    elif file_type == 'folder.h5': #seq file is a folder containing h5 files
        train_seq_files = []
        for (dirpath, dirnames, filenames) in os.walk(train_seq_folder):
            train_seq_files.extend(os.path.join(dirpath, x) for x in filenames)
        if args.base_data_dir != '':
            filenames = [args.base_data_dir + x for x in filenames]
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
                read_h5_w_labels, [filename, labels, h, w, clip_width, depth_range, num_classes, add_g_noise], [tf.uint8, tf.uint8])), num_parallel_calls=1)

    elif file_type == '.h5': #seq file is a single h5 file
        train_seq = h5py.File(args.base_data_dir + file, "r")
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
                read_h5_w_labels, [filename, labels, h, w, clip_width, depth_range, num_classes, add_g_noise], [tf.uint8, tf.uint8])), num_parallel_calls=1)

    else:
        assert 0==1 #invalid file type

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset
