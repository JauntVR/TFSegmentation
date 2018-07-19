
# coding: utf-8

# In[21]:


import os
import h5py
import tensorflow as tf
import numpy as np
from train.basic_train import BasicTrain
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit
from utils.average_meter import FPSMeter

MIN_DEPTH = 300
MAX_DEPTH = 1500


# In[22]:


sess = tf.Session()


# In[23]:


train_seq_folder = 'c:\\seq\\train_seq\\'
print(train_seq_folder)


# In[24]:


train_seq_files = []
for (dirpath, dirnames, filenames) in os.walk(train_seq_folder):
    train_seq_files.extend(os.path.join(dirpath, x) for x in filenames)
print(train_seq_files)


# In[25]:


filenames = []
for train_seq_name in train_seq_files:
    train_seq = h5py.File(train_seq_name, "r")
    num_cameras = train_seq['INFO']['NUM_CAMERAS'].value[0]
    num_frames = train_seq['INFO']['COUNT'].value[0]
    train_seq.close()
    for frame_idx in range(num_frames):
        for cam_idx in range(num_cameras):
            filename_str = train_seq_name + '__' + 'FRAME{:04d}/RAW/CAM{:d}/'.format(frame_idx, cam_idx)
            filenames.append(filename_str)
            


# In[26]:


def _read_hdf5_func(filename, label):
    filename_decoded = filename.decode("utf-8")
    print(filename_decoded)
    h5_file_name, group_name = filename_decoded.split('__')
    h5_file = h5py.File(h5_file_name, "r")
    #print(group_name)
    
    # Read depth image
    depth_image_path = group_name + 'Z'
    depth_image = h5_file[depth_image_path].value
    depth_image_scaled = np.array(depth_image, copy=False)
    depth_image_scaled.clip(MIN_DEPTH, MAX_DEPTH, out=depth_image_scaled)
    depth_image_scaled -= MIN_DEPTH
    np.floor_divide(depth_image_scaled, (MAX_DEPTH - MIN_DEPTH + 1) / 256,
                    out=depth_image_scaled, casting='unsafe')
    
    depth_image_scaled = depth_image_scaled.astype(np.uint8)
    
    # Read labels
    label_image_path = group_name + 'LABEL'
    label_image = h5_file[label_image_path].value
    h5_file.close()
    return depth_image_scaled, label_image

labels = [0]*len(filenames)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_hdf5_func, [filename, labels], [tf.uint8, tf.uint8])), num_parallel_calls=1)


dataset = dataset.batch(1)
dataset = dataset.repeat()
dataset = dataset.prefetch(1)


# In[27]:


iterator = dataset.make_one_shot_iterator()
next_example = iterator.get_next()

sess.run(next_example)

