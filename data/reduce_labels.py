#Script to generate and save label images with a reduced label scheme
#run in folder with h5 files to add labels to all

import h5py
import os
import numpy as np
from tqdm import tqdm

dict_29_17 = {0:0,  #background
                       1:1,  #torso front
                       2:2, #torso back
                       3:3, #neck
                       4:4, 5:4, 6:4, #right upper arm
                       7:5, 8:5, #right lower arm
                       9:6, #right hand
                       10:7, 11:7, 12:7, #left upper arm
                       13:8, 14:8, #left lower arm
                       15:9, #left hand
                       16:10,#front of head
                       17:11, #back of head
                       18:12, #hips
                       19:13, 20:13, #right upper leg
                       21:14, 22:14, 23:14, #right lower leg
                       24:15, 25:15, #left upper leg
                       26:16, 27:16, 28:16, #left lower leg
                       'n_classes':17}

dict_29_15 = {0:0,  #background
                       1:1, 2:1, #torso back
                       3:2, #neck
                       4:3, 5:3, 6:3, #right upper arm
                       7:4, 8:4, #right lower arm
                       9:5, #right hand
                       10:6, 11:6, 12:6, #left upper arm
                       13:7, 14:7, #left lower arm
                       15:8, #left hand
                       16:9, 17:9, #back of head
                       18:10, #hips
                       19:11, 20:11, #right upper leg
                       21:12, 22:12, 23:12, #right lower leg
                       24:13, 25:13, #left upper leg
                       26:14, 27:14, 28:14, #left lower leg
                       'n_classes':15}

dict_17_15 = {0:0,   #Converts from coloring where front and back of head/torso are diff to coloring where they are the same
                        1:1, 2:1, #torso
                        3:2,
                        4:3,
                        5:4,
                        6:5,
                        7:6,
                        8:7,
                        9:8,
                        10:9, 11:9, #head
                        12:10,
                        13:11,
                        14:12,
                        15:13,
                        16:14,
                        'n_classes':15}

dict_19_17 = {0:0,   #Converts from coloring where front and back of head/torso are diff to coloring where they are the same (with 2 extra categories for noise)
                        1:1, 2:1, #torso
                        3:2,
                        4:3,
                        5:4,
                        6:5,
                        7:6,
                        8:7,
                        9:8,
                        10:9, 11:9, #head
                        12:10,
                        13:11,
                        14:12,
                        15:13,
                        16:14,
                        17:15, #noise (holes)
                        18:16, #noise (floaters)
                        'n_classes':17}

dict_29_11 = {0:0, #nada  #No longer valid
                       1:1, 2:1, #torso
                       3:2, #r_shoulder
                       4:3, 5:3, 6:3, 7:3, 8:3, #r_arm
                       9:4, #l_shoulder
                       10:5, 11:5, 12:5, 13:5, 14:5, #l_arm
                       15:6, #head
                       16:7, #waist
                       17:8, 18:8,19:8, 20:8, 21:8, #r_leg
                       22:9, 23:9, 24:9, 25:9, 26:9, #r_leg
                       27: 10, #background
                       'n_classes': 11}
dicts = {(29,17): dict_29_17,
         (29,15): dict_29_15,
         (17,15): dict_17_15,
         (19,17): dict_19_17,
         (29,11): dict_29_11}

def reduce_labels(img, final_num, starting_num = 29):
    dict = dicts[(starting_num, final_num)]
    return np.vectorize(lambda x: dict[x])(img)

def reduce_labels_h5(dir):
    for filename in tqdm(os.listdir(dir)):
        if '.h5' in filename:
            f = h5py.File(filename)
            for key in list(f.keys()):
                if 'FRAME' in key:
                    data = f[key]['RAW']
                    for cam_key in list(data.keys()):
                        if 'CAM' in cam_key:
                            imgs = data[cam_key]
                            for dict in dicts:
                                new_name = 'LABEL_' + str(dict['n_classes'])
                                if new_name not in list(imgs.keys()):
                                    old_labeled_img = np.array(imgs['LABEL'])
                                    new_labeled_img = np.vectorize(lambda x: dict[x])(old_labeled_img)
                                    imgs.create_dataset(new_name, data = new_labeled_img)

def reduce_labels_npy(dir):
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith('.npy') and 'label' in filename:
            old_labeled_img = np.load(dir + filename)
            for dict in dicts:
                new_labeled_img = np.vectorize(lambda x: dict[x])(old_labeled_img)
                new_name = (dir + filename).replace('label_','label_' + str(dict['n_classes']) + '_')
                np.save(new_name, new_labeled_img)

def collapse_front_back(dir):
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith('.npy') and 'label_17' in filename:
            old_labeled_img = np.load(dir + filename)
            dict = correspondance_dict3
            new_labeled_img = np.vectorize(lambda x: dict[x])(old_labeled_img)
            new_name = (dir + filename).replace('label_17_','label_' + str(dict['n_classes']) + '_')
            np.save(new_name, new_labeled_img)

def collapse_front_back_w_noise(dir):
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith('.npy') and 'label_19' in filename:
            old_labeled_img = np.load(dir + filename)
            dict = correspondance_dict4
            new_labeled_img = np.vectorize(lambda x: dict[x])(old_labeled_img)
            new_name = (dir + filename).replace('label_19_','label_' + str(dict['n_classes']) + '_')
            np.save(new_name, new_labeled_img)
