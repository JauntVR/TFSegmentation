import numpy as np
import copy
import os
from tqdm import tqdm

#Gets dimensions of non-background part of image
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

#Generates a splotch with the pixels concentrated around the n_loci randomly generated loci
def make_splotch_loci(noise_x, noise_y, n_loci):
    loci = np.concatenate((np.random.randint(int(noise_x/4),int(3*noise_x/4),(n_loci,1)), np.random.randint(int(noise_y/4),int(3*noise_y/4),(n_loci,1))), axis = 1)
    offset = np.zeros((noise_y, noise_x))
    for y, arry in enumerate(offset):
        for x, val in enumerate(arry):
            dist = min([(x-locus[0])**2 + (y - locus[1])**2 for locus in loci])
            offset[y,x] = .5 - 3*dist**.5/(noise_x*noise_y)**.5
    noise_map = np.random.random((noise_y,noise_x))
    noise_splotch = (offset + noise_map) > .5
    return noise_splotch

def make_splotch_strand(noise_x, noise_y, n_lines):
    lines = []
    for _ in range(n_lines):
        x1, x2 = np.random.randint(0,noise_x,2)
        y1, y2 = np.random.randint(0, noise_y,2)
        lines.append([x1,x2,y1,y2])
    offset = np.zeros((noise_y, noise_x))

    for y, arry in enumerate(offset):
        for x, val in enumerate(arry):
            edge_dist = min(min(x,noise_x-x)**2, min(y, noise_y-y)**2)**.5
            dist = min([abs((y2 - y1)*x - (x2 - x1)*y + x2*y1-y2*x1)/np.sqrt((y2-y1)**2 + (x2-x1)**2 + 1) for x1,x2,y1,y2 in lines])
            offset[y,x] = .5 - 10*dist/(noise_x*noise_y)**.5
            if edge_dist < 10:
                offset[y,x] = offset[y,x]-1 + (edge_dist)/10
    noise_map = np.random.random((noise_y,noise_x))
    noise_splotch = (offset + noise_map) > .5
    return noise_splotch

#Adds randomly generated floaters nad holes to a single image depending on the parameters in the floaters and holes lists passed
def add_noise_imgs(depth, labels, holes, floaters, hole_label, floater_label, bg_label = 0, bg_depth = 0):
    y_dim, x_dim = depth.shape[:2]
    new_depth = copy.copy(depth)
    new_labels = copy.copy(labels)
    for hole in holes:
        y_noise, x_noise, depth = hole
        splotch = make_splotch_strand(x_noise, y_noise,5)
        for _ in range(100):
            x = np.random.randint(0, x_dim-x_noise)
            y = np.random.randint(0, y_dim-y_noise)
            if new_depth[y,x] != bg_depth and new_depth[y+y_noise, x+x_noise] != bg_depth:
                break
        aligned_splotch = splotch*(new_labels[y:y+y_noise, x:x+x_noise] != bg_label)
        new_depth[y:y+y_noise, x:x+x_noise] = aligned_splotch*depth + new_depth[y:y+y_noise, x:x+x_noise] * (1-aligned_splotch)
        new_labels[y:y+y_noise, x:x+x_noise] = aligned_splotch*hole_label + new_labels[y:y+y_noise, x:x+x_noise] * (1-aligned_splotch)
    for floater in floaters:
        y_noise, x_noise, depth_val = floater
        splotch = make_splotch_strand(x_noise, y_noise,5)
        if depth_val == -1:
            y_min, y_max, x_min, x_max = get_dims(labels, bg_label)
            if np.random.randint(0,2):
                x = np.random.randint(x_min, min(x_max, x_dim-x_noise))
                if np.random.randint(0,2):
                    ys = reversed(range(0, y_dim-y_noise))
                else:
                    ys = range(0, y_dim-y_noise)
                for y in ys:
                    if (new_depth[y,x] == bg_depth) != (new_depth[y+y_noise,x] == bg_depth):
                        depth_val = new_depth[y,x] + new_depth[y+y_noise,x] - bg_depth
                        break
            else:
                y = np.random.randint(y_min, min(y_max, y_dim-y_noise))
                if np.random.randint(0,2):
                    xs = reversed(range(0, x_dim-x_noise))
                else:
                    xs = range(0, x_dim-x_noise)
                for x in xs:
                    if (new_depth[y,x] == bg_depth) != (new_depth[y,x+x_noise] == bg_depth):
                        depth_val = new_depth[y,x] + new_depth[y,x+x_noise] - bg_depth
                        break
        else:
            x = np.random.randint(0, x_dim-x_noise)
            y = np.random.randint(0, y_dim-y_noise)
            for _ in range(100):
                if new_depth[y,x] == bg_depth:
                    break
        if splotch.shape != new_labels[y:y+y_noise, x:x+x_noise].shape:
            import ipdb; ipdb.set_trace()
        aligned_splotch = splotch*(new_labels[y:y+y_noise, x:x+x_noise] == bg_label)
        new_depth[y:y+y_noise, x:x+x_noise] = aligned_splotch*depth_val + new_depth[y:y+y_noise, x:x+x_noise] * (1-aligned_splotch)
        new_labels[y:y+y_noise, x:x+x_noise] = aligned_splotch*floater_label + new_labels[y:y+y_noise, x:x+x_noise] * (1-aligned_splotch)

    return new_depth, new_labels

#Adds random holes and floaters to two photos trying to mimic true noise in depth data
def add_hf_noise(depth, label, max_label):
    max_depth = np.max(depth)
    holes = []
    #small holes w/ z = 0
    for _ in range(np.random.randint(4,6)):
        holes.append([np.random.randint(3,11),np.random.randint(3,11),0])
    #larger holes w/ z = 0
    for _ in range(np.random.randint(2,4)):
        holes.append([np.random.randint(10,20),np.random.randint(10,20),0])
    #small holes with random values
    for _ in range(np.random.randint(2,6)):
        holes.append([np.random.randint(3,11),np.random.randint(3,11),np.random.randint(0,max_depth)])

    floaters = []
    #large solid floaters away from body
    for _ in range(np.random.randint(1,3)):
        if np.random.random() > .9:
            depth_val = np.random.randint(0, int(max_depth/4))
        else:
            depth_val = max_depth
        floaters.append([np.random.randint(20,100),np.random.randint(20,100),depth_val])
    #small solid floaters away from the body
    for _ in range(np.random.randint(2,4)):
        if np.random.random() > .9:
            depth_val = np.random.randint(0, int(max_depth/4))
        else:
            depth_val = max_depth
        floaters.append([np.random.randint(5,20),np.random.randint(5,20),depth_val])
    #small solid floaters close to the body
    for _ in range(np.random.randint(3,6)):
        floaters.append([np.random.randint(7,17),np.random.randint(7,17),-1])
    #large solid floater close to the body in 1/2 of the images
    if np.random.random() > .5:
        floaters.append([np.random.randint(20,70),np.random.randint(20,70),-1])
    new_depth, new_label = add_noise_imgs(depth, label, holes, floaters, max_label+1, max_label+2)

    return new_depth, new_label

def add_gaussian_noise(img, avg, std, bg_val = 0):

    noise = np.random.normal(avg, std, img.shape)

    out = (img + noise*(img != bg_val)*(img+noise >= 0)*(img+noise <= 255)).astype('uint8')
    return out
#Adds holes and floaters to all images in a given directory and saves the noisy images to another directory
def add_noise_npy_files(directory, outdir, max_label):
    files = os.listdir(directory)
    already_written = open(outdir+'already_written.txt', 'w')
    for file in tqdm(files):
        if file.endswith('.npy') and 'depth' in file:
            prefix, suffix = file.split('_depth_')
            label_filename = prefix + '_label_' + str(max_label + 1) + '_' + suffix

            depth_img = np.load(directory + file)
            label_img = np.load(directory + label_filename)

            new_depth, new_label = add_random_noise(depth_img, label_img, max_label)
            new_label_filename = prefix + '_noise_label_' + str(max_label + 3) + '_' + suffix
            new_depth_filename = prefix + '_noise_depth_' + suffix
            np.save(outdir+new_label_filename, new_label)
            np.save(outdir+new_depth_filename, new_depth)
            already_written.write(file+'\n')
