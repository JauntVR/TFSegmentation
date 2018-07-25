def _read_hdf5_func(filename, label):
    filename_decoded = filename.decode("utf-8")
    h5_file_name, group_name = filename_decoded.split('__')
    h5_file = h5py.File(h5_file_name, "r")
    #print(group_name)

    # Read depth image
    depth_image_path = group_name + 'Z'
    depth_image = h5_file[depth_image_path].value

    # Read labels
    label_image_path = group_name + 'LABEL'
    label_image = h5_file[label_image_path].value
    h5_file.close()
    return depth_image, label_image


def generate_datasets(train_seq_folder, val_seq_folder, params):
    print(train_seq_folder)
    print(val_seq_folder)

    train_seq_files = []
    for (dirpath, dirnames, filenames) in os.walk(train_seq_folder):
        train_seq_files.extend(os.path.join(dirpath, x) for x in filenames)
    print(train_seq_files)

    val_seq_files = []
    for (dirpath, dirnames, filenames) in os.walk(val_seq_folder):
        val_seq_files.extend(os.path.join(dirpath, x) for x in filenames)
    print(val_seq_files)

    train_filenames = []
    for train_seq_name in train_seq_files:
        train_seq = h5py.File(train_seq_name, "r")
        num_cameras = train_seq['INFO']['NUM_CAMERAS'].value[0]
        num_frames = train_seq['INFO']['COUNT'].value[0]
        train_seq.close()
        for frame_idx in range(num_frames):
            for cam_idx in range(num_cameras):
                train_filename_str = train_seq_name + '__' + 'FRAME{:04d}/RAW/CAM{:d}/'.format(frame_idx, cam_idx)
                train_filenames.append(train_filename_str)

    val_filenames = []
    for val_seq_name in val_seq_files:
        val_seq = h5py.File(val_seq_name, "r")
        num_cameras = val_seq['INFO']['NUM_CAMERAS'].value[0]
        num_frames = val_seq['INFO']['COUNT'].value[0]
        val_seq.close()
        for frame_idx in range(num_frames):
            for cam_idx in range(num_cameras):
                val_filename_str = val_seq_name + '__' + 'FRAME{:04d}/RAW/CAM{:d}/'.format(frame_idx, cam_idx)
                val_filenames.append(val_filename_str)



    parse_fn = lambda filename, label: tuple(tf.py_func(
                        read_hdf5_func, [filename, label], [tf.int16, tf.uint8]))

    val_labels = [0]*len(val_filenames) 
    val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        .shuffle(buffer_size=10000)
        .map(parse_fn, num_parallel_calls=params.num_parallel_calls)

    train_labels = [0]*len(train_filenames)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        .shuffle(buffer_size=10000)
        .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
        .batch(params.batch_size)
        .repeat()
        .prefetch(1)
    
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
    
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)
    
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 
              'labels': labels, 
              'train_init_op': train_init_op, 
              'val_init_op': val_init_op}
    
    return inputs

