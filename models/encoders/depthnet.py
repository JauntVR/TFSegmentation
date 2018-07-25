import tensorflow as tf
from layers.convolution import depthwise_separable_conv2d, conv2d
import os
from utils.misc import load_obj, save_obj

class DepthNet:
    """
    DepthNet Encoder class
    """
    MEAN = 1000.0

    def __init__(self, x_input,
                 num_classes,
                 pretrained_path,
                 train_flag,
                 width_multipler=1.0,
                 weight_decay=5e-4):

        # init parameters and input
        self.x_input = x_input
        self.num_classes = num_classes
        self.train_flag = train_flag
        self.wd = weight_decay
        self.pretrained_path = os.path.realpath(os.getcwd()) + "/" + pretrained_path
        self.width_multiplier = width_multipler

        # All layers
        self.conv1_1 = None

        self.conv2_1 = None
        self.conv2_2 = None

        self.conv3_1 = None
        self.conv3_2 = None

        self.conv4_1 = None
        self.conv4_2 = None

        self.conv5_1 = None
        self.conv5_2 = None
        self.conv5_3 = None
        self.conv5_4 = None
        self.conv5_5 = None
        self.conv5_6 = None

        self.conv6_1 = None
        self.flattened = None

        self.score_fr = None

        # These feed layers are for the decoder
        self.feed1 = None
        self.feed2 = None

    def build(self):
        self.encoder_build()

    @staticmethod
    def _debug(operation):
        print("Layer_name: " + operation.op.name + " -Output_Shape: " + str(operation.shape.as_list()))

    def encoder_build(self):
        print("Building the DepthNet..")
        with tf.variable_scope('mobilenet_encoder'):
            with tf.name_scope('Pre_Processing'):
                preprocessed_input = self.x_input - DepthNet.MEAN

            self.conv1_1 = conv2d('conv_1', preprocessed_input, num_filters=int(round(32 * self.width_multiplier)),
                                  kernel_size=(3, 3),
                                  padding='SAME', stride=(2, 2), activation=tf.nn.relu6, batchnorm_enabled=True,
                                  is_training=self.train_flag, l2_strength=self.wd)
            
            self._debug(self.conv1_1)
            self.conv2_1 = depthwise_separable_conv2d('conv_2_1', self.conv1_1, width_multiplier=self.width_multiplier,
                                                      num_filters=64, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd, activation=tf.nn.relu6)
            self._debug(self.conv2_1)
            self.conv2_2 = depthwise_separable_conv2d('conv_2_2', self.conv2_1, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv2_2)
            self.conv3_1 = depthwise_separable_conv2d('conv_3_1', self.conv2_2, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv3_1)
            self.conv3_2 = depthwise_separable_conv2d('conv_3_2', self.conv3_1, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv3_2)
            self.conv4_1 = depthwise_separable_conv2d('conv_4_1', self.conv3_2, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv4_1)
            self.conv4_2 = depthwise_separable_conv2d('conv_4_2', self.conv4_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv4_2)
            self.conv5_1 = depthwise_separable_conv2d('conv_5_1', self.conv4_2, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_1)
            self.conv5_2 = depthwise_separable_conv2d('conv_5_2', self.conv5_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_2)
            self.conv5_3 = depthwise_separable_conv2d('conv_5_3', self.conv5_2,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_3)
            self.conv5_4 = depthwise_separable_conv2d('conv_5_4', self.conv5_3,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_4)
            self.conv5_5 = depthwise_separable_conv2d('conv_5_5', self.conv5_4,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_5)
            self.conv5_6 = depthwise_separable_conv2d('conv_5_6', self.conv5_5,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_6)
            self.conv6_1 = depthwise_separable_conv2d('conv_6_1', self.conv5_6,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv6_1)
            # Pooling is removed.
            self.score_fr = conv2d('conv_1c_1x1', self.conv6_1, num_filters=self.num_classes, l2_strength=self.wd,
                                   kernel_size=(1, 1))

            self._debug(self.score_fr)
            self.feed1 = self.conv4_2
            self.feed2 = self.conv3_2

            print("\nEncoder DepthNet is built successfully\n\n")
