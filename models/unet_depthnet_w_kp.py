from models.basic.basic_model_w_kp import BasicModel_w_kp
from models.encoders.depthnet import DepthNet
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf


class UNetDepthNet_w_kp(BasicModel_w_kp):
    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

    @staticmethod
    def _debug(operation):
        print("Layer_name: " + operation.op.name + " -Output_Shape: " + str(operation.shape.as_list()))

    def init_network(self):
        self.init_encoder()
        self.init_seg_branch()
        self.init_kp_branch()
        self.init_kp_refinement_branch()

    def init_encoder(self):
        batch_size = self.x_pl.shape[0]
        #TODO test this size once running on a strong enough cmputer
        batchnorm_enabled = batch_size > 10
        # Init DepthNet as an encoder
        self.encoder = DepthNet(x_input=self.x_pl, num_classes=self.params.num_classes,
                                 pretrained_path=self.args.pretrained_path, batchnorm_enabled = batchnorm_enabled,
                                 train_flag=self.is_training, width_multipler=1.0, weight_decay=self.args.weight_decay,
                                 dropout_keep_prob=self.args.dropout_keep_prob,)

        # Build Encoding part
        self.encoder.build()
    def init_seg_branch(self):
        """
        Building the segmentation branch here
        :return:
        """
        batch_size = self.x_pl.shape[0]
        #TODO test this size once running on a strong enough cmputer
        batchnorm_enabled = batch_size > 10
        # Build Decoding part
        with tf.name_scope('upscale_1'):
            self.expand11 = conv2d('expand1_1', x=self.encoder.conv5_6, batchnorm_enabled=batchnorm_enabled, is_training= self.is_training,
                                      num_filters=self.encoder.conv5_5.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.expand11)
            self.upscale1 = conv2d_transpose('upscale1', x=self.expand11,is_training= self.is_training,
                                             output_shape=self.encoder.conv5_5.shape.as_list(), batchnorm_enabled=batchnorm_enabled,
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob,l2_strength=self.encoder.wd)
            self._debug(self.upscale1)
            self.add1 = tf.add(self.upscale1, self.encoder.conv5_5)
            self._debug(self.add1)
            self.expand12 = conv2d('expand1_2', x=self.add1, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv5_5.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.expand12)

        with tf.name_scope('upscale_2'):
            self.expand21 = conv2d('expand2_1', x=self.expand12, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv4_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.expand21)
            self.upscale2 = conv2d_transpose('upscale2', x=self.expand21,is_training= self.is_training,
                                             output_shape=self.encoder.conv4_1.shape.as_list(),batchnorm_enabled=batchnorm_enabled,
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.upscale2)
            self.add2 = tf.add(self.upscale2, self.encoder.conv4_1)
            self._debug(self.add2)
            self.expand22 = conv2d('expand2_2', x=self.add2, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                              num_filters=self.encoder.conv4_1.shape.as_list()[3], kernel_size=(1, 1),
                              l2_strength=self.encoder.wd)
            self._debug(self.expand22)

        with tf.name_scope('upscale_3'):
            self.expand31 = conv2d('expand3_1', x=self.expand22, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv3_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.expand31)
            self.upscale3 = conv2d_transpose('upscale3', x=self.expand31, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                             output_shape=self.encoder.conv3_1.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.upscale3)
            self.add3 = tf.add(self.upscale3, self.encoder.conv3_1)
            self._debug(self.add3)
            self.expand32 = conv2d('expand3_2', x=self.add3, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                num_filters=self.encoder.conv3_1.shape.as_list()[3], kernel_size=(1, 1),
                                dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.expand32)

        with tf.name_scope('upscale_4'):
            self.expand41 = conv2d('expand4_1', x=self.expand32, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv2_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.expand41)
            self.upscale4 = conv2d_transpose('upscale4', x=self.expand41, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                             output_shape=self.encoder.conv2_1.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.upscale4)
            self.add4 = tf.add(self.upscale4, self.encoder.conv2_1)
            self._debug(self.add4)
            self.expand42 = conv2d('expand4_2', x=self.add4, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv2_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.expand42)

        with tf.name_scope('upscale_5'):
            self.upscale5 = conv2d_transpose('upscale5', x=self.expand42, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [
                                                 self.encoder.conv2_1.shape.as_list()[3]],
                                             dropout_keep_prob=self.args.dropout_keep_prob,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale5)
            self.expand5 = conv2d('expand5', x=self.upscale5, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv1_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob,
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand5)

        with tf.name_scope('final_score'):
            self.fscore = conv2d('fscore', x=self.expand5,
                                      num_filters=self.params.num_classes,
                                      kernel_size=(1, 1), l2_strength=self.encoder.wd)
            self._debug(self.fscore)

        self.logits = self.fscore

    def init_kp_branch(self):
        """
        Building the keypoint heatmap branch here
        :return:
        """
        batch_size = self.x_pl.shape[0]
        #TODO test this size once running on a strong enough cmputer
        batchnorm_enabled = batch_size > 10
        # Build Decoding part
        with tf.name_scope('kp_upscale_1'):
            self.kp_expand11 = conv2d('kp_expand1_1', x=self.encoder.conv5_6, batchnorm_enabled=batchnorm_enabled, is_training= self.is_training,
                                      num_filters=self.encoder.conv5_5.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_expand11)
            self.kp_upscale1 = conv2d_transpose('kp_upscale1', x=self.kp_expand11,is_training= self.is_training,
                                             output_shape=self.encoder.conv5_5.shape.as_list(), batchnorm_enabled=batchnorm_enabled,
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob,l2_strength=self.encoder.wd)
            self._debug(self.kp_upscale1)
            self.kp_add1 = tf.add(self.kp_upscale1, self.encoder.conv5_5)
            self._debug(self.kp_add1)
            self.kp_expand12 = conv2d('kp_expand1_2', x=self.kp_add1, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv5_5.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_expand12)

        with tf.name_scope('kp_upscale_2'):
            self.kp_expand21 = conv2d('kp_expand2_1', x=self.kp_expand12, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv4_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_expand21)
            self.kp_upscale2 = conv2d_transpose('kp_upscale2', x=self.kp_expand21,is_training= self.is_training,
                                             output_shape=self.encoder.conv4_1.shape.as_list(),batchnorm_enabled=batchnorm_enabled,
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_upscale2)
            self.kp_add2 = tf.add(self.kp_upscale2, self.encoder.conv4_1)
            self._debug(self.kp_add2)
            self.kp_expand22 = conv2d('kp_expand2_2', x=self.kp_add2, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                              num_filters=self.encoder.conv4_1.shape.as_list()[3], kernel_size=(1, 1),
                              l2_strength=self.encoder.wd)
            self._debug(self.kp_expand22)

        with tf.name_scope('kp_upscale_3'):
            self.kp_expand31 = conv2d('kp_expand3_1', x=self.kp_expand22, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv3_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_expand31)
            self.kp_upscale3 = conv2d_transpose('kp_upscale3', x=self.kp_expand31, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                             output_shape=self.encoder.conv3_1.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_upscale3)
            self.add3 = tf.add(self.kp_upscale3, self.encoder.conv3_1)
            self._debug(self.add3)
            self.kp_expand32 = conv2d('kp_expand3_2', x=self.add3, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                num_filters=self.encoder.conv3_1.shape.as_list()[3], kernel_size=(1, 1),
                                dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_expand32)

        with tf.name_scope('kp_upscale_4'):
            self.kp_expand41 = conv2d('kp_expand4_1', x=self.kp_expand32, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv2_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_expand41)
            self.kp_upscale4 = conv2d_transpose('kp_upscale4', x=self.kp_expand41, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                             output_shape=self.encoder.conv2_1.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2),
                                             dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_upscale4)
            self.add4 = tf.add(self.kp_upscale4, self.encoder.conv2_1)
            self._debug(self.add4)
            self.kp_expand42 = conv2d('kp_expand4_2', x=self.add4, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv2_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob, l2_strength=self.encoder.wd)
            self._debug(self.kp_expand42)

        with tf.name_scope('kp_upscale_5'):
            self.kp_upscale5 = conv2d_transpose('kp_upscale5', x=self.kp_expand42, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [
                                                 self.encoder.conv2_1.shape.as_list()[3]],
                                             dropout_keep_prob=self.args.dropout_keep_prob,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.kp_upscale5)
            self.kp_expand5 = conv2d('kp_expand5', x=self.kp_upscale5, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                      num_filters=self.encoder.conv1_1.shape.as_list()[3], kernel_size=(1, 1),
                                      dropout_keep_prob=self.args.dropout_keep_prob,
                                      l2_strength=self.encoder.wd)
            self._debug(self.kp_expand5)

        with tf.name_scope('kp_pre_final_score'):
            self.kp_fscore = conv2d('kp_fscore', x=self.kp_expand5,
                                      num_filters=self.params.num_keypoints,
                                      kernel_size=(1, 1), l2_strength=self.encoder.wd)
            self._debug(self.kp_fscore)

        self.kp_logits_rough = self.kp_fscore

    def init_kp_refinement_branch(self):
        """
        Building the keypoint heatmap branch here
        :return:
        """
        batch_size = self.x_pl.shape[0]
        #TODO test this size once running on a strong enough cmputer
        batchnorm_enabled = batch_size > 10

        with tf.name_scope('kp_refine'):
            self.concat = tf.concat([self.x_pl, self.kp_logits_rough],3)

            self.refine1 = conv2d('refine1', x=self.concat, batchnorm_enabled=batchnorm_enabled, is_training=self.is_training,
                                num_filters=32, kernel_size=(9,9), activation=tf.nn.relu6,
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                l2_strength=self.encoder.wd)
            self._debug(self.refine1)
            self.refine2 = conv2d('refine2', x=self.refine1, batchnorm_enabled=batchnorm_enabled, is_training=self.is_training,
                                num_filters=32, kernel_size=(13,13), activation=tf.nn.relu6,
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                l2_strength=self.encoder.wd)
            self._debug(self.refine2)
            self.refine3 = conv2d('refine3', x=self.refine2, batchnorm_enabled=batchnorm_enabled, is_training=self.is_training,
                                num_filters=64, kernel_size=(13,13), activation=tf.nn.relu6,
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                l2_strength=self.encoder.wd)
            self._debug(self.refine3)
            self.refine4 = conv2d('refine4', x=self.refine3, batchnorm_enabled=batchnorm_enabled, is_training=self.is_training,
                                num_filters=128, kernel_size=(15,15), activation=tf.nn.relu6,
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                l2_strength=self.encoder.wd)
            self._debug(self.refine4)
            self.refine5 = conv2d('refine5', x=self.refine4, batchnorm_enabled=batchnorm_enabled, is_training=self.is_training,
                                num_filters=256, kernel_size=(1,1), activation=tf.nn.relu6,
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                l2_strength=self.encoder.wd)
            self._debug(self.refine5)
            self.refine6 = conv2d('refine6', x=self.refine5, batchnorm_enabled=batchnorm_enabled, is_training=self.is_training,
                                num_filters=16, kernel_size=(1,1), activation=tf.nn.relu6,
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                l2_strength=self.encoder.wd)
            self._debug(self.refine6)
            self.refine7 = conv2d('refine7', x=self.refine6, batchnorm_enabled=batchnorm_enabled, is_training=self.is_training,
                                num_filters=64, kernel_size=(1,1), activation=tf.nn.relu6,
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                l2_strength=self.encoder.wd)
            self._debug(self.refine7)
            self.refine_upscale = conv2d_transpose('refine_upscale', x=self.refine7, batchnorm_enabled=batchnorm_enabled,is_training= self.is_training,
                                output_shape=self.x_pl.shape.as_list()[0:3] + [
                                    self.encoder.conv2_1.shape.as_list()[3]],
                                dropout_keep_prob=self.args.dropout_keep_prob,
                                kernel_size=(4, 4), activation=tf.nn.relu6, stride=(1, 1), l2_strength=self.encoder.wd)
            self._debug(self.refine_upscale)
            self.kp_fscore_refined = conv2d('kp_fscore_refined', x=self.refine_upscale,
                                      num_filters=self.params.num_keypoints,
                                      kernel_size=(1, 1), l2_strength=self.encoder.wd)
            self._debug(self.kp_fscore_refined)
            self.kp_logits_refined = self.kp_fscore_refined
