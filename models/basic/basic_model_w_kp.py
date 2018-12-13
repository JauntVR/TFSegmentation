"""
This file will contain the basic model of the segmentation problem
Any segmentation model will inherit from it .
You can override any function you want
"""

import utils.img_utils as imu
import numpy as np
import tensorflow as tf


class Params:
    """
    Class to hold BasicModel parameters
    """

    def __init__(self):
        self.img_width = None
        self.img_height = None
        self.num_channels = None
        self.num_classes = None
        self.weighted_loss = True
        self.class_weights = None


class BasicModel_w_kp:
    """
    Base class for any segmentation model
    """

    def __init__(self, args):
        self.args = args
        self.params = Params()
        # Init parameters
        self.params.img_width = self.args.img_width
        self.params.img_height = self.args.img_height
        self.params.num_channels = self.args.num_channels
        self.params.num_classes = self.args.num_classes
        self.params.num_keypoints = self.args.num_keypoints
        self.params.class_weights = np.ones(self.params.num_classes) * 1.0 / (self.params.num_classes)
        #np.load(self.args.data_dir + 'weights.npy')
        self.params.weighted_loss = self.args.weighted_loss
        # Input
        self.x_pl = None
        self.y_seg_pl = None
        self.y_seg_rough_pl = None
        self.y_seg_refined_pl = None
        self.is_training = None
        self.curr_learning_rate = None
        # Output
        self.logits = None
        self.out_softmax = None
        self.out_argmax = None
        self.out_one_hot = None
        self.y_argmax = None

        self.kp_logits_refined = None
        self.kp_logits_rough = None
        # Train
        self.cross_entropy_loss = None
        self.regularization_loss = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        # Summaries
        self.accuracy = None
        self.segmented_summary = None
        self.merged_summaries = None
        # Init global step
        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None
        self.init_global_step()
        # Init global epoch
        self.global_epoch_tensor = None
        self.global_epoch_input = None
        self.global_epoch_assign_op = None
        self.init_global_epoch()
        # Init Best iou tensor
        self.best_iou_tensor = None
        self.best_iou_input = None
        self.best_iou_assign_op = None
        #########################################

    def init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def build(self):
        """
        Override it in model class
        It will contain the logic of building a model
        :return:
        """
        raise NotImplementedError("build function is not implemented in the model")

    def init_input(self):
        with tf.name_scope('input'):
            self.x_pl = tf.placeholder(tf.float32,
                                       [self.args.batch_size,
                                        self.params.img_height,
                                        self.params.img_width, self.params.num_channels])
            self.y_seg_pl = tf.placeholder(tf.int32, [self.args.batch_size, self.params.img_height, self.params.img_width])
            self.y_kp_rough_pl = tf.placeholder(tf.int32, [self.args.batch_size,self.params.img_height, self.params.img_width, self.params.num_keypoints])
            self.y_kp_refined_pl = tf.placeholder(tf.float32, [self.args.batch_size,self.params.img_height, self.params.img_width, self.params.num_keypoints])
            #            self.curr_learning_rate= tf.placeholder(tf.float32)

            if self.params.weighted_loss:
                self.wghts = np.zeros((self.args.batch_size, 2, self.params.img_height, self.params.img_width),
                                      dtype=np.float32)
            self.is_training = tf.placeholder(tf.bool)

    def init_network(self):
        raise NotImplementedError("init_network function is not implemented in the model")

    def init_output(self):
        with tf.name_scope('output'):
            self.out_softmax = tf.nn.softmax(self.logits)
            self.out_argmax = tf.argmax(self.out_softmax, axis=3, output_type=tf.int32)
            self.out_conf =  tf.reduce_max(self.out_softmax, axis = 3)


    # def bootstrapped_ce_loss(self, ce, fraction):
    #     # only consider k worst pixels (lowest posterior probability) per image
    #     assert fraction is not None
    #     batch_size = ce.get_shape().as_list()[0]
    #     if batch_size is None:
    #         batch_size = tf.shape(ce)[0]
    #     k = tf.cast(tf.cast(tf.shape(ce)[1] * tf.shape(ce)[2], tf.float32) * fraction, tf.int32)
    #     bs_ce, _ = tf.nn.top_k(tf.reshape(ce, shape=[batch_size, -1]), k=k, sorted=False)
    #     bs_ce = tf.reduce_mean(bs_ce, axis=1)
    #     bs_ce = tf.reduce_sum(bs_ce, axis=0)
    #     return bs_ce

    def init_train(self):
        with tf.name_scope('loss'):

                #                self.ce = tf.reduce_mean(
                #                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_pl))
            seg_losses = tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.y_seg_pl)
            kp_losses_rough = tf.losses.sigmoid_cross_entropy(self.y_kp_rough_pl, self.kp_logits_rough)
            kp_losses_refined = tf.nn.l2_loss(self.y_kp_refined_pl - self.kp_logits_refined)

            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = tf.reduce_mean(seg_losses) + self.args.kp_loss_weight_rough*tf.reduce_mean(kp_losses_rough) + self.args.kp_loss_weight_refined*tf.reduce_mean(kp_losses_refined)+ self.regularization_loss

        with tf.name_scope('train-operation'):
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)

    def init_summaries(self):
        with tf.name_scope('pixel_wise_accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_seg_pl, self.out_argmax), tf.float32))

        with tf.name_scope('segmented_output'):
            input_summary = tf.py_func(imu.decode_input, [self.x_pl], tf.uint8)
#            labels_summary = tf.py_func(imu.decode_labels, [self.y_pl, self.params.num_classes], tf.uint8)
            preds_summary = tf.py_func(imu.decode_labels, [self.out_argmax, self.params.num_classes], tf.uint8)
#            conf_summary = tf.py_func(imu.decode_conf, [self.out_argmax], tf.uint8)
            self.segmented_summary = tf.concat(axis=1, values=[preds_summary,input_summary])
            self.test_segmented_summary = tf.concat(axis = 1, values = [preds_summary, input_summary])

        # Every step evaluate these summaries
        with tf.name_scope('train-summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('pixel_wise_accuracy', self.accuracy)

        self.merged_summaries = tf.summary.merge_all()

        # Save the best iou on validation
        self.best_iou_tensor = tf.Variable(0.0, trainable=False, name='best_iou')
        self.best_iou_input = tf.placeholder('float32', None, name='best_iou_input')
        self.best_iou_assign_op = self.best_iou_tensor.assign(self.best_iou_input)
