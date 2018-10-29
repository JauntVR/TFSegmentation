"""
Trainer class to train Segmentation models
"""
import os
import h5py
import tensorflow as tf
import numpy as np
from train.basic_train import BasicTrain
from data.data_load_psy import load_dataset
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit
from utils.average_meter import FPSMeter
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.augmentation import flip_randomly_left_right_image_with_annotation, \
    scale_randomly_image_with_annotation_with_fixed_size_output
import scipy.misc as misc


class TrainPsy(BasicTrain):
    """
    Trainer class
    """


    def __init__(self, args, sess, train_model, test_model):
        """
        Call the constructor of the base class
        init summaries
        init loading data
        :param args:
        :param sess:
        :param model:
        :return:
        """
        super().__init__(args, sess, train_model, test_model)
        ##################################################################################
        # Init summaries

        # Summary variables
        self.scalar_summary_tags = ['mean_iou_on_val',
                                    'train-loss-per-epoch', 'val-loss-per-epoch',
                                    'train-acc-per-epoch', 'val-acc-per-epoch']
        self.images_summary_tags = [
            ('train_prediction_sample', [None, self.params.img_height * 2,
                                         self.params.img_width, 3]),
            ('val_prediction_sample', [None, self.params.img_height * 2,
                                       self.params.img_width, 3])]
        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}
        # init summaries and it's operators
        self.init_summaries()
        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)
        self.num_iterations_training_per_epoch = int((self.args.train_data_len - 1) / self.args.batch_size) + 1
        self.num_iterations_validation_per_epoch = 1
        ##################################################################################
        # Init metrics class
        self.metrics = Metrics(self.args.num_classes)
        # Init reporter class
        if self.args.mode == 'train' or 'overfit':
            self.reporter = Reporter(self.args.out_dir + 'report_train.json', self.args)
        elif self.args.mode == 'test':
            self.reporter = Reporter(self.args.out_dir + 'report_test.json', self.args)
            ##################################################################################
        train_seq_folder = self.args.data_dir + 'train_seq'
        test_seq_folder = self.args.data_dir + 'test_seq'
        valid_seq_folder = test_seq_folder #TODO create validation folder
        self.train_dataset = load_dataset(train_seq_folder,
                                             self.args.batch_size,
                                             self.args.img_height,
                                             self.args.img_width)
        self.valid_dataset = load_dataset(valid_seq_folder,
                                             self.args.batch_size,
                                             self.args.img_height,
                                             self.args.img_width)
        self.test_dataset = load_dataset(test_seq_folder,
                                             self.args.batch_size,
                                             self.args.img_height,
                                             self.args.img_width)

        self.dataset_train_iterator = self.train_dataset.make_one_shot_iterator()
        self.dataset_valid_iterator = self.valid_dataset.make_one_shot_iterator()
        self.dataset_test_iterator = self.test_dataset.make_one_shot_iterator()

    def init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        with tf.variable_scope('train-summary-per-epoch'):
            for tag in self.scalar_summary_tags:
                self.summary_tags += tag
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            for tag, shape in self.images_summary_tags:
                self.summary_tags += tag
                self.summary_placeholders[tag] = tf.placeholder('float32', shape, name=tag)
                self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=10)

    def add_summary(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

    def train(self):
        print("Training mode will begin NOW ..")
        # curr_lr= self.model.args.learning_rate
        next_element = self.dataset_train_iterator.get_next()
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            tt = tqdm(range(self.num_iterations_training_per_epoch), total=self.num_iterations_training_per_epoch,
                      desc="epoch-" + str(cur_epoch) + "-")
            # init acc and loss lists
            loss_list = []
            acc_list = []
            for cur_iteration in tt:
                # get the cur_it for the summary
                cur_it = self.model.global_step_tensor.eval(self.sess)
                next_batch = next_element
                x_batch, y_batch = self.sess.run(next_batch)

                # Feed this variables to the network
                feed_dict = {self.model.x_pl: x_batch,
                             self.model.y_pl: y_batch,
                             self.model.is_training: True
                             #self.model.curr_learning_rate:curr_lr
                             }
                save_image = (cur_iteration == self.num_iterations_training_per_epoch - 1) and \
                             (cur_epoch % self.args.save_every == 0)
                if not save_image:
                    # run the feed_forward
                    _, loss, acc, summaries_merged = self.sess.run(
                            [self.model.train_op, self.model.loss, self.model.accuracy,
                             self.model.merged_summaries],
                             feed_dict=feed_dict)
                    self.add_summary(cur_it, summaries_merged=summaries_merged)
                else:
                    #also get images
                    _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                            [self.model.train_op, self.model.loss, self.model.accuracy,
                             self.model.merged_summaries, self.model.segmented_summary],
                             feed_dict=feed_dict)

                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]

            total_loss = np.mean(loss_list)
            total_acc = np.mean(acc_list)

            # summarize
            summaries_dict = dict()
            summaries_dict['train-loss-per-epoch'] = total_loss
            summaries_dict['train-acc-per-epoch'] = total_acc
            if cur_epoch % self.args.save_every == 0:
                summaries_dict['train_prediction_sample'] = segmented_imgs
            self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

            # report
            self.reporter.report_experiment_statistics('train-acc', 'epoch-' + str(cur_epoch), str(total_acc))
            self.reporter.report_experiment_statistics('train-loss', 'epoch-' + str(cur_epoch), str(total_loss))
            self.reporter.finalize()

            # Update the Global step
            self.model.global_step_assign_op.eval(session=self.sess,
                                                  feed_dict={self.model.global_step_input: cur_it + 1})

            # Update the Cur Epoch tensor
            # it is the last thing because if it is interrupted it repeat this
            self.model.global_epoch_assign_op.eval(session=self.sess,
                                                   feed_dict={self.model.global_epoch_input: cur_epoch + 1})

            # print in console
            tt.close()
            print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" + " acc:" + str(total_acc)[
                                                                                                :6])

            # Save the current checkpoint
            if cur_epoch % self.args.save_every == 0:
                self.save_model()

            # Test the model on validation
            if cur_epoch % self.args.test_every == 0:
                self.test_per_epoch(step=self.model.global_step_tensor.eval(self.sess),
                                    epoch=self.model.global_epoch_tensor.eval(self.sess))

        print("Training Finished")

    def test_per_epoch(self, step, epoch):
        print("Validation at step:" + str(step) + " at epoch:" + str(epoch) + " ..")
        # init tqdm and get the epoch value
        tt = tqdm(range(self.num_iterations_validation_per_epoch), total=self.num_iterations_validation_per_epoch,
                  desc="Val-epoch-" + str(epoch) + "-")

        # init acc and loss lists
        loss_list = []
        acc_list = []

        # reset metrics
        self.metrics.reset()

        # get the maximum iou to compare with and save the best model
        max_iou = self.model.best_iou_tensor.eval(self.sess)
        next_element = self.dataset_valid_iterator.get_next()
        # loop by the number of iterations
        for cur_iteration in tt:
            # Feed this variables to the network
            next_batch = next_element
            x_batch, y_batch = self.sess.run(next_batch)
            feed_dict = {self.model.x_pl: x_batch,
                         self.model.y_pl: y_batch,
                         self.model.is_training: False
                         }

            # run the feed_forward
            if cur_iteration < self.num_iterations_validation_per_epoch - 1:
                out_argmax, loss, acc, summaries_merged = self.sess.run(
                        [self.model.out_argmax,
                         self.model.loss, self.model.accuracy,
                         self.model.merged_summaries],
                         feed_dict=feed_dict)
            else:
                out_argmax, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                        [self.test_model.out_argmax,
                         self.test_model.loss, self.model.accuracy,
                         self.test_model.merged_summaries,
                         self.test_model.test_segmented_summary],
                         feed_dict=feed_dict)

            # log loss and acc
            loss_list += [loss]
            acc_list += [acc]

            # log metrics
            self.metrics.update_metrics_batch(out_argmax, y_batch)

        # mean over batches
        total_acc = np.mean(acc_list)
        mean_iou = self.metrics.compute_final_metrics(self.num_iterations_validation_per_epoch)
        mean_iou_arr = self.metrics.iou

        # summarize
        summaries_dict = dict()
        summaries_dict['val-acc-per-epoch'] = total_acc
        summaries_dict['mean_iou_on_val'] = mean_iou
        summaries_dict['val_prediction_sample'] = segmented_imgs
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

        # report
        self.reporter.report_experiment_statistics('validation-acc', 'epoch-' + str(epoch), str(total_acc))
#        self.reporter.report_experiment_statistics('avg_inference_time_on_validation', 'epoch-' + str(epoch),
#                                                   str(mean_inference))
        self.reporter.report_experiment_validation_iou('epoch-' + str(epoch), str(mean_iou), mean_iou_arr)
        self.reporter.finalize()

        # print in console
        tt.close()
        print("Val-epoch-" + str(epoch) + "-" +
              "acc:" + str(total_acc)[:6] + "-mean_iou:" + str(mean_iou))
        print("Last_max_iou: " + str(max_iou))
        if mean_iou > max_iou:
            print("This validation got a new best iou. so we will save this one")
            # save the best model
            self.save_best_model()
            # Set the new maximum
            self.model.best_iou_assign_op.eval(session=self.sess,
                                               feed_dict={self.model.best_iou_input: mean_iou})
        else:
            print("hmm not the best validation epoch :/..")

    def test_eval(self, pkl=False):
        print("Testing mode will begin NOW..")

        # load the best model checkpoint to test on it
        if not pkl:
            self.load_best_model()
        next_element = self.dataset_test_iterator.get_next()

        # init tqdm and get the epoch value
        tt = tqdm(range(1))


        # loop by the number of iterations
        for _ in tt:
            next_batch = next_element
            x_batch, y_batch = self.sess.run(next_batch)
            # Feed this variables to the network
            if self.args.random_cropping:
                feed_dict = {self.test_model.x_pl_before: x_batch,
                             self.test_model.is_training: False,
                             }
            else:
                feed_dict = {self.test_model.x_pl: x_batch,
                             self.test_model.is_training: False
                             }

            # run the feed_forward
            out_argmax, segmented_imgs = self.sess.run(
                [self.test_model.out_argmax,
                 self.test_model.segmented_summary],
                feed_dict=feed_dict)

            # Colored results for visualization
            for i, img in enumerate(segmented_imgs):
                colored_save_path = self.args.out_dir + 'imgs/' + 'test_'+str(i) #str(self.names_mapper['Y'][idx])
                if not os.path.exists(os.path.dirname(colored_save_path)):
                   os.makedirs(os.path.dirname(colored_save_path))
                plt.imsave(colored_save_path, segmented_imgs[0])

            # Results for official evaluation
            #save_path = self.args.out_dir + 'results/' + str(self.names_mapper['Y'][idx])
            #if not os.path.exists(os.path.dirname(save_path)):
            #    os.makedirs(os.path.dirname(save_path))
            #output = postprocess(out_argmax[0])
            #misc.imsave(save_path, misc.imresize(output, [1024, 2048], 'nearest'))


        # print in console
        tt.close()


    def finalize(self):
        self.reporter.finalize()
        self.summary_writer.close()
        self.save_model()
