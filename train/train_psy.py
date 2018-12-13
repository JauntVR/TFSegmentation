"""
Trainer class to train Segmentation models
"""
import os
import h5py
import csv
import tensorflow as tf
import numpy as np
from train.basic_train import BasicTrain
from data.data_load_psy import load_dataset_w_labels, load_dataset_no_labels
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit
from utils.average_meter import FPSMeter
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.augmentation import flip_randomly_left_right_image_with_annotation, \
    scale_randomly_image_with_annotation_with_fixed_size_output
import scipy.misc as misc
import time
from utils.img_utils import decode_labels

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
        train_seq_file = self.args.train_data_file
        train_seq_type = self.args.train_data_type
        test_seq_file = self.args.test_data_file
        test_seq_type = self.args.test_data_type
        if self.args.valid_data_file is not None:
            valid_seq_file = self.args.valid_data_file
            valid_seq_type = self.args.valid_data_type
        else:
            valid_seq_file = test_seq_file
            valid_seq_type = test_seq_type

        self.train_dataset = load_dataset_w_labels(train_seq_file, train_seq_type, self.args)
        self.valid_dataset = load_dataset_w_labels(valid_seq_file, valid_seq_type, self.args)
        self.test_dataset = load_dataset_w_labels(test_seq_file, test_seq_type, self.args)


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
                # cur_it = self.model.global_step_tensor.eval(self.sess)
                # if cur_it == 0:
                #     tf.train.write_graph(self.sess.graph.as_graph_def(), self.args.exp_dir, 'input_graph_def.pb')

                next_batch = next_element
                if self.args.num_keypoints > 0:

                    x_batch, y_batch, y_rough_kp, y_refined_kp = self.sess.run(next_batch)
                    # Feed this variables to the network
                    feed_dict = {self.model.x_pl: x_batch,
                                 self.model.y_seg_pl: y_batch,
                                 self.model.y_kp_rough_pl: y_rough_kp,
                                 self.model.y_kp_refined_pl: y_refined_kp,
                                 self.model.is_training: True
                                 #self.model.curr_learning_rate:curr_lr
                                 }
                else:
                    x_batch, y_batch= self.sess.run(next_batch)
                    # Feed this variables to the network
                    feed_dict = {self.model.x_pl: x_batch,
                                 self.model.y_pl: y_batch,
                                 self.model.is_training: True
                                 #self.model.curr_learning_rate:curr_lr
                                                     }
                save_image = cur_iteration%200 == 0
                if not save_image or True:
                    # run the feed_forward
                    _, loss, acc, summaries_merged = self.sess.run(
                            [self.model.train_op, self.model.loss, self.model.accuracy,
                             self.model.merged_summaries],
                             feed_dict=feed_dict)
                    #self.add_summary(cur_it, summaries_merged=summaries_merged)
                # else:
                #     #also get images
                #     out_argmax, _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                #             [self.model.out_argmax, self.model.train_op, self.model.loss, self.model.accuracy,
                #              self.model.merged_summaries, self.model.segmented_summary],
                #              feed_dict=feed_dict)
                #
                #     for i, img in enumerate(out_argmax):
                #         label = decode_labels(np.expand_dims(y_batch[i],0),self.args.num_classes)[0]
                #         color = decode_labels(np.expand_dims(img,0),self.args.num_classes)[0]
                #         colored_save_path = self.args.out_dir + 'train_outputs/' + str(cur_epoch) + '_' + str(cur_iteration) + '_' + str(i)+ '_argmax.png'
                #         depth_save_path = self.args.out_dir + 'train_outputs/' + str(cur_epoch) + '_' +  str(cur_iteration) + '_' + str(i)+ '_depth.png'
                #         label_save_path = self.args.out_dir + 'train_outputs/' + str(cur_epoch) + '_' + str(cur_iteration) + '_' + str(i)+ '_label.png'
                #         if not os.path.exists(os.path.dirname(colored_save_path)):
                #             os.makedirs(os.path.dirname(colored_save_path))
                #         plt.imsave(colored_save_path, color)
                #         plt.imsave(depth_save_path, np.squeeze(x_batch[i]))
                #         plt.imsave(label_save_path, label)

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
        #
        #self.reporter.report_experiment_validation_iou('epoch-' + str(epoch), str(mean_iou), mean_iou_arr)
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
        tt = tqdm(range(10))


        # loop by the number of iterations
        for ind in tt:
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
            a = time.time()
            out_argmax, segmented_imgs = self.sess.run(
                [self.test_model.out_argmax,
                 self.test_model.segmented_summary],
                feed_dict=feed_dict)
            b = time.time()
            print("time = ", b-a)
            # Colored results for visualization
            for i, pred_img in enumerate(out_argmax):
                true_img = y_batch[i]
                out_img = np.concatenate((out_argmax[i],y_batch[i]),axis=1)
                colored_save_path = self.args.out_dir + 'imgs/' + 'batch_'+str(ind)+'test_'+str(i)+'_labels' #str(self.names_mapper['Y'][idx])
                depth_save_path = self.args.out_dir + 'imgs/' + 'batch_'+str(ind)+'test_'+str(i)+'_depth'
                if not os.path.exists(os.path.dirname(colored_save_path)):
                   os.makedirs(os.path.dirname(colored_save_path))
                plt.imsave(colored_save_path, out_img)
                if not os.path.exists(os.path.dirname(depth_save_path)):
                   os.makedirs(os.path.dirname(depth_save_path))
                plt.imsave(depth_save_path, np.squeeze(x_batch[i]))

            # Results for official evaluation
            #save_path = self.args.out_dir + 'results/' + str(self.names_mapper['Y'][idx])
            #if not os.path.exists(os.path.dirname(save_path)):
            #    os.makedirs(os.path.dirname(save_path))
            #output = postprocess(out_argmax[0])
            #misc.imsave(save_path, misc.imresize(output, [1024, 2048], 'nearest'))


        # print in console
        tt.close()


    def test(self, pkl=False):   #Runs the network on every single animation and measures the performance
        # print("Testing mode will begin NOW..")
        # files_dir = "/jaunt/users/trevor/volcap/SyntheticData/blender/outputs/front_cam_imgs/"
        # out_file = "/jaunt/users/trevor/TFSegmentation/file_performance.txt"
        # # load the best model f.write("\n".join(data))checkpoint to test on it
        # data = []
        # with open(out_file, 'w') as f:
        #     for name in os.listdir(files_dir):
        #         if name.endswith(".h5"):
        #             file_dataset = load_dataset_file(files_dir+name,
        #                                          self.args.batch_size,
        #                                          self.args.img_height,
        #                                          self.args.img_width,
        #                                          self.args.num_channels,
        #                                          self.args.num_classes)
        #             file_dataset_iterator = file_dataset.make_one_shot_iterator()
        #             next_element = file_dataset_iterator.get_next()
        #
        #             # init tqdm and get the epoch value
        #             tt = tqdm(range(10))
        #             loss_list = []
        #             acc_list = []
        #
        #             # loop by the number of iterations
        #             for ind in tt:
        #                 next_batch = next_element
        #                 x_batch, y_batch = self.sess.run(next_batch)
        #
        #                 feed_dict = {self.model.x_pl: x_batch,
        #                              self.model.y_pl: y_batch,
        #                              self.model.is_training: False
        #                              }
        #
        #                 # run the feed_forward
        #                 a = time.time()
        #                 out_argmax, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
        #                         [self.test_model.out_argmax,
        #                          self.test_model.loss, self.model.accuracy,
        #                          self.test_model.merged_summaries,
        #                          self.test_model.test_segmented_summary],
        #                          feed_dict=feed_dict)
        #                 b = time.time()
        #                 print("time = ", b-a)
        #
        #                 loss_list += [loss]
        #                 acc_list += [acc]
        #
        #                 # log metrics
        #                 self.metrics.update_metrics_batch(out_argmax, y_batch)
        #
        #             # mean over batches
        #             total_acc = np.mean(acc_list)
        #             total_loss = np.mean(loss)
        #             mean_iou = self.metrics.compute_final_metrics(self.num_iterations_validation_per_epoch)
        #             mean_iou_arr = self.metrics.iou
        #             info = [name, total_acc, total_loss, mean_iou]
        #             data.append(info)
        #             f.write(','.join([str(x) for x in info])+'\n')
        #             # print in console
        #             tt.close()
        # init tqdm and get the epoch value
        if not pkl:
            self.load_best_model()
        tt = tqdm(range(int(2000/self.args.batch_size)),
                  desc="Testing")
        # init acc and loss lists
        loss_list = []
        acc_list = []
        no_bg_acc_list = []
        # reset metrics
        self.metrics.reset()

        # get the maximum iou to compare with and save the best model
        next_element = self.dataset_test_iterator.get_next()
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
            if cur_iteration % 50 == 0: #save images
                for i, img in enumerate(out_argmax):
                    label = decode_labels(np.expand_dims(y_batch[i],0),self.args.num_classes)[0]
                    color = decode_labels(np.expand_dims(img,0),self.args.num_classes)[0]
                    colored_save_path = self.args.out_dir + 'test_outputs/' + str(cur_iteration) + '_' + str(i)+ 'argmax.png'
                    depth_save_path = self.args.out_dir + 'test_outputs/' + str(cur_iteration) + '_' + str(i)+ 'depth.png'
                    label_save_path = self.args.out_dir + 'test_outputs/' + str(cur_iteration) + '_' + str(i)+ 'label.png'
                    if not os.path.exists(os.path.dirname(colored_save_path)):
                        os.makedirs(os.path.dirname(colored_save_path))
                    plt.imsave(colored_save_path, color)
                    plt.imsave(depth_save_path, np.squeeze(x_batch[i]))
                    plt.imsave(label_save_path, label)

            # log loss and acc
            loss_list += [loss]
            acc_list += [acc]
            n_bg = np.sum(y_batch == 0)
            a,b,c = y_batch.shape
            n_tot = a*b*c
            n_wrong = (1 - acc)*n_tot
            no_bg_acc = 1 - n_wrong/(n_tot - n_bg)
            no_bg_acc_list += [no_bg_acc]

            # log metrics
            self.metrics.update_metrics_batch(out_argmax, y_batch)

        # mean over batches
        total_acc = np.mean(acc_list)
        total_no_bg_acc = np.mean(no_bg_acc_list)
        mean_iou = self.metrics.compute_final_metrics(self.num_iterations_validation_per_epoch)
        mean_iou_arr = self.metrics.iou


        # print in console
        tt.close()
        print(no_bg_acc_list)
        print(
              "acc:" + str(total_acc)[:6] + "-mean_iou:" + str(mean_iou) + "-no_bg_acc:" + str(total_no_bg_acc))

    def test_inference(self):
        print("Inference mode will begin NOW..")
        direct = 'raycast_real_data/'
        seq_file = '/jaunt/users/trevor/volcap/SyntheticData/blender/outputs/' + direct
        file_type = 'folder.npy'

        self.load_best_model()
        data_iterator =  load_dataset_no_labels(seq_file, file_type, self.args)
        dataset_iterator = data_iterator.make_one_shot_iterator()
        next_element = dataset_iterator.get_next()

        # init tqdm and get the epoch value
        tt = tqdm(range(10))


        # loop by the number of iterations
        for ind in tt:
            next_batch = next_element
            x_batch, filenames = self.sess.run(next_batch)
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
            a = time.time()
            out_softmax, out_argmax, segmented_imgs = self.sess.run(
                [self.test_model.out_softmax,
                self.test_model.out_argmax,
                 self.test_model.segmented_summary],
                feed_dict=feed_dict)
            b = time.time()
            print("time = ", b-a)
            # Colored results for visualization
            for i, pred_img in enumerate(out_argmax):
                f_name = "".join(filenames[i].decode('utf-8').split('/')[-1].split('.npy'))
                out_img = decode_labels(np.expand_dims(pred_img,0),self.args.num_classes)
                colored_save_path = self.args.out_dir + direct + f_name+'_labels_col.png' #str(self.names_mapper['Y'][idx])
                softmax_save_path = self.args.out_dir + direct + f_name+'_softmax.npy' #str(self.names_mapper['Y'][idx])
                conf_save_path = self.args.out_dir + direct + f_name+'_confidence.png'
                depth_save_path = self.args.out_dir + direct + f_name+'_depth.png'
                if not os.path.exists(os.path.dirname(colored_save_path)):
                   os.makedirs(os.path.dirname(colored_save_path))
                plt.imsave(colored_save_path, np.squeeze(out_img))
                if not os.path.exists(os.path.dirname(depth_save_path)):
                   os.makedirs(os.path.dirname(depth_save_path))
                plt.imsave(depth_save_path, np.squeeze(x_batch[i]))
                if not os.path.exists(os.path.dirname(softmax_save_path)):
                   os.makedirs(os.path.dirname(softmax_save_path))
                np.save(softmax_save_path, out_softmax[i])
                if not os.path.exists(os.path.dirname(conf_save_path)):
                   os.makedirs(os.path.dirname(conf_save_path))
                plt.imsave(softmax_save_path, np.max(out_softmax[i], axis=2))

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
