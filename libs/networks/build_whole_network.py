# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.configs import cfgs
from libs.losses import losses_fcos
from libs.detection_oprations.proposal_opr import postprocess_detctions
from libs.detection_oprations.fcos_target import get_fcos_target_batch
from libs.networks.ops import norm_


def debug(tensor):
    tensor = tf.Print(tensor, [tensor], 'tensor', summarize=50)
    return tensor


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training, batch_size):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.batch_size = batch_size

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def linspace(self, start, end, num):
        return np.array(np.linspace(start, end, num), np.float32)

    def get_rpn_bbox(self, offsets, stride):

        batch, fm_height, fm_width = tf.shape(offsets)[0], tf.shape(offsets)[1], tf.shape(offsets)[2]
        offsets = tf.reshape(offsets, [self.batch_size, -1, 4])

        y_list = tf.py_func(self.linspace, inp=[tf.constant(0.5), tf.cast(fm_height, tf.float32)-tf.constant(0.5),
                                                tf.cast(fm_height, tf.float32)],
                            Tout=[tf.float32])
        y_list = tf.broadcast_to(tf.reshape(y_list, [1, fm_height, 1, 1]), [1, fm_height, fm_width, 1])

        x_list = tf.py_func(self.linspace, inp=[tf.constant(0.5), tf.cast(fm_width, tf.float32)-tf.constant(0.5),
                                                tf.cast(fm_width, tf.float32)],
                            Tout=[tf.float32])
        x_list = tf.broadcast_to(tf.reshape(x_list, [1, 1, fm_width, 1]), [1, fm_height, fm_width, 1])

        xy_list = tf.concat([x_list, y_list], axis=3) * stride

        center = tf.reshape(tf.broadcast_to(xy_list, [self.batch_size, fm_height, fm_width, 2]),
                            [self.batch_size, -1, 2])

        xmin = tf.expand_dims(center[:, :, 0] - offsets[:, :, 0], axis=2)
        ymin = tf.expand_dims(center[:, :, 1] - offsets[:, :, 1], axis=2)
        xmax = tf.expand_dims(center[:, :, 0] + offsets[:, :, 2], axis=2)
        ymax = tf.expand_dims(center[:, :, 1] + offsets[:, :, 3], axis=2)
        all_boxes = tf.concat([xmin, ymin, xmax, ymax], axis=2)
        return all_boxes

    def rpn_cls_ctn_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(4):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                         scope='{}_{}'.format(scope_list[0], i),
                                         reuse=reuse_flag)
            # rpn_conv2d_3x3 = norm_(rpn_conv2d_3x3, 'group', is_train=self.is_training)
            # rpn_conv2d_3x3 = tf.contrib.layers.group_norm(rpn_conv2d_3x3)
            # rpn_conv2d_3x3 = tf.nn.relu(rpn_conv2d_3x3)

        rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=cfgs.CLASS_NUM,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=cfgs.FINAL_CONV_WEIGHTS_INITIALIZER,
                                     biases_initializer=cfgs.FINAL_CONV_BIAS_INITIALIZER,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        rpn_ctn_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=1,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                     scope=scope_list[3],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        rpn_box_scores = tf.reshape(rpn_box_scores, [self.batch_size, -1, cfgs.CLASS_NUM],
                                    name='rpn_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.nn.sigmoid(rpn_box_scores, name='rpn_{}_classification_sigmoid'.format(level))

        rpn_ctn_scores = tf.reshape(rpn_ctn_scores, [self.batch_size, -1],
                                    name='rpn_{}_centerness_reshape'.format(level))
        return rpn_box_scores, rpn_box_probs, rpn_ctn_scores

    def rpn_reg_net(self, inputs, scope_list, reuse_flag):
        rpn_box_offset = inputs
        for i in range(4):
            rpn_box_offset = slim.conv2d(inputs=rpn_box_offset,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                         stride=1,
                                         activation_fn=tf.nn.relu,
                                         scope='{}_{}'.format(scope_list[1], i),
                                         reuse=reuse_flag)
            # rpn_box_offset = norm_(rpn_box_offset, 'group', is_train=self.is_training)
            # rpn_box_offset = tf.contrib.layers.group_norm(rpn_box_offset)
            # rpn_box_offset = tf.nn.relu(rpn_box_offset)

        rpn_box_offset = slim.conv2d(rpn_box_offset,
                                     num_outputs=4,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                     scope=scope_list[4],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        # rpn_box_offset = tf.reshape(rpn_box_offset, [self.batch_size, -1, 4],
        #                             name='rpn_{}_regression_reshape'.format(level))
        return rpn_box_offset

    def rpn_net(self, feature_pyramid):

        rpn_box_list = []
        rpn_box_scores_list = []
        rpn_box_probs_list = []
        rpn_cnt_scores_list = []
        with tf.variable_scope('rpn_net'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                for level, stride in zip(cfgs.LEVLES, cfgs.ANCHOR_STRIDE_LIST):

                    if cfgs.SHARE_HEADS:
                        reuse_flag = None if level == 'P3' else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'rpn_classification',
                                      'rpn_centerness', 'rpn_regression']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'rpn_classification_' + level, 'rpn_centerness' + level,
                                      'rpn_regression_' + level]

                    rpn_box_scores, rpn_box_probs, rpn_ctn_scores = self.rpn_cls_ctn_net(feature_pyramid[level],
                                                                                         scope_list, reuse_flag, level)
                    rpn_box_offset = self.rpn_reg_net(feature_pyramid[level], scope_list, reuse_flag)

                    # si = tf.Variable(tf.constant(1.0),
                    #                  name='rpn_bbox_offsets_scale_'.format(level),
                    #                  dtype=tf.float32, trainable=True)
                    rpn_box_offset = tf.exp(rpn_box_offset) * stride

                    rpn_bbox = self.get_rpn_bbox(rpn_box_offset, stride)

                    rpn_box_scores_list.append(rpn_box_scores)
                    rpn_box_probs_list.append(rpn_box_probs)
                    rpn_cnt_scores_list.append(rpn_ctn_scores)
                    rpn_box_list.append(rpn_bbox)

                all_rpn_box_scores_list = tf.concat(rpn_box_scores_list, axis=1)
                all_rpn_box_probs_list = tf.concat(rpn_box_probs_list, axis=1)
                all_rpn_cnt_scores_list = tf.concat(rpn_cnt_scores_list, axis=1)
                all_rpn_box_list = tf.concat(rpn_box_list, axis=1)

            return all_rpn_box_scores_list, all_rpn_box_probs_list, all_rpn_cnt_scores_list, all_rpn_box_list

    def _fcos_target(self, feature_pyramid, img_batch, gtboxes_batch):
        with tf.variable_scope('fcos_target'):
            fm_size_list = []
            for level in cfgs.LEVLES:
                featuremap_height, featuremap_width = tf.shape(feature_pyramid[level])[1], tf.shape(feature_pyramid[level])[2]
                featuremap_height = tf.cast(featuremap_height, tf.int32)
                featuremap_width = tf.cast(featuremap_width, tf.int32)
                fm_size_list.append([featuremap_height, featuremap_width])

            fcos_target_batch = tf.py_func(get_fcos_target_batch,
                                           inp=[gtboxes_batch, img_batch, fm_size_list],
                                           Tout=[tf.float32])
            fcos_target_batch = tf.reshape(fcos_target_batch, [self.batch_size, -1, 6])
            return fcos_target_batch

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch):

        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [self.batch_size, -1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(input_img_batch)

        feature_pyramid = self.build_base_network(input_img_batch)  # [P3, P4, P5, P6, P7]

        rpn_cls_score, rpn_cls_prob, rpn_cnt_scores, rpn_box = self.rpn_net(feature_pyramid)

        rpn_cnt_prob = tf.nn.sigmoid(rpn_cnt_scores)
        rpn_cnt_prob = tf.expand_dims(rpn_cnt_prob, axis=2)
        rpn_cnt_prob = tf.broadcast_to(rpn_cnt_prob,
                                       [self.batch_size, tf.shape(rpn_cls_prob)[1], tf.shape(rpn_cls_prob)[2]])
        rpn_prob = rpn_cls_prob * rpn_cnt_prob

        if not self.is_training:
            with tf.variable_scope('postprocess_detctions'):
                boxes, scores, category = postprocess_detctions(rpn_bbox=rpn_box[0, :, :],
                                                                rpn_cls_prob=rpn_prob[0, :, :],
                                                                img_shape=img_shape)
                return boxes, scores, category
        else:
            with tf.variable_scope('postprocess_detctions'):
                boxes, scores, category = postprocess_detctions(rpn_bbox=rpn_box[0, :, :],
                                                                rpn_cls_prob=rpn_prob[0, :, :],
                                                                img_shape=img_shape)
            with tf.variable_scope('build_loss'):
                fcos_target_batch = self._fcos_target(feature_pyramid, input_img_batch, gtboxes_batch)

                cls_gt = tf.stop_gradient(fcos_target_batch[:, :, 0])
                ctr_gt = tf.stop_gradient(fcos_target_batch[:, :, 1])
                gt_boxes = tf.stop_gradient(fcos_target_batch[:, :, 2:])

                rpn_cls_loss = losses_fcos.focal_loss(rpn_cls_prob, cls_gt, alpha=cfgs.ALPHA, gamma=cfgs.GAMMA)
                # rpn_bbox_loss = losses_fcos.iou_loss(rpn_box, gt_boxes[:, :, :4], cls_gt, weight=ctr_gt)
                rpn_bbox_loss = losses_fcos.iou_loss(rpn_box, gt_boxes[:, :, :4], cls_gt)
                rpn_ctr_loss = losses_fcos.centerness_loss(rpn_cnt_scores, ctr_gt, cls_gt)
                loss_dict = {
                    'rpn_cls_loss': rpn_cls_loss,
                    'rpn_bbox_loss': rpn_bbox_loss,
                    'rpn_ctr_loss': rpn_ctr_loss
                }

            return boxes, scores, category, loss_dict

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])
            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith(self.base_network_name):
                    var_name_in_ckpt = name_in_ckpt_rpn(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients




















