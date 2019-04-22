# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from libs.configs import cfgs


def focal_loss(pred, label, ignore_label=-1, background=0, alpha=0.5, gamma=2.0):

    mask = 1 - tf.cast(tf.equal(label, ignore_label), tf.int32)
    vlabel = tf.cast(label, tf.int32) * mask

    one_hot = tf.one_hot(vlabel, cfgs.CLASS_NUM+1, axis=2)
    onehot = one_hot[:, :, 1:]
    pos_part = tf.pow(1 - pred, gamma) * onehot * tf.log(pred)
    neg_part = tf.pow(pred, gamma) * (1 - onehot) * tf.log(1 - pred)
    loss = tf.reduce_sum(-(alpha * pos_part + (1 - alpha) * neg_part), axis=2) * tf.cast(mask, tf.float32)
    positive_mask = tf.cast(tf.greater(label, background), tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(positive_mask), 1)


def centerness_loss(pred, label, cls_gt, background=0):
    mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label) * tf.cast(mask, tf.float32)
    # not_neg_mask = tf.cast(tf.greater_equal(pred, 0), tf.float32)
    # loss = (pred * not_neg_mask - pred * label + tf.log(1 + tf.exp(-tf.abs(pred)))) * tf.cast(mask, tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1)


def iou_loss(pred, gt, cls_gt, background=0, weight=None):
    mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)

    aog = tf.abs(gt[:, :, 2] - gt[:, :, 0] + 1) * tf.abs(gt[:, :, 3] - gt[:, :, 1] + 1)
    aop = tf.abs(pred[:, :, 2] - pred[:, :, 0] + 1) * tf.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

    iw = tf.minimum(pred[:, :, 2], gt[:, :, 2]) - tf.maximum(pred[:, :, 0], gt[:, :, 0]) + 1
    ih = tf.minimum(pred[:, :, 3], gt[:, :, 3]) - tf.maximum(pred[:, :, 1], gt[:, :, 1]) + 1
    inter = tf.maximum(iw, 0) * tf.maximum(ih, 0)

    union = aog + aop - inter
    iou = tf.maximum(inter / union, 0)
    if weight is not None:
        iou *= weight
    loss = - tf.log(iou + cfgs.EPSILON) * tf.cast(mask, tf.float32)

    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
