# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops

from libs.configs import cfgs


def focal_loss(pred, label, background=0, alpha=0.5, gamma=2.0):

    label = tf.cast(label, tf.int32)
    one_hot = tf.one_hot(label, cfgs.CLASS_NUM+1, axis=2)
    onehot = one_hot[:, :, 1:]
    pos_part = tf.pow(1 - pred, gamma) * onehot * tf.log(pred)
    neg_part = tf.pow(pred, gamma) * (1 - onehot) * tf.log(1 - pred)
    loss = tf.reduce_sum(-(alpha * pos_part + (1 - alpha) * neg_part), axis=2)
    positive_mask = tf.cast(tf.greater(label, background), tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(positive_mask), 1)


def focal_loss_(pred, label, alpha=0.25, gamma=2.0):

    with tf.name_scope("focal_loss"):
        label = tf.cast(label, tf.int32)
        one_hot = tf.one_hot(label, cfgs.CLASS_NUM + 1, axis=2)
        onehot_labels = one_hot[:, :, 1:]

        logits = tf.cast(pred, tf.float32)
        onehot_labels = tf.cast(onehot_labels, tf.float32)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
        predictions = tf.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        loss = ce * tf.pow(1-predictions_pt, gamma) * alpha_t
        positive_mask = tf.cast(tf.greater(label, 0), tf.float32)
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(positive_mask), 1)


def focal_loss__(pred, label, alpha=0.25, gamma=2):

    label = tf.cast(label, tf.int32)
    one_hot = tf.one_hot(label, cfgs.CLASS_NUM + 1, axis=2)
    onehot_labels = one_hot[:, :, 1:]

    sigmoid_p = tf.nn.sigmoid(pred)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(onehot_labels > zeros, onehot_labels - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(onehot_labels > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    positive_mask = tf.stop_gradient(tf.cast(tf.greater(label, 0), tf.float32))
    return tf.reduce_sum(per_entry_cross_ent) / tf.maximum(tf.reduce_sum(positive_mask), 1)


def centerness_loss_(pred, label, cls_gt, background=0):
    mask = tf.stop_gradient(1 - tf.cast(tf.equal(cls_gt, background), tf.int32))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label)
    # not_neg_mask = tf.cast(tf.greater_equal(pred, 0), tf.float32)
    # loss = (pred * not_neg_mask - pred * label + tf.log(1 + tf.exp(-tf.abs(pred)))) * tf.cast(mask, tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1)


def iou_loss_(pred, gt, cls_gt, background=0, weight=None):
    mask = tf.stop_gradient(1 - tf.cast(tf.equal(cls_gt, background), tf.int32))

    area_gt = tf.abs(gt[:, :, 2] - gt[:, :, 0] + 1) * tf.abs(gt[:, :, 3] - gt[:, :, 1] + 1)
    area_pred = tf.abs(pred[:, :, 2] - pred[:, :, 0] + 1) * tf.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

    iw = tf.minimum(pred[:, :, 2], gt[:, :, 2]) - tf.maximum(pred[:, :, 0], gt[:, :, 0]) + 1
    ih = tf.minimum(pred[:, :, 3], gt[:, :, 3]) - tf.maximum(pred[:, :, 1], gt[:, :, 1]) + 1
    inter = tf.maximum(iw, 0) * tf.maximum(ih, 0)

    union = area_gt + area_pred - inter
    iou = tf.maximum(inter / union, 0)
    if weight is not None:
        iou *= weight
    loss = - tf.log(iou + cfgs.EPSILON) * tf.cast(mask, tf.float32)

    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)


def centerness_loss(pred, label, cls_gt, background=0):
    mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label) * tf.cast(mask, tf.float32)
    # not_neg_mask = tf.cast(tf.greater_equal(pred, 0), tf.float32)
    # loss = (pred * not_neg_mask - pred * label + tf.log(1 + tf.exp(-tf.abs(pred)))) * tf.cast(mask, tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1)


def iou_loss(pred, gt, cls_gt, background=0, weight=None):
    mask = 1 - tf.cast(tf.equal(cls_gt, background), tf.int32)

    area_gt = tf.abs(gt[:, :, 2] - gt[:, :, 0] + 1) * tf.abs(gt[:, :, 3] - gt[:, :, 1] + 1)
    area_pred = tf.abs(pred[:, :, 2] - pred[:, :, 0] + 1) * tf.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

    iw = tf.minimum(pred[:, :, 2], gt[:, :, 2]) - tf.maximum(pred[:, :, 0], gt[:, :, 0]) + 1
    ih = tf.minimum(pred[:, :, 3], gt[:, :, 3]) - tf.maximum(pred[:, :, 1], gt[:, :, 1]) + 1
    inter = tf.maximum(iw, 0) * tf.maximum(ih, 0)

    union = area_gt + area_pred - inter
    iou = tf.maximum(inter / union, 0)
    if weight is not None:
        iou *= weight
    loss = - tf.log(iou + cfgs.EPSILON) * tf.cast(mask, tf.float32)

    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
