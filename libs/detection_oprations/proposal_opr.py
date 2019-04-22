# encoding: utf-8
from libs.configs import cfgs
from libs.box_utils import boxes_utils
import tensorflow as tf


def filter_detections(boxes, scores):
    """
    :param boxes: [-1, 4]
    :param scores: [-1, ]
    :param labels: [-1, ]
    :return:
    """
    indices = tf.reshape(tf.where(tf.greater(scores, cfgs.FILTERED_SCORES)), [-1, ])

    if cfgs.NMS:
        filtered_boxes = tf.gather(boxes, indices)
        filtered_scores = tf.gather(scores, indices)

        # perform NMS
        nms_indices = tf.image.non_max_suppression(boxes=filtered_boxes,
                                                   scores=filtered_scores,
                                                   max_output_size=cfgs.MAXIMUM_DETECTIONS,
                                                   iou_threshold=cfgs.NMS_IOU_THRESHOLD)

        # filter indices based on NMS
        indices = tf.gather(indices, nms_indices)

    # add indices to list of all indices
    return indices


def postprocess_detctions(rpn_bbox, rpn_cls_prob, img_shape):
    '''
    :param rpn_bbox: [-1, 4]
    :param rpn_cls_prob: [-1, NUM_CLASS]
    :param img_shape:
    :return:
    '''

    boxes = boxes_utils.clip_boxes_to_img_boundaries(boxes=rpn_bbox,
                                                     img_shape=img_shape)

    return_boxes = []
    return_scores = []
    return_labels = []
    for j in range(0, cfgs.CLASS_NUM):
        indices = filter_detections(boxes, rpn_cls_prob[:, j])
        tmp_boxes = tf.reshape(tf.gather(boxes, indices), [-1, 4])
        return_boxes.append(tmp_boxes)
        tmp_scores = tf.gather(rpn_cls_prob[:, j], indices)
        tmp_scores = tf.reshape(tmp_scores, [-1, ])
        return_scores.append(tmp_scores)
        return_labels.append(tf.ones_like(tmp_scores)*(j+1))

    return_boxes = tf.concat(return_boxes, axis=0)
    return_scores = tf.concat(return_scores, axis=0)
    return_labels = tf.concat(return_labels, axis=0)

    return return_boxes, return_scores, return_labels

