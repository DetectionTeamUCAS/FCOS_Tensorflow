# encoding: utf-8
import tensorflow as tf

from libs.configs import cfgs
from libs.box_utils import boxes_utils
# from libs.box_utils.py_cpu_nms import soft_nms


def debug(tensor):
    tensor = tf.Print(tensor, [tensor], 'tensor', summarize=200)
    return tensor


def filter_detections(boxes, scores, is_training):
    """
    :param boxes: [-1, 4]
    :param scores: [-1, ]
    :param labels: [-1, ]
    :return:
    """
    if is_training:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.VIS_SCORE)), [-1, ])
    else:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.FILTERED_SCORE)), [-1, ])

    if cfgs.NMS:
        filtered_boxes = tf.gather(boxes, indices)
        filtered_scores = tf.gather(scores, indices)

        if cfgs.NMS_TYPE == 'NMS':
            nms_indices = tf.image.non_max_suppression(boxes=filtered_boxes,
                                                       scores=filtered_scores,
                                                       max_output_size=cfgs.MAXIMUM_DETECTIONS,
                                                       iou_threshold=cfgs.NMS_IOU_THRESHOLD)
        else:
            # det = tf.concat([filtered_boxes, tf.expand_dims(filtered_scores, axis=-1)], axis=1)
            # nms_indices = tf.py_func(soft_nms, inp=[det, 0.5, 0.5, 0.001, 2],
            #                          Tout=[tf.int32])
            pass

        indices = tf.gather(indices, nms_indices)
    return indices


def postprocess_detctions(rpn_bbox, rpn_cls_prob, img_shape, is_training):
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
        indices = filter_detections(boxes, rpn_cls_prob[:, j], is_training)
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

