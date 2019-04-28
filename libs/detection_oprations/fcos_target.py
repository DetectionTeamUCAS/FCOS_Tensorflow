# encoding: utf-8

import numpy as np
import sys
import cv2
sys.path.append('../../')
from libs.configs import cfgs


def fcos_target(gt_boxes, image_batch, fm_size_list):
    gt_boxes = np.array(gt_boxes, np.int32)
    raw_height, raw_width = image_batch.shape[:2]

    gt_boxes = np.concatenate([np.zeros((1, 5)), gt_boxes])
    gt_boxes_area = (np.abs(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])))
    gt_boxes = gt_boxes[np.argsort(gt_boxes_area)]
    boxes_cnt = len(gt_boxes)

    shift_x = np.arange(0, raw_width).reshape(-1, 1)
    shift_y = np.arange(0, raw_height).reshape(-1, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    off_l = (shift_x[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
    off_t = (shift_y[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
    off_r = -(shift_x[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
    off_b = -(shift_y[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])

    center = ((np.minimum(off_l, off_r) * np.minimum(off_t, off_b)) / (
        np.maximum(off_l, off_r) * np.maximum(off_t, off_b) + cfgs.EPSILON))
    center = np.squeeze(np.sqrt(np.abs(center)))
    center[:, :, 0] = 0

    offset = np.concatenate([off_l, off_t, off_r, off_b], axis=3)
    cls = gt_boxes[:, 4]

    cls_res_list = []
    ctr_res_list = []
    gt_boxes_res_list = []

    for fm_i, stride in enumerate(cfgs.ANCHOR_STRIDE_LIST):
        fm_height = fm_size_list[fm_i][0]
        fm_width = fm_size_list[fm_i][1]

        shift_x = np.arange(0, fm_width)
        shift_y = np.arange(0, fm_height)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        xy = np.vstack((shift_y.ravel(), shift_x.ravel())).transpose()

        off_xy = offset[xy[:, 0] * stride, xy[:, 1] * stride]
        # off_xy = offset[xy[:, 0] * stride + int(0.5 * stride), xy[:, 1] * stride + int(0.5 * stride)]

        off_max_xy = off_xy.max(axis=2)
        off_valid = np.zeros((fm_height, fm_width, boxes_cnt))

        is_in_boxes = (off_xy > 0).all(axis=2)
        is_in_layer = (off_max_xy <= cfgs.SET_WIN[fm_i]) & \
                      (off_max_xy >= cfgs.SET_WIN[fm_i - 1])
        off_valid[xy[:, 0], xy[:, 1], :] = is_in_boxes & is_in_layer
        off_valid[:, :, 0] = 0

        hit_gt_ind = np.argmax(off_valid, axis=2)

        # gt_boxes
        gt_boxes_res = np.zeros((fm_height, fm_width, 4))
        gt_boxes_res[xy[:, 0], xy[:, 1]] = \
            gt_boxes[hit_gt_ind[xy[:, 0], xy[:, 1]], :4]
        gt_boxes_res_list.append(gt_boxes_res.reshape(-1, 4))

        # cls
        cls_res = np.zeros((fm_height, fm_width))
        cls_res[xy[:, 0], xy[:, 1]] = cls[hit_gt_ind[xy[:, 0], xy[:, 1]]]
        # cv2.imwrite('./cls.jpg', cls_res * 255)
        cls_res_list.append(cls_res.reshape(-1))

        # centerness
        center_res = np.zeros((fm_height, fm_width))
        center_res[xy[:, 0], xy[:, 1]] = center[
            xy[:, 0] * stride, xy[:, 1] * stride,
            hit_gt_ind[xy[:, 0], xy[:, 1]]]
        # center_res[xy[:, 0], xy[:, 1]] = center[
        #     xy[:, 0] * stride + int(0.5 * stride), xy[:, 1] * stride + int(0.5 * stride),
        #     hit_gt_ind[xy[:, 0], xy[:, 1]]]
        # cv2.imwrite('./centerness.jpg', center_res * 255)
        ctr_res_list.append(center_res.reshape(-1))

    cls_res_final = np.concatenate(cls_res_list, axis=0)[:, np.newaxis]
    ctr_res_final = np.concatenate(ctr_res_list, axis=0)[:, np.newaxis]
    gt_boxes_res_final = np.concatenate(gt_boxes_res_list, axis=0)
    return np.concatenate(
        [cls_res_final, ctr_res_final, gt_boxes_res_final], axis=1)


def get_fcos_target_batch(gtboxes_batch, img_batch, fm_size_list):
    fcos_target_batch = []
    for i in range(cfgs.BATCH_SIZE):
        gt_target = fcos_target(gtboxes_batch[i, :, :], img_batch[i, :, :, :], fm_size_list)
        fcos_target_batch.append(gt_target)
    return np.array(fcos_target_batch, np.float32)


def anchor_target(gt_boxes, resized_image):
    fm_stride = [128, 64, 32, 16, 8]

    sep_win = np.asarray([100000, 512, 256, 128, 64, 0]) * 600 / 800
    eps = 1e-5

    raw_height, raw_width = resized_image.shape[:2]

    # l, t, r, b
    gt_boxes = np.concatenate([np.zeros((1, 5)), gt_boxes])
    gt_boxes_area = (np.abs(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])))
    gt_boxes = gt_boxes[np.argsort(gt_boxes_area)]
    boxes_cnt = len(gt_boxes)

    shift_x = np.arange(0, raw_width).reshape(-1, 1)
    shift_y = np.arange(0, raw_height).reshape(-1, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    off_l = (shift_x[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
    off_t = (shift_y[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
    off_r = -(shift_x[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
    off_b = -(shift_y[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])

    center = ((np.minimum(off_l, off_r) * np.minimum(off_t, off_b)) / (
        np.maximum(off_l, off_r) * np.maximum(off_t, off_b) + eps))
    center = np.squeeze(np.sqrt(np.abs(center)))
    center[:, :, 0] = 0

    offset = np.concatenate([off_l, off_t, off_r, off_b], axis=3)
    cls = gt_boxes[:, 4]

    cls_res_list = []
    ctr_res_list = []
    gt_boxes_res_list = []

    fm_height = int(np.ceil(((raw_height + 1) / 2 // 2) / 2))
    fm_width = int(np.ceil(((raw_width + 1) / 2 // 2) / 2))

    fm_size_list = []
    for fm_i in range(len(fm_stride)):
        fm_size_list.append([fm_height, fm_width])
        fm_height = int(np.ceil(fm_height / 2))
        fm_width = int(np.ceil(fm_width / 2))

    fm_size_list = fm_size_list[::-1]
    print(fm_size_list)
    for fm_i, stride in enumerate(fm_stride):
        fm_height = fm_size_list[fm_i][0]
        fm_width = fm_size_list[fm_i][1]

        shift_x = np.arange(0, fm_width)
        shift_y = np.arange(0, fm_height)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        xy = np.vstack((shift_y.ravel(), shift_x.ravel())).transpose()
        off_xy = offset[xy[:, 0] * stride, xy[:, 1] * stride]
        off_max_xy = off_xy.max(axis=2)
        off_valid = np.zeros((fm_height, fm_width, boxes_cnt))

        is_in_boxes = (off_xy > 0).all(axis=2)
        is_in_layer = (off_max_xy <= sep_win[fm_i]) & \
                      (off_max_xy >= sep_win[fm_i + 1])
        off_valid[xy[:, 0], xy[:, 1], :] = is_in_boxes & is_in_layer
        off_valid[:, :, 0] = 0

        hit_gt_ind = np.argmax(off_valid, axis=2)

        # gt_boxes
        gt_boxes_res = np.zeros((fm_height, fm_width, 4))
        gt_boxes_res[xy[:, 0], xy[:, 1]] = \
            gt_boxes[hit_gt_ind[xy[:, 0], xy[:, 1]], :4]
        gt_boxes_res_list.append(gt_boxes_res.reshape(-1, 4))

        # cls
        cls_res = np.zeros((fm_height, fm_width))
        cls_res[xy[:, 0], xy[:, 1]] = cls[hit_gt_ind[xy[:, 0], xy[:, 1]]]
        cls_res_list.append(cls_res.reshape(-1))

        # center
        center_res = np.zeros((fm_height, fm_width))
        center_res[xy[:, 0], xy[:, 1]] = center[
            xy[:, 0] * stride, xy[:, 1] * stride,
            hit_gt_ind[xy[:, 0], xy[:, 1]]]
        ctr_res_list.append(center_res.reshape(-1))

    cls_res_final = np.concatenate(cls_res_list, axis=0)[:, np.newaxis]
    ctr_res_final = np.concatenate(ctr_res_list, axis=0)[:, np.newaxis]
    gt_boxes_res_final = np.concatenate(gt_boxes_res_list, axis=0)
    return np.concatenate(
        [cls_res_final, ctr_res_final, gt_boxes_res_final], axis=1)


if __name__ == '__main__':
    image = np.zeros([512, 512, 3])
    gt_boxes = np.array([[11, 11, 511, 511, 1],
                         [127+64, 127+64, 383-64, 383-64, 2],
                         [0, 0, 60, 60, 4]])
    # gt_boxes = np.array([[127, 127, 383, 383, 2]])
    fm_size_list = [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
    res = fcos_target(gt_boxes, image, fm_size_list)
    print(res.shape)
    # print(res[7542, :])
    print(np.argmax(res[:, 1]))

