# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
from tqdm import tqdm
import numpy as np
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.val_libs import voc_eval
from libs.box_utils import draw_box_in_img
import argparse


def eval_with_plac(det_net, real_test_imgname_list, img_root, draw_imgs=False):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        all_boxes = []
        pbar = tqdm(real_test_imgname_list)
        for a_img_name in pbar:
            raw_img = cv2.imread(os.path.join(img_root, a_img_name))
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            if draw_imgs:
                show_indices = detected_scores >= cfgs.VIS_SCORE
                show_scores = detected_scores[show_indices]
                show_boxes = detected_boxes[show_indices]
                show_categories = detected_categories[show_indices]

                draw_img = np.squeeze(resized_img, 0)
                if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
                    draw_img = (draw_img * np.array(cfgs.PIXEL_STD) + np.array(cfgs.PIXEL_MEAN_)) * 255
                else:
                    draw_img = draw_img + np.array(cfgs.PIXEL_MEAN)
                final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,
                                                                                    boxes=show_boxes,
                                                                                    labels=show_categories,
                                                                                    scores=show_scores,
                                                                                    in_graph=False)
                if not os.path.exists(cfgs.TEST_SAVE_PATH):
                    os.makedirs(cfgs.TEST_SAVE_PATH)

                cv2.imwrite(cfgs.TEST_SAVE_PATH + '/' + a_img_name,
                            final_detections[:, :, ::-1])

            xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                     detected_boxes[:, 2], detected_boxes[:, 3]

            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]

            xmin = xmin * raw_w / resized_w
            xmax = xmax * raw_w / resized_w

            ymin = ymin * raw_h / resized_h
            ymax = ymax * raw_h / resized_h

            boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
            dets = np.hstack((detected_categories.reshape(-1, 1),
                              detected_scores.reshape(-1, 1),
                              boxes))
            all_boxes.append(dets)

            pbar.set_description("Eval image %s" % a_img_name)

        # save_dir = os.path.join(cfgs.EVALUATE_DIR, cfgs.VERSION)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # fw1 = open(os.path.join(save_dir, 'detections.pkl'), 'w')
        # pickle.dump(all_boxes, fw1)
        return all_boxes


def eval(num_imgs, eval_dir, annotation_dir, showbox):

    test_imgname_list = [item for item in os.listdir(eval_dir)
                         if item.endswith(('.jpg', 'jpeg', '.png', '.tif', '.tiff'))]
    if num_imgs == np.inf:
        real_test_imgname_list = test_imgname_list
    else:
        real_test_imgname_list = test_imgname_list[: num_imgs]

    fcos = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                is_training=False)
    all_boxes = eval_with_plac(det_net=fcos, real_test_imgname_list=real_test_imgname_list,
                               img_root=eval_dir, draw_imgs=showbox)

    # save_dir = os.path.join(cfgs.EVALUATE_DIR, cfgs.VERSION)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(os.path.join(save_dir, 'detections.pkl')) as f:
    #     all_boxes = pickle.load(f)
    #
    #     print(len(all_boxes))

    voc_eval.voc_evaluate_detections(all_boxes=all_boxes,
                                     test_annotation_path=annotation_dir,
                                     test_imgid_list=real_test_imgname_list)


def parse_args():

    parser = argparse.ArgumentParser('evaluate the result with Pascal2007 stdand')

    parser.add_argument('--eval_imgs', dest='eval_imgs',
                        help='evaluate imgs dir ',
                        default='/data/yangxue/dataset/VOC2007/VOC2007_test/VOCdevkit/VOC2007/JPEGImages', type=str)
    parser.add_argument('--annotation_dir', dest='test_annotation_dir',
                        help='the dir save annotations',
                        default='/data/yangxue/dataset/VOC2007/VOC2007_test/VOCdevkit/VOC2007/Annotations', type=str)
    parser.add_argument('--showbox', dest='showbox',
                        help='whether show detecion results when evaluation',
                        default=False, type=bool)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu id',
                        default='0', type=str)
    parser.add_argument('--eval_num', dest='eval_num',
                        help='the num of eval imgs',
                        default=np.inf, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(20*"--")
    print(args)
    print(20*"--")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    eval(args.eval_num,  # use np.inf to test all the imgs. use 10 to test 10 imgs.
         eval_dir=args.eval_imgs,
         annotation_dir=args.test_annotation_dir,
         showbox=args.showbox)
















