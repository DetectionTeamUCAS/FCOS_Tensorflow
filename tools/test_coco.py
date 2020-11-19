# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import time
import cv2
import numpy as np
import json
import math
from tqdm import tqdm
import argparse
from multiprocessing import Queue, Process
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.networks import build_whole_network
from help_utils import tools
from libs.label_name_dict.label_dict import *


def worker(gpu_id, images, det_net, eval_data, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
            print('restore model %d ...' % gpu_id)
        for a_img in images:
            raw_img = cv2.imread(os.path.join(eval_data, a_img['file_name']))
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}
                )

            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
            scales = [raw_w / resized_w, raw_h / resized_h]
            result_dict = {'scales': scales, 'boxes': detected_boxes,
                           'scores': detected_scores, 'labels': detected_categories,
                           'image_id': a_img['id']}
            result_queue.put_nowait(result_dict)


def test_coco(det_net, real_test_img_list, eval_data, gpu_ids):

    save_path = os.path.join('./eval_coco', cfgs.VERSION)
    tools.mkdir(save_path)
    fw_json_dt = open(os.path.join(save_path, 'coco_test-dev.json'), 'w')
    coco_det = []

    nr_records = len(real_test_img_list)
    pbar = tqdm(total=nr_records)
    gpu_num = len(gpu_ids.strip().split(','))

    nr_image = math.ceil(nr_records / gpu_num)
    result_queue = Queue(500)
    procs = []

    for i in range(gpu_num):
        start = i * nr_image
        end = min(start + nr_image, nr_records)
        split_records = real_test_img_list[start:end]
        proc = Process(target=worker, args=(i, split_records, det_net, eval_data, result_queue))
        print('process:%d, start:%d, end:%d' % (i, start, end))
        proc.start()
        procs.append(proc)

    for i in range(nr_records):
        res = result_queue.get()

        xmin, ymin, xmax, ymax = res['boxes'][:, 0], res['boxes'][:, 1], \
                                 res['boxes'][:, 2], res['boxes'][:, 3]

        xmin = xmin * res['scales'][0]
        xmax = xmax * res['scales'][0]

        ymin = ymin * res['scales'][1]
        ymax = ymax * res['scales'][1]

        boxes = np.transpose(np.stack([xmin, ymin, xmax-xmin, ymax-ymin]))

        for j, box in enumerate(boxes):
            coco_det.append({'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                             'score': float(res['scores'][j]), 'image_id': res['image_id'],
                             'category_id': int(classes_originID[LABEL_NAME_MAP[res['labels'][j]]])})

        pbar.set_description("Test image %s" % res['image_id'])

        pbar.update(1)

    for p in procs:
        p.join()

    json.dump(coco_det, fw_json_dt)
    fw_json_dt.close()


def eval(num_imgs, eval_data, json_file, gpu_ids):

    with open(json_file) as f:
        test_img_list = json.load(f)['images']

    if num_imgs == np.inf:
        real_test_img_list = test_img_list
    else:
        real_test_img_list = test_img_list[: num_imgs]

    fcos = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                is_training=False)
    test_coco(det_net=fcos, real_test_img_list=real_test_img_list, eval_data=eval_data, gpu_ids=gpu_ids)


def parse_args():

    parser = argparse.ArgumentParser('evaluate the result with Pascal2007 stdand')

    parser.add_argument('--eval_data', dest='eval_data',
                        help='evaluate imgs dir ',
                        default='/data/COCO/test2017', type=str)
    parser.add_argument('--json_file', dest='json_file',
                        help='test-dev json file',
                        default='image_info_test-dev2017.json', type=str)
    parser.add_argument('--showbox', dest='showbox',
                        help='whether show detecion results when evaluation',
                        default=False, type=bool)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpu id',
                        default='0,1,2,3,4,5,6,7', type=str)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    eval(np.inf,  # use np.inf to test all the imgs. use 10 to test 10 imgs.
         eval_data=args.eval_data,
         json_file=args.eval_gt,
         gpu_ids=args.gpus)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # eval(np.inf,  # use np.inf to test all the imgs. use 10 to test 10 imgs.
    #      eval_data='/data/COCO/test2017',
    #      json_file='/data/COCO/annotation/image_info_test-dev2017.json',
    #      gpu_ids='0,1,2,3')

    # cocoval('./eval_coco/FPN_Res101_20190108_v1/coco_res.json',
    #         '/unsullied/sharefs/_research_detection/GeneralDetection/COCO/data/MSCOCO/instances_minival2014.json')



