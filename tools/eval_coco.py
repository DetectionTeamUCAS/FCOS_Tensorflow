# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import json
import math
from tqdm import tqdm
from multiprocessing import Queue, Process
import argparse
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.networks import build_whole_network
from help_utils import tools
from libs.label_name_dict.label_dict import *

from data.lib_coco.PythonAPI.pycocotools.coco import COCO
from data.lib_coco.PythonAPI.pycocotools.cocoeval import COCOeval


def cocoval(detected_json, eval_json):
    eval_gt = COCO(eval_json)

    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    # cocoEval.params.imgIds = eval_gt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def worker(gpu_id, images, det_net, result_queue):
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
            record = json.loads(a_img)
            img_path = os.path.join('/data/dataset/COCO/val2017', record['fpath'].split('_')[-1])
            raw_img = cv2.imread(img_path)
            # raw_img = cv2.imread(record['fpath'])
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
                           'image_id': record['ID']}
            result_queue.put_nowait(result_dict)


def eval_coco(det_net, real_test_img_list, gpu_ids):

    save_path = os.path.join('./eval_coco', cfgs.VERSION)
    tools.mkdir(save_path)
    fw_json_dt = open(os.path.join(save_path, 'coco_minival.json'), 'w')
    coco_det = []

    nr_records = len(real_test_img_list)
    pbar = tqdm(total=nr_records)
    gpu_num = len(gpu_ids.strip().split(','))

    nr_image = math.ceil(nr_records / gpu_num)
    result_queue = Queue(5000)
    procs = []

    for i in range(gpu_num):
        start = i * nr_image
        end = min(start + nr_image, nr_records)
        split_records = real_test_img_list[start:end]
        proc = Process(target=worker, args=(i, split_records, det_net, result_queue))
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

        sort_scores = np.array(res['scores'])
        sort_labels = np.array(res['labels'])
        sort_boxes = np.array(boxes)

        # if len(res['scores']) > cfgs.MAXIMUM_DETECTIONS:
        #     sort_indx = np.argsort(np.array(res['scores']) * -1)[:cfgs.MAXIMUM_DETECTIONS]
        #     # print(sort_indx)
        #     sort_scores = np.array(res['scores'])[sort_indx]
        #     sort_labels = np.array(res['labels'])[sort_indx]
        #     sort_boxes = np.array(boxes)[sort_indx]

        for j, box in enumerate(sort_boxes):
            coco_det.append({'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                             'score': float(sort_scores[j]), 'image_id': int(res['image_id'].split('.jpg')[0].split('_000000')[-1]),
                             'category_id': int(classes_originID[LABEL_NAME_MAP[sort_labels[j]]])})

        pbar.set_description("Eval image %s" % res['image_id'])

        pbar.update(1)

    for p in procs:
        p.join()

    json.dump(coco_det, fw_json_dt)
    fw_json_dt.close()
    return os.path.join(save_path, 'coco_minival.json')


def eval(num_imgs, eval_data, eval_gt, gpu_ids):

    with open(eval_data) as f:
        test_img_list = f.readlines()

    if num_imgs == np.inf:
        real_test_img_list = test_img_list
    else:
        real_test_img_list = test_img_list[: num_imgs]

    fcos = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                is_training=False)
    detected_json = eval_coco(det_net=fcos, real_test_img_list=real_test_img_list, gpu_ids=gpu_ids)

    # save_path = os.path.join('./eval_coco', cfgs.VERSION)
    # detected_json = os.path.join(save_path, 'coco_minival.json')
    cocoval(detected_json, eval_gt)


def parse_args():

    parser = argparse.ArgumentParser('evaluate the result with Pascal2007 stdand')

    parser.add_argument('--eval_data', dest='eval_data',
                        help='evaluate imgs dir, download link: https://drive.google.com/file/d/1Au55e6lqvuTunNBZO2Cj4Kh9XySyM3ZN/view?usp=sharing',
                        default='/data/dataset/COCO/coco_minival2014.odgt', type=str)
    parser.add_argument('--eval_gt', dest='eval_gt',
                        help='eval gt, download link: https://drive.google.com/file/d/1cgyEzdGVfx7zPNUO0lLfm8pu0HfIj3Xv/view?usp=sharing',
                        default='/data/dataset/COCO/instances_minival2014.json',
                        type=str)
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
    eval(args.eval_num,  # use np.inf to test all the imgs. use 10 to test 10 imgs.
         eval_data=args.eval_data,
         eval_gt=args.eval_gt,
         gpu_ids=args.gpus)

    # os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP
    # eval(np.inf,  # use np.inf to test all the imgs. use 10 to test 10 imgs.
    #      eval_data='/data/COCO/coco_minival2014.odgt',
    #      eval_gt='/data/COCO/instances_minival2014.json',
    #      gpu_ids='0,1,2,3,4,5,6,7')




