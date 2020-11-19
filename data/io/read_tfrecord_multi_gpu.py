# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import sys
sys.path.append('../../')

from data.io import image_preprocess_multi_gpu as image_preprocess
from libs.configs import cfgs


def read_single_example_and_decode(filename_queue):

    # tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    # reader = tf.TFRecordReader(options=tfrecord_options)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)

    img = tf.reshape(img, shape=[img_height, img_width, 3])

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])

    num_objects = tf.cast(features['num_objects'], tf.int32)
    return img_name, img, gtboxes_and_label, num_objects


def read_and_prepocess_single_img(filename_queue, shortside_len, is_training):

    img_name, img, gtboxes_and_label, num_objects = read_single_example_and_decode(filename_queue)

    img = tf.cast(img, tf.float32)

    if is_training:
        img, gtboxes_and_label, img_h, img_w = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                                  target_shortside_len=shortside_len,
                                                                                  length_limitation=cfgs.IMG_MAX_LENGTH)
        img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img,
                                                                         gtboxes_and_label=gtboxes_and_label)

    else:
        img, gtboxes_and_label, img_h, img_w = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                                  target_shortside_len=shortside_len,
                                                                                  length_limitation=cfgs.IMG_MAX_LENGTH)
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img = img / 255 - tf.constant([[cfgs.PIXEL_MEAN_]])
    else:
        img = img - tf.constant([[cfgs.PIXEL_MEAN]])  # sub pixel mean at last
    return img_name, img, gtboxes_and_label, num_objects, img_h, img_w


def next_batch(dataset_name, batch_size, shortside_len, is_training):
    '''
    :return:
    img_name_batch: shape(1, 1)
    img_batch: shape:(1, new_imgH, new_imgW, C)
    gtboxes_and_label_batch: shape(1, Num_Of_objects, 5] .each row is [x1, y1, x2, y2, label]
    '''
    # assert batch_size == 1, "we only support batch_size is 1.We may support large batch_size in the future"

    if dataset_name not in ['ship', 'spacenet', 'pascal', 'coco', 'bdd100k', 'DOTA']:
        raise ValueError('dataSet name must be in pascal, coco spacenet and ship')

    if is_training:
        pattern = os.path.join('../data/tfrecord', dataset_name + '_train*')
    else:
        pattern = os.path.join('../data/tfrecord', dataset_name + '_test*')

    print('tfrecord path is -->', os.path.abspath(pattern))

    filename_tensorlist = tf.train.match_filenames_once(pattern)

    filename_queue = tf.train.string_input_producer(filename_tensorlist)

    # shortside_len = tf.constant(shortside_len)
    # shortside_len = tf.random_shuffle(shortside_len)[0]

    img_name, img, gtboxes_and_label, num_obs, img_h, img_w = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                                            is_training=is_training)

    img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch, img_h_batch, img_w_batch = \
        tf.train.batch(
                       [img_name, img, gtboxes_and_label, num_obs, img_h, img_w],
                       batch_size=batch_size,
                       capacity=16,
                       num_threads=16,
                       dynamic_pad=True)

    return img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch, img_h_batch, img_w_batch


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    num_gpu = len(cfgs.GPU_GROUP.strip().split(','))
    img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, img_h_batch, img_w_batch = \
        next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                   batch_size=cfgs.BATCH_SIZE * 8,
                   shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                   is_training=True)
    gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        img_name_batch_, img_batch_, gtboxes_and_label_batch_, num_objects_batch_, img_h_batch_, img_w_batch_ \
            = sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, img_h_batch, img_w_batch])

        print(img_name_batch_.shape)
        print(img_batch_.shape)
        print(gtboxes_and_label_batch_.shape)
        print(num_objects_batch_.shape)
        print(img_h_batch_.shape)
        print('debug')

        coord.request_stop()
        coord.join(threads)