# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import math
import tensorflow as tf
import numpy as np

"""
cls : pottedplant|| Recall: 0.8125 || Precison: 0.0019816066256795896|| AP: 0.3682586113336541
____________________
cls : horse|| Recall: 0.9597701149425287 || Precison: 0.004477571922674746|| AP: 0.7940057638622982
____________________
cls : diningtable|| Recall: 0.8058252427184466 || Precison: 0.0006661236020433141|| AP: 0.5744291963472873
____________________
cls : bus|| Recall: 0.9483568075117371 || Precison: 0.0022466411602455735|| AP: 0.7551722474583373
____________________
cls : aeroplane|| Recall: 0.9157894736842105 || Precison: 0.00149109626997412|| AP: 0.7629470300020763
____________________
cls : cow|| Recall: 0.9549180327868853 || Precison: 0.0027648238463090194|| AP: 0.7163776059087306
____________________
cls : sheep|| Recall: 0.9297520661157025 || Precison: 0.002474185992808366|| AP: 0.7017814559336306
____________________
cls : person|| Recall: 0.8827296819787986 || Precison: 0.012277118240597115|| AP: 0.6732339345178622
____________________
cls : boat|| Recall: 0.8783269961977186 || Precison: 0.0018205892088712346|| AP: 0.5256596783771708
____________________
cls : bird|| Recall: 0.9259259259259259 || Precison: 0.003277500154235301|| AP: 0.7423172903696731
____________________
cls : car|| Recall: 0.9475437135720233 || Precison: 0.008746243649750601|| AP: 0.8375589859016583
____________________
cls : motorbike|| Recall: 0.96 || Precison: 0.004014668982821849|| AP: 0.740476470279191
____________________
cls : sofa|| Recall: 0.8786610878661087 || Precison: 0.0024223967885939716|| AP: 0.5864924424317157
____________________
cls : train|| Recall: 0.9361702127659575 || Precison: 0.003226003543716014|| AP: 0.7916057371506752
____________________
cls : dog|| Recall: 0.9734151329243353 || Precison: 0.007610155400652299|| AP: 0.8172101767123812
____________________
cls : bicycle|| Recall: 0.9436201780415431 || Precison: 0.003495235268902298|| AP: 0.787456868078286
____________________
cls : bottle|| Recall: 0.7334754797441365 || Precison: 0.0021579710054012006|| AP: 0.30458876836486803
____________________
cls : tvmonitor|| Recall: 0.9155844155844156 || Precison: 0.0018705723155297302|| AP: 0.6746970738847704
____________________
cls : chair|| Recall: 0.8677248677248677 || Precison: 0.003887061890794892|| AP: 0.4268281119933386
____________________
cls : cat|| Recall: 0.952513966480447 || Precison: 0.004570616698165051|| AP: 0.8413117684989478
____________________
mAP is : 0.6711204608703275

"""

# ------------------------------------------------
VERSION = 'FCOS_Res50_20201118'
NET_NAME = 'resnet50_v1d'  # 'resnet_v1_50'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 5000 * 2

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise NotImplementedError

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = False

MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

BATCH_SIZE = 4
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4 * NUM_GPU * BATCH_SIZE
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(0.125 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 20

# --------------------------------------------- Network_config
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
FINAL_CONV_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-np.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001

# ---------------------------------------------Anchor config
USE_CENTER_OFFSET = True
LEVLES = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE_LIST = [8, 16, 32, 64, 128]
SET_WIN = np.asarray([0, 64, 128, 256, 512, 1e5]) * IMG_SHORT_SIDE_LEN / 800

# --------------------------------------------FPN config
SHARE_HEADS = True
ALPHA = 0.25
GAMMA = 2
USE_P5 = True
USE_GN = False

NMS = True
NMS_IOU_THRESHOLD = 0.5
NMS_TYPE = 'NMS'
MAXIMUM_DETECTIONS = 300
FILTERED_SCORE = 0.001
VIS_SCORE = 0.2
