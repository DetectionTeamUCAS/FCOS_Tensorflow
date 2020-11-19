# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import math
import tensorflow as tf
import numpy as np

"""

v2 + 896 * 896

cls : tvmonitor|| Recall: 0.9577922077922078 || Precison: 0.0013966149840217777|| AP: 0.7664093115125827
____________________
cls : bird|| Recall: 0.9738562091503268 || Precison: 0.001657649104980735|| AP: 0.7627812295518228
____________________
cls : aeroplane|| Recall: 0.9649122807017544 || Precison: 0.0028677199019761196|| AP: 0.8483397103898138
____________________
cls : dog|| Recall: 0.9938650306748467 || Precison: 0.0019292068419360345|| AP: 0.8390880455505264
____________________
cls : pottedplant|| Recall: 0.925 || Precison: 0.0014933455312307656|| AP: 0.46948451330535623
____________________
cls : car|| Recall: 0.9800166527893422 || Precison: 0.0016859324194634236|| AP: 0.8880484673426596
____________________
cls : horse|| Recall: 0.9913793103448276 || Precison: 0.0009294857115145767|| AP: 0.8342729244615834
____________________
cls : sheep|| Recall: 0.9669421487603306 || Precison: 0.00043450892135945586|| AP: 0.7435088950875767
____________________
cls : diningtable|| Recall: 0.9271844660194175 || Precison: 0.00022820820713180518|| AP: 0.6168236330937569
____________________
cls : train|| Recall: 0.9716312056737588 || Precison: 0.002156784030352406|| AP: 0.8361595464607916
____________________
cls : boat|| Recall: 0.9467680608365019 || Precison: 0.00163322598206731|| AP: 0.6441541349312472
____________________
cls : bus|| Recall: 0.9953051643192489 || Precison: 0.0011203475191172508|| AP: 0.8269117140003418
____________________
cls : chair|| Recall: 0.9391534391534392 || Precison: 0.0009086790366466416|| AP: 0.4815866899543183
____________________
cls : bottle|| Recall: 0.9019189765458422 || Precison: 0.0010706069050348897|| AP: 0.5046887179731494
____________________
cls : cow|| Recall: 0.9836065573770492 || Precison: 0.0010339435036037239|| AP: 0.7777682207404092
____________________
cls : motorbike|| Recall: 0.9661538461538461 || Precison: 0.0013214709570984874|| AP: 0.7775758152921273
____________________
cls : bicycle|| Recall: 0.9762611275964391 || Precison: 0.0006883161986534945|| AP: 0.8243515498470947
____________________
cls : sofa|| Recall: 0.9205020920502092 || Precison: 0.0006549997320455642|| AP: 0.6115495824746827
____________________
cls : cat|| Recall: 0.994413407821229 || Precison: 0.001651412746494227|| AP: 0.8681401509332181
____________________
cls : person|| Recall: 0.9617932862190812 || Precison: 0.0032269609437693718|| AP: 0.7886623688810717
____________________
mAP is : 0.7355152610892064

"""

# ------------------------------------------------
VERSION = 'FCOS_Res50_20201120'
NET_NAME = 'resnet50_v1d'  # 'resnet_v1_50'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "1,2,3"
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

BATCH_SIZE = 2
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
IMG_SHORT_SIDE_LEN = 896
IMG_MAX_LENGTH = 896
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

# -------------------------------------------- FPN config
SHARE_HEADS = True
ALPHA = 0.25
GAMMA = 2
USE_P5 = True
USE_GN = True

NMS = True
NMS_IOU_THRESHOLD = 0.5
NMS_TYPE = 'NMS'
MAXIMUM_DETECTIONS = 300
FILTERED_SCORE = 0.001
VIS_SCORE = 0.2
