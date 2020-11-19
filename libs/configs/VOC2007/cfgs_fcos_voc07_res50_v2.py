# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import math
import tensorflow as tf
import numpy as np

"""

v1+GN

cls : horse|| Recall: 0.9568965517241379 || Precison: 0.004740550928891736|| AP: 0.8120262016438552
____________________
cls : cat|| Recall: 0.9748603351955307 || Precison: 0.004712395355117472|| AP: 0.8580301734885958
____________________
cls : chair|| Recall: 0.8690476190476191 || Precison: 0.003712053155245182|| AP: 0.4338180787986161
____________________
cls : boat|| Recall: 0.8935361216730038 || Precison: 0.0035753951952774356|| AP: 0.5505445984636399
____________________
cls : tvmonitor|| Recall: 0.9383116883116883 || Precison: 0.003202287031291552|| AP: 0.6995652206185272
____________________
cls : person|| Recall: 0.8780918727915195 || Precison: 0.015182004589730763|| AP: 0.670838573161257
____________________
cls : car|| Recall: 0.9425478767693589 || Precison: 0.004985707930887165|| AP: 0.830416978111336
____________________
cls : motorbike|| Recall: 0.9476923076923077 || Precison: 0.003979276753530316|| AP: 0.7682180369740779
____________________
cls : train|| Recall: 0.9397163120567376 || Precison: 0.003298604628004531|| AP: 0.8043909646326083
____________________
cls : bicycle|| Recall: 0.9436201780415431 || Precison: 0.0023746938287830814|| AP: 0.8086468541469074
____________________
cls : cow|| Recall: 0.9713114754098361 || Precison: 0.0020537795609937865|| AP: 0.7556387831542724
____________________
cls : bottle|| Recall: 0.7526652452025586 || Precison: 0.0025633019395409295|| AP: 0.3111857817328514
____________________
cls : sofa|| Recall: 0.8870292887029289 || Precison: 0.0013511877067412794|| AP: 0.5857960018750333
____________________
cls : bus|| Recall: 0.971830985915493 || Precison: 0.001719240544177007|| AP: 0.7945002445411151
____________________
cls : diningtable|| Recall: 0.8252427184466019 || Precison: 0.000416670751674036|| AP: 0.5594467828722313
____________________
cls : aeroplane|| Recall: 0.9192982456140351 || Precison: 0.005216941120248501|| AP: 0.7870797923000652
____________________
cls : pottedplant|| Recall: 0.8 || Precison: 0.0030462893181547736|| AP: 0.3848575445835274
____________________
cls : sheep|| Recall: 0.9338842975206612 || Precison: 0.001781042146076979|| AP: 0.6985141045118906
____________________
cls : dog|| Recall: 0.9795501022494888 || Precison: 0.0055871134802234846|| AP: 0.8294480228959785
____________________
cls : bird|| Recall: 0.9259259259259259 || Precison: 0.0030731185283739224|| AP: 0.7593940815952821
____________________
mAP is : 0.6851178410050832
"""

# ------------------------------------------------
VERSION = 'FCOS_Res50_20201119'
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
