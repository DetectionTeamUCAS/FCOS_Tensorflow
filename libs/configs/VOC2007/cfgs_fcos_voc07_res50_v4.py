# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import math
import tensorflow as tf
import numpy as np

"""

v2 + 896 * 896 + fix fcos_target.py bug

cls : person|| Recall: 0.9757067137809188 || Precison: 0.003594690143079733|| AP: 0.8034730720377914
____________________
cls : boat|| Recall: 0.9581749049429658 || Precison: 0.0005172063468607421|| AP: 0.5994630812893729
____________________
cls : bird|| Recall: 0.9694989106753813 || Precison: 0.0011239076627771885|| AP: 0.7794772924383294
____________________
cls : train|| Recall: 0.9929078014184397 || Precison: 0.00145036388593924|| AP: 0.8786161655096263
____________________
cls : pottedplant|| Recall: 0.9291666666666667 || Precison: 0.0009246647053823364|| AP: 0.47343885670470587
____________________
cls : bottle|| Recall: 0.9466950959488273 || Precison: 0.0006450066461833476|| AP: 0.5937734525972473
____________________
cls : tvmonitor|| Recall: 0.974025974025974 || Precison: 0.0006375599306334796|| AP: 0.7378064702147478
____________________
cls : cow|| Recall: 0.9918032786885246 || Precison: 0.0005160486877009817|| AP: 0.8139563088296033
____________________
cls : aeroplane|| Recall: 0.9719298245614035 || Precison: 0.0015950340888151832|| AP: 0.8157759165748075
____________________
cls : car|| Recall: 0.9891756869275604 || Precison: 0.0013678488734244695|| AP: 0.8959909639418537
____________________
cls : horse|| Recall: 1.0 || Precison: 0.0010865696247588004|| AP: 0.8254924125968947
____________________
cls : chair|| Recall: 0.9788359788359788 || Precison: 0.0008863570485747618|| AP: 0.49630409854675017
____________________
cls : diningtable|| Recall: 0.9660194174757282 || Precison: 0.0002147244890582227|| AP: 0.6354922604959689
____________________
cls : bicycle|| Recall: 0.9910979228486647 || Precison: 0.0005678369661864987|| AP: 0.8084494051443152
____________________
cls : sofa|| Recall: 0.9832635983263598 || Precison: 0.0005724377321722185|| AP: 0.6477540512002553
____________________
cls : motorbike|| Recall: 0.9876923076923076 || Precison: 0.00102992229059851|| AP: 0.7754105508297362
____________________
cls : bus|| Recall: 0.9953051643192489 || Precison: 0.0005510544114245016|| AP: 0.8442613936643346
____________________
cls : cat|| Recall: 0.9972067039106145 || Precison: 0.0021300080546523075|| AP: 0.8697736220269141
____________________
cls : dog|| Recall: 0.9938650306748467 || Precison: 0.00246340375491667|| AP: 0.844876629430716
____________________
cls : sheep|| Recall: 0.9834710743801653 || Precison: 0.0003620327989546683|| AP: 0.7341811847740185
____________________
mAP is : 0.7436883594423994

"""

# ------------------------------------------------
VERSION = 'FCOS_Res50_20201122'
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
SET_WIN = np.asarray([-1, 64, 128, 256, 512, 1e7]) * IMG_SHORT_SIDE_LEN / 800

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
