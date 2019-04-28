# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

from libs.box_utils.nms.gpu_nms import gpu_nms
from libs.box_utils.nms.cpu_nms import cpu_nms, cpu_soft_nms
import numpy as np

USE_GPU_NMS = False


def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=2):

    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return np.array(keep, np.int32)


# Original NMS implementation
def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if USE_GPU_NMS and not force_cpu:
        return np.array(gpu_nms(dets, thresh, device_id=1))
    else:
        return np.array(cpu_nms(dets, thresh))