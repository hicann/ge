#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
## TF1.X environment
from npu_bridge.npu_init import *

if __main__ == ‘__main__’:
    data1 = tf.placeholder(tf.float32, shape=[1024, 128])
    input_data1 = np.random.rand(1024, 128).astype(np.float32)
    ## 构造模型结构
    abs_0 = tf.abs(data1)
    reduce_0 = tf.reduce_sum(abs_0, axis=0)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = “NpuOptimizer”
    custom_op.parameter_map[“use_off_line”].b = True
    custom_op.parameter_map[“graph_run_mode”].i = 0

    feed_dict = {data1: input_data1}
    step = 100
    ## 执行模型
    with tf.compat.v1.Session(config=sess_config) as sess:
        for _ in range(step):
            sess.run(reduce_0, feed_dict=feed_dict)
