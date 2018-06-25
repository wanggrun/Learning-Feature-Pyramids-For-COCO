#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: gr_attention.py
# Author: Guangrun Wang <wanggrun@mail2.sysu.edu.cn>

import tensorflow as tf
from tensorpack.models.common import layer_register
import numpy as np

__all__ = ['Grresize']
@layer_register(log_shape=True)
def Grresize(x, resize_shape, nl=tf.identity):
	#x = tf.transpose(x, [0,2,3,1])
	outputs = tf.image.resize_images(x, resize_shape)
	#outputs = tf.transpose(outputs, [0,3,1,2])
	ret = nl(outputs if True else outputs, name='output')
	return ret
