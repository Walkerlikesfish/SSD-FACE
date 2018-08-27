from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/playground/tutorial/models-master/research/slim') # append the slim package path

import tensorflow as tf
from nets.mobilenet import mobilenet_v2


def inspect_module():
	features = tf.zeros([8,224,224,3], name='input')
	with tf.variable_scope('TestSSD', default_name=None, values=[features], reuse=tf.AUTO_REUSE):
		with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
		  logits, endpoints = mobilenet_v2.mobilenet(features)
		  for key in endpoints:
		  	print(key, endpoints[key])


class MobileNetv2SSD(object):
	def __init__(self, features, data_format='channels_first'):
		if data_format == 'channels_first':
			features = tf.transpose(features, [0, 2, 3, 1])
		with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
			self.logits, self.endpoints = mobilenet_v2.mobilenet(features)

	def get_feature_layers(self):
		feats = []
		feats.append(self.endpoints['layer_4/output'])
		feats.append(self.endpoints['layer_7/output'])
		feats.append(self.endpoints['layer_14/output'])
		feats.append(self.endpoints['layer_18/output'])
		return feats


def multibox_head(feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first'):
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers):
            loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
                        name='loc_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))
            cls_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, (3, 3), use_bias=True,
                        name='cls_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))

        return loc_preds, cls_preds


def ssd_conv_block(self, filters, strides, name, padding='same', reuse=None):
    with tf.variable_scope(name):
        conv_blocks = []
        conv_blocks.append(
                tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding,
                    data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                    kernel_initializer=self._conv_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None)
            )
        conv_blocks.append(
                tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding=padding,
                    data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                    kernel_initializer=self._conv_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None)
            )
        return conv_blocks

if __name__ == '__main__':
	inspect_module()