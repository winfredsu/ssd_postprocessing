#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import PIL

INFERENCE_GRAPH = 'oiltank_mnetv2_0.5_ssd.pb'
TEST_IMAGE = './test.jpg'
IMAGE_SIZE = [270,480]
ENDPOINTS = []

# load graph
f = open(INFERENCE_GRAPH, 'rb')
gd = tf.GraphDef.FromString(f.read())
tf.import_graph_def(gd, name='')

# prepare input image ([-1,1), NHWC)
test_image = np.array(PIL.Image.open(TEST_IMAGE).resize(IMAGE_SIZE)).astype(np.float)/128-1
test_image = test_image.reshape(1,IMAGE_SIZE[0],IMAGE_SIZE[1],3)

# eval the endpoints
g = tf.get_default_graph()
with tf.Session() as sess:
    return sess.run(tensors, feed_dict={g.get_tensor_by_name(self.input_tensor_name): self.test_img})

