#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import PIL
import PIL.ImageDraw as ImageDraw
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable GPU info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do ssd postprocessing')
    parser.add_argument('-i', '--input_path', default='.', help='input image')
    parser.add_argument('-o', '--output_path', default='.', help='output image')
    args = parser.parse_args()

##################################################################
# GENERAL DEFINES
INFERENCE_GRAPH = 'oiltank_mnetv2_0.5_ssd.pb'
INPUT_IMAGE = args.input_path
OUTPUT_IMAGE = args.output_path
IMAGE_SIZE = [270,480] # [H,W]
INPUT_TENSOR_NAME = 'normalized_input_image_tensor:0'

# SSD
NUM_SSD_LAYERS = 6
NUM_CLASSES = 1   # excludes background
MIN_SCALE = 0.2
MAX_SCALE = 0.95
ASPECT_RATIOS = (1.0, 2.0, 3.0, 1.0/2, 1.0/3)
BOX_CODER_SCALE = [10.0, 10.0, 5.0, 5.0]  # [y_scale, x_scale, h_scale, w_scale]
ENDPOINT_CLASS = ['BoxPredictor_'+str(i)+'/ClassPredictor/act_quant/FakeQuantWithMinMaxVars:0' for i in range(NUM_SSD_LAYERS)]
ENDPOINT_BOX   = ['BoxPredictor_'+str(i)+'/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars:0' for i in range(NUM_SSD_LAYERS)]

# CHANGE THESE PARAMS FOR BETTER DEMONSTRATION
SCORE_THRESHOLD = -0.1
MAX_NMS_CANDIDATES = 50
NMS_THRESHOLD = 0.6
##################################################################

def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                       base_anchor_size=None,
                       anchor_strides=None,
                       anchor_offsets=None,
                       reduce_boxes_in_lowest_layer=True):
  """Creates MultipleGridAnchorGenerator for SSD anchors.

  This function instantiates a MultipleGridAnchorGenerator that reproduces
  ``default box`` construction proposed by Liu et al in the SSD paper.
  See Section 2.2 for details. Grid sizes are assumed to be passed in
  at generation time from finest resolution to coarsest resolution --- this is
  used to (linearly) interpolate scales of anchor boxes corresponding to the
  intermediate grid sizes.

  Anchors that are returned by calling the `generate` method on the returned
  MultipleGridAnchorGenerator object are always in normalized coordinates
  and clipped to the unit square: (i.e. all coordinates lie in [0, 1]x[0, 1]).

  Args:
    num_layers: integer number of grid layers to create anchors for (actual
      grid sizes passed in at generation time)
    min_scale: scale of anchors corresponding to finest resolution (float)
    max_scale: scale of anchors corresponding to coarsest resolution (float)
    scales: As list of anchor scales to use. When not None and not empty,
      min_scale and max_scale are not used.
    aspect_ratios: list or tuple of (float) aspect ratios to place on each
      grid point.
    interpolated_scale_aspect_ratio: An additional anchor is added with this
      aspect ratio and a scale interpolated between the scale for a layer
      and the scale for the next layer (1.0 for the last layer).
      This anchor is not included if this value is 0.
    base_anchor_size: base anchor size as [height, width].
      The height and width values are normalized to the minimum dimension of the
      input height and width, so that when the base anchor height equals the
      base anchor width, the resulting anchor is square even if the input image
      is not square.
    anchor_strides: list of pairs of strides in pixels (in y and x directions
      respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]
      means that we want the anchors corresponding to the first layer to be
      strided by 25 pixels and those in the second layer to be strided by 50
      pixels in both y and x directions. If anchor_strides=None, they are set to
      be the reciprocal of the corresponding feature map shapes.
    anchor_offsets: list of pairs of offsets in pixels (in y and x directions
      respectively). The offset specifies where we want the center of the
      (0, 0)-th anchor to lie for each layer. For example, setting
      anchor_offsets=[(10, 10), (20, 20)]) means that we want the
      (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space
      and likewise that we want the (0, 0)-th anchor of the second layer to lie
      at (25, 25) in pixel space. If anchor_offsets=None, then they are set to
      be half of the corresponding anchor stride.
    reduce_boxes_in_lowest_layer: a boolean to indicate whether the fixed 3
      boxes per location is used in the lowest layer.

  Returns:
    a MultipleGridAnchorGenerator
  """
  if base_anchor_size is None:
    base_anchor_size = [1.0, 1.0]
  box_specs_list = []
  if scales is None or not scales:
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]
  else:
    # Add 1.0 to the end, which will only be used in scale_next below and used
    # for computing an interpolated scale for the largest scale in the list.
    scales += [1.0]

  for layer, scale, scale_next in zip(
      range(num_layers), scales[:-1], scales[1:]):
    layer_box_specs = []
    if layer == 0 and reduce_boxes_in_lowest_layer:
      layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
    else:
      for aspect_ratio in aspect_ratios:
        layer_box_specs.append((scale, aspect_ratio))
      # Add one more anchor, with a scale between the current scale, and the
      # scale for the next layer, with a specified aspect ratio (1.0 by
      # default).
      if interpolated_scale_aspect_ratio > 0.0:
        layer_box_specs.append((np.sqrt(scale*scale_next),
                                interpolated_scale_aspect_ratio))
    box_specs_list.append(layer_box_specs)

  return box_specs_list

def draw_bbox(size, draw, box):
    ymin = box[0]
    xmin = box[1]
    ymax = box[2]
    xmax = box[3]

    im_width, im_height = size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=4, fill='blue')

def draw_bboxes(boxes):
    """
    input image: path to image
    input boxes: 2d list, [NUM_BOXES, 4]
    """
    image = PIL.Image.open(INPUT_IMAGE)
    draw = PIL.ImageDraw.Draw(image)

    for box in boxes:
        draw_bbox(image.size, draw, box)

    image.show()
    image.save(OUTPUT_IMAGE)

def decode(encoded_box, anchor_box):
    """
    input: encoded_box [ty,tx,th,tw]
    input: anchor_box [ycenter_a,xcenter_a,ha,wa]
    return: decoded_box [ymin,xmin,ymax,xmax]
    """
    ty = encoded_box[0]
    tx = encoded_box[1]
    th = encoded_box[2]
    tw = encoded_box[3]
    
    ycenter_a = anchor_box[0]
    xcenter_a = anchor_box[1]
    ha        = anchor_box[2]
    wa        = anchor_box[3]
    
    scale_factors = BOX_CODER_SCALE
    
    ty /= scale_factors[0]
    tx /= scale_factors[1]
    th /= scale_factors[2]
    tw /= scale_factors[3]
    
    w = np.exp(tw) * wa
    h = np.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = max(0., ycenter - h / 2.)
    xmin = max(0., xcenter - w / 2.)
    ymax = min(1., ycenter + h / 2.)
    xmax = min(1., xcenter + w / 2.)   
    
    return [ymin,xmin,ymax,xmax]

def get_anchor_box_from_preds(fmap_shape, pred_id, box_spec):
    """
    input: fmap_shape [H,W,ANCHORS]
    input: pred_id [y,x,boxid]
    input: box_spec, a list of (scale, ratio) for the current ssd layer
    return: anchor_box [ymin,xmin,ymax,xmax]
    """
    anchor_box_center = [(pred_id[i]+0.5)/fmap_shape[i] for i in range(2)]
    ycenter_a = anchor_box_center[0]
    xcenter_a = anchor_box_center[1]

    scale, ratio = box_spec[pred_id[2]]
    wa = scale*np.sqrt(ratio)
    ha = scale/np.sqrt(ratio)

    # handle the non-square input with square anchor boxes
    # IMAGE_SIZE = [H,W]
    # the anchor boxes in the training process are square and relative to min(H,W)
    # we need to re-scale the ha or wa here
    # so that the follow-up scripts can treat (ha,wa) as ratios relative to (H,W) 
    if IMAGE_SIZE[0]>IMAGE_SIZE[1]: # H>W
      ha *= (IMAGE_SIZE[1]/IMAGE_SIZE[0])
    else: # W>H
      wa *= (IMAGE_SIZE[0]/IMAGE_SIZE[1])

    return [ycenter_a, xcenter_a, ha, wa]

def get_anchor_box_from_candidate(id, box_specs, endpoint_shapes):
    """
    input: candidate id in range(sum(H*W*ANCHORS))
    input: box_specs, a list of (scale, ratio) for all ssd layers
    input: endpoint_shapes, a list of (H, W, ANCHORS) for all ssd layers
    return: anchor_box [ymin, xmin, ymax, xmax]
    """
    threshold = 0
    for layer_id in range(NUM_SSD_LAYERS):
        endpoint_shape = endpoint_shapes[layer_id]
        endpoint_shape_flatten = endpoint_shape[0]*endpoint_shape[1]*endpoint_shape[2] # H*W*ANCHORS
        threshold += endpoint_shape_flatten
        if id < threshold:
            id -= (threshold-endpoint_shape_flatten)
            break

    # now we have layer_id and the flattened pixel id in that layer
    # 1. get the feature map shape
    fmap_shape = endpoint_shapes[layer_id]
    # 2. get the pred_id in [y,x,boxid]
    pred_id = np.unravel_index(id, fmap_shape)
    # 3. get the box_spec of the current layer
    box_spec = box_specs[layer_id]
    
    anchor_box_center = [(pred_id[i]+0.5)/fmap_shape[i] for i in range(2)]
    ycenter_a = anchor_box_center[0]
    xcenter_a = anchor_box_center[1]

    scale, ratio = box_spec[pred_id[2]]
    wa = scale*np.sqrt(ratio)
    ha = scale/np.sqrt(ratio)

    # handle the non-square input with square anchor boxes
    # IMAGE_SIZE = [H,W]
    # the anchor boxes in the training process are square and relative to min(H,W)
    # we need to re-scale the ha or wa here
    # so that the follow-up scripts can treat (ha,wa) as ratios relative to (H,W) 
    if IMAGE_SIZE[0]>IMAGE_SIZE[1]: # H>W
      ha *= (IMAGE_SIZE[1]/IMAGE_SIZE[0])
    else: # W>H
      wa *= (IMAGE_SIZE[0]/IMAGE_SIZE[1])

    return [ycenter_a, xcenter_a, ha, wa]

def calc_iou(box1, box2):
    """
    calculate the iou of two boxes
    Args:
        box1, box2: [ymin,xmin,ymax,xmax]
    Return: 
        iou
    """
    # get the coordinates
    ymin1,xmin1,ymax1,xmax1 = box1
    ymin2,xmin2,ymax2,xmax2 = box2

    # calc the areas of two boxes
    area1 = (xmax1-xmin1)*(ymax1-ymin1)
    area2 = (xmax2-xmin2)*(ymax2-ymin2)

    # get the posible intersection boundries
    xx1 = max(xmin1, xmin2)
    yy1 = max(ymin1, ymin2)
    xx2 = min(xmax1, xmax2)
    yy2 = min(ymax1, ymax2)

    # calc the intersection area
    intersection_area = max(0., xx2-xx1) * max(0., yy2-yy1)

    return intersection_area / (area1+area2-intersection_area)+1e-6

def non_maximum_supression(candidates, threshold=NMS_THRESHOLD):
    """
    do NMS on the candidates
    Args:
        candidates: array of bboxes of a specific class sorted by score in descending order
        threshold: nms threshold
    Returns:
        results: final predictions
    """
    for idx,m in enumerate(candidates):
        for c in candidates[idx+1:]:
            if calc_iou(m,c) > NMS_THRESHOLD:
                candidates.remove(c)
    
    return candidates


# get box specs (scale, aspect_ratio)
# for SSDs with 6 layers, the number of anchor boxes are [3,6,6,6,6,6]
box_specs = create_ssd_anchors(num_layers=NUM_SSD_LAYERS, min_scale=MIN_SCALE, max_scale=MAX_SCALE)

# load graph
f = open(INFERENCE_GRAPH, 'rb')
gd = tf.GraphDef.FromString(f.read())
tf.import_graph_def(gd, name='')

# prepare input image ([-1,1), NHWC)
# NOTE: PIL.resize() has [W,H] as param, but the shape of the result is [H,W,C]
test_image = np.array([np.array(PIL.Image.open(INPUT_IMAGE).resize([480,270])).astype(np.float)/128-1])

# eval the endpoints
g = tf.get_default_graph()
with tf.Session() as sess:
    classes = sess.run(ENDPOINT_CLASS, feed_dict={INPUT_TENSOR_NAME: test_image})
    boxes   = sess.run(ENDPOINT_BOX  , feed_dict={INPUT_TENSOR_NAME: test_image})

# get the endpoint shapes
endpoint_shapes = [] 
for p in classes:
    endpoint_shapes.append(list(p.shape[1:3])+[int(p.shape[3]/(NUM_CLASSES+1))]) # (H,W,ANCHORS)

# reshape the output tensors
pred_class = np.zeros((0,NUM_CLASSES+1))
pred_box = np.zeros((0,4))
for pc,pb in zip(classes, boxes):
    pred_class = np.concatenate((pred_class, pc.reshape(-1, NUM_CLASSES+1)))
    pred_box = np.concatenate((pred_box, pb.reshape(-1, 4)))
# discard the backgrounds
pred_class = pred_class[:,1:]

#################################################
# FROM NOW ON WE FOCUS ON THE FIRST CLASS
pred_class = pred_class[:,0]

# argsort the pred_class (decend)
pred_class_argsort = pred_class.argsort()[::-1]
# find the indexes of largest scores that are greater than SCORE_THRESHOLD
pred_class_candidates = []
for i in range(MAX_NMS_CANDIDATES):
    if pred_class[pred_class_argsort[i]] < SCORE_THRESHOLD:
        break
    pred_class_candidates.append(pred_class_argsort[i])

# for each candidate, get its encoding and anchor box, and then decode
decoded_candidates = []
for c in pred_class_candidates:
    # get the encoding
    encoding = pred_box[c,:]
    # get the anchor box from c
    anchorbox = get_anchor_box_from_candidate(c, box_specs, endpoint_shapes)
    # decode the box
    decoded_candidates.append(decode(encoding, anchorbox))

# do NMS (the decoded_candidates is already sorted in descending order)
print('Number of candidates before NMS: ', len(decoded_candidates))
boxes_nms = non_maximum_supression(decoded_candidates)
print('Number of boxes after NMS: ', len(boxes_nms))

# draw the boxes
draw_bboxes(boxes_nms)



