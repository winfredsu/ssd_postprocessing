# How Anchor Boxes Are Generated
See `create_ssd_anchors` in [Multiple_grid Anchor Generator in TensorFlow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/anchor_generators/multiple_grid_anchor_generator.py)

# Tensor Format
## Class predictors
- Before reshape & concat: `H x W x Anchors x (Classes+1)`
- After reshape & concat: `[H0xW0xAnchors0 ... H5xW5xAnchors5] x (Classes+1)`
## Box predictors
- Before reshape & concat: `H x W x Anchors x 4 (y,x,h,w)`
- After reshape & concat: `[H0xW0xAnchors0 ... H5xW5xAnchors5] x 4`
