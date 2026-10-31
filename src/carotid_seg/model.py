"""Lightweight attention segmentation network for carotid-artery ultrasound.

Ported from ``notebooks/carotid-artery-segmentation-attention-u-net.ipynb`` (function
``build_caroid_seg_model``) and checked against its ``model.summary()`` (171,653 parameters).

A depthwise-separable encoder (16, 32, 64 filters) feeds an ASPP-like multi-scale context
module. The decoder uses attention gates on the skip connections, squeeze-and-excitation and
coordinate attention. A boundary-refinement branch predicts an edge map from the highest-
resolution features and feeds it back into the decoder through a final attention gate. The
model has two named outputs, ``segmentation`` (the mask) and ``edge_pred`` (the edge map).
"""

from __future__ import annotations

import keras
from keras import Model, layers, ops

ENCODER_FILTERS = (16, 32, 64)


@keras.saving.register_keras_serializable(package="carotid_seg")
class CoordAtt(layers.Layer):
    """Coordinate attention: direction-aware channel attention along height and width."""

    def __init__(self, filters: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction
        self.conv1 = layers.Conv2D(
            max(filters // reduction, 1), 1, activation="relu", padding="same"
        )
        self.conv2 = layers.Conv2D(filters, 1, activation="sigmoid", padding="same")

    def build(self, input_shape):
        batch, _, _, channels = input_shape
        # The two pooled descriptors are concatenated along the height axis: (B, H + W, 1, C).
        self.conv1.build((batch, None, 1, channels))
        self.conv2.build((batch, None, 1, max(channels // self.reduction, 1)))
        super().build(input_shape)

    def call(self, x):
        height = x.shape[1]
        pooled_h = ops.mean(x, axis=2, keepdims=True)  # (B, H, 1, C)
        pooled_w = ops.transpose(ops.mean(x, axis=1, keepdims=True), (0, 2, 1, 3))  # (B, W, 1, C)
        attention = self.conv2(self.conv1(ops.concatenate([pooled_h, pooled_w], axis=1)))
        attn_h, attn_w = ops.split(attention, [height], axis=1)
        attn_w = ops.transpose(attn_w, (0, 2, 1, 3))  # back to (B, 1, W, C)
        return x * attn_h * attn_w

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "filters": self.filters, "reduction": self.reduction}


def dw_sep_conv(x, filters, kernel_size=3, strides=1, activation="relu", name=None):
    """Depthwise conv, BN, pointwise conv, BN, activation."""
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation(activation, name=name)(x)


def efficient_encoder(inputs, filter_list=ENCODER_FILTERS):
    """Returns the feature map of every encoder stage (full, 1/2 and 1/4 resolution)."""
    x, features = inputs, []
    for i, filters in enumerate(filter_list):
        x = dw_sep_conv(x, filters, name=f"encoder_{i + 1}")
        features.append(x)
        if i < len(filter_list) - 1:
            x = layers.MaxPool2D(2)(x)
    return features


def multi_scale_context(x, filters):
    """ASPP-like module: dilation rates 1, 2, 4, 8 plus global pooling."""
    d1 = dw_sep_conv(x, filters)
    branches = [d1]
    for rate, name in ((2, "dilated1"), (4, "dilated2"), (8, "dilated3")):
        d = dw_sep_conv(x, filters, name=name)
        branches.append(layers.Conv2D(filters, 3, dilation_rate=rate, padding="same")(d))
    pooled = layers.GlobalAveragePooling2D()(x)
    pooled = layers.Reshape((1, 1, filters))(pooled)
    pooled = layers.Conv2D(filters, 1, activation="relu", padding="same")(pooled)
    pooled = layers.UpSampling2D(size=x.shape[1:3], interpolation="bilinear")(pooled)
    branches.append(pooled)
    merged = layers.Concatenate()(branches)
    return layers.Conv2D(filters, 1, activation="relu", padding="same")(merged)


def se_block(x, ratio=8):
    """Squeeze-and-excitation channel re-weighting."""
    filters = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(filters // ratio, activation="relu")(s)
    s = layers.Dense(filters, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, filters))(s)
    return layers.Multiply()([x, s])


def boundary_refinement(x):
    """Predict a one-channel edge map from high-resolution features."""
    edge = layers.DepthwiseConv2D(7, padding="same")(x)
    edge = layers.BatchNormalization()(edge)
    edge = layers.Activation("relu")(edge)
    return layers.Conv2D(1, 1, activation="sigmoid", padding="same", name="edge_pred")(edge)


def attention_gate(x, g, filters):
    """Additive attention gate: re-weights ``x`` using the gating signal ``g``."""
    theta_x = layers.Conv2D(filters, 1, padding="same")(x)
    phi_g = layers.Conv2D(filters, 1, padding="same")(g)
    activated = layers.Activation("relu")(layers.Add()([theta_x, phi_g]))
    psi = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(activated)
    return layers.Multiply()([x, psi])


def decoder_block(x, skip, filters):
    """Upsample, gate the skip connection, concatenate, then conv + SE + coordinate attention."""
    x = layers.UpSampling2D(2)(x)
    skip = attention_gate(skip, x, filters)
    x = layers.Concatenate()([x, skip])
    x = dw_sep_conv(x, filters)
    x = se_block(x)
    return CoordAtt(filters)(x)


def segmentation_model(model: Model) -> Model:
    """A single-output view of ``model`` that returns only the segmentation mask.

    It shares every layer and weight with ``model``. The edge branch stays in the graph because
    it feeds the decoder through the final attention gate; it just has no output of its own.
    """
    return Model(model.inputs, model.output["segmentation"], name="carotid_seg_mask")


def build_model(input_shape: tuple[int, int, int] = (256, 256, 3)) -> Model:
    """Build the network. Outputs: a dict ``{"segmentation", "edge_pred"}`` of ``(H, W, 1)`` maps.

    The input height and width must be divisible by 4.
    """
    inputs = layers.Input(input_shape)
    feats = efficient_encoder(inputs)
    bridge = multi_scale_context(feats[-1], ENCODER_FILTERS[-1])
    edge_pred = boundary_refinement(feats[0])

    d = decoder_block(bridge, feats[1], ENCODER_FILTERS[1])
    d = decoder_block(d, feats[0], ENCODER_FILTERS[0])

    edge_attn = attention_gate(d, edge_pred, ENCODER_FILTERS[0])
    d = layers.Concatenate()([d, edge_attn])
    segmentation = layers.Conv2D(1, 1, activation="sigmoid", padding="same", name="segmentation")(d)
    return Model(inputs, {"segmentation": segmentation, "edge_pred": edge_pred}, name="carotid_seg")
