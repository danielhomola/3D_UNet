"""
Simple implementation of 3D U-Net building function.
Original paper: https://arxiv.org/abs/1606.06650
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

log = logging.getLogger('tensorflow')


def unet_3d_network(inputs, params, training):
    """
    Simple implementation of 3D U-Net building function.
    Original paper: https://arxiv.org/abs/1606.06650

    Inspiration was taken from the following repos:
    TF implementation: https://github.com/zhengyang-wang/3D-Unet--Tensorflow
    Keras implementation: https://github.com/ellisdg/3DUnetCNN

    If logging is set to info, it will print out the size of each layer.

    Args:
        inputs (:class:`tf.Tensor`): 5D tensor input to the network.
        params (dict): Params for setting up the model. Expected keys are:
            depth (int): Depth of the architecture.
            n_base_filters (int): Number of conv3d filters in the first layer.
            num_classes (int): Number of mutually exclusive output classes.
            batch_norm (bool): Whether to use batch_norm in the conv3d blocks.
        training (bool): Whether we are training or not, important for BN.

    Returns:
        :class:`tf.Tensor`: Final logits layer of the network.
    """

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------

    # extract model params
    depth = params['depth']
    n_base_filters = params['n_base_filters']
    num_classes = params['num_classes']
    batch_norm = params['batch_norm']

    # additional model params that are currently baked into the model_fn
    conv_size = 3
    conv_strides = 1
    pooling_size = 2
    pooling_strides = 2
    padding = 'same'

    # variables to track architecture building with loops
    current_layer = inputs
    levels = list()
    concat_layer_sizes = list()

    # -------------------------------------------------------------------------
    # network architecture: analysis path
    # -------------------------------------------------------------------------

    for layer_depth in range(depth):
        # two conv3d > batch norm > relu blocks
        conv1_filters = n_base_filters * (2 ** layer_depth)
        conv2_filters = conv1_filters * 2
        layer1 = conv3d_bn_relu(
            inputs=current_layer,
            filters=conv1_filters,
            kernel=conv_size,
            strides=conv_strides,
            padding=padding,
            batch_norm=batch_norm,
            training=training,
            part='analysis',
            layer_depth=layer_depth,
        )
        log.info('conv1: %s' % layer1.shape)

        layer2 = conv3d_bn_relu(
            inputs=layer1,
            filters=conv2_filters,
            kernel=conv_size,
            strides=conv_strides,
            padding=padding,
            batch_norm=batch_norm,
            training=training,
            part='analysis',
            layer_depth=layer_depth,
        )
        concat_layer_sizes.append(conv2_filters)
        log.info('conv2: %s' % layer2.shape)

        # add max pooling unless we're at the end of the bottleneck
        if layer_depth < depth - 1:
            current_layer = tf.layers.max_pooling3d(
                inputs=layer2,
                pool_size=pooling_size,
                strides=pooling_strides,
                padding=padding,
                name=create_name('analysis', 'maxpool', layer_depth)
            )
            log.info('maxpool layer: %s' % current_layer.shape)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # -------------------------------------------------------------------------
    # network architecture: synthesis path
    # -------------------------------------------------------------------------

    for layer_depth in range(depth - 2, -1, -1):
        # add up-conv
        n_filters = n_base_filters * (2 ** layer_depth) * 2
        up_conv = tf.layers.conv3d_transpose(
            inputs=current_layer,
            filters=n_filters * 2,
            kernel_size=pooling_size,
            strides=pooling_strides,
            padding=padding,
            # see https://github.com/tensorflow/tensorflow/issues/10520
            use_bias=False,
            name=create_name('synthesis', 'upconv', layer_depth)
        )
        log.info('upconv layer: %s' % up_conv.shape)
        log.info('concat layer: %s' % levels[layer_depth][1].shape)
        # concat with appropriate layer of analysis path
        concat = tf.concat(
            [up_conv, levels[layer_depth][1]],
            axis=-1,
            name=create_name('synthesis', 'concat', layer_depth)
        )

        # two conv3d > batch norm > relu blocks
        current_layer = conv3d_bn_relu(
            inputs=concat,
            filters=concat_layer_sizes[layer_depth],
            kernel=conv_size,
            strides=conv_strides,
            padding=padding,
            batch_norm=batch_norm,
            training=training,
            part='synthesis',
            layer_depth=layer_depth
        )

        log.info('up_conv layer1: %s' % current_layer.shape)
        current_layer = conv3d_bn_relu(
            inputs=current_layer,
            filters=concat_layer_sizes[layer_depth],
            kernel=conv_size,
            strides=conv_strides,
            padding=padding,
            batch_norm=batch_norm,
            training=training,
            part='synthesis2',
            layer_depth=layer_depth
        )
        log.info('up_conv layer2 : %s' % current_layer.shape)

    # final 1 x 1 x 1 conv3d layer
    logits = tf.layers.conv3d(
        inputs=current_layer,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        use_bias=True,
        name=create_name('synthesis', 'final_conv3d', 0)
    )
    log.debug('output layer:: %s' % logits.shape)
    return logits


def conv3d_bn_relu(inputs, filters, kernel, strides, padding,
                   batch_norm, training, part, layer_depth):
    """
    Basic conv3d > Batch Normalisation > Relu building block for the network.

    Args:
        inputs (:class:`tf.Tensor`): 5D tensor input to the block.
        filters (int): See conv3D TF docs.
        kernel (int): See conv3D TF docs.
        strides (int): See conv3D TF docs.
        padding (str): See conv3D TF docs.
        batch_norm (bool): Whether to use batch_norm in the conv3d blocks.
        training (bool): Whether we are training or not, important for the
            batch normalisation layer. At inference time we need the
            population mean and variance instead of the batch one.
        part (str): Needed for name generation.
        layer_depth (int): Needed for name generation.


    Returns:
        :class:`tf.Tensor`: Relu(BatchNorm(Conv3D))
    """
    layer = tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=create_name(part, 'conv3d_%d' % filters, layer_depth)
    )
    if batch_norm:
        layer = tf.layers.batch_normalization(
            inputs=layer,
            training=training,
            axis=-1,
            fused=True,
            name=create_name(part, 'batch_norm_%d' % filters, layer_depth)
        )
    layer = tf.nn.relu(
        layer,
        name=create_name(part, 'relu_%d' % filters, layer_depth)
    )
    return layer


def create_name(part, layer, i):
    """
    Helper function for generating names for layers.

    Args:
        part (str): Part/path the layer belongs to.
        layer (str): The function of the layer, .e.g conv3d.
        i (int): The layer depth of the layer.

    Returns:
        str: Concatenated layer name.
    """
    return "%s_%s_l%d" % (part, layer, i)
