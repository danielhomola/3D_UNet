"""
This module builds a 3D U-Net network.
Original paper: https://arxiv.org/abs/1606.06650
"""

import logging
import tensorflow as tf
log = logging.getLogger('tensorflow')


def unet_3d(inputs, num_classes=3, depth=4, n_base_filters=16, training=True):
    """
    Simple implementation of 3D U-Net building function.
    Original paper: https://arxiv.org/abs/1606.06650

    Inspiration was taken from several repos:
    TF implementation: https://github.com/zhengyang-wang/3D-Unet--Tensorflow
    Keras implementation: https://github.com/ellisdg/3DUnetCNN

    Args:
        inputs (:class:`tf.Tensor`): 5D tensor input to the network.
        num_classes (int): Number of mutually exclusive output classes.
        depth (int): Depth of the architecture.
        n_base_filters (int): Number of conv3d filters in the first layer.
        training (bool): Whether we are training or not, important for the
            batch normalisation layer. At inference time we need the population
            mean and variance instead of the batch one.

    Returns:
        :class:`tf.Tensor`: 3D U-Net network.
    """
    current_layer = inputs
    levels = list()
    concat_layer_sizes = list()

    # -------------------------------------------------------------------------
    # ANALYSIS PATH
    # -------------------------------------------------------------------------

    for layer_depth in range(depth):
        # two conv > batch norm > relu blocks
        n_filters = n_base_filters * (2 ** layer_depth)
        layer1 = conv3d_bn_relu(
            inputs=current_layer,
            part='analysis',
            layer_depth=layer_depth,
            filters=n_filters,
            training=training
        )
        log.debug('conv1: %s' % layer1.shape)

        layer2 = conv3d_bn_relu(
            inputs=layer1,
            part='analysis2',
            layer_depth=layer_depth,
            filters=n_filters * 2,
            training=training
        )
        concat_layer_sizes.append(n_filters * 2)
        log.debug('conv2: %s' % layer2.shape)

        # add max pooling unless we're at the end of the bottle-neck
        if layer_depth < depth - 1:
            current_layer = tf.layers.max_pooling3d(
                inputs=layer2,
                pool_size=(2, 2, 2),
                strides=(2, 2, 2),
                padding='valid',
                name=create_name('analysis', 'maxpool', layer_depth)
            )
            log.debug('maxpool layer: %s' % current_layer.shape)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # -------------------------------------------------------------------------
    # SYNTHESIS PATH
    # -------------------------------------------------------------------------

    for layer_depth in range(depth - 2, -1, -1):
        # add up-conv
        n_filters = n_base_filters * (2 ** layer_depth) * 2
        up_conv = tf.layers.conv3d_transpose(
            inputs=current_layer,
            filters=n_filters * 2,
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='same',
            # see https://github.com/tensorflow/tensorflow/issues/10520
            use_bias=False,
            name=create_name('synthesis', 'upconv', layer_depth)
        )
        log.debug('upconv layer: %s' % up_conv.shape)
        log.debug('concat layer: %s' % levels[layer_depth][1].shape)
        # concat with appropriate layer of analysis path
        concat = tf.concat(
            [up_conv, levels[layer_depth][1]],
            axis=-1,
            name=create_name('synthesis', 'concat', layer_depth)
        )

        # two conv > batch norm > relu blocks
        current_layer = conv3d_bn_relu(
            inputs=concat,
            part='synthesis',
            layer_depth=layer_depth,
            filters=concat_layer_sizes[layer_depth],
            training=training
        )
        log.debug('up_conv layer1: %s' % current_layer.shape)
        current_layer = conv3d_bn_relu(
            inputs=current_layer,
            part='synthesis2',
            layer_depth=layer_depth,
            filters=concat_layer_sizes[layer_depth],
            training=training
        )
        log.debug('up_conv layer2 : %s' % current_layer.shape)

    # add final conv layer
    output_layer = tf.layers.conv3d(
        inputs=current_layer,
        filters=num_classes,
        kernel_size=(1, 1, 1),
        strides=(1, 1, 1),
        use_bias=True,
        name=create_name('synthesis', 'final_conv3d', 0)
    )
    log.debug('output layer:: %s' % output_layer.shape)
    return output_layer


def conv3d_bn_relu(inputs, filters, part, layer_depth,
                   kernel=(3, 3, 3), strides=(1, 1, 1),
                   padding='same', training=True):
    """
    Basic conv3d > batch normalisation > relu building block for the network.

    Args:
        inputs (:class:`tf.Tensor`): 5D tensor input to the block.
        filters (tuple): See conv3D TF docs.
        part (str): Needed for name generation.
        layer_depth (int): Needed for name generation.
        kernel (tuple): See conv3D TF docs.
        strides (tuple): See conv3D TF docs.
        padding (str): See conv3D TF docs.
        training (bool): Whether we are training or not, important for the
            batch normalisation layer. At inference time we need the population
            mean and variance instead of the batch one.

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
        name=create_name(part, 'conv3d', layer_depth)
    )
    layer = tf.layers.batch_normalization(
        inputs=layer,
        training=training,
        axis=-1,
        fused=True,
        name=create_name(part, 'batch_norm', layer_depth)
    )
    layer = tf.nn.relu(
        layer,
        name=create_name(part, 'relu', layer_depth)
    )
    return layer


def create_name(part, layer, i):
    """
    Helper function for generating name strings for layers.

    Args:
        part (str): Part/path the layer belongs to.
        layer (str): The function of the layer, .e.g conv3d.
        i (int): The layer depth of the layer.

    Returns:
        str: Concatenated layer name.
    """
    return "%s_%s_%d" % (part, layer, i)
