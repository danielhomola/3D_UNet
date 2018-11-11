import tensorflow as tf


def unet_3d(inputs, num_classes=3, depth=4, n_base_filters=16, training=True):
    """

    Args:
        inputs:
        num_classes:
        depth:
        n_base_filters:
        training:

    Returns:

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
        print('conv1:', layer1.shape)
        layer2 = conv3d_bn_relu(
            inputs=layer1,
            part='analysis2',
            layer_depth=layer_depth,
            filters=n_filters * 2,
            training=training
        )
        print('conv2:', layer2.shape)
        concat_layer_sizes.append(n_filters * 2)

        # add max pooling unless we're at the end of the bottle-neck
        if layer_depth < depth - 1:
            current_layer = tf.layers.max_pooling3d(
                inputs=layer2,
                pool_size=(2, 2, 2),
                strides=(2, 2, 2),
                padding='valid',
                name=create_name('analysis', 'maxpool', layer_depth)
            )
            print('maxpool layer:', current_layer.shape)
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
            name=create_name('synthesis', 'upconv', layer_depth)
        )
        print('upconv layer:', up_conv.shape)
        print('concat layer:', levels[layer_depth][1].shape)
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
        print('up_conv layer1:', current_layer.shape)
        current_layer = conv3d_bn_relu(
            inputs=current_layer,
            part='synthesis2',
            layer_depth=layer_depth,
            filters=concat_layer_sizes[layer_depth],
            training=training
        )
        print('up_conv layer2:', current_layer.shape)

    # add final conv layer
    output_layer = tf.layers.conv3d(
        inputs=current_layer,
        filters=num_classes,
        kernel_size=(1, 1, 1),
        strides=(1, 1, 1),
        use_bias=True,
        name=create_name('synthesis', 'final_conv3', 0)
    )
    print('output layer:', output_layer.shape)
    return output_layer


def conv3d_bn_relu(inputs, filters, part, layer_depth,
                   kernel=(3, 3, 3), strides=(1, 1, 1),
                   padding='same', training=True):
    """

    Args:
        inputs:
        filters:
        part:
        layer_depth:
        kernel:
        strides:
        padding:
        training:

    Returns:

    """
    layer = tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        name=create_name(part, 'conv3d', layer_depth)
    )
    layer = tf.layers.batch_normalization(
        inputs=layer,
        training=training,
        name=create_name(part, 'batch_norm', layer_depth)
    )
    layer = tf.nn.relu(
        layer,
        name=create_name(part, 'relu', layer_depth)
    )
    return layer


def create_name(part, layer, i):
    """

    Args:
        part:
        layer:
        i:

    Returns:

    """
    return "%s_%s_%d" % (part, layer, i)
