import tensorflow as tf


def create_conv_layer(input_data, num_channels, num_filters, name, filter_shape=[3, 3], pool_shape=[2, 2]):
    conv_filt_shape = [
        filter_shape[0],
        filter_shape[1],
        num_channels,
        num_filters
    ]

    # setup weights and biases
    weights = tf.Variable(tf.truncated_normal(
        conv_filt_shape,
        stddev=0.03,
        name=name + '_weights'
    ))

    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_biases')

    # setup the convolutional layer operation
    # TODO: may be we can do 0 padding?
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)

    # perform max_pooling
    kside = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, kside=kside, strides=strides, padding='SAME')
    return out_layer


def unpool_2d(pool,
              ind,
              stride=[1, 2, 2, 1],
              scope='unpool_2d'):
    """Adds a 2D unpooling op.
    https://arxiv.org/abs/1505.04366
    Unpooling layer after max_pool_with_argmax.
         Args:
             pool:        max pooled output tensor
             ind:         argmax indices
             stride:      stride is the same as for the pool
         Return:
             unpool:    unpooling tensor
    https://github.com/rayanelleuch/tensorflow/blob/b46d50583d8f4893f1b1d629d0ac9cb2cff580af/tensorflow/contrib/layers/python/layers/layers.py#L2291-L2327
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret


def create_upconv_layer(input_data, num_channels, num_filters, name, filter_shape=[3, 3], pool_shape=[2, 2]):
    apply_unpool = unpool_2d(input_data, 0)
    return create_conv_layer(apply_unpool, num_channels, num_filters, name, filter_shape, pool_shape)

input_data = tf.placeholder(tf.float32, shape=(None, 384, 512, 6))
layer_1 = create_conv_layer(input_data, 6, 64, 'conv1', filter_shape=[7, 7])
layer_2 = create_conv_layer(layer_1, 64, 128, 'conv2', filter_shape=[5, 5])
layer_3 = create_conv_layer(layer_2, 128, 256, 'conv3', filter_shape=[5, 5])
layer_4 = create_conv_layer(layer_3, 256, 256, 'conv4')
layer_4_1 = create_conv_layer(layer_4, 256, 512, 'conv_4_1')
layer_5 = create_conv_layer(layer_4_1, 512, 512, 'conv_5')
layer_5_1 = create_conv_layer(layer_5, 512, 512, 'conv_5_1')
layer_6 = create_conv_layer(layer_5_1, 512, 1024, 'conv_6')
layer_refine_1 = create_upconv_layer(layer_6, 1024, 512, 'deconv_5', filter_shape=[5, 5])
layer_refine_2 = create_upconv_layer(layer_refine_1, 512, 256, 'deconv_4', filter_shape=[5, 5])
layer_refine_3 = create_upconv_layer(layer_refine_2, 256, 128, 'deconv_3', filter_shape=[5, 5])
layer_refine_4 = create_upconv_layer(layer_refine_3, 128, 64, 'deconv_2', filter_shape=[5, 5])
output = create_upconv_layer(layer_refine_4, 64, 1, 'prediction', filter_shape=[5, 5])

