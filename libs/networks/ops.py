import tensorflow as tf
import tensorflow.contrib.slim as slim


def norm(x, norm_type, is_train, G=32, esp=1e-5):
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.999,
                is_training=is_train, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()

            G = min(G, C)
            x = tf.reshape(x, [-1, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta
            gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output


def norm_(x, norm_type, is_train, G=32, esp=1e-5):
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.999,
                is_training=is_train, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()

            G = min(G, C)
            x_gn = []
            for i in range(C // G):
                x_split = x[:, i*32:(i+1)*32, :, :]
                mean, var = tf.nn.moments(x_split, [1, 2, 3], keep_dims=True)
                x_split = (x_split - mean) / tf.sqrt(var + esp)
                x_gn.append(x_split)
            x_gn = tf.concat(x_gn, axis=1)

            # per channel gamma and beta
            gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = x_gn * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

