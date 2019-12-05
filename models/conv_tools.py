import cv2
import tensorflow as tf


def conv(input_tensor, output_channel, name, activation="relu"):
    input_channel = input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if activation == "relu":
            kernel = tf.Variable(tf.truncated_normal([3, 3, input_channel, output_channel], stddev=0.1), name="widgets")
            bias = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name="bias")
            conv_t = tf.nn.conv2d(input_tensor, kernel, padding="VALID")
            out = tf.nn.relu(conv_t + bias, name=scope + "out")
            return out


def max_pool(input_tensor, name):
    pool = tf.nn.max_pool(input_tensor, [1, 2, 2, 1], [1, 2, 2, 1], name=name, padding="VALID")
    return pool


def conv_and_max_pool(input_tensor, output_channel, name, activation="relu"):
    conv_out = conv(input_tensor, output_channel, name, activation)
    pool_out = max_pool(conv_out, name)
    return pool_out


def full_connection(input_tensor, output_channel, name, activation="relu"):
    if len(input_tensor.get_shape()) > 3:
        input_chnnel = input_tensor.get_shape()[-1].value
        feature_size_x = input_tensor.get_shape()[-3].value
        feature_size_y = input_tensor.get_shape()[-2].value
        flat_size = feature_size_x * feature_size_y * input_chnnel
        flatten = tf.reshape(input_tensor, [-1, flat_size])
    else:
        flatten = input_tensor
        flat_size = input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if activation == "relu":
            weight = tf.Variable(tf.truncated_normal([flat_size, output_channel], stddev=0.1), name=scope + "weight")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[output_channel]), trainable=True,
                               name=scope + "bias")
        out = tf.nn.relu_layer(flatten, weight, bias, name=name)
        return out


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img

