import tensorflow as tf

from models.conv_tools import conv_and_max_pool, conv, full_connection


class LandmarkModel(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, input_tensor):

        layer1 = conv_and_max_pool(input_tensor, 32, "Layer_1", activation="relu")
        layer2 = conv(layer1, 64, "Layer_2", activation="relu")
        layer3 = conv_and_max_pool(layer2, 64, "Layer_3", activation="relu")
        layer4 = conv(layer3, 64, "Layer_4", activation="relu")
        layer5 = conv_and_max_pool(layer4, 64, "Layer_5", activation="relu")
        layer6 = conv(layer5, 128, "Layer_6", activation="relu")
        layer7 = conv_and_max_pool(layer6, 128, "Layer_7", activation="relu")
        layer8 = conv(layer7, 256, "Layer_8", activation="relu")
        layer9 = full_connection(layer8, 1024, 'fc1')
        logits = full_connection(layer9, self.output_size, 'fc2')

        return logits


if __name__ == '__main__':
    input_tensor = tf.Variable(tf.truncated_normal([1, 227, 227, 3], stddev=0.1), dtype=tf.float32)
    model = LandmarkModel(136)
    a = model(input_tensor)
