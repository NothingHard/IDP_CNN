from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops.gen_nn_ops import *
import os
import numpy as np
import tensorflow as tf

def my_profile(C, prof_type):
    def half_exp(n, k=1, dtype='float32'):
        n_ones = int(n/2)
        n_other = n - n_ones
        return np.append(np.ones(n_ones, dtype=dtype), np.exp((1-k)*np.arange(n_other), dtype=dtype))
    if prof_type == "linear":
        profile = np.linspace(1.0,0.0, num=C, endpoint=False, dtype='float32')
    elif prof_type == "all-one":
        profile = np.ones(C, dtype='float32')
    elif prof_type == "half-exp":
        profile = half_exp(C, 2.0)
    elif prof_type == "harmonic":
        profile = np.array(1.0/(np.arange(C)+1))
    else:
        raise ValueError("prof_type must be \"all-one\", \"half-exp\", \"harmonic\" or \"linear\".")
    return profile

def idp_conv2d(input, filter, strides, padding,
               use_cudnn_on_gpu=True, data_format='NHWC',
               name=None, prof_type=None, dp=1.0):
    with ops.name_scope(name, "idp_convolution", [input, filter]) as scope:
        if not (data_format == "NHWC" or data_format == "NCHW"):
            raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
        
        # do conv2d
        conv2d_res = gen_nn_ops.conv2d(input, filter, strides, padding,
                                       use_cudnn_on_gpu=True,
                                       data_format='NHWC',
                                       name=None)
        B,H,W,C = conv2d_res.get_shape().as_list()
        
        # get profile
        profile = my_profile(C, prof_type)
        # tensor_profile = tf.get_variable(initializer=profile,name="tensor_profile",dtype='float32')

        # create a mask determined by the dot product percentage
        n1 = int(C * dp)
        n0 = C - n1
        mask = np.append(np.ones(n1, dtype='float32'), np.zeros(n0, dtype='float32'))
        if len(profile) == len(mask):
            profile *= mask
        else:
            raise ValueError("profile and mask must have the same shape.")

        # create a profile coefficient, gamma
        conv2d_profile = np.stack([profile for i in range(B*H*W)])
        conv2d_profile = np.reshape(conv2d_profile, newshape=(B, H, W, C))
        
        gamma = tf.get_variable(initializer=conv2d_profile, name="gamma"+str(dp*100))
        
        # IDP conv2d output
        idp_conv2d_res = tf.multiply(conv2d_res, gamma, name="idp"+str(dp*100))
        
        return idp_conv2d_res


VGG_MEAN = [123.68, 116.779, 103.939] # [R, G, B]

class vgg16:
    def __init__(self, vgg16_npy_path=None):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16_weights.npz")
            vgg16_npy_path = path
            print(path)
        
        self.data_dict = np.load(vgg16_npy_path).item()
        print("npy file loaded")

        # cnn layer
        self.cnn_layer = []

        # loss
        self.loss_list = []

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # normalization
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        rgb = tf.concat(axis=3, values=[
            red - VGG_MEAN[0],
            green - VGG_MEAN[1],
            blue - VGG_MEAN[2],
        ])
        assert rgb.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(rgb, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_filter(self, name):
        self.cnn_layer = self.cnn_layer.append(name)
        return tf.get_variable(initializer=self.data_dict[name+"_W"], name="conv_W")
    def get_bias(self, name):
        return tf.get_variable(initializer=self.data_dict[name+"_b"], name="conv_b")
    def get_fc_weight(self, name):
        return tf.get_variable(initializer=self.data_dict[name+"_W"], name="fc_W")

    def aggregate_loss(self,idp):
        for layer in self.cnn_layer:
            with tf.variable_scope(layer,reuse=True):
                conv_W = tf.get_variable(name="conv_W")
                conv_b = tf.get_variable(name="conv_b")
                