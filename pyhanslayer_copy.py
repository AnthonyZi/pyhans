import numpy as np

from variables import *


# general idea of a neuron:
#   inp[0]---{*w[x][0]}---
#                         \
#                          {+}---{+bias[x]}---"lay_out_pre_activation"---{act(x)}---"lay_out"
#                         /
#   inp[1]---{*w[x][1]}---


# HansLayer
# - HansLayer_1D
# - - Hanslayer_Dense

class HansLayer(object):
    funct_activation = None
    funct_activation_derivative = None

    weights_randrange = layer_weight_init_range
    bias_init = layer_bias_init_value

    input_vector_dimension = None
    output_vector_dimension = None

    input_dimensions = []
    output_dimensions = []

    def __init__(self, p_inp_dims, p_out_dims, p_inp_vec_dim, p_out_vec_dim, p_funct_activation, p_funct_activation_deriv, p_weights_min_init = None, p_weights_max_init = None, p_bias_init = None):

        if not isinstance(p_inp_dims, list):
            exit(100)
        if not isinstance(p_out_dims, list):
            exit(101)
        if not isinstance(p_inp_vec_dim, int):
            exit(102)
        if not isinstance(p_out_vec_dim, int):
            exit(103)

        self.input_dimensions = p_inp_dims
        self.output_dimensions = p_out_dims
        self.input_vector_dimension = p_inp_vec_dim
        self.output_vector_dimension = p_out_vec_dim

        self.funct_activation = p_funct_activation
        self.funct_activation_derivative = p_funct_activation_deriv

        if p_weights_min_init:
            self.weights_randrange = (p_weights_min_init, self.weights_randrange[1])
        if p_weights_max_init:
            self.weights_randrange = (self.weights_randrange[0], p_weights_max_init)
        if p_bias_init:
            self.bias_init = p_bias_init

class HansLayer_Dense(HansLayer):

    weights = None
    delta_weights = None
    biases = None
    delta_biases = None

    lay_input = None
    lay_out_pre_activation = None
    lay_out = None

    def __init__(self, p_inp_vec_dim, p_out_vec_dim, p_funct_activation, p_funct_activation_deriv, p_weights_min_init = None, p_weights_max_init = None, p_bias_init = None):
        HansLayer.__init__(self, [1], [1], p_inp_vec_dim, p_out_vec_dim, p_funct_activation, p_funct_activation_deriv, p_weights_min_init, p_weights_max_init, p_bias_init)

        self.weights = np.random.random((self.output_vector_dimension, self.input_vector_dimension))
        self.weights *= (self.weights_randrange[1] - self.weights_randrange[0])
        self.weights += self.weights_randrange[0]
        self.biases = np.full(self.output_vector_dimension, self.bias_init)

    def feedforward(self, p_input):
        self.lay_input = p_input
        self.lay_out_pre_activation = np.dot(self.weights, self.lay_input) + self.biases
        self.lay_out = self.funct_activation(self.lay_out_pre_activation)

        if verbosemax:
            print "feedforward: ", p_input
            print "weights: "
            print self.weights
            print "-> ", np.dot(self.weights, self.lay_input)
            print "biases: ", self.biases
            print "-> ", self.lay_out_pre_activation
            print "funct_activation(x)"
            print "-> ", self.lay_out

        return self.lay_out

    def backprop(self, p_input_right):
        tmp_backprop = self.funct_activation_derivative(self.lay_out_pre_activation, self.lay_out) * p_input_right

        self.delta_biases = tmp_backprop 
        self.delta_weights = np.dot(self.delta_biases.reshape(self.output_vector_dimension, 1), self.lay_input.reshape(1, self.input_vector_dimension))


        if verbosemax:
            print "p_input_right: ", p_input_right
            print "delta_biases: ", self.delta_biases
            print "delta_weights: "
            print self.delta_weights

        return np.dot(self.delta_biases, self.weights)

    def update_vanilla(self, p_learning_rate):
        self.weights -= self.delta_weights*p_learning_rate
        self.biases -= self.delta_biases*p_learning_rate

class HansLayer_Conv_2D(HansLayer):

    kernel_dimensions = None
    stride = None
    padding = None

    weights = None
    delta_weights = None
    biases = None
    delta_biases = None

    lay_input = None
    lay_out_pre_activation = None
    lay_out = None

    def __init__(self, p_inp_dims, p_inp_vec_dim, p_out_vec_dim, p_kernel_dims, p_stride, p_padding, p_funct_activation, p_funct_activation_deriv, p_weights_min_init = None, p_weights_max_init = None, p_bias_init = None):

        if p_padding == None:
            p_padding = [0,0,0]

        if not isinstance(p_kernel_dims, list):
            exit(104)
        if not isinstance(p_stride, list):
            exit(105)
        if not isinstance(p_padding, list):
            exit(106)

        if not len(p_inp_dims) == len(p_kernel_dims):
            exit(120)
        if not len(p_inp_dims) == len(p_stride):
            exit(121)
        if not len(p_inp_dims) == len(p_padding[:-1]):
            exit(122)

        variations = np.array(p_inp_dims) - np.array(p_kernel_dims) + (2*np.array(p_padding[:-1]))
        if any(variations%p_stride) != 0:
            print "warning, the convolution-layer will ignore some inputs"
        output_dimensions = list(variations/np.array(p_stride)+1)

        HansLayer.__init__(self, p_inp_dims, output_dimensions, p_inp_vec_dim, p_out_vec_dim, p_funct_activation, p_funct_activation_deriv, p_weights_min_init, p_weights_max_init, p_bias_init)

        self.kernel_dimensions = p_kernel_dims
        self.stride = p_stride
        self.padding = p_padding

        self.weights = np.random.random([self.output_vector_dimension] + self.kernel_dimensions + [self.input_vector_dimension])
        #self.weights = np.random.random([self.output_vector_dimension] + self.kernel_dimensions)
        self.biases = np.full([self.output_vector_dimension] + [self.input_vector_dimension], self.bias_init)

    def feedforward(self, p_input):
        pass

    def backprop(self, p_input_right):
        pass

    def update_vanilla(self, p_learning_rate):
        pass
