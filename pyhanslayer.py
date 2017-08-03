import numpy as np

from pyhansmath import *

from variables import *

if verbose:
    from myhelp import *


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

    input_dimensions = None
    output_dimensions = None
    verbose0 = verbose

    def __init__(self,
            p_inp_dims,
            p_out_dims,
            p_inp_vec_dim,
            p_out_vec_dim,
            p_funct_activation,
            p_funct_activation_deriv,
            p_weights_min_init = None,
            p_weights_max_init = None,
            p_bias_init = None):

        assert isinstance(p_inp_dims, list), "p_inp_dims is not a list"
        assert isinstance(p_out_dims, list), "p_out_dims is not a list"
        assert isinstance(p_inp_vec_dim, int), "p_inp_vec_dim is not a int"
        assert isinstance(p_out_vec_dim, int), "p_out_vec_dim is not a int"

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

        if self.verbose0:
            print "funct_activation: ", type(self.funct_activation)
            print "funct_activation_derivative: ", type(self.funct_activation_derivative)
            print "input_vector_dimension: ", self.input_vector_dimension
            print "output_vector_dimension: ", self.output_vector_dimension
            print "input_dimensions: ", self.input_dimensions
            print "output_dimensions: ", self.output_dimensions

class HansLayer_Dense(HansLayer):

    weights = None
    delta_weights = None
    biases = None
    delta_biases = None

    lay_input = None
    lay_out_pre_activation = None
    lay_out = None

    def __init__(self,
            p_inp_vec_dim,
            p_out_vec_dim,
            p_funct_activation,
            p_funct_activation_deriv,
            p_weights_min_init = None,
            p_weights_max_init = None,
            p_bias_init = None):

        HansLayer.__init__(self,
                [1],
                [1],
                p_inp_vec_dim,
                p_out_vec_dim,
                p_funct_activation,
                p_funct_activation_deriv,
                p_weights_min_init,
                p_weights_max_init,
                p_bias_init)

        self.weights = np.random.random((self.output_vector_dimension, self.input_vector_dimension))
        self.weights *= (self.weights_randrange[1] - self.weights_randrange[0])
        self.weights += self.weights_randrange[0]
        self.biases = np.full(self.output_vector_dimension, self.bias_init)

    def feedforward(self, p_input):
        self.lay_input = p_input
        self.lay_out_pre_activation = np.dot(self.weights, self.lay_input) + self.biases
        self.lay_out = self.funct_activation(self.lay_out_pre_activation)

        if self.verbose0:
            print "feedforward: "
            print p_input
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


        if self.verbose0:
            print "p_input_right: ", p_input_right
            print "delta_biases: ", self.delta_biases
            print "delta_weights: "
            print self.delta_weights

        return np.dot(self.delta_biases, self.weights)

    def update_vanilla(self, p_learning_rate):
        self.weights -= self.delta_weights*p_learning_rate
        self.biases -= self.delta_biases*p_learning_rate

class HansLayer_Conv_2D(HansLayer):

    stride = None
    padding = None
    padding_value = None

    # weights =^ kernels
    weights = None
    delta_weights = None
    biases = None
    delta_biases = None

    lay_input = None
    lay_out_pre_activation = None
    lay_out = None

    def __init__(self,
            p_inp_dims,
            p_inp_vec_dim,
            p_out_vec_dim,
            p_kernel_dims,
            p_stride,
            p_padding,
            p_funct_activation,
            p_funct_activation_deriv,
            p_weights_min_init = None,
            p_weights_max_init = None,
            p_bias_init = None):

        if p_padding == None:
            p_padding = [0,0,0]

        assert isinstance(p_kernel_dims, list), "p_kernel_dims is not a list"
        assert isinstance(p_stride, list), "p_stride is not a list"
        assert isinstance(p_padding, list), "p_padding is not a list"

        assert len(p_inp_dims) == 2, "p_inp_dims wrong dimension"
        assert len(p_kernel_dims) == 2, "p_kernel_dims wrong dimension"
        assert len(p_stride) == 2, "p_stride wrong dimension"
        #p_padding : [padding_horizontally, padding_vertically, padding_value]
        assert len(p_padding) == 3, "p_padding wrong dimension"

        kernel_dims = np.array(p_kernel_dims)
        stride = np.array(p_stride)
        padding = np.array(p_padding[:-1])
        padding_value = p_padding[-1]

        translations = np.array(p_inp_dims) - kernel_dims + (2*padding)
        if any(translations%p_stride) != 0:
            print "warning, the convolution-layer will ignore some inputs"
        output_dimensions = list(translations/stride + 1)

        HansLayer.__init__(self,
                p_inp_dims,
                output_dimensions,
                p_inp_vec_dim,
                p_out_vec_dim,
                p_funct_activation,
                p_funct_activation_deriv,
                p_weights_min_init,
                p_weights_max_init,
                p_bias_init)

        self.stride = p_stride
        self.padding = p_padding[:-1]
        self.padding_value = p_padding[-1]

        self.weights = np.random.random([self.output_vector_dimension] + [self.input_vector_dimension] + p_kernel_dims)
        self.weights *= (self.weights_randrange[1] - self.weights_randrange[0])
        self.weights += self.weights_randrange[0]

        self.weights = np.array(self.weights, dtype='int')
        self.weights = np.array(self.weights, dtype='float')


        self.biases = np.full([self.output_vector_dimension] + self.output_dimensions, self.bias_init)
        self.delta_weights = np.zeros_like(self.weights)
        self.delta_biases = np.zeros_like(self.biases)

        self.lay_input = np.zeros([self.input_vector_dimension] + self.input_dimensions)
        self.lay_out_pre_activation = np.zeros([self.output_vector_dimension] + self.output_dimensions)
        self.lay_out = np.zeros([self.output_vector_dimension] + self.output_dimensions)

        if self.verbose0:
            print "-----------------------------------------"
            print "kernel_dimensions: ", p_kernel_dims
            print "stride: ", self.stride
            print "padding: ", self.padding
            print "weights.shape: ", self.weights.shape
            print "delta_weights.shape: ", self.delta_weights.shape
            print "biases.shape: ", self.biases.shape
            print "delta_biases.shape: ", self.delta_biases.shape
            print "lay_input.shape: ", self.lay_input.shape
            print "lay_out_pre_activation.shape: ", self.lay_out_pre_activation.shape
            print "lay_out.shape: ", self.lay_out.shape
            print "-----------------------------------------"



    def feedforward(self, p_input):
        if self.verbose0 or self.verbose0:
            starttime = time.time()
        self.lay_input = p_input
        if self.verbose0:
            lay_out_z = np.zeros_like(self.lay_out)
            lay_out_z = crosscorr_vec2d_input_feedforward(p_input, self.weights, self.stride)
            self.lay_out_pre_activation = lay_out_z + self.biases
#        for i in xrange(self.output_vector_dimension):
#            if self.verbose0:
#                lay_out_z[:,:,i] = crosscorr2dvec(p_input, self.weights[i], self.stride)
#                self.lay_out_pre_activation[:,:,i] = lay_out_z[:,:,i] + self.biases[:,:,i]
#            else:
#                self.lay_out_pre_activation[:,:,i] = crosscorr2dvec(p_input, self.weights[i], self.stride) + self.biases[:,:,i]


        else:
            self.lay_out_pre_activation = crosscorr_vec2d_input_feedforward(p_input, self.weights, self.stride) + self.biases

        self.lay_out = self.funct_activation(self.lay_out_pre_activation)
        if self.verbose0 or self.verbose0:
            stoptime = time.time()
            print "feedforward-time: ", (stoptime-starttime)


        if self.verbose0:
            print "----------------------"
            print "feedforward: "
            varprint(p_input)
            print "weights: "
            varprint(self.weights)
            print "-> "
            varprint(lay_out_z)
            print "biases: "
            varprint(self.biases)
            print "-> "
            varprint(self.lay_out_pre_activation)
            print "funct_activation(x)"
            print "-> "
            varprint(self.lay_out)
            print "______________________"
            print ""

        return self.lay_out

    def backprop(self, p_input_right):

        if self.verbose0:
            print "----------------------"
            print "backprop: - deltas:"
            varprint(p_input_right)
            print "p_input_right.shape: ", p_input_right.shape
            print "self.lay_out.shape: ", self.lay_out.shape
            print "self.lay_out_pre_activation.shape: ", self.lay_out_pre_activation.shape
            print "self.biases.shape: ", self.biases.shape
            print "self.weights.shape: ", self.weights.shape
            print "self.lay_input.shape: ", self.lay_input.shape
            print "xxxxxxxxxxxxxxxxxxxxxx"

        if self.verbose0 or self.verbose0:
            starttime = time.time()

        # gradients biases
        tmp_backprop = self.funct_activation_derivative(self.lay_out_pre_activation, self.lay_out) * p_input_right
        self.delta_biases = tmp_backprop

        # gradients weights
        self.delta_weights = crosscorr_vec2d_weights_backprop(self.lay_input, self.weights.shape, tmp_backprop, self.stride)
#        for i in xrange(self.output_vector_dimension):
#            for j in xrange(self.input_vector_dimension):
#                self.delta_weights[i,:,:,j] = crosscorr2dvec(self.lay_input[:,:,j:j+1], tmp_backprop[:,:,i:i+1], self.stride)

        # gradients input
#        da0(i) = sum_j[da1(j) (x) w1(j,i)]
        out = convolution_vec2d_input_backprop(tmp_backprop, self.weights, self.stride)

#        out = np.zeros_like(self.lay_input)
#        for i in xrange(self.input_vector_dimension):
#            for j in xrange(self.output_vector_dimension):
#                out[:,:,i] += conv2dvec_full(tmp_backprop[:,:,j:j+1], self.weights[j,:,:,i:i+1], self.stride)

        if self.verbose0:
            stoptime = time.time()
            print "backprop-time: ", (stoptime-starttime)

        if self.verbose0:
            print "xxxxxxxxxxxxxxxxxxxxxx"
            print "activation backprop:"
            varprint(tmp_backprop)
            print "weigths backprop:"
            print "layer.input:"
            varprint(self.lay_input)
            print "->"
            print4dmat(self.delta_weights)
            print "input backprop:"
            print "layer.weights:"
            print4dmat(self.weights)
            print "->"
            varprint(out)
            print "______________________"
            print ""

        return out

    def update_vanilla(self, p_learning_rate):
        if self.verbose0 or self.verbose0:
            starttime = time.time()
        self.weights -= self.delta_weights*p_learning_rate
        self.biases -= self.delta_biases*p_learning_rate
        if self.verbose0 or self.verbose0:
            stoptime = time.time()
            print "update-time: ", (stoptime-starttime)

