import numpy as np

from variables import *

class HansNet(object):

    layers = []

    fcnt_cost_derivative = None

    def __init__(self, p_fcnt_cost_derivative):
        self.fcnt_cost_derivative = p_fcnt_cost_derivative

    def add_layer(self, p_layer):
        self.layers.append(p_layer)

    def feed_net(self, p_input):
        current_vector = np.array(p_input)
        for lay in self.layers:
            current_vector = lay.feedforward(current_vector)
        return current_vector

    def backpropagate(self, p_network_out, p_desired_out):
        current_vector = self.fcnt_cost_derivative(p_network_out, p_desired_out)
        if verbosemax:
            print "backprop - cost: ", current_vector
        for lay in reversed(self.layers):
            current_vector = lay.backprop(current_vector)
        if verbosemax:
            print "input_derivatives:"
            print current_vector

    def update_vanilla(self, p_learning_rate):
        for lay in self.layers:
            lay.update_vanilla(p_learning_rate)
