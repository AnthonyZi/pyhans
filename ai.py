import pyhans
import pyhanslayer

import transferer
import cost

from variables import *

from test import *

if __name__ == "__main__":

#    net = pyhans.HansNet(cost.quadratic_derivative)
#
#
#    l1 = pyhanslayer.HansLayer_Dense(3, 2, transferer.t_pass, transferer.t_bp_pass)
#
#    net.add_layer(l1)
#
#    backprop_set = [ ([1,0,1],[1,0]) , ([1,0,0],[1,0]) , ([0,1,0],[0,1]) ,       ([1,1,0],[1,1]) ]
#
#    for a in xrange(10000):
#        for bs in backprop_set:
#            net.backpropagate(net.feed_net(bs[0]), bs[1])
#            net.update_vanilla(learning_rate)
#
##    net.feed_net([1,0,1])
##    net.feed_net([1,0,0])
##    net.feed_net([0,1,0])
##    net.feed_net([1,1,0])
#
#    print net.feed_net([0,1,1])
#    print net.feed_net([0,0,0])
#    print net.feed_net([0,0,1])

    net2 = pyhans.HansNet(cost.quadratic_derivative)

    l12 = pyhanslayer.HansLayer_Conv_2D(
            p_inp_dims = [5, 5],
            p_inp_vec_dim = 2,
            p_out_vec_dim = 3,
            p_kernel_dims = [2,2],
            p_stride = [1,1],
            p_padding = [0,0,0],
            p_funct_activation = transferer.t_pass,
            p_funct_activation_deriv = transferer.t_bp_pass,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
            p_weights_min_init = -2,
            p_weights_max_init = 2,
            p_bias_init = 1)

    net2.add_layer(l12)

    net2.feed_net(inp)
