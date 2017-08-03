import pyhans
import pyhanslayer

import transferer
import cost

from variables import *

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

    l2_1 = pyhanslayer.HansLayer_Conv_2D(
            p_inp_dims = [20,20],
            p_inp_vec_dim = 1,
            p_out_vec_dim = 1,
            p_kernel_dims = [3, 3],
            p_stride = [1,1],
            p_padding = [0,0,0],
            p_funct_activation = transferer.t_tanh,
            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
            p_weights_min_init = -2,
            p_weights_max_init = 2,
            p_bias_init = 0)
#    l2_2 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [2,2],
#            p_inp_vec_dim = 1,
#            p_out_vec_dim = 1,
#            p_kernel_dims = [2, 2],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
##            p_weights_min_init = -0.0001,
##            p_weights_max_init = +0.0001,
#            p_weights_min_init = -2,
#            p_weights_max_init = 2,
#            p_bias_init = 0)

    net2.add_layer(l2_1)
#    net2.add_layer(l2_2)



####cifart_10_reconstruction of lenet
#    l2_1 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [32,32],
#            p_inp_vec_dim = 1,
#            p_out_vec_dim = 32,
#            p_kernel_dims = [18,18],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
##            p_weights_min_init = -2,
##            p_weights_max_init = 2,
#            p_bias_init = 0)
#    l2_2 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [15,15],
#            p_inp_vec_dim = 32,
#            p_out_vec_dim = 32,
#            p_kernel_dims = [6,6],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
##            p_weights_min_init = -2,
##            p_weights_max_init = 2,
#            p_bias_init = 0)
#    l2_3 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [10,10],
#            p_inp_vec_dim = 32,
#            p_out_vec_dim = 48,
#            p_kernel_dims = [5,5],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
##            p_weights_min_init = -2,
##            p_weights_max_init = 2,
#            p_bias_init = 0)
#    l2_4 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [6,6],
#            p_inp_vec_dim = 48,
#            p_out_vec_dim = 48,
#            p_kernel_dims = [3,3],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
##            p_weights_min_init = -2,
##            p_weights_max_init = 2,
#            p_bias_init = 0)
#    l2_5 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [4,4],
#            p_inp_vec_dim = 48,
#            p_out_vec_dim = 768,
#            p_kernel_dims = [4,4],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
##            p_weights_min_init = -2,
##            p_weights_max_init = 2,
#            p_bias_init = 0)
#    l2_6 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [1,1],
#            p_inp_vec_dim = 768,
#            p_out_vec_dim = 500,
#            p_kernel_dims = [1,1],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
##            p_weights_min_init = -2,
##            p_weights_max_init = 2,
#            p_bias_init = 0)
#    l2_7 = pyhanslayer.HansLayer_Conv_2D(
#            p_inp_dims = [1,1],
#            p_inp_vec_dim = 500,
#            p_out_vec_dim = 1,
#            p_kernel_dims = [1,1],
#            p_stride = [1,1],
#            p_padding = [0,0,0],
#            p_funct_activation = transferer.t_tanh,
#            p_funct_activation_deriv = transferer.t_bp_tanh,
#            p_weights_min_init = -0.0001,
#            p_weights_max_init = +0.0001,
##            p_weights_min_init = -2,
##            p_weights_max_init = 2,
#            p_bias_init = 0)
#
#
#
#
#
#
#
#
#    net2.add_layer(l2_1)
#    net2.add_layer(l2_2)
#    net2.add_layer(l2_3)
#    net2.add_layer(l2_4)
#    net2.add_layer(l2_5)
#    net2.add_layer(l2_6)
#    net2.add_layer(l2_7)

#    out = net2.feed_net(inp[:,:,0:1])
#    des = np.array([1]).reshape(1,1,1)
#    net2.backpropagate(out, des)

#    net2.verbosemax0 = False
#    for lay in net2.layers:
#        lay.verbosemax0 = False
    net2.verbose0 = False
    for lay in net2.layers:
        lay.verbose0 = False
#    net2.verbose0 = True
    for a in xrange(1000):
        randi = np.random.randint(inp.shape[0])
#        if (a+1)%5 == 0:
#            print des[randi]
#            net2.verbosemax0 = True
#            for lay in net2.layers:
#                lay.verbosemax0 = True
#            net2.verbose0 = True
#            for lay in net2.layers:
#                lay.verbose0 = True
#
#        else:
#            if (a+1)%1 == 0:
#                print a+1
        print a+1

        net2.backpropagate(net2.feed_net(inp[randi]), des[randi])
        net2.update_vanilla(learning_rate)

#        if (a+1)%5 == 0:
#            net2.verbosemax0 = False
#            for lay in net2.layers:
#                lay.verbosemax0 = False
#            net2.verbose0 = False
#            for lay in net2.layers:
#                lay.verbose0 = False

#    net2.verbosemax0 = True
#    for lay in net2.layers:
#        lay.verbosemax0 = True
    net2.verbose0 = True
    for lay in net2.layers:
        lay.verbose0 = True

    for a in xrange(5):
        randi = np.random.randint(inp.shape[0])
        bla = net2.feed_net(inp[randi])
        print des[randi], bla

