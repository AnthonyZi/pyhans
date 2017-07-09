import numpy as np

def t_tanh(p_lay_out_pre_activation):
    return np.tanh(p_lay_out_pre_activation)

def t_bp_tanh(p_lay_out_pre_activation, p_out):
    return 1 - p_out**2

def t_pass(p_lay_out_pre_activation):
    return p_lay_out_pre_activation

def t_bp_pass(p_lay_out_pre_activation, p_out):
    return 1
