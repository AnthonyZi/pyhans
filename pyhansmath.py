import numpy as np
import scipy.signal as scipysig

from variables import *

#crosscorrelation performant implementation for convnet
def crosscorr_vec2d_input_feedforward(p_input, p_weights, p_stride):
    iz,iy,ix = p_input.shape
    wout,wz,wy,wx = p_weights.shape

    row_extent = (iy - wy)/p_stride[0] +1
    col_extent = (ix - wx)/p_stride[1] +1

    colinput = imvec2col_vert(p_input, (wy,wx), p_stride)
    weightsmat = p_weights.reshape(wout, wx*wy*wz)

    return np.dot(weightsmat, colinput).reshape(wout, row_extent, col_extent)

#crosscorrelation performant implementation for convnet using scipy
def crosscorr_vec2d_input_feedforward_scipy(p_input, p_weights, p_stride):
    crosscorr_method = "auto" #"direct", "fft"
    iz,iy,ix = p_input.shape
    wo,wz,wy,wx = p_weights.shape

    row_extent = (iy-wy)/p_stride[0] +1
    col_extent = (ix-wx)/p_stride[1] +1
    col_extent_no_stride = (ix-wx+1)

    row_idx = np.arange(row_extent)*p_str[0]*col_extent_no_stride
    col_idx = np.arange(col_extent)*p_str[1]

    mask = row_idx.ravel()[:,np.newaxis] + col_idx.ravel()

    out = []
    for ww in p_weights:
        out.append(np.take(scipysig.correlate(p_input, ww, mode="valid", method=crosscorr_method), mask))
    out = np.array(out)

    return out

#crosscorrelation performant implementation for convnet
def crosscorr_vec2d_weights_backprop(p_lay_input, p_weights_size, p_input_right, p_stride):
    iz,iy,ix = p_lay_input.shape
    vert_stride, horiz_stride = p_stride
    ko,kz,ky,kx = p_weights_size
    rz,ry,rx = p_input_right.shape

    col_lay_input = imvec2col_weights_backprop(p_lay_input, (ky,kx), p_stride)

    input_right_mat = p_input_right.reshape(rz, rx*ry)

    return np.dot(input_right_mat, col_lay_input).reshape(ko,kz,ky,kx)

#crosscorrelation performant implementation for convnet using scipy
def crosscorr_vec2d_weights_backprop_scipy(p_lay_input, p_weights_size, p_input_right, p_stride):
    crosscorr_method = "auto" #"direct", "fft"
    iz,iy,ix = p_lay_input.shape
    ko,kz,ky,kx = p_weights_size
    input_right_stretched = stretch2dvec(p_input_right, p_stride, (iy-ky+1,ix-kx+1))
    rz,ry,rx = input_right_stretched.shape

#    row_extent = iy-ry +1
#    col_extent = ix-rx +1

#    out = np.zeros([rz, iz, row_extent, col_extent])
    out = np.zeros([rz,iz,ky,kx])
    for o, right_inp in enumerate(input_right_stretched):
        for z, lay_inp in enumerate(p_lay_input):
            out[o,z,:,:] = scipysig.correlate(lay_inp, right_inp, mode="valid", method=crosscorr_method)[:ky,:kx]

    return out


#crosscorrelation performant implementation for convnet
def convolution_vec2d_input_backprop(p_input_right, p_weights, p_stride):
    weights_rotated_kernels = rotkernels(p_weights)
    wout,wz,wy,wx = weights_rotated_kernels.shape
    #padding: [left, top, right, bottom]
    padding = [wx-1, wy-1, wx-1, wy-1]
    padded_input_right = pad2dvec(p_input_right, padding)

    rpz,rpy,rpx = padded_input_right.shape

    row_extent = rpy - wy +1
    col_extent = rpx - wx +1

    print "padded_input_right.shape: ", padded_input_right.shape
    print "p_weights.shape: ", p_weights.shape
    print "col_extent", col_extent
    print "row_extent", row_extent

    col_input_right = imvec2col_vert(padded_input_right, (wy,wx), p_stride)
    weightsmat = np.rollaxis(weights_rotated_kernels,1).reshape(wz,wout*wy*wx)

    print "weightsmat.shape: ", weightsmat.shape
    print "col_input_right.shape ", col_input_right.shape

    return np.dot(weightsmat, col_input_right).reshape(wz, row_extent, col_extent)

##cross correlation (edge-handling: crop)
#def crosscorr2dvec(p_data, p_kernel, p_stride):
#    d_z,d_y,d_x = p_data.shape
#    k_out,k_z,k_y,k_x = p_kernel.shape
#
#    if verbose:
#        print p_data.shape
#        print p_kernel.shape
#
#    col_extent = d_x - k_x + 1
#    row_extent = d_y - k_y + 1
#
#    colim = imvec2col(p_data, (k_y,k_x), p_stride)
#    weightsmat = p_kernel.reshape(k_out, k_x*k_y*k_z)
#
#    return np.dot(weightsmat, colim).reshape(k_out,row_extent,col_extent)
#
#def crosscorr2dvec_full(p_data, p_kernel, p_stride):
#    kernel_dims = p_kernel.shape[:-1]
#    data = pad2dvec(p_data, list(np.array(kernel_dims)-1) + list(np.array(kernel_dims)-1), 0)
#    return crosscorr2dvec(data, p_kernel, p_stride)
#
##def rot2dvec(p_mat):
##    out = np.zeros_like(p_mat)
##    for i in xrange(len(p_mat)):
##        for j in xrange(len(p_mat[0])):
##            out[-(i+1)][-(j+1)] = p_mat[i][j]
##    return out

def rot180(p_mat):
    return np.rot90(p_mat, 2)

def rotkernels(p_weights):
    out = np.zeros_like(p_weights)
    for outd, w_row in enumerate(p_weights):
        for zd, weights_set in enumerate(w_row):
            out[outd][zd] = rot180(weights_set)
    return out

#def conv2dvec(p_data, p_kernel, p_stride):
#    kernel = rot2dvec(p_kernel)
#    return crosscorr2dvec(p_data, kernel, p_stride)
#
##convolution (edge-handling) padding with 0
#def conv2dvec_full(p_data, p_kernel, p_stride):
#    kernel_dims = p_kernel.shape[:-1]
#    if p_stride != [1,1]:
#        pass
#    kernel = rot2dvec(p_kernel)
#    data = pad2dvec(p_data, list(np.array(kernel_dims)-1) + list(np.array(kernel_dims)-1), 0)
#    return crosscorr2dvec(data, kernel, p_stride)


def pad2dvec(p_mat, p_pad_dimensions, p_values=0):
    #p_pad_dimensions: 0-left, 1-top, 2-right, 3-bottom
    l,t,r,b = p_pad_dimensions

    return np.pad(p_mat, [(0,0), (t,b), (l,r)], 'constant', constant_values=p_values)

def stretch2dvec(p_mat, p_pad_dimensions, p_out_dimensions):
    assert len(p_mat.shape) in [3,4], "p_mat-dimension not supported"
    pad_vert,pad_horiz = p_pad_dimensions

    if len(p_mat.shape) == 3:
        mz,my,mx = p_mat.shape
        outy = (my-1)*pad_vert +1
        outx = (mx-1)*pad_horiz +1
        out = np.zeros( (mz, outy, outx) )
        out[:,::pad_vert,::pad_horiz] = p_mat
        return out

    if len(p_mat.shape) == 4:
        mo,mz,my,mx = p_mat.shape
        outy = (my-1)*pad_vert +1
        outx = (mx-1)*pad_horiz +1
        out = np.zeros( (mo,mz, outy, outx) )
        out[:,:,::pad_vert,::pad_horiz] = p_mat
        return out




def imvec2col_vert(p_mat, p_kernel_size, p_stride=[1,1]):
    assert len(p_mat.shape) == 3, "p_mat: wrong dimension"
    m_z,m_y,m_x = p_mat.shape
    k_y,k_x = p_kernel_size
    vert_stride,horiz_stride = p_stride
    col_extent = m_x - k_x + 1
    row_extent = m_y - k_y + 1

    #Get Block indices
    block_idx = np.arange(k_y)[:,np.newaxis]*m_x*vert_stride + np.arange(k_x)*horiz_stride

#    mat_start_idx = np.arange(row_extent)[:,np.newaxis]*

    #Get Starting indices of blocks for all the image-layers
    row_start_idx = np.arange(m_z)[:,np.newaxis]*m_x*m_y + block_idx.ravel()

    #Get indices for block upper-left-corner indices strided for one image
    block_offset_idx = np.arange(row_extent)[::p_stride[0],np.newaxis]*m_x + np.arange(col_extent)[::p_stride[1]]

    return np.take(p_mat, row_start_idx.ravel()[:,np.newaxis] + block_offset_idx.ravel())

#def imvec2col_horiz(p_mat, p_kernel_size, p_stride=[1,1]):
#    assert len(p_mat.shape) == 3, "p_mat: wrong dimension"
#    m_z,m_y,m_x = p_mat.shape
#    k_y,k_x = p_kernel_size
#    horiz_stride,vert_stride = p_stride
#    col_extent = (m_x - k_x)/horiz_stride + 1
#    row_extent = (m_y - k_y)/vert_stride + 1
#
#    #Get Block indices
#    block_idx = np.arange(k_y)[:,np.newaxis]*m_x + np.arange(k_x)
#    print block_idx
#
#    lay_start_idx = np.arange(m_z)*m_x*m_y
#    stride_idx = np.arange(row_extent)[:,np.newaxis]*m_x*vert_stride + np.arange(col_extent)*horiz_stride
#
#    col_start_idx = lay_start_idx.ravel()[:,np.newaxis] + stride_idx.ravel()
#
#    return np.take(p_mat, block_idx.ravel()[:,np.newaxis] + col_start_idx.ravel())

def imvec2col_weights_backprop(p_mat, p_weights_size, p_stride=[1,1]):
    assert len(p_mat.shape) == 3, "p_mat: wrong dimension"
    m_z,m_y,m_x = p_mat.shape
    r_y,r_x = p_weights_size
    vert_stride,horiz_stride = p_stride
    row_extent = (m_y-r_y)/vert_stride +1
    col_extent = (m_x-r_x)/horiz_stride +1

    #Get Block indices
    lay_start_idx = np.arange(m_z)*m_x*m_y
    block_idx = np.arange(r_y)[:,np.newaxis]*m_x + np.arange(r_x)
    block_idx_per_layer = lay_start_idx.ravel()[:,np.newaxis] + block_idx.ravel()

    stride_idx = np.arange(row_extent)[:,np.newaxis]*m_x*vert_stride + np.arange(col_extent)*horiz_stride

    return np.take(p_mat, stride_idx.ravel()[:,np.newaxis] + block_idx_per_layer.ravel())


#another attempt could be:
#    shp = list(p_kernel_size) + list(np.array(p_mat.shape) - p_kernel_size +1)
#    strd = list(p_mat.strides) + list(p_mat.strides)
#    out_view = np.lib.stride_tricks.as_strided(p_mat, shape=shp, strides=strd)
#    return out_view.reshape(shp[0]*shp[1],-1)[:,::p_stepsize]

#def imvec2col_vert_padded(p_mat, p_kernel_size, p_stride=[1,1], p_padval=0):
