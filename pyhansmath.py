import numpy as np


#cross correlation (edge-handling: crop)
def crosscorr2dvec(p_data, p_kernel, p_stride):
    if p_stride != [1,1]:
        pass

#    output_dim = [(len(p_data)-len(p_kernel))/p_stride[0]+1, (len(p_data[0])-len(p_kernel[0]))/p_stride[1]+1]

    kernel_dim = p_kernel.shape[:-1]
    data_dim = p_data.shape[:-1]

    output_dim = [((data_dim[0]-kernel_dim[0])/p_stride[0]+1) , ((data_dim[1]-kernel_dim[1])/p_stride[1]+1) ]

    output = np.zeros(output_dim)

### (1)
    for i in xrange(output_dim[0]):
        for j in xrange(output_dim[1]):
            output[i][j] = sum(sum(sum(p_data[i:i+kernel_dim[0],j:j+kernel_dim[1]] * p_kernel)))
### (2) : three times slower than (1)
#    for i in xrange(output_dim[0]):
#        for j in xrange(output_dim[1]):
#            for a in xrange(kernel_dim[0]):
#                for b in xrange(kernel_dim[1]):
#                    output[i][j] += sum(p_data[i+a][j+b] * p_kernel[a][b])

    return output

def crosscorr2dvec_full(p_data, p_kernel, p_stride):
    kernel_dims = p_kernel.shape[:-1]
    data = pad2dvec(p_data, list(np.array(kernel_dims)-1) + list(np.array(kernel_dims)-1), 0)
    return crosscorr2dvec(data, p_kernel, p_stride)

#def rot2dvec(p_mat):
#    out = np.zeros_like(p_mat)
#    for i in xrange(len(p_mat)):
#        for j in xrange(len(p_mat[0])):
#            out[-(i+1)][-(j+1)] = p_mat[i][j]
#    return out

def rot2dvec(p_mat):
    return np.rot90(p_mat, 2)

def conv2dvec(p_data, p_kernel, p_stride):
    kernel = rot2dvec(p_kernel)
    return crosscorr2dvec(p_data, kernel, p_stride)

#convolution (edge-handling) padding with 0
def conv2dvec_full(p_data, p_kernel, p_stride):
    kernel_dims = p_kernel.shape[:-1]
    if p_stride != [1,1]:
        pass
    kernel = rot2dvec(p_kernel)
    data = pad2dvec(p_data, list(np.array(kernel_dims)-1) + list(np.array(kernel_dims)-1), 0)
    return crosscorr2dvec(data, kernel, p_stride)


def pad2dvec(p_mat, p_pad_dimensions, p_values):
    #p_pad_dimensions: 0-left, 1-top, 2-right, 3-bottom
    l = p_pad_dimensions[0]
    t = p_pad_dimensions[1]
    r = p_pad_dimensions[2]
    b = p_pad_dimensions[3]
    return np.pad(p_mat, [(t,b), (l,r), (0,0)], 'constant', constant_values=p_values)


def im2col_sliding_strided(p_mat, p_kernel_size, p_stride=[1,1]):
#    shp = list(p_kernel_size) + list(np.array(p_mat.shape) - p_kernel_size +1)
#    strd = list(p_mat.strides) + list(p_mat.strides)
#    out_view = np.lib.stride_tricks.as_strided(p_mat, shape=shp, strides=strd)
#    return out_view.reshape(shp[0]*shp[1],-1)[:,::p_stepsize]
    m,n = p_mat.shape
    col_extent = n - p_kernel_size[1] + 1
    row_extent = m - p_kernel_size[0] + 1

    #Get Starting block indices
    start_idx = np.arange(p_kernel_size[0])[:,np.newaxis]*n + np.arange(p_kernel_size[1])

    #Get offseted indices across the height and widht of input array
    offset_idx = np.arange(row_extent)[::p_stride[0],np.newaxis]*n + np.arange(col_extent)[::p_stride[1]]

    #Get all actual indices & index into input array for final output
    return np.take(p_mat, start_idx.ravel()[:,np.newaxis] + offset_idx.ravel())
