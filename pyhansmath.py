import numpy as np

def crosscorr2dvec(p_data, p_kernel, p_stride):
    if p_stride != [1,1]:
        pass

    output_dim = [(len(p_data)-len(p_kernel))/p_stride[0]+1, (len(p_data[0])-len(p_kernel[0]))/p_stride[1]+1]
    kernel_dim = [len(p_kernel), len(p_kernel[0])]

    output = np.zeros(output_dim)

    for i in xrange(output_dim[0]):
        for j in xrange(output_dim[1]):
            for a in xrange(kernel_dim[0]):
                for b in xrange(kernel_dim[1]):
                    output[i][j] += sum(p_data[i+a][j+b] * p_kernel[a][b])

    return output

def rot2dvec(p_mat):
    out = np.zeros_like(p_mat)
    for i in xrange(len(p_mat)):
        for j in xrange(len(p_mat[0])):
            out[-(i+1)][-(j+1)] = p_mat[i][j]
    return out

def conv2dvec(p_data, p_kernel, p_stride):
    kernel = rot2dvec(p_kernel)
    return crosscorr2dvec(p_data, kernel, p_stride)

def conv2dvec_full(p_deltas, p_kernel, p_kernel_dims, p_stride):
    if p_stride != [1,1]:
        pass
    kernel = rot2dvec(p_kernel)
    deltas = pad2dvec(p_deltas, p_kernel_dims-1, 0)
    return crosscorr2dvec(deltas, kernel, p_stride)


def pad2dvec(p_mat, p_pad_dimensions, p_values):
    #p_pad_dimensions: 0-left, 1-top, 2-right, 3-bottom
    l = p_pad_dimensions[0]
    t = p_pad_dimensions[1]
    r = p_pad_dimensions[2]
    b = p_pad_dimensions[3]
    return np.pad(p_mat, [(t,b), (l,r), (0,0)], 'constant', constant_values=p_values)
