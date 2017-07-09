import numpy as np

def print3dmat(p_mat):
    if type(p_mat) != np.ndarray:
        exit(1)
    if len(p_mat.shape) != 3:
        exit(2)
    mats = []
    for i in xrange(p_mat.shape[2]):
        mats.append(p_mat[:,:,i])

    mats = np.array(mats)
    out = ""
    for i in xrange(mats.shape[1]):
        for j in xrange(mats.shape[0]):
            out += str(mats[j,i,:]).strip("\r\n") + "\t"
        out += "\n"
    print out

def print4dmat(p_mat):
    for mat in p_mat:
        print3dmat(mat)
