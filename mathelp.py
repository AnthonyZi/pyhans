import numpy as np

def print3dmat(p_mat):
    assert isinstance(p_mat, np.ndarray), "p_mat wrong type"
    assert len(p_mat.shape) == 3, "p_mat wrong dimensions"

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

def varprint(p_mat):
    assert isinstance(p_mat, np.ndarray), "p_mat wrong type"
    assert len(p_mat.shape) in [3,4], "p_mat-dimension not supported"
    options = { 3 : print3dmat,
                4 : print4dmat }
    options[len(p_mat.shape)](p_mat)
