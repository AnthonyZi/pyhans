#!/usr/bin/python
import numpy as np

from myhelp import *
from pyhansmath import *

#inp = np.random.randint(2, size=[4, 4, 2])
#inp2 = np.random.randint(2, size=[5, 4, 2])
#ker = np.random.randint(3, size=[3, 2, 2, 2])
#
#a = np.random.randint(3, size=[3,3,1])
#k = np.random.randint(3, size=[2,2,1])
#
def newimg(p_width, p_height):
    return np.zeros([p_height, p_width])

def add_plus(img, p_x, p_y):
    height = img.shape[0]
    width = img.shape[1]
    plus = np.array([[0,1,0],[1,1,1],[0,1,0]])
    if p_x+2 >= width or p_y+2 >= height:
        return img
    for i in xrange(plus.shape[1]):
        for j in xrange(plus.shape[0]):
            img[p_x+i][p_y+j] = min(img[p_x+i][p_y+j] + plus[i][j], 1)
    return img
def add_pixel_dim(img):
    out = []
    img = np.array(img)
    for h in xrange(img.shape[0]):
        row = []
        for w in xrange(img.shape[1]):
            row.append([img[h][w]])
        out.append(row)
    out = np.array(out)
    return out


inp = []
des = []
for i in xrange(100):
    random_var = np.random.randint(3)
    if random_var == 0:
        n = newimg(32,32)
        n = add_pixel_dim(n)
        inp.append(n)
        des.append(add_pixel_dim([[0]]))
    else:
        n = newimg(32,32)
        n = add_pixel_dim(n)
        offset = np.random.randint(0,32-2, (2))
#        offset = [np.random.randint(32-2), np.random.randint(32-2)]
        inp.append(add_plus(n, offset[0], offset[1]))
        des.append(add_pixel_dim([[1]]))

inp = np.array(inp)
des = np.array(des)

arr = np.random.randint(3, size=[4,4,2])
weights = np.random.randint(3, size=[2, 2,2, 3])
reord = split_3d_z_dim(arr)



#
#
#def padwithtens(vector, pad_width, iaxis, kwargs):
#    if iaxis == 0:
#        vector[:pad_width[0]] += 0
#        vector[-pad_width[1]:] += 1
#    if iaxis == 1:
#        vector[:pad_width[0]] += 20
#        vector[-pad_width[1]:] += 21
#    if iaxis > 1:
#        vector[:pad_width[0]] += 100
#        vector[-pad_width[1]:] += 101
#    return vector
#
#test2d = np.arange(6).reshape((2,3))
#


if __name__ == "__main__":
    pass

#    aa = np.arange(9).reshape(3,3,1)
#    af = rot2dvec(aa)
#    kk = np.arange(4).reshape(2,2,1)
#    kf = rot2dvec(kk)
#
#    print "aa"
#    print3dmat(aa)
#    print "af"
#    print3dmat(af)
#    print "kk"
#    print3dmat(kk)
#    print "kf"
#    print3dmat(kf)
#
#    print "aa cross kk"
#    print crosscorr2dvec(aa, kk, [1,1]).shape
#    print3dmat(crosscorr2dvec(aa, kk, [1,1]).reshape(2,2,1))
#
#    print "aa cross kk"
#    print crosscorr2dvec_full(aa, kk, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(aa, kk, [1,1]).reshape(4,4,1))
#
#    print "aa cross kf"
#    print crosscorr2dvec_full(aa, kf, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(aa, kf, [1,1]).reshape(4,4,1))
#
#    print "kk cross aa"
#    print crosscorr2dvec_full(kk, aa, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(kk, aa, [1,1]).reshape(4,4,1))
#
#    print "kf cross aa"
#    print crosscorr2dvec_full(kf, aa, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(kf, aa, [1,1]).reshape(4,4,1))
#
#    print "af cross kk"
#    print crosscorr2dvec_full(af, kk, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(af, kk, [1,1]).reshape(4,4,1))
#
#    print "af cross kf"
#    print crosscorr2dvec_full(af, kf, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(af, kf, [1,1]).reshape(4,4,1))
#
#    print "kk cross af"
#    print crosscorr2dvec_full(kk, af, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(kk, af, [1,1]).reshape(4,4,1))
#
#    print "kf cross af"
#    print crosscorr2dvec_full(kf, af, [1,1]).shape
#    print3dmat(crosscorr2dvec_full(kf, af, [1,1]).reshape(4,4,1))
#
#
##   => (*) = crosscorr2dvec(op_a, op_b)
##   aa (*) kk == kf (*) af == rot180( kk (*) aa ) == rot180( af (*) kf )
##   aa (*) kf == kk (*) af == rot180( kf (*) aa ) == rot180( af (*) kk )
##   => (x) = conv2dvec(op_a, op_b)
##   aa (*) kf == aa (x) kk == kk (x) aa == kk (*) af

#### gradients on weights ####
##  aa0 (x) ww1 = aa1       aa0 (*) wf1  = aa1      | wf1 == yy1
##  dd1 = oo1-aa1           dd1 = oo1-aa1
##  dd1 (x) af1 = dww1      dd1 (*) aa1 = dww1
##                          aa1 (*) dd1 = dwf1
#
##  aa0 (*) yy1 = aa1
##  dd1 = oo1-aa1
##  aa1 (*) dd1 = dyy1
#### gradients on input ####
##  aa0 (x) ww1 = aa1       aa0 (*) wf1 = aa1
##  dd1 = oo1-aa1           dd1 = oo1-aa1
##  dd1 (x) wf1 = daa0      dd1 (*) ww1 = daa0
#
##  aa0 (*) yy1 = aa1
##  dd1 = oo1-aa1
##  dd1 (x) yy1 = daa0
##  -> dd1 (*) yf1 = daa0
##  -> dd1 (*) ww1 = daa0
#############################################
###### pyhans ###############################
## aa0 (*) ww1 = aa1   # cross correlation ##
## dd1 = oo1-aa1                           ##
#### gradients on weights ####             ##
## dww1 = aa1 (*) dd1                      ##
#### gradients on input ####               ##
## daa0 = dd1 (x) ww1  # convolution full  ##
## daa0 = dd1 (*) wr1  # wr1 = rot180(ww1) ##
#############################################
#############################################
