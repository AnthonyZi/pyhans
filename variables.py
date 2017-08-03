import time

#all
verbose = False


#pyhanslayer.py
layer_weight_init_range = (-0.0005, 0.0005)
layer_bias_init_value = 0

#ai.py
learning_rate = 0.1




###############################################################################

import numpy as np

#def add_pixel_dim(img):
#out = []
#    img = np.array(img)
#    for h in xrange(img.shape[0]):
#        row = []
#        for w in xrange(img.shape[1]):
#            row.append([img[h][w]])
#        out.append(row)
#    out = np.array(out)
#    return out

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

def add_square(img, p_x, p_y):
    height,width = img.shape
    square = np.array([[1,1,1],[1,0,1],[1,1,1]])
    if p_x+2 >= width or p_y+2 >= height:
        return img
    for i in xrange(square.shape[1]):
        for j in xrange(square.shape[0]):
            img[p_x+i][p_y+j] = min(img[p_x+i][p_y+j] + square[i][j], 1)
    return img

inp = []
des = []
for i in xrange(100):
    random_var = np.random.randint(2)
    x,y = (500,500)
    if random_var == 0:
        n = newimg(x,y)
        offx,offy = np.random.randint(0,x-2, (2))
        inp.append([add_square(n, offx, offy)])
        des.append([[-1]])
    else:
        n = newimg(x,y)
        offx,offy = np.random.randint(0,x-2, (2))
        inp.append([add_plus(n, offx, offy)])
        des.append([[1]])

inp = np.array(inp)
des = np.array(des)


i1 = np.array(np.arange(500*500).reshape(1,500,500), dtype='float') / (500*500)
w1 = np.array(np.arange(20*20).reshape(1,1,20,20), dtype='float') / (500*500)
