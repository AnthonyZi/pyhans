import pyhans
import pyhanslayer

import transferer
import cost

import scipy.signal as ss

from variables import *
from myhelp import *
from pyhansmath import *
import time
import numpy as np

np.set_printoptions(linewidth=160, precision=2)
#r = np.random.randint(len(inp))

#net = pyhans.HansNet(cost.quadratic_derivative)
#
#l1 = pyhanslayer.HansLayer_Conv_2D(
#        p_inp_dims = [500,500],
#        p_inp_vec_dim = 1,
#        p_out_vec_dim = 2,
#        p_kernel_dims = [3, 3],
#        p_stride = [1,1],
#        p_padding = [0,0,0],
#        p_funct_activation = transferer.t_tanh,
#        p_funct_activation_deriv = transferer.t_bp_tanh,
#        p_weights_min_init = -0.01,
#        p_weights_max_init = +0.01,
#        p_bias_init = 0)
#
#l2 = pyhanslayer.HansLayer_Conv_2D(
#        p_inp_dims = [498,498],
#        p_inp_vec_dim = 2,
#        p_out_vec_dim = 1,
#        p_kernel_dims = [498,498],
#        p_stride=[1,1],
#        p_padding = [0,0,0],
#        p_funct_activation = transferer.t_tanh,
#        p_funct_activation_deriv = transferer.t_bp_tanh,
#        p_weights_min_init = -0.01,
#        p_weights_max_init = +0.01,
#        p_bias_init = 0)
#
#net.add_layer(l1)
#net.add_layer(l2)





def mycross(p_inp, p_weights_size, p_right, p_str):
    a = time.clock()
    out = crosscorr_vec2d_weights_backprop(p_inp, p_weights_size, p_right, p_str)
    b = time.clock()
    return out, b-a


#def sscross(p_inp, p_wei, m="d", p_str=[1,1]):
#    meto = {"d": "direct", "a": "auto", "f": "fft"}
#    a = time.clock()
#    out = ss.correlate(p_inp, p_wei, mode="valid", method=meto[m])
#    b = time.clock()
#    return out, b-a

def mysscross(p_inp, p_weights_size, p_right, p_str):
    a = time.clock()
#    meto = {"d": "direct", "a": "auto", "f": "fft"}
#
#    iz,iy,ix = p_inp.shape
#    rz,ry,rx = p_right.shape
#
#    col_extent_no_stride= (ix-wx+1)
#    col_extent = (ix-wx)/p_str[1] +1
#    row_extent = (iy-wy)/p_str[0] +1
#
#    col_idx = np.arange(col_extent)*p_str[1]
#    row_idx = np.arange(row_extent)*p_str[0]*col_extent_no_stride
#    mask = row_idx.ravel()[:,np.newaxis] + col_idx.ravel()
#
#    out = []
#    for ww in p_wei:
#        out.append(np.take(ss.correlate(p_inp, ww, mode="valid", method=meto[m])[0], mask))
#    out = np.array(out)
    out = crosscorr_vec2d_weights_backprop_scipy(p_inp, p_weights_size, p_right, p_str)

    b = time.clock()
    return out, b-a


def timebench(inp, wei_size, right_inp, p_stride):
    t2 = -1
    t1 = -1
    try:
        y2, t2 = mysscross(inp, wei_size, right_inp, p_stride)
    except Exception as inst:
#        print "inp.shape: ", inp.shape
#        print "wei_size: ", wei_size
#        print "right_inp.shape: ", right_inp.shape
#        print "p_stride: ", p_stride
#        print "m: ", m
        print inst.args
        y2 = None
    try:
        y1, t1 = mycross(inp, wei_size, right_inp, p_stride)
    except Exception as inst:
#        print "inp.shape: ", inp.shape
#        print "wei_size: ", wei_size
#        print "right_inp.shape: ", right_inp.shape
#        print "p_stride: ", p_stride
        print inst.args
        y1 = None

    return [(t1,t2), (y1,y2)]


def accbench(inp, w1, m="a"):
    y1 = mycross(inp, w1)
    y2 = mysscross(inp, w1, m)
    print "y2.shape: ", y2.shape
    print "y1.shape: ", y1.shape

    acc = 0.0000000001
    for i in xrange(8):
        print "acc: ", acc

        okay = []
        not_okay = []

        dz,dy,dx = y1.shape
        for z in xrange(dz):
            for y in xrange(dy):
                for x in xrange(dx):
                    if abs(y1[z,y,x]-y2[z,y,x]) > acc:
                        not_okay.append((z,y,x))
                    else:
                        okay.append((z,y,x))
        print "1.0*len(okay)/(len(okay)+len(not_okay)): ", 1.0*len(okay)/(len(okay)+len(not_okay))
        if len(not_okay) < 100:
            for z,y,x in not_okay:
                print "my != scy: ", y1[z,y,x], " != ", y2[z,y,x]
        acc /= 10

def create_scipy_numpy_bench_vars():
    varsiz = [1] + list((np.arange(4)+1)*50)
#    varsiy = [5,10,15,20,30,50,75] + list((np.arange(20)+1)*100)
    varsiy = [1] + list((np.arange(4)+1)*25)
    varsix = varsiy


    varswo = [1,3,5] #scales more or less linear in numpy and clearly linear in scipy
    varswz = varsiz
    varswy = [1] + list((np.arange(4)+1)*25)
    varswx = varswy

    stridex = [1,3,5]
    stridey = [1,3,5]

    print "varsiz: ", varsiz
    print "varsiy: ", varsiy
    print "varsix: ", varsix

    print "varswo: ", varswo
    print "varswz: ", varswz
    print "varswy: ", varswy
    print "varswx: ", varswx

    outlist = []

    for ix in varsix:
#     for iy in varsiy:
      for iz in varsiz:
       for wx in varswx:
#        for wy in varswy:
         for wo in varswo:
          for sx in stridex:
    #       for sy in stridey:
            sy = sx
            iy = ix
            wy = wx
            if wx <= ix and wy <= iy and sx <= ix and sy <= iy:
             wz = iz
             outvars = [(ix,iy,iz), (wx,wy,wz,wo), (sx,sy)]
             outlist.append(outvars)
    return outlist

#def create_scipy_numpy_bench_vars_back_wei():
#    varsiz = [1] + list((np.arange(4)+1)*50)
##    varsiy = [5,10,15,20,30,50,75] + list((np.arange(20)+1)*100)
#    varsiy = [1] + list((np.arange(4)+1)*500)
#    varsix = varsiy
#
#    varsrz = varsiz
#    varsry = varsiy
#    varsrx = varsix
#
#    print "varsiz: ", varsiz
#    print "varsiy: ", varsiy
#    print "varsix: ", varsix
#    print "varsrz: ", varsrz
#    print "varsry: ", varsry
#    print "varsrx: ", varsrx
#
#    outlist = []
#    for ix in varsix:
#     for iz in varsiz:
#      for rx in varsrx:
#       for rz in varsrz:
#           iy = ix
#           ry = rx
#           if rx <= ix and ry <= iy:
#               outvars = [(ix,iy,iz), (rx,ry,rz)]
#               outlist.append(outvars)
#    return outlist

def scipy_numpy_bench_batch(varset, batch_no):
    outlist = []

    outfile_counter = 0
    batch_size = 50
    low_bound = (batch_no-1)*batch_size
    upp_bound = (batch_no)*batch_size

    variables_no = low_bound
    for variables in varset[low_bound:upp_bound]:
        ix,iy,iz = variables[0]
        wx,wy,wz,wo = variables[1]
        sumixy = iy*ix
        sumixyz = iz*iy*ix
        sumwxy = wy*wx
        sumwxyz = wz*wy*wx
        sumwxyzo = wo*wz*wy*wx
        inp = np.array(np.arange(sumixyz).reshape(iz,iy,ix), dtype='float') / sumixyz -0.5
        wei = np.array(np.arange(sumwxyzo).reshape(wo,wz,wy,wx), dtype='float') / sumwxyzo -0.5
        inp *= 4
        wei *= 4
        outvars = [(ix,iy,iz), (wx,wy,wz,wo)]
        tmy,tsci = timebench(inp,wei,"a")
        outsums = (sumixy, sumixyz, sumwxy, sumwxyz, sumwxyzo)
        print "(" + str(variables_no) + ")  " + '{:010.6f}'.format(tmy), '{:010.6f}'.format(tsci), outvars, outsums
        variables_no += 1
        outlist.append((tmy, tsci, outvars, outsums))

    outstringlist = ['{:010.6f}'.format(item[0]) + " " + '{:010.6f}'.format(item[1]) + " " + str(item[2]) + " " + str(item[3]) + "\n" for item in outlist]
#    outstringlist = [str(outitem) + "\n" for outitem in outlist]
    outstring = "dataset no: " + str(batch_no) + " varset[" + str(low_bound+1) + "," + str(upp_bound) + "]\n\n"
    for line in outstringlist:
        outstring += line
    write_file(outstring, batch_no)

def scipy_numpy_bench_batch_back_wei(varset, batch_no):
    outlist = []

    outfile_counter = 0
    batch_size = 50
    low_bound = (batch_no-1)*batch_size
    upp_bound = (batch_no)*batch_size

    variables_no = low_bound
    for variables in varset[low_bound:upp_bound]:
        ix,iy,iz = variables[0]
        wx,wy,wz,wo = variables[1]
        sx,sy = variables[2]
        sumixy = iy*ix
        sumixyz = iz*iy*ix
        sumwxy = wy*wx
        sumwxyz = wz*wy*wx
        sumwxyzo = wo*wz*wy*wx
        rx = (ix-wx)/sx +1
        ry = (iy-wy)/sy +1
        rz = wo
        sumrxy = rx*ry
        sumrxyz = rx*ry*rz
        inp = np.array(np.arange(sumixyz).reshape(iz,iy,ix), dtype='float') / sumixyz -0.5
        right = np.array(np.arange(sumrxyz).reshape(rz,ry,rx), dtype='float') / sumrxyz -0.5
        inp *= 4
        right *= 4
        outvars = [(iz,iy,ix), (rz,ry,rx), (sy,sx)]
        [(tmy,tsci), (y1,y2)] = timebench(inp, (wo,wz,wy,wx), right, (sy,sx))
        if not tmy<0 or tsci<0:
            if not test_equality(y1,y2):
                print "result is not the same!!!!!!!"
        outsums = (sumixy, sumixyz, sumrxy, sumrxyz)
        if tmy <= tsci:
            if tsci/tmy >= 8:
                side = "<<<<x     "
            elif tsci/tmy >= 4:
                side = " <<<x     "
            elif tsci/tmy >= 2:
                side = "  <<x     "
            else:
                side = "   <x     "
        else:
            if tmy/tsci >= 8:
                side = "    x>>>> "
            elif tmy/tsci >= 4:
                side = "    x>>>  "
            elif tmy/tsci >= 2:
                side = "    x>>   "
            else:
                side = "    x>    "
        print "(" + str(variables_no) + ")  " + side + '{:010.6f}'.format(tmy), '{:010.6f}'.format(tsci), outvars, outsums
        variables_no += 1
        outlist.append((tmy, tsci, outvars, outsums))

    outstringlist = ['{:010.6f}'.format(item[0]) + " " + '{:010.6f}'.format(item[1]) + " " + str(item[2]) + " " + str(item[3]) + "\n" for item in outlist]
#    outstringlist = [str(outitem) + "\n" for outitem in outlist]
    outstring = "dataset no: " + str(batch_no) + " varset[" + str(low_bound+1) + "," + str(upp_bound) + "]\n\n"
    for line in outstringlist:
        outstring += line
    write_file(outstring, batch_no)



def scipy_numpy_bench_batches(varset, batch_no_list):
    for batch_no in batch_no_list:
        scipy_numpy_bench_batch_back_wei(varset, batch_no)

def write_file(data_batch_str, counter):
    outfile_name = "benchmark_data/data_" + str(counter)
    outfile = open(outfile_name, "w")
    outfile.write(data_batch_str)
    outfile.close()
#
#
#
#def load_bench(filestring):
#    data = []
#    for line in filestring.split("\r\n"):
#        data = []
#        ldata = [float(dat.strip("[()],")) for dat in line.strip("\n\r").split(" ")]
#        data.append(ldata)
#    return data
#
#def load_data(data_arr):
#    for dat in data_arr:
#        tmy_tsci = dat[0:2]
#        outvars = dat[2:9]
#        outsums = dat[9:14]
def test_equality(inp1, inp2):
    assert isinstance(inp1, np.ndarray)
    assert isinstance(inp2, np.ndarray)
    assert inp1.shape == inp2.shape
    out = abs(inp1-inp2)
    os = out.shape
    for i in xrange(os[0]):
        for j in xrange(os[1]):
            for k in xrange(os[2]):
                for l in xrange(os[3]):
                    if out[i,j,k,l] >= 0.00000001:
                        return False
    return True



insize = [4,4]
indepth = [2]
outdepth = [2]
kernelsize = [2,2]
vstride,hstride = [2,2]

iz,iy,ix = indepth + insize
weights_size = outdepth + indepth + kernelsize
wo,wz,wy,wx = weights_size
outsize = [((iy-wy)/vstride+1),((ix-wx)/hstride+1)]
az,ay,ax = outdepth + outsize

inp = np.array(np.arange(ix*iy*iz).reshape(iz,iy,ix), dtype='float')# / (ix*iy*iz) -0.5
right = np.array(np.arange(az*ay*ax).reshape(az,ay,ax), dtype='float')# / (wx*wy*wz*wo) -0.5
print "inp: "
varprint(inp)
print "right: "
varprint(right)
y1, t1 = mycross(inp, weights_size, right, [vstride,hstride])
print "y1: "
varprint(y1)
y2, t2 = mysscross(inp, weights_size, right, [vstride,hstride])
print "y2: "
varprint(y2)
if not test_equality(y1,y2):
    print "not equal"

myvarset = create_scipy_numpy_bench_vars()
print "len(myvarset): ", len(myvarset)
for line in myvarset:
    print line

scipy_numpy_bench_batches(myvarset, [1,2,3,4,5,6,7,8,9,10,11,12,13])
