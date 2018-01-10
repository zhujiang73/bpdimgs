#!/usr/bin/env python

import numpy as np
import os
import sys
import glob
import time
import copy
import shutil
import lmdb

import sys
sys.path.append('c:\\mingw\\python')
#print(sys.path)

import caffe
from   skimage import transform as sktr
from   skimage import io as skio

import gi 
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

def convert_mean(binMean, npyMean):
	blob = caffe.proto.caffe_pb2.BlobProto()
	bin_mean = open(binMean, 'rb' ).read()
	blob.ParseFromString(bin_mean)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	npy_mean = arr[0]
	np.save(npyMean, npy_mean )

def caffe_image_match(str_img_fns, caffe_net, transformer, wh, str_id_conv):

    h_res = wh
    w_res = wh

    img_dats = []

    for str_fn in str_img_fns:
        img = caffe.io.load_image(str_fn, color = False)
        img_res = sktr.resize(img, (h_res, w_res), mode = 'constant' )   
        data = transformer.preprocess('data', img_res)     
        img_dats.append(data)

    caffe_data = np.asarray(img_dats)

    caffe_net.blobs['data'].reshape(len(str_img_fns), 1, wh, wh)

    start = time.time()
    caffe_out = caffe_net.forward_all(data = caffe_data)
    #caffe_out = net_full_conv.forward_backward_all(data = caffe_data)
    time_caffe = time.time() - start 

    print("Caffe net forward in %.2f s." % (time.time() - start))
 
    input_caffe_data = caffe_net.blobs['data'].data[0][0]
    #print ("input_caffe_data : {0}".format(input_caffe_data.shape))
    #print ("labels: {}".format(labels) )
 
    #fulconv_data = caffe_net.blobs['conv3'].data
    fulconv_data = caffe_net.blobs[str_id_conv].data
    fulconv_out = caffe_net.blobs['prob'].data

    print("fulconv_data.shape: {}".format(fulconv_data.shape) )
    print("fulconv_out.shape: {}".format(fulconv_out.shape) )

    return fulconv_data, fulconv_out

def conv_to_lmdb(str_db, str_img_fns, conv_datas):
    lmdb_env = lmdb.open(str_db, map_size=int(1e7))
    lmdb_txn = lmdb_env.begin(write=True)

    n = len(str_img_fns)

    for idx in range(0, n):
        str_fn = str_img_fns[idx]
        conv_vector = conv_datas[idx]
        str_data = conv_vector.tobytes()
        lmdb_txn.put(str_fn.encode('utf8'), str_data)

    lmdb_txn.commit()
    lmdb_env.close()

set_gpu_mode = 0

if set_gpu_mode:
	caffe.set_mode_gpu()
	caffe.set_device(0)
	#caffe.set_device(1)
	caffe.select_device(0, True)
	print("GPU mode")
else:
	caffe.set_mode_cpu()
	print("CPU mode")

caffe.init_log()

net_file="..\\bpcv_cloths\\data\\caffenet_places.prototxt"
caffe_model="..\\bpcv_cloths\\models\\caffenet_train_quick_iter_15000.caffemodel"

mean_bin="..\\bpcv_cloths\\data\\mean.binaryproto"
mean_npy="..\\bpcv_cloths\\data\\mean.npy"

convert_mean(mean_bin, mean_npy)

imagenet_labels_filename = "..\\bpcv_cloths\\data\\synset_places.txt"
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

str_short_labs = []
for str_lab in labels:
    idx = str_lab.rfind(" ")
    str_short = str_lab[idx+1:]
    str_short_labs.append(str_short)

wh = 137
#batch_size = 50 
batch_size = 120

imgs_dir = "..\\bpcv_cloths\\imgs"

imgfns = []

for parent,dirnames,filenames in os.walk(imgs_dir): 
    for filename in filenames: 
        fn = imgs_dir + "\\" + filename
        imgfns.append(fn)
        
str_fns = []
for i in range(0, len(imgfns)):
    fn = imgfns[i]
    #print(fn)
    str_fns.append(fn)

print("str_fns size : {}".format(len(str_fns)))

str_convn = "conv4"
str_db = "imgs_{}_db".format(str_convn)

convs = []

idx_fn01 = 0
idx_fn02 = idx_fn01 + batch_size

caffe_net = caffe.Net(network_file = net_file, phase = caffe.TEST, weights = caffe_model) 
caffe_net.blobs['data'].reshape(1, 1, wh, wh)
transformer = caffe.io.Transformer({'data': caffe_net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_npy).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

while True :

    if idx_fn02 > len(str_fns): idx_fn02 = len(str_fns)

    print("idx_fn01 idx_fn02 : {}  {}".format(idx_fn01, idx_fn02))

    str_caffe_fns =  str_fns[idx_fn01 : idx_fn02]

    fulconv_data, fulconv_out = caffe_image_match(str_caffe_fns, caffe_net, transformer, wh, str_convn)

    for i in range(0, fulconv_data.shape[0]):
        conv_data = fulconv_data[i]
        conv = conv_data.reshape(conv_data.shape[0]*conv_data.shape[1]*conv_data.shape[2])
        convs.append(conv)
    
    if idx_fn02 >= len(str_fns) : break
    idx_fn01 = idx_fn02
    idx_fn02 = idx_fn01 + batch_size

    caffe_net = caffe.Net(network_file = net_file, phase = caffe.TEST, weights = caffe_model) 
    caffe_net.blobs['data'].reshape(1, 1, wh, wh)

print("str_fns size : {}".format(len(str_fns)))
print("convs size : {}".format(len(convs)))

convs_map = {}

for idx in range(len(str_fns)):
    fn = str_fns[idx]
    conv = convs[idx]
    convs_map[fn] = conv

conv_to_lmdb(str_db, str_fns, convs)

exit()






