#!/usr/bin/env python

import numpy as np
import os
import sys
import glob
import time
import lmdb

import sys
sys.path.append('c:\\mingw\\python')
#print(sys.path)

import caffe
from   skimage import transform as sktr
from   skimage import io as skio

from  scipy import spatial

import gi 
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

def print_data(str_name, data):
	print ("{0} : {1}".format(str_name, data) ) 
	
def convert_mean(binMean, npyMean):
	blob = caffe.proto.caffe_pb2.BlobProto()
	bin_mean = open(binMean, 'rb' ).read()
	blob.ParseFromString(bin_mean)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	npy_mean = arr[0]
	np.save(npyMean, npy_mean )

def caffe_image(str_img_fn, caffe_net, transformer, wh, str_id_conv):
    img = caffe.io.load_image(str_img_fn, color = False)
    #img_show = skio.imread(str_img_fn)

    h_res = wh
    w_res = wh

    img_res = sktr.resize(img, (h_res, w_res), mode = 'constant' )

    caffe_data = np.asarray([transformer.preprocess('data', img_res)])
        
    start = time.time()
    caffe_out = caffe_net.forward_all(data = caffe_data)
    #caffe_out = net_full_conv.forward_backward_all(data = caffe_data)
    time_caffe = time.time() - start 

    #print("Caffe net forward in %.2f s." % (time.time() - start))
 
    input_caffe_data = caffe_net.blobs['data'].data[0][0]
    #print ("input_caffe_data : {0}".format(input_caffe_data.shape))
    #print ("labels: {}".format(labels) )
 
    #fulconv_data = caffe_net.blobs['conv3'].data
    fulconv_data = caffe_net.blobs[str_id_conv].data[0]
    fulconv_out = caffe_net.blobs['prob'].data

    #print(fulconv_data.shape)
    conv_vector = fulconv_data.reshape(fulconv_data.shape[0] * fulconv_data.shape[1] * fulconv_data.shape[2])

    idx_class = fulconv_out[0].argmax(axis=0)[0][0]
    value = fulconv_out[0][idx_class][0][0]
    
    return idx_class, value, conv_vector

def read_db(str_db):

        convs_map = {}

        env = lmdb.open(str_db, readonly=True)
        txn = env.begin()
        cursor = txn.cursor()

        for key, value in cursor:
            conv = np.frombuffer(value, dtype=np.float32)
            str_fn = key.decode()
            convs_map[str_fn] = conv

        env.close()

        return  convs_map
    
def conv_knn_match(convs_map, conv01):

        str_fns = list( convs_map.keys() )
        convs = list( convs_map.values() )

        kt_conv = spatial.cKDTree(data = convs)
        #kt_conv = spatial.cKDTree(data = convs, leafsize = 32)

        #knn_vas, knn_idxs = kt_conv.query(conv01, k=[1,3,5])
        knn_vas, knn_idxs = kt_conv.query(conv01, k=5)

        knn_convs_map = {}

        for idx in knn_idxs:
            str_fn = str_fns[idx]
            conv = convs_map[str_fn]
            knn_convs_map[str_fn] = conv

        return knn_idxs, knn_vas, knn_convs_map

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

caffe_net = caffe.Net(network_file = net_file, phase = caffe.TEST, weights = caffe_model) 

wh = 137

caffe_net.blobs['data'].reshape(1, 1, wh, wh)
transformer = caffe.io.Transformer({'data': caffe_net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_npy).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

str_img_fn01 = "..\\bpcv_cloths\\imgs\\2007_000256.jpg"
#str_img_fn01 = "..\\bpcv_cloths\\imgs\\2007_002024.jpg"
#str_img_fn01 = "..\\bpcv_cloths\\imgs\\2012_000960.jpg"

img01 = skio.imread(str_img_fn01, as_grey=True)

img_show01 = sktr.resize(img01, (wh, wh), mode = 'constant' )

str_convn = "conv4"
str_db = "imgs_{}_db".format(str_convn)

idx_class01, value01, conv01 = caffe_image(str_img_fn01, caffe_net, transformer, wh, str_convn)

str_class01 = str_short_labs[idx_class01]
#print("{0} : {1},  {2:.5f} ".format(str_img_fn01, str_class01, value01))

convs_map = read_db(str_db)
#str_fns = convs_map.keys()

start = time.time()
knn_idxs, knn_vas, knn_convs_map = conv_knn_match(convs_map, conv01)
print("knn_match: %.2f s." % (time.time() - start))

print("knn_convs_map size : {}".format(len(knn_convs_map)) )
print("knn_idxs: [ {0[0]}  {0[1]}  {0[2]}  {0[3]}  {0[4]}]".format(knn_idxs))
print("knn_vas: [ {0[0]:.5}  {0[1]:.5}  {0[2]:.5}  {0[3]:.5}  {0[4]:.5}]".format(knn_vas))
 
str_kfns = list(knn_convs_map.keys())

simils = []

for str_fn in str_kfns:
    conv02 = knn_convs_map[str_fn]
    simil = 1.0 - spatial.distance.braycurtis(conv01, conv02)
    simils.append(simil)

print("simils: [ {0[0]:.5}  {0[1]:.5}  {0[2]:.5}  {0[3]:.5}  {0[4]:.5}]".format(simils))

img02 = skio.imread(str_kfns[0], as_grey=True)
img_show02 = sktr.resize(img02, (wh, wh), mode = 'constant' )

img03 = skio.imread(str_kfns[1], as_grey=True)
img_show03 = sktr.resize(img03, (wh, wh), mode = 'constant' )

img04 = skio.imread(str_kfns[2], as_grey=True)
img_show04 = sktr.resize(img04, (wh, wh), mode = 'constant' )

img05 = skio.imread(str_kfns[3], as_grey=True)
img_show05 = sktr.resize(img05, (wh, wh), mode = 'constant' )

img06 = skio.imread(str_kfns[4], as_grey=True)
img_show06 = sktr.resize(img06, (wh, wh), mode = 'constant' )

fig = Figure(figsize=(5, 4), dpi=80)

ax01 = fig.add_subplot(2,3,1)
ax01.axis('on')
ax01.imshow(img_show01, cmap = 'gray', origin = 'lower')

ax02 = fig.add_subplot(2,3,2)
ax02.axis('on')
ax02.imshow(img_show02, cmap = 'gray', origin = 'lower')

ax03 = fig.add_subplot(2,3,3)
ax03.axis('on')
ax03.imshow(img_show03, cmap = 'gray', origin = 'lower')

ax04 = fig.add_subplot(2,3,4)
ax04.axis('on')
ax04.imshow(img_show04, cmap = 'gray', origin = 'lower')

ax05 = fig.add_subplot(2,3,5)
ax05.axis('on')
ax05.imshow(img_show05, cmap = 'gray', origin = 'lower')

ax06 = fig.add_subplot(2,3,6)
ax06.axis('on')
ax06.imshow(img_show06, cmap = 'gray', origin = 'lower')

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit)
win.set_default_size(800, 600)
win.set_title("caffe convs knn images match")

sw = Gtk.ScrolledWindow()
win.add(sw)
# A scrolled window border goes outside the scrollbars and viewport
sw.set_border_width(6)

canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
canvas.set_size_request(800, 600)
sw.add_with_viewport(canvas)

win.show_all()
Gtk.main()




