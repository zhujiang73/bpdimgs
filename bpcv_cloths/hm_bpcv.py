#!/usr/bin/env python

import numpy as np
import os
import sys
import glob
import time
import copy

import sys
sys.path.append('c:\\mingw\\python')
print(sys.path)

import caffe
import matplotlib.pyplot as plt
from   skimage import transform as sktr
from   skimage import io as skio

def print_data(str_name, data):
	print ("{0} : {1}".format(str_name, data) ) 
	

def convert_mean(binMean, npyMean):
	blob = caffe.proto.caffe_pb2.BlobProto()
	bin_mean = open(binMean, 'rb' ).read()
	blob.ParseFromString(bin_mean)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	npy_mean = arr[0]
	np.save(npyMean, npy_mean )

def show_data(str_title, data_fp, padsize=1, padval=0):
	data = copy.deepcopy(data_fp)
	
	data -= data.min()
	data /= data.max()
	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	plt.figure(str_title),plt.title(str_title)
	plt.imshow(data,cmap='gray')
	plt.axis('off')

	plt.rcParams['figure.figsize'] = (8, 8)
	# plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'

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

net_file=".\\data\\caffenet_places.prototxt"
#caffe_model=".\\models\\caffenet_train_quick_iter_4000.caffemodel"
caffe_model=".\\models\\caffenet_train_quick_iter_5000.caffemodel"

mean_bin=".\\data\\mean.binaryproto"
mean_npy=".\\data\\mean.npy"

convert_mean(mean_bin, mean_npy)

imagenet_labels_filename = ".\\data\\synset_places.txt"
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

net_full_conv = caffe.Net(net_file, caffe_model, caffe.TEST) 

# load input and configure preprocessing

#str_img_fn = ".\\imgs\\2008_001042.jpg";  h_res = int(800) ;  w_res = int(960)
#str_img_fn = ".\\imgs\\fish-bike.jpg"; h_res = int(500) ;  w_res = int(720) 
#str_img_fn = ".\\imgs\\SSDB00253.jpg";   h_res = int(1200);  w_res = int(1600)
str_img_fn = ".\\imgs\\SSDB00266.jpg";   h_res = int(480) ;  w_res = int(600)
#str_img_fn = ".\\imgs\\SSDB00266.jpg";   h_res = int(300) ;  w_res = int(360)

img = caffe.io.load_image(str_img_fn, color = False)
img_show = skio.imread(str_img_fn)

img_res = sktr.resize(img, (h_res, w_res), mode = 'constant' )

print("heatmap ...... ")

start = time.time()
net_full_conv.blobs['data'].reshape(1, 1, h_res, w_res)
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_npy).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
#transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', img_res)]))
print("Caffe net forward in %.2f s." % (time.time() - start))
 
input_caffe_data = net_full_conv.blobs['data'].data[0][0]
print ("input_caffe_data : {0}".format(input_caffe_data.shape))
show_data("conv1 params", net_full_conv.params['conv1'][0].data.reshape(128*1,11,11))

print_data("blobs['conv7'].data.shape", net_full_conv.blobs['conv7'].data.shape)       
print_data("blobs['pool7'].data.shape", net_full_conv.blobs['pool7'].data.shape)       
print_data("blobs['prob'].data.shape", net_full_conv.blobs['prob'].data.shape)     

fulconv_data = net_full_conv.blobs['conv7'].data
#fulconv_data = net_full_conv.blobs['pool7'].data
fulconv_data = net_full_conv.blobs['prob'].data

str_title = "net_full_conv"
plt.figure(str_title),plt.title(str_title)
#plt.axis('off')

idx_fn = str_img_fn.rfind("\\")
str_title_fn = "image: {0}".format(str_img_fn[idx_fn+1:])

#print ("fulconv_data ...")
#print fulconv_data[0]
#print (fulconv_data[0].argmax(axis=0) )

if (h_res == 115 and w_res == 115):
	idx = fulconv_data[0].argmax(axis=0)[0][0]
	str_class = labels[idx]
	print(fulconv_data[0])
	print("classification : {0}".format(str_class))
	str_title = "image:  {0},   class:  {1}".format(str_img_fn, str_class)
	plt.title(str_title)
	plt.imshow(img_show)
	plt.show()
	exit()

plt.subplot(2, 2, 1),plt.title(str_title_fn)
plt.imshow(input_caffe_data, plt.cm.gray)

plt.subplot(2, 2, 2),plt.title(labels[0])
plt.imshow(fulconv_data[0,0], plt.cm.hot)

plt.subplot(2, 2, 3),plt.title(labels[1])
plt.imshow(fulconv_data[0,1], plt.cm.hot)

plt.subplot(2, 2, 4),plt.title(labels[1])
plt.imshow(fulconv_data[0,1], plt.cm.cool)
#plt.imshow(fulconv_data[0,1], plt.cm.hot)
plt.imshow(fulconv_data[0,1], plt.cm.gray)
plt.show()


