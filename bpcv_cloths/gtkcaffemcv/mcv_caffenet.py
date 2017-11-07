import os
import sys
import glob
import time
import copy
import numpy as np

import caffe
from   skimage import transform as sktr
from   skimage import io as skio

def convert_mean(binMean, npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )

def show_data(ax, str_title, p_data, padsize=1, padval=0):
    #data = copy.deepcopy(p_data)	
    data = p_data.copy()
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    ax.imshow(data, cmap = 'gray', origin = 'lower')

class McmCaffeNet():
    def __init__(self, p_call, str_net_file, str_caffe_model,
                       str_mean_npy, str_labels_fn):
        
        self.p_call = p_call

        self.rf_conv      = 137
        self.set_gpu_mode = 0
        self.height_plot  = 600
        self.width_plot   = 620
        self.str_mean_npy = str_mean_npy

        self.str_caffe_model = str_caffe_model
        self.str_net_file = str_net_file

        caffe.init_log()

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

        self.labels = np.loadtxt(str_labels_fn, str, delimiter='\t')

        self.net_full_conv = caffe.Net(network_file = self.str_net_file, 
                                        phase = caffe.TEST, 
                                        weights = self.str_caffe_model) 

        #self.net_params = self.net_full_conv.params.copy()

        data_blobs = self.net_full_conv.blobs

        self.data_shapes = {}

        for str_key in data_blobs.keys():
            blob = data_blobs[str_key]
            self.data_shapes[str_key] = blob.shape

    def img_full_conv_hm(self, str_img_fn, w_res, h_res):

        img = caffe.io.load_image(str_img_fn, color = False)
        #img_show = skio.imread(str_img_fn)

        img_res = sktr.resize(img, (h_res, w_res), mode = 'constant' )        

        conv_params = self.net_full_conv.params
        data_blobs = self.net_full_conv.blobs
               
        for str_key in conv_params.keys():
            conv_param = conv_params[str_key]
            data_blob = data_blobs[str_key]
            print ("conv_params {0} : {1}".format(str_key, conv_param[0].data.shape) )
            #print ("data_blob {0} : {1}".format(str_key, data_blob.data[0].shape) )
        

        self.net_full_conv.blobs['data'].reshape(1, 1, h_res, w_res)
        #self.net_full_conv.reshape() ????????????????????????????????????????????????????????????????????????
        self.transformer = caffe.io.Transformer({'data': self.net_full_conv.blobs['data'].data.shape})
        self.transformer.set_mean('data', np.load(self.str_mean_npy).mean(1).mean(1))
        self.transformer.set_transpose('data', (2,0,1))
        #transformer.set_channel_swap('data', (2,1,0))
        self.transformer.set_raw_scale('data', 255.0)

        caffe_data = np.asarray([self.transformer.preprocess('data', img_res)])
        
        start = time.time()
        #caffe_out = self.net_full_conv.forward(data = caffe_data)conv1
        #caffe_out = self.net_full_conv.forward(data = caffe_data, start="conv1", end="conv7")
        #caffe_out = self.net_full_conv.forward(data = caffe_data, start="conv7", end="prob")
        caffe_out = self.net_full_conv.forward_all(data = caffe_data)
        #caffe_out = self.net_full_conv.forward_backward_all(data = caffe_data)
        time_caffe = time.time() - start 

        #self.net_full_conv.params = self.net_params

        """
        for str_key in conv_params.keys():
            conv_param = conv_params[str_key]
            data_blob = data_blobs[str_key]
            print ("conv_params {0} : {1}".format(str_key, conv_param[0].data.shape) )
            print ("data_blob {0} : {1}".format(str_key, data_blob.data[0].shape) )
        """

        input_caffe_data = self.net_full_conv.blobs['data'].data[0][0].copy()
        fulconv_data = self.net_full_conv.blobs['prob'].data.copy()
        
        """
        for str_key in self.net_params.keys():
            param = conv_params[str_key]
            self.net_full_conv.params[str_key][0] = param[0]
            self.net_full_conv.params[str_key][1] = param[1]

        data_blobs = self.net_full_conv.blobs

        for str_key in data_blobs.keys():
            blob = data_blobs[str_key]
            shape = list(self.data_shapes[str_key] )
            blob.reshape(shape[0], shape[1], shape[2], shape[3])
        """
        
        for str_key in data_blobs.keys():
            data_blob = data_blobs[str_key]
            print ("data_blob {0} : {1}".format(str_key, data_blob.data.shape) )

        print("Caffe net forward in {0:.3}s ".format( time_caffe ) )
 
        self.net_full_conv = caffe.Net(network_file = self.str_net_file, phase = caffe.TEST, weights = self.str_caffe_model) 

        return input_caffe_data, fulconv_data


