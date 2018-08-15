import os 
import sys
import time
import threading
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from skimage import io as skio
from skimage import exposure as skie
from skimage import transform as sktr

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gtk, GObject
from gi.repository.GdkPixbuf import Pixbuf


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,      out_planes, 
                     kernel_size=3,  stride=stride,
                     padding=1,      bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BpcvNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(BpcvNet, self).__init__()

        conv1 = nn.Conv2d(1, 96, kernel_size=3, stride=2, padding=0, bias=False)
        norm1 = nn.BatchNorm2d(96)
        relu1 = nn.ReLU(inplace=True)

        conv2 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0, bias=False)
        norm2 = nn.BatchNorm2d(128)
        relu2 = nn.ReLU(inplace=True)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        resnet01 = BasicBlock(128, 128)
        resnet02 = BasicBlock(128, 128)
        resnet03 = BasicBlock(128, 128)
    
        drop8 = nn.Dropout(p=0.5)

        conv9 = nn.Conv2d(128, num_classes, kernel_size=1, padding=0, bias=False)
        relu9 = nn.ReLU(inplace=True)

        pool  = nn.AvgPool2d(kernel_size=13, stride=1)
        prob  = nn.Softmax2d()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            conv1, norm1, relu1,
            conv2, norm2, relu2, pool2)
        

        self.resnets = nn.Sequential(
            resnet01,
            resnet02,
            resnet03)

        self.classifier = nn.Sequential(
            drop8,
            conv9,
            relu9)

        self.prob_class = nn.Sequential(
            pool,
            prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                if m is conv1:
                    m.weight.data.normal_(mean=0.0, std=0.01/50.0)
                else:
                    m.weight.data.normal_(mean=0.0, std=0.01/10.0)

                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x_conv = self.features(x)
        x_resnet = self.resnets(x_conv)
        x_conv9 = self.classifier(x_resnet)
        x_prob = self.prob_class(x_conv9)
        return x_prob, x_conv9, x_conv


def draw_data(data_p, padsize=1, padval=0):
    data = data_p.copy()
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data

class McThread(threading.Thread):
    def __init__(self, p_caller_slot, p_lock, str_on_fun):
        super(McThread, self).__init__()
        self.caller_slot = p_caller_slot
        self.lock = p_lock
        self.str_on_fun = str_on_fun

    def run(self):
        self.thread_fun(self.caller_slot, self.lock, self.str_on_fun)
        
    def thread_fun(self, caller_slot, lock, str_on_fun):
        lock.acquire()       
        for i in range(10):
            str_txt = "{0} str_timer = {1}".format(str_on_fun, i)
            GLib.idle_add(caller_slot, str_txt)
            time.sleep(1.0)
        lock.release()


class FnsListThread(threading.Thread):
    def __init__(self, p_caller_slot, p_fns_lock, str_path, imgs_max_num, img_wh,
                        p_caller_img_paths, p_caller_img_pixbufs):
        super(FnsListThread, self).__init__()
        self.caller_slot = p_caller_slot
        self.fns_lock = p_fns_lock
        self.str_path = str_path
        self.imgs_max_num = imgs_max_num
        self.img_wh = img_wh
        self.caller_img_paths = p_caller_img_paths
        self.caller_img_pixbufs = p_caller_img_pixbufs        

    def run(self):
        self.thread_fun(self.caller_slot, self.fns_lock, self.str_path, self.imgs_max_num, self.img_wh,
                        self.caller_img_paths, self.caller_img_pixbufs)
        
    def thread_fun(self, caller_slot, fns_lock, str_path, imgs_max_num, img_wh,
                         caller_img_paths, caller_img_pixbufs):

        list_fns = []
        for parent,dirnames,filenames in os.walk(str_path):
            for filename in filenames:
                list_fns.append(filename)

        str_iexs = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

        list_img_fns = []

        for fn in list_fns: #{
            for str_ex in str_iexs:
                idx = fn.rfind(str_ex)
                if (idx > 0 and len(str_ex)+idx == len(fn) ):
                    str_img_fn = str_path + "/" + fn
                    img_fat,w,h = Pixbuf.get_file_info(str_img_fn)
                    if (img_fat != None and w >= img_wh and h >= img_wh):
                        list_img_fns.append(fn)
                        break                     
            
            if (len(list_img_fns) >= imgs_max_num):
                break;
        #}

        n = imgs_max_num
        idx = 0

        if (n > len(list_img_fns) - idx):
            n = len(list_img_fns) - idx

        for i in range(n):
            fn = list_img_fns[idx + i]
            fns_lock.acquire()
            img_fn = str_path + "/" + fn
            pixbuf = Pixbuf.new_from_file_at_size(img_fn, img_wh, img_wh)
            caller_img_pixbufs.append(pixbuf)
            fns_lock.release()

        for fn in list_img_fns:
            fns_lock.acquire()
            img_fn = str_path + "/" + fn
            caller_img_paths.append(img_fn) 
            fns_lock.release()
        
        GLib.idle_add(caller_slot, "img_fns ok")

class Torch_Hm_Thread(threading.Thread):
    def __init__(self, p_caller_slot, p_lock, p_model, p_show_data,
                       str_img_fn, w_res, h_res):
        super(Torch_Hm_Thread, self).__init__()
        self.caller_slot = p_caller_slot
        self.lock = p_lock
        self.model = p_model
        self.show_data = p_show_data
        self.str_img_fn = str_img_fn
        self.w_res = w_res
        self.h_res = h_res

    def img_full_conv_hm(self, model, str_img_fn, w_res, h_res):

        img = skio.imread(str_img_fn, as_grey=True)

        img_res = sktr.resize(img, (h_res, w_res), mode = 'constant' )

        img_res = skie.rescale_intensity(img_res, in_range="image", out_range=(0.0, 1.0) )

        img_data = img_res - 0.5
        img_data = img_data / 0.5

        img_ten = torch.zeros(1,1,h_res,w_res)
        img_ten[0][0] = torch.from_numpy(img_data)

        x_prob, x_conv9, x_convs = model(img_ten)

        """
        pool_show = nn.AvgPool2d(kernel_size=12, stride=1)
        prob_show  = nn.Softmax2d()

        pool_data = pool_show(x_conv9)
        out_data = prob_show(pool_data)
        """

        np_out_data = x_prob.detach().numpy()
        #np_out_data = out_data.detach().numpy()

        return  img_res, np_out_data

    def run(self):
        input_data, out_data = self.img_full_conv_hm(self.model, self.str_img_fn, self.w_res, self.h_res)
        self.lock.acquire()
        self.show_data["torch_in"] = input_data
        self.show_data["torch_out"] = out_data
        self.lock.release()
        GLib.idle_add(self.caller_slot, "torch data")



