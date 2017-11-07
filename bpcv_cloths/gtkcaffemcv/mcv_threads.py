import os 
import sys
import time
import threading

from   skimage import io as skio

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gtk, GObject
from gi.repository.GdkPixbuf import Pixbuf

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

class Caffe_Hm_Thread(threading.Thread):
    def __init__(self, p_caller_slot, p_lock, p_caffe_net, p_show_data,
                       str_img_fn, w_res, h_res):
        super(Caffe_Hm_Thread, self).__init__()
        self.caller_slot = p_caller_slot
        self.lock = p_lock
        self.caffe_net = p_caffe_net
        self.show_data = p_show_data
        self.str_img_fn = str_img_fn
        self.w_res = w_res
        self.h_res = h_res

    def run(self):
        input_caffe_data, fulconv_data = self.caffe_net.img_full_conv_hm(self.str_img_fn, self.w_res, self.h_res)
        self.lock.acquire()
        self.show_data["caffe_in"] = input_caffe_data
        self.show_data["caffe_out"] = fulconv_data
        self.lock.release()
        GLib.idle_add(self.caller_slot, "caffe data")



