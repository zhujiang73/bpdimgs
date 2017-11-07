import sys
import time
import threading

sys.path.append('c:/mingw/python')
#sys.path.append('/usr/local/python')
#print(sys.path)

import caffe
#import matplotlib.pyplot as plt
from   skimage import transform as sktr
from   skimage import io as skio

import matplotlib.cm as cm
from matplotlib.figure import Figure

import numpy as np
from numpy import arange, sin, pi
#from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gtk, GObject
from gi.repository.GdkPixbuf import Pixbuf


from  gtkcaffemcv  import  convert_mean, show_data
from  gtkcaffemcv  import  ImgsListView, McmCaffeNet
from  gtkcaffemcv  import  McThread, FnsListThread, Caffe_Hm_Thread

class main:
    def __init__(self):
        self.builder = Gtk.Builder()
        self.builder.add_from_file("./data/gtkwin.glade")

        self.imgs_show_idx = 0
        self.imgs_list_size = 8

        self.img_width_1x = 0
        self.img_height_1x = 0

        self.img_width = self.img_width_1x
        self.img_height = self.img_height_1x

        self.str_img_fn = ""

        self.toolbutton1 = self.builder.get_object('toolbutton1')
        self.toolbutton3 = self.builder.get_object('toolbutton3')
        self.toolbutton4 = self.builder.get_object('toolbutton4')
        self.toolbutton5 = self.builder.get_object('toolbutton5')
        self.toolbutton6 = self.builder.get_object('toolbutton6')

        self.toolbutton1.connect("clicked", self.on_toolbutton1)
        self.toolbutton3.connect("clicked", self.on_toolbutton3)
        self.toolbutton4.connect("clicked", self.on_toolbutton4)
        self.toolbutton5.connect("clicked", self.on_toolbutton5)
        self.toolbutton6.connect("clicked", self.on_toolbutton6)

        self.window = self.builder.get_object('applicationwindow1')
        self.window.set_title("Python Gtk3 Caffe Heat Map")
        self.window.resize(820,620)
        self.window.connect("delete-event", self.onDeleteWindow)

        self.imgs_win = self.builder.get_object('IMG_LIST_VIEW')

        self.imgs_list_view = ImgsListView(self)

        self.imgs_win.connect("edge_reached", self.on_edge_reached)

        self.show_text_view = self.builder.get_object('textview01')
        self.show_entry = self.builder.get_object('entry1')

        self.image_view = Gtk.Image()
        self.show_view = self.image_view

        self.show_win = self.builder.get_object('IMAGE_SHOW')
        self.show_win.add(self.image_view)

        self.imgs_win.add(self.imgs_list_view)

        self.lock = threading.Lock()
        #self.lock = threading.RLock()

        self.fns_lock = threading.Lock()
        self.img_paths = []
        self.img_pixbufs = []

        fns_thread = FnsListThread(self.update_img_fns, self.fns_lock, "./imgs", self.imgs_list_size, 240,
                                        self.img_paths, self.img_pixbufs)
        fns_thread.start()

        net_file="./data/caffenet_places.prototxt"

        #caffe_model="./models/caffenet_train_quick_iter_4000.caffemodel"
        caffe_model="./models/caffenet_train_quick_iter_5000.caffemodel"
        #caffe_model="./models/caffenet_train_quick_iter_9000.caffemodel"

        mean_bin="./data/mean.binaryproto"
        mean_npy="./data/mean.npy"

        convert_mean(mean_bin, mean_npy)

        imagenet_labels_filename = "./data/synset_places.txt"
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

        self.caffe_lock = threading.Lock()
        self.caffe_show_data = {}
        self.caffe_busy = 0

        self.caffe_net = McmCaffeNet(self, net_file, caffe_model, mean_npy, imagenet_labels_filename)

        self.fig = Figure(figsize=(5, 4), dpi=80)

        self.window.show_all()

    def img_caffe_full_conv(self, str_img_fn, img_width, img_height):

        if (self.caffe_busy == 1):
            strbuf = Gtk.TextBuffer()
            strbuf.set_text("message : caffe_net busy ..")
            self.show_text_view.set_buffer(strbuf)
            return

        self.caffe_busy = 1

        strbuf = Gtk.TextBuffer()
        strbuf.set_text("message : caffe_net runing ..")
        self.show_text_view.set_buffer(strbuf)

        thread_caffe = Caffe_Hm_Thread(self.update_caffe_data, self.caffe_lock, self.caffe_net,
                                        self.caffe_show_data, str_img_fn, img_width, img_height)
        thread_caffe.start()    
        
        print("caffe_net runing ..")  


    def thread_fun(self, str_call_id):
        for i in range(10):
            str_txt = "{0} str_timer = {1}".format(str_call_id, i)
            GLib.idle_add(self.update_progess, str_txt)
            time.sleep(1.0)

    def update_progess(self, str_txt):
        #progress.set_text(str(i))
        strbuf = Gtk.TextBuffer()
        strbuf.set_text("thread message : {}".format(str_txt))
        self.show_text_view.set_buffer(strbuf)
        print("thread message : {}".format(str_txt))
        return False

    def update_show(self, str_txt):
        progress.set_text(str(i))
        self.window.show_all()
        return False

    def update_caffe_data(self, str_txt):

        self.caffe_lock.acquire()
        input_caffe_data = self.caffe_show_data["caffe_in"]
        fulconv_data = self.caffe_show_data["caffe_out"]
        self.caffe_show_data.clear()
        self.caffe_lock.release()
        
        self.caffe_busy = 0

        strbuf = Gtk.TextBuffer()
        strbuf.set_text("")
        self.show_text_view.set_buffer(strbuf)

        ax01 = self.fig.add_subplot(2,2,1)
        ax01.set_title(self.str_img_fn)
        ax01.imshow(input_caffe_data, cmap = 'gray', origin = 'lower')
        #ax01.axis('off')

        ax02 = self.fig.add_subplot(2,2,2)
        show_data(ax02, "conv1 params", self.caffe_net.net_full_conv.params['conv1'][0].data.reshape(128*1,9,9))
        ax02.set_title("conv1 params")

        ax03 = self.fig.add_subplot(2,2,3)
        ax03.set_title(self.caffe_net.labels[0])
        ax03.imshow(fulconv_data[0,0], cmap = 'hot', origin = 'lower')

        ax04 = self.fig.add_subplot(2,2,4)

        ax04.imshow(fulconv_data[0,1], cmap = 'hot', origin = 'lower')
        ax04.set_title(self.caffe_net.labels[1])
        
        canvas = FigureCanvas(self.fig)  # a Gtk.DrawingArea
        canvas.set_size_request(900, 720)
        self.show_win.remove(self.show_view)
        self.show_win.add(canvas)
        self.show_win.show_all()
        self.show_view = canvas

        return False


    def update_img_fns(self, str_txt):
        #print(str_txt)
        
        self.imgs_list_view.update_lists()
        
        return False

    def onDeleteWindow(self, window, *args):
        Gtk.main_quit(*args)

    def on_edge_reached(self, scr_win, pos):
        print ("imgs_list : {0}".format(pos))

    def on_selection_changed(self, selection):#{ on_selection_changed start
        #print (type(selection))
        self.image_view = Gtk.Image()
        iters = selection.get_selected_items()
        if len(iters)>0:
            str_idx_sel = iters[0].to_string()
            idx_sel = int(str_idx_sel)	
            #print (type(iters[0]),",",type(iters[0].to_string()), " : ", str_idx_sel)
            #print (idx_sel, img_fns[idx_sel])
            self.str_img_fn = self.img_paths[self.imgs_show_idx + idx_sel]
            pixbuf = Pixbuf.new_from_file_at_size(self.img_paths[self.imgs_show_idx + idx_sel], 900, 800)
            self.image_view.set_from_pixbuf(pixbuf)
            self.show_entry.set_text(self.img_paths[idx_sel])

            self.img_width_1x = pixbuf.get_width()
            self.img_height_1x = pixbuf.get_height()

            self.img_width = self.img_width_1x
            self.img_height = self.img_height_1x

            print ("img_{0} w_h:{1}x{2} {3}".format(idx_sel, 
                    self.img_width, self.img_height,
                    self.img_paths[self.imgs_show_idx + idx_sel])) 

        self.show_win.remove(self.show_view)
        self.show_win.add(self.image_view)
        self.show_win.show_all()
        self.show_view = self.image_view


        strbuf = Gtk.TextBuffer()
        strbuf.set_text("")
        self.show_text_view.set_buffer(strbuf)

        """
        self.thread = threading.Thread(target=self.thread_fun, args=("on_selection_changed",))
        self.thread.daemon = True
        self.thread.start()
        """
        #} on_selection_changed end


    def on_toolbutton1(self, toolbutton1):
        #print("toolbutton1")
        """
        thread01 = McThread(self.update_progess, self.lock, "on_toolbutton1 th01")
        thread02 = McThread(self.update_progess, self.lock, "on_toolbutton1 th02")
        thread02.start()      
        thread01.start()
        """

    def on_toolbutton3(self, toolbutton3):
        #print("toolbutton3")
        self.img_width = self.img_width_1x
        self.img_height = self.img_height_1x
        self.img_caffe_full_conv(self.str_img_fn, self.img_width, self.img_height)

    def on_toolbutton4(self, toolbutton4):
        #print("toolbutton4")
        self.img_width = int(self.img_width*1.1 + 0.5)
        self.img_height = int(self.img_height*1.1 + 0.5)
        self.img_caffe_full_conv(self.str_img_fn, self.img_width, self.img_height)

    def on_toolbutton5(self, toolbutton5):
        #print("toolbutton5")
        self.img_width = int(self.img_width*0.9 + 0.5)
        self.img_height = int(self.img_height*0.9 + 0.5)
        self.img_caffe_full_conv(self.str_img_fn, self.img_width, self.img_height)

    def on_toolbutton6(self, toolbutton6):
        #print("exit")
        self.window.hide()
        Gtk.main_quit()


if __name__ == "__main__":
    my_main = main()
  
    #my_main.toolbutton3.connect("clicked", my_main.on_toolbutton3)
    #my_main.toolbutton6.connect("clicked", my_main.on_toolbutton6)
    #print(sys.path)

    Gtk.main()


