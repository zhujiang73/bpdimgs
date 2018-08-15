import sys
import time
import threading

import torch
import torch.nn as nn
from torch.autograd import Variable

from   skimage import transform as sktr
from   skimage import io as skio

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gtk, GObject
from gi.repository.GdkPixbuf import Pixbuf

import matplotlib.cm as cm
from matplotlib.figure import Figure

import numpy as np
from numpy import arange, sin, pi
#from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

from  mcvtorch  import  ImgsListView, BpcvNet, draw_data
from  mcvtorch  import  McThread, FnsListThread, Torch_Hm_Thread

class main:
    def __init__(self):
        self.builder = Gtk.Builder()
        self.builder.add_from_file("gtkwin.glade")

        self.imgs_show_idx = 0
        self.imgs_list_size = 32

        self.img_wh = 620

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
        self.window.set_title("Gtk3 Pytorch Heat Map")
        self.window.resize(1080,720)
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

        fns_thread = FnsListThread(self.update_img_fns, self.fns_lock, "./imgs01", self.imgs_list_size, 240,
                                        self.img_paths, self.img_pixbufs)
        fns_thread.start()

        self.torch_lock = threading.Lock()
        self.torch_show_data = {}
        self.torch_busy = 0

        self.str_labels = ["cloth", "place"]
        
        #str_pth_fn = "./models/bpcv_net_n1_04000.pth"
        #str_pth_fn = "./models/bpcv_net_n2_02000.pth"
        str_pth_fn = "./models/bpcv_net_n3_01000.pth"

        #self.bpcv_net = BpcvNet()
        self.bpcv_net = torch.load(str_pth_fn)
        self.bpcv_net.eval()

        self.np_convs_weights = []

        for m in self.bpcv_net.modules():
            if isinstance(m, nn.Conv2d):
                np_weight = m.weight.data.detach().numpy()
                print("np_weight.shape : {}".format(np_weight.shape))
                self.np_convs_weights.append(np_weight)
                break

        self.window.show_all()

    def img_torch_full_conv(self, str_img_fn, img_width, img_height):

        if (self.torch_busy == 1):
            strbuf = Gtk.TextBuffer()
            strbuf.set_text("message : torch_net busy ..")
            self.show_text_view.set_buffer(strbuf)
            return

        self.torch_busy = 1

        strbuf = Gtk.TextBuffer()
        strbuf.set_text("message : torch_net runing ..")
        self.show_text_view.set_buffer(strbuf)

        thread_torch = Torch_Hm_Thread(self.update_torch_data, self.torch_lock, self.bpcv_net,
                                        self.torch_show_data, str_img_fn, img_width, img_height)
        thread_torch.start()    
        
        print("torch_net runing ..")  


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

    def update_torch_data(self, str_txt):

        self.torch_lock.acquire()
        input_data = self.torch_show_data["torch_in"]
        fulconv_data = self.torch_show_data["torch_out"]
        self.torch_show_data.clear()
        self.torch_lock.release()
        
        self.torch_busy = 0

        strbuf = Gtk.TextBuffer()
        strbuf.set_text("")
        self.show_text_view.set_buffer(strbuf)

        fig = Figure(figsize=(5, 4), dpi=80)

        ax01 = fig.add_subplot(2,2,1)
        ax01.set_title(self.str_img_fn)
        ax01.imshow(input_data, cmap = 'gray', origin = 'lower')
        #ax01.axis('off')

        ax02 = fig.add_subplot(2,2,2)

        np_weight = self.np_convs_weights[0]
        wpe = np_weight.shape
        draw_conv = draw_data(np_weight.reshape(wpe[0]*wpe[1],wpe[2],wpe[3]) )
        #draw_conv = draw_data(np_weight.reshape(wpe[0],wpe[1],wpe[2]) )
        ax02.imshow(draw_conv, cmap = 'gray', origin = 'lower')
        ax02.set_title("conv_weight")

        ax03 = fig.add_subplot(2,2,3)
        ax03.set_title(self.str_labels[0])
        ax03.imshow(fulconv_data[0][0], cmap = 'hot', origin = 'lower')

        ax04 = fig.add_subplot(2,2,4)

        ax04.imshow(fulconv_data[0][1], cmap = 'cool', origin = 'lower')
        ax04.set_title(self.str_labels[1])
        
        canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
        canvas.set_size_request(800, 620)
        self.show_win.remove(self.show_view)
        self.show_win.add(canvas)
        self.show_win.show_all()
        self.show_view = canvas

        return False


    def update_img_fns(self, str_txt):
        #print(str_txt)
        
        self.imgs_list_view.update_lists()

        self.str_img_fn = self.img_paths[0]
        pixbuf = Pixbuf.new_from_file_at_size(self.img_paths[0], self.img_wh, self.img_wh)
        self.image_view.set_from_pixbuf(pixbuf)
        self.show_entry.set_text(self.img_paths[0])

        self.img_width_1x = pixbuf.get_width()
        self.img_height_1x = pixbuf.get_height()

        self.img_width = self.img_width_1x
        self.img_height = self.img_height_1x

        self.show_win.remove(self.show_view)
        self.show_win.add(self.image_view)
        self.show_win.show_all()
        self.show_view = self.image_view
        
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
            pixbuf = Pixbuf.new_from_file_at_size(self.img_paths[self.imgs_show_idx + idx_sel], self.img_wh, self.img_wh)
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
        self.img_torch_full_conv(self.str_img_fn, self.img_width, self.img_height)

    def on_toolbutton4(self, toolbutton4):
        #print("toolbutton4")
        self.img_width = int(self.img_width*1.1 + 0.5)
        self.img_height = int(self.img_height*1.1 + 0.5)
        self.img_torch_full_conv(self.str_img_fn, self.img_width, self.img_height)

    def on_toolbutton5(self, toolbutton5):
        #print("toolbutton5")
        self.img_width = int(self.img_width*0.9 + 0.5)
        self.img_height = int(self.img_height*0.9 + 0.5)
        self.img_torch_full_conv(self.str_img_fn, self.img_width, self.img_height)

    def on_toolbutton6(self, toolbutton6):
        #print("exit")
        self.window.hide()
        Gtk.main_quit()


if __name__ == "__main__":
    win_main = main()

    Gtk.main()


