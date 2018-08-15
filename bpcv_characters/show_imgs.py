import os 
import os.path 
import shutil 
import hashlib
import time
import lmdb
import numpy as np

import img_data_pb2

from skimage import io as skio
from skimage import transform as sktr

import gi 
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

str_db_fn = "train_dbs"
#str_db_fn = "test_dbs"

str_key01 = "{0:0>8d}".format(12480)
str_key02 = "{0:0>8d}".format(12480+1)

env = lmdb.open(str_db_fn, readonly=True)
txn = env.begin()

img_po01 = img_data_pb2.ImgData()
pb_buf01 = txn.get(str_key01.encode('utf8'))
img_po01.ParseFromString(pb_buf01)
img_data01 = np.frombuffer(img_po01.data, dtype=np.uint8)
img_data01 = img_data01.reshape(img_po01.wh, img_po01.wh)
img_data01 = img_data01.astype(np.float32) / 255.0

img_po02 = img_data_pb2.ImgData()
pb_buf02 = txn.get(str_key02.encode('utf8'))
img_po02.ParseFromString(pb_buf02)
img_data02 = np.frombuffer(img_po02.data, dtype=np.uint8)
img_data02 = img_data02.reshape(img_po02.wh, img_po02.wh)
img_data02 = img_data02.astype(np.float32) / 255.0

env.close()

fig = Figure(figsize=(5, 4), dpi=80)

ax01 = fig.add_subplot(1,2,1)
ax01.axis('on')
ax01.imshow(img_data01, cmap = 'gray', origin = 'lower')                   
ax01.set_title("class_id : {}  {}".format(img_po01.class_id, img_po01.class_name))

ax02 = fig.add_subplot(1,2,2)
ax02.axis('on')
ax02.imshow(img_data02, cmap = 'gray', origin = 'lower')                   
ax02.set_title("class_id : {}  {}".format(img_po02.class_id, img_po02.class_name))

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit)
win.set_default_size(600, 360)
win.set_title("show imgs")

sw = Gtk.ScrolledWindow()
win.add(sw)
sw.set_border_width(2)

canvas = FigureCanvas(fig) 
canvas.set_size_request(600, 360)
sw.add_with_viewport(canvas)

win.show_all()
Gtk.main()






