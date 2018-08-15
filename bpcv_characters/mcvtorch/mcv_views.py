import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository.GdkPixbuf import Pixbuf

class ImgsListView(Gtk.IconView):
    def __init__(self, p_obj):
        Gtk.IconView.__init__(self)
        self.p_obj = p_obj
        #self.set_default_size(260, 600)

        self.liststore = Gtk.ListStore(Pixbuf)    
        self.set_model(self.liststore)
        self.set_selection_mode(Gtk.SelectionMode.BROWSE)
        #self.set_selection_mode(Gtk.SelectionMode.SINGLE)
        #self.set_selection_mode(Gtk.SelectionMode.MULTIPLE)
        self.set_pixbuf_column(0)

        self.connect("selection-changed", p_obj.on_selection_changed)
        #self.show()
 
    def update_lists(self):
        #print("img_pixbufs size : {}".format(len(img_pixbufs)))
        self.liststore.clear()
        for pixbuf in self.p_obj.img_pixbufs:
            self.liststore.append([pixbuf])


