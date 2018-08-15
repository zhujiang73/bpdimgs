import os 
import os.path 
import shutil 
import hashlib
import time
import lmdb
import numpy as np

import img_data_pb2

from skimage import io as skio
from skimage import exposure as skie
from skimage import transform as sktr

def  imgs_to_dbs(imgs_dirs, num_samples, wh, str_db_fn):

    imgs_class_fns = {}
    imgs_class = []

    for class_name, class_dir in imgs_dirs.items():

        list_fns = []

        for parent,dirnames,filenames in os.walk(class_dir):
            for filename in filenames:
                str_fn = class_dir + "/" + filename
                list_fns.append(str_fn)    

        imgs_class.append(class_name)
        imgs_class_fns[class_name] = list_fns
        
    num_class = len(imgs_class)

    lmdb_env = lmdb.open(str_db_fn, map_size = (wh*wh*2 + 96)*num_samples*num_class )
    lmdb_txn = lmdb_env.begin(write=True)

    idx_key = 0
    for n in range(num_samples):
        for m in range(num_class):
            class_name = imgs_class[m]
            str_fn = imgs_class_fns[class_name][n]

            #print ("class {} {} : {}".format(m, class_name, str_fn))

            img_grey = skio.imread(str_fn, as_grey=True)
            img_regrey = sktr.resize(img_grey, (wh, wh), mode = 'constant' )
            img_rescale = skie.rescale_intensity(img_regrey, in_range="image", out_range=(0.0, 1.0) )
            img_rescale = img_rescale * 255.0
            img_rescale_uint8 = img_rescale.astype(np.uint8)
            img_po = img_data_pb2.ImgData()
            img_po.class_id = m
            img_po.class_name = class_name
            img_po.wh = wh
            img_po.data = img_rescale_uint8.tobytes()
            img_buf = img_po.SerializeToString()

            str_key = "{0:0>8d}".format(idx_key)
            print("{} : {}".format(str_key, str_fn))
            lmdb_txn.put(str_key.encode('utf8'), img_buf)
            idx_key += 1

            if (idx_key % 1000 == 0):
                lmdb_txn.commit()
                lmdb_txn = lmdb_env.begin(write=True)

    lmdb_txn.commit()
    lmdb_env.close()


wh = 57

num_samples = 1000
str_db_fn = "test_dbs"
imgs_dirs_test = { "cloth"     :  "d:/caffe_net/images/bpcvs_test/cloths",
                   "place"  :  "d:/caffe_net/images/bpcvs_test/places" } 
imgs_to_dbs(imgs_dirs_test, num_samples, wh, str_db_fn)

num_samples = 20*1000
str_db_fn = "train_dbs"
imgs_dirs_train = { "cloth"     :  "d:/caffe_net/images/bpcvs_train/cloths", 
                    "place"  :  "d:/caffe_net/images/bpcvs_train/places"} 
imgs_to_dbs(imgs_dirs_train, num_samples, wh, str_db_fn)






