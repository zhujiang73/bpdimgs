import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

import lmdb
import math
import numpy as np
import img_data_pb2

from   skimage import transform as sktr
from   skimage import io as skio

from   mcvtorch  import BpcvNet

from   train_funs  import  train_m,  test_m,  test

n_class = 2
wh = 57

if __name__ == "__main__":
    str_train_mdb = "train_dbs" 
    str_test_mdb = "test_dbs" 

    env_train = lmdb.open(str_train_mdb, readonly=True)
    env_test = lmdb.open(str_test_mdb, readonly=True)

    np.set_printoptions(formatter={'float': lambda x: format(x, '9.8f')})

    str_pth_fn = "./models/bpcv_net_n1_04000.pth"
    bpcv_model = torch.load(str_pth_fn)

    np_convs_weights = []

    for m in bpcv_model.modules():
        if isinstance(m, nn.Conv2d):
            np_weight = m.weight.data.detach().numpy()
            np_convs_weights.append(np_weight)

    bz = 20
    num_loop = int(20*1000*n_class / bz)
    print("n2 num_loop : {}".format(num_loop))
    optimizer = optim.SGD(bpcv_model.parameters(), lr=0.002)

    for n in range(num_loop):
        idx = bz * n
        str_loop = "train n2 {0:0>5d}".format(n)
        train_m(bpcv_model, n_class, wh, optimizer, env_train, n, bz, str_loop, np_convs_weights)
        m = 500
        if ( (n+1) % m == 0):
            test(bpcv_model, n_class, wh, env_test, np_convs_weights)
            str_pth_fn = "./models/bpcv_net_n2_{0:0>5d}.pth".format(n+1)
            print("model save to : {}".format(str_pth_fn))
            torch.save(bpcv_model, str_pth_fn)
    

    bz = 40
    num_loop = int(20*1000*n_class / bz)
    print("n3 num_loop : {}".format(num_loop))
    optimizer = optim.SGD(bpcv_model.parameters(), lr=0.001)

    for n in range(num_loop):
        idx = bz * n
        str_loop = "train n3 {0:0>5d}".format(n)
        train_m(bpcv_model, n_class, wh, optimizer, env_train, n, bz, str_loop, np_convs_weights)
        m = 200
        if ( (n+1) % m == 0):
            test(bpcv_model, n_class, wh, env_test, np_convs_weights)
            str_pth_fn = "./models/bpcv_net_n3_{0:0>5d}.pth".format(n+1)
            print("model save to : {}".format(str_pth_fn))
            torch.save(bpcv_model, str_pth_fn)

    env_train.close()
    env_test.close()




