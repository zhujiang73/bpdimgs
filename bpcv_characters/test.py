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

    #str_pth_fn = "./models/bpcv_net_n1_04000.pth"
    #str_pth_fn = "./models/bpcv_net_n2_02000.pth"
    str_pth_fn = "./models/bpcv_net_n3_01000.pth"

    bpcv_model = torch.load(str_pth_fn)

    np_convs_weights = []

    for m in bpcv_model.modules():
        if isinstance(m, nn.Conv2d):
            np_weight = m.weight.data.detach().numpy()
            np_convs_weights.append(np_weight)

    test(bpcv_model, n_class, wh, env_test, np_convs_weights)

    env_train.close()
    env_test.close()




