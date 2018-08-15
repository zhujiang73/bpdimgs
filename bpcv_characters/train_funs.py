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

from  mcvtorch  import BpcvNet  

loss_func = torch.nn.MSELoss()

def train_m(model, n_class, wh, optimizer, env_train, n_loop, bz, str_loop, np_convs_weights):

    model.train()

    txn_train = env_train.begin()

    targets_data = torch.zeros(bz, n_class)
    imgs_ten = torch.Tensor(bz,1,wh,wh)

    imgs_pb = []

    idxz = n_loop * bz 

    for i in range(bz):
                        
        str_key = "{0:0>8d}".format(idxz + i)

        img_po = img_data_pb2.ImgData()
        pb_buf = txn_train.get(str_key.encode('utf8'))
        img_po.ParseFromString(pb_buf)
        img_data = np.frombuffer(img_po.data, dtype=np.uint8) 
        img_data = img_data.reshape(img_po.wh, img_po.wh)

        img_data = img_data.astype(np.float32) / 255.0

        img_data = img_data - 0.5
        img_data = img_data / 0.5

        imgs_ten[i][0] = torch.from_numpy(img_data)

        imgs_pb.append(img_po)
        targets_data[i][img_po.class_id] = 1.0    

    txn_train.commit()

    optimizer.zero_grad()
    x_prob, x_conv9, x_convs = model(imgs_ten)

    #print("out_ten.shape",out_ten.shape)

    loss = loss_func(x_prob.view(x_prob.size(0), n_class), targets_data)
    loss.backward()
    optimizer.step()

    np1 = np_convs_weights[0]
    np2 = np_convs_weights[1]

    print("{0} :  convs_weight[...] : {1}  loss : {2:9.8f}".format(str_loop, np1[32][0][2][0:5], loss))
    #print("{0} :  convs_weight[...] : {1}  loss : {2:9.8f}".format(str_loop, np2[32][32][1][0:], loss))

def test_m(model, n_class, wh, env_db, n_loop, bz, str_loop, np_convs_weights):

    model.eval()

    txn_db = env_db.begin()

    targets_data = torch.zeros(bz, n_class)
    imgs_ten = torch.Tensor(bz,1,wh,wh)

    imgs_pb = []

    idxz = n_loop * bz 

    for i in range(bz):
                        
        str_key = "{0:0>8d}".format(idxz + i)

        img_po = img_data_pb2.ImgData()
        pb_buf = txn_db.get(str_key.encode('utf8'))
        img_po.ParseFromString(pb_buf)
        img_data = np.frombuffer(img_po.data, dtype=np.uint8) 
        img_data = img_data.reshape(img_po.wh, img_po.wh)

        img_data = img_data.astype(np.float32) / 255.0

        img_data = img_data - 0.5
        img_data = img_data / 0.5

        imgs_ten[i][0] = torch.from_numpy(img_data)

        imgs_pb.append(img_po)
        targets_data[i][img_po.class_id] = 1.0    

    txn_db.commit()
    x_prob, x_conv9, x_convs = model(imgs_ten)

    np_out = x_prob.view(x_prob.size(0), n_class).detach().numpy()
    #print("{0}, {1}".format(np_out[0], np.argmax(np_out[0], axis=None)))

    np_out_bins = np.zeros((bz, n_class))

    for i in range(bz):
        idx_max = np.argmax(np_out[i], axis=None)
        np_out_bins[i][idx_max] = 1.0

    np_out_bins = np_out_bins.astype(np.int32)

    np_targets = targets_data.detach().numpy()
    np_targets = np_targets.astype(np.int32)

    num_right = 0
    for i in range(bz):
        if (np_out_bins[i] == np_targets[i]).all():    
            num_right = num_right + 1

    np1 = np_convs_weights[0]

    print("{0} convs_weight[...] : {1} num_right : {2}".format(str_loop, np1[32][0][2][0:5], num_right))

    return  num_right 

def test(model, n_class, wh, env_db, np_convs_weights):
    bz = n_class*10
    num_samples = 1000
    num_loop = int(num_samples*n_class / bz)
    print("n1 num_loop : {}".format(num_loop))

    sum_right = 0

    for n in range(num_loop):
        idx = bz * n
        str_loop = "test {0:0>5d}".format(n)
        num_right = test_m(model, n_class, wh, env_db, n, bz, str_loop, np_convs_weights)
        sum_right = sum_right + num_right

    accuracy = sum_right * 1.0 / (num_samples*n_class)

    print("sum_right : ", sum_right)
    print("accuracy : ", accuracy)



