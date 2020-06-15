#!/usr/bin/env python

import argparse
import os
from tqdm import tqdm 

import pprint as pp
import numpy as np

import torch
print(torch.__version__)
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value

from neural_combinatorial_rl import NeuralCombOptRL
from plot_attention import plot_attention
from dubins_path import DubinsPath
import pandas as pd

def str2bool(v):
      return v.lower() in ('true', '1')

def reward(bat_, index, USE_CUDA=False):
    """
    Args:
        List of length sourceL of [batch_size] Tensors
    Returns:
        Tensor of shape [batch_size] containins rewards
    """

    # sourceL = bat_[0].size(1)
    sourceL = 15
    batch_size = len(bat_)
    # print(index)
    
    bat = np.array(bat_)
    # batch_size = bat.shape(0)
    
    # solution_batch = np.ones((n,batch_size,3))
    # for city in range(n):
    #     solution_batch[city] = np.array(sample_solution[city]) #[batch_size x input_dim]
    # # print(solution_batch)

    # path = []
    # cost = np.zeros(batch_size)
    tour_len_ = 0
    cost = 0


    for j in range(sourceL-1):
        for i in range(batch_size):
            city1 = [bat[i,0,int(index[j])], bat[i,1,int(index[j])], bat[i,2,int(index[j])]]
            city2 = [bat[i,0,int(index[j+1])], bat[i,1,int(index[j+1])], bat[i,2,int(index[j+1])]]
            # print(city1, city2)
            dubins = DubinsPath(city1, city2, 0.1)
            dubins.calc_paths()
            path, cost = dubins.get_shortest_path()
            # tour_len += torch.norm(path, dim=2)
            tour_len_ += cost

    for i in range(batch_size):
        city1 = [bat[i,0,int(index[sourceL-1])], bat[i,1,int(index[sourceL-1])], bat[i,2,int(index[sourceL-1])]]
        city2 = [bat[i,0,int(index[0])], bat[i,1,int(index[0])], bat[i,2,int(index[j+1])]]
        # print(city1, city2)
        dubins = DubinsPath(city1, city2, 0.1)
        dubins.calc_paths()
        path, cost = dubins.get_shortest_path()
        # tour_len += torch.norm(path, dim=2)
        tour_len_ += cost



    return tour_len_
    # return tour_len


def test1():
    bat_1 = np.array([[[ 0.1830,  0.9410,  0.0990,  0.2210,  0.1480,  0.3650,  0.7010,
           0.6630,  0.3960,  0.5140,  0.4850,  0.4300,  0.8960,  0.6010,
           0.8990],
         [ 0.4050,  0.0480,  0.5640,  0.0260,  0.4760,  0.5400,  0.3560,
           0.0490,  0.9800,  0.7300,  0.4670,  0.0530,  0.8530,  0.4230,
           0.7980],
         [ 5.2360,  1.0472,  5.2360,  5.7596,  5.2360,  2.0944,  2.6180,
           0.0000,  3.6652,  0.0000,  2.6180,  0.0000,  1.5708,  2.6180,
           1.5708]]])
    index_optim = np.array([ 3, 11,  7,  1,  6, 13, 10,  5,  9, 14, 12,  8,  2,  4,  0])
    index_algo = np.array([4, 13, 6, 0, 10, 11, 3, 5, 8, 2, 7, 1, 12, 9, 14])

    R_optim = reward(bat_1, index_optim)
    R_out = reward(bat_1, index_algo)

    # bat_2 = [[[ 0.1600,  0.2400,  0.6500,  0.9100,  0.3200,  0.1160,  0.1420,
    #        0.3730,  0.4010,  0.1910,  0.6070,  0.8790,  0.6860,  0.0010,
    #        0.4190],
    #      [ 0.3970,  0.8360,  0.6670,  0.1400,  0.8350,  0.9110,  0.8400,
    #        0.7020,  0.8040,  0.8190,  0.0040,  0.1180,  0.5150,  0.5290,
    #        0.7840],
    #      [ 5.7596,  3.1416,  2.0944,  0.5236,  3.1416,  2.0944,  3.6652,
    #        1.0472,  1.5708,  4.1888,  0.0000,  0.5236,  2.0944,  5.2360,
    #        2.6180]]]
    # bat_3 =[[[ 0.5660,  0.9550,  0.2640,  0.4180,  0.1460,  0.7060,  0.3070,
    #        0.3370,  0.3940,  0.0260,  0.1710,  0.6180,  0.5000,  0.0300,
    #        0.6400],
    #      [ 0.9820,  0.2350,  0.1290,  0.4000,  0.6040,  0.4110,  0.0730,
    #        0.4090,  0.8120,  0.2390,  0.9460,  0.8970,  0.5720,  0.6780,
    #        0.9980],
    #      [ 0.0000,  1.5708,  5.2360,  2.6180,  2.6180,  3.1416,  5.2360,
    #        3.6652,  4.1888,  5.2360,  0.5236,  2.6180,  4.7124,  2.0944,
    #        0.5236]]]


    print('Example test input: {}'.format(index_optim))
    print('Example test output: {}'.format(index_algo))
    print('Example test input reward: {}'.format(R_optim))
    print('Example test output reward: {}'.format(R_out))

if __name__ == "__main__":
    test1()