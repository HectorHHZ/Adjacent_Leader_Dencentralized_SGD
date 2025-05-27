import os
import numpy as np
import time
import argparse

#from mpi4py import MPI
from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models

from models import *



"""
NOTE: Following util is only used for MACHA Paper. It's not general method will be used by others.
"""

def select_graph(graphid):
    # pre-defined base network topologies
    # you can add more by extending the list
    Graphs = [
        # graph 0: 100% degree, graph 1
        # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
        [[(1, 5), (6, 7), (0, 4), (2, 3)],
         [(1, 7), (3, 6)],
         [(1, 0), (3, 7), (5, 6)],
         [(1, 2), (7, 0)],
         [(3, 1)]],
        
        # graph 1: 100% degree, graph 2
        # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
        [[(4, 6), (5, 2), (3, 1), (7, 0)],
         [(4, 2), (0, 5)],
         [(4, 3), (0, 2), (6, 5)],
         [(4, 7), (2, 3)],
         [(0, 4)]],

        # graph 2: 100% degree, graph 2
        # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
        [[(2, 3), (0, 4), (6, 7), (1, 5)],
         [(4, 3), (1, 0)],
         [(6, 3), (1, 4), (0, 2)],
         [(3, 5), (4, 6)],
         [(3, 1)]],

        # graph 3: 84.6% degree, graph 1
        [[(1, 5), (0, 4), (2, 3)],
         [(1, 7), (3, 6)],
         [(1, 0), (3, 7), (5, 6)],
         [(1, 2), (7, 0)]],

        # graph 4: 84.6% degree, graph 2
        [[(4, 6), (3, 1), (7, 0)],
         [(4, 2), (0, 5)],
         [(4, 3), (0, 2), (6, 5)],
         [(4, 7), (2, 3)]],

        # graph 5: 84.6% degree, graph 2
        [[(2, 3), (6, 7), (1, 5)],
         [(4, 3), (1, 0)],
         [(6, 3), (1, 4), (0, 2)],
         [(3, 5), (4, 6)]],

        # graph 6: 69.2% degree, graph 1
        [[(1, 5), (0, 4), (2, 3)],
         [(3, 6)],
         [(1, 0), (3, 7), (5, 6)],
         [(1, 2), (7, 0)]],

        # graph 7: 69.2% degree, graph 2
        [[(4, 6), (3, 1), (7, 0)],
         [(0, 5)],
         [(4, 3), (0, 2), (6, 5)],
         [(4, 7), (2, 3)]],

        # graph 8: 69.2% degree, graph 2
        [[(2, 3), (6, 7), (1, 5)],
         [(1, 0)],
         [(6, 3), (1, 4), (0, 2)],
         [(3, 5), (4, 6)]],

        # graph 9: 53.8% degree, graph 1
        [[(0, 4), (2, 3)],
         [(3, 6)],
         [(1, 0), (3, 7), (5, 6)],
         [(1, 2)]],

        # graph 10: 53.8% degree, graph 1
        [[(3, 1), (7, 0)],
         [(0, 5)],
         [(4, 3), (0, 2), (6, 5)],
         [(4, 7)]],

        # graph 11: 53.8% degree, graph 1
        [[(6, 7), (1, 5)],
         [(1, 0)],
         [(6, 3), (1, 4), (0, 2)],
         [(3, 5)]],

        # graph 12: 38.5% degree, graph 1
        [[(2, 3)],
         [(3, 6)],
         [(1, 0), (5, 6)],
         [(1, 2)]],

        # graph 13: 38.5% degree, graph 1
        [[(7, 0)],
         [(0, 5)],
         [(4, 3), (6, 5)],
         [(4, 7)]],

        # graph 14: 38.5% degree, graph 1
        [[(1, 5)],
         [(1, 0)],
         [(6, 3), (0, 2)],
         [(3, 5)]],
############################# 16-nodes #####################################
        # graph 10:
        # 16-node gemetric graph as shown in Fig. A.3(a) in Appendix
        [[(4, 8), (6, 11), (7, 13), (0, 12), (5, 14), (10, 15), (2, 3), (1, 9)],
         [(11, 13), (14, 2), (5, 6), (15, 3), (10, 9)],
         [(11, 13), (14, 2), (5, 6), (15, 3), (10, 9)],
         [(11, 8), (2, 5), (13, 4), (14, 3), (0, 10)],
         [(11, 5), (15, 14), (13, 8)],
         [(2, 11)]],

        # graph 11:
        # 16-node gemetric graph as shown in Fig. A.3(b) in Appendix
        [[(2, 7), (12, 15), (3, 13), (5, 6), (8, 0), (9, 4), (11, 14), (1, 10)],
         [(8, 6), (0, 11), (3, 2), (5, 4), (15, 14), (1, 9)],
         [(8, 3), (0, 6), (11, 2), (4, 1), (12, 14)],
         [(8, 11), (6, 3), (0, 5)],
         [(8, 2), (0, 3), (6, 7), (11, 12)],
         [(8, 5), (6, 4), (0, 2), (11, 7)],
         [(8, 15), (3, 7), (0, 4), (6, 2)],
         [(8, 14), (5, 3), (11, 6), (0, 9)],
         [(8, 7), (15, 11), (2, 5), (4, 3), (1, 0), (13, 6)],
         [(12, 8)]],

        # graph 12:
        # 16-node gemetric graph as shown in Fig. A.3(c) in Appendix
        [[(3, 12), (4, 8), (1, 13), (5, 7), (9, 10), (11, 14), (6, 15), (0, 2)],
         [(7, 14), (2, 6), (5, 13), (8, 10), (1, 15), (0, 11), (3, 9), (4, 12)],
         [(2, 7), (3, 15), (9, 13), (6, 11), (4, 14), (10, 12), (1, 8), (0, 5)],
         [(5, 14), (1, 12), (13, 8), (9, 4), (2, 11), (7, 0)],
         [(5, 1), (14, 8), (13, 12), (10, 4), (6, 7)],
         [(5, 9), (14, 1), (13, 3), (8, 2), (11, 7)],
         [(5, 12), (14, 13), (1, 9), (8, 0)],
         [(5, 2), (14, 10), (1, 3), (9, 8), (13, 15)],
         [(5, 8), (14, 12), (1, 4), (13, 10)],
         [(5, 3), (14, 2), (9, 12), (1, 10), (13, 4)],
         [(5, 6), (14, 0), (8, 12), (1, 2)],
         [(5, 15), (9, 14)],
         [(11, 5)]],

        # graph 13:
        # 16-node erdos-renyi graph as shown in Fig 3.(b) in the main paper
        [[(2, 7), (3, 15), (13, 14), (8, 9), (1, 5), (0, 10), (6, 12), (4, 11)],
         [(12, 11), (5, 6), (14, 1), (9, 10), (15, 2), (8, 13)],
         [(12, 5), (11, 6), (1, 8), (9, 3), (2, 10)],
         [(12, 14), (11, 9), (5, 15), (0, 6), (1, 7)],
         [(12, 8), (5, 2), (11, 14), (1, 6)],
         [(12, 15), (13, 11), (10, 5), (3, 14)],
         [(12, 9)],
         [(0, 12)]],

        # graph 14, 8-node ring
        [[(0, 1), (2, 3), (4, 5), (6, 7)],
         [(0, 7), (2, 1), (4, 3), (6, 5)]]

    ]

    return Graphs[graphid]