import os
import numpy as np
import time
import argparse
import sys
import copy
import random


from math import ceil
from random import Random
import networkx as nx

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
from torch._C._distributed_c10d import (
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    PrefixStore,
    ProcessGroup,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    DebugLevel,
    get_debug_level,
)

cudnn.benchmark = True
from models import resnet
from models import vggnet
from models import wrn
import MACHA_util_triple
import util
from graph_manager import FixedProcessor, MatchaProcessor
from communicator import *


## SYNC_ALLREDUCE: gaurantee all models have the same initialization
def sync_allreduce(model, rank, size):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype = senddata[param].dtype)
    torch.cuda.synchronize()
    comm.barrier()

    comm_start = time.time()
    for param in model.parameters():
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    torch.cuda.synchronize()
    comm.barrier()

    comm_end = time.time()
    comm_t = (comm_end - comm_start)

    for param in model.parameters():
        param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = param.data/float(size)
    return comm_t

def run(rank, size):
    """
    load topology structure from ring, PS(parameter server), topology{centralized(all_reduce), and decentralied(MATCHA)}
    load communication scheme from MATCHA, local leader decentralized SGD(LLD-SGD), and D-PSGD
    topology: ring,                   communication scheme: baseline
    topology: PS(parameter server),   communication scheme: baseline
    topology: topology,               communication scheme: MATCHA, DPSGD, LLDSGD(MATCHA), LLDSGD(DPSGD)

    :param rank: 8 gpus, from 0 - 7
    :param size: 8 gpus, size = 8
    """

    # set random seed
    print("random seed is:", args.randomSeed)
    print("rank number is:",rank)
    torch.manual_seed(args.randomSeed+rank*15)
    np.random.seed(args.randomSeed)
    torch.cuda.manual_seed(args.randomSeed +rank*15)


    # load data
    partitioner = util.partition_dataset(rank, size, args)
    test_loader = partitioner.get_test()
    partitioner.init_train()
    train_loader = partitioner.get_train(rank)
    num_batches = ceil(len(train_loader.dataset) / float(args.bs))

    # bug test
    # True to rotate graph
    # args.graph_rotate = False

    ##  (computation and communication overlap not done yet)
    if (args.topology == 'ring'):
        communicator = ringCommunicator(rank, size)

    if (args.topology == 'PS'):
        communicator = PScommunicator(rank, size)

    if (args.topology == "all reduce sync"):
            communicator = centralizedCommunicator(rank, size)

    if (args.topology == 'topology'):
        # load base network topology
        subGraphs = MACHA_util_triple.select_graph(args.graphid)
        if args.graph_rotate == True:
            subGraphs1 = MACHA_util_triple.select_graph(args.graphid + 1)
            subGraphs2 = MACHA_util_triple.select_graph(args.graphid + 2)

        # define graph activation scheme
        if args.matcha:
            GP = MatchaProcessor(subGraphs, args.budget, rank, size, args.epoch * num_batches, True)
            if args.graph_rotate == True:
                GP1 = MatchaProcessor(subGraphs1, args.budget, rank, size, args.epoch * num_batches, True)
                GP2 = MatchaProcessor(subGraphs2, args.budget, rank, size, args.epoch * num_batches, True)
                print('graph rotation is active, use MATCHA processor')
            else:
                print('graph rotation is not active, use MATCHA processor')
        else:
            GP = FixedProcessor(subGraphs, args.budget, rank, size, args.epoch * num_batches, True)
            if args.graph_rotate == True:
                GP1 = FixedProcessor(subGraphs1, args.budget, rank, size, args.epoch * num_batches, True)
                GP2 = FixedProcessor(subGraphs2, args.budget, rank, size, args.epoch * num_batches, True)
                print('graph rotation is active, use fix processor')
            else:
                print('graph rotation is not active, use fix processor')

        # define communicator
        if args.compress:
            communicator = ChocoCommunicator(rank, size, GP, 0.9, args.consensus_lr)

        elif args.LLDSGD:
            communicator = LLDSGDCommunicator(rank, size, GP)
            if args.graph_rotate == True:
                communicator1 = LLDSGDCommunicator(rank, size, GP1)
                communicator2 = LLDSGDCommunicator(rank, size, GP2)
                print('graph rotation is active and use LLDSGD')
            else:
                print('graph rotation is NOT active and use LLDSGD')
        else:
            communicator = decenCommunicator(rank, size, GP)
            if args.graph_rotate == True:
                communicator1 = decenCommunicator(rank, size, GP1)
                communicator2 = decenCommunicator(rank, size, GP2)
                print('graph rotation is active')
            else:
                print('graph rotation is NOT active')

    # select neural network model
    model = util.select_model(args.numClass, args)
    # random model
    for param in model.parameters():
        param.data = param.data * random.uniform(0, 2)


    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=5e-4,
                          nesterov=args.nesterov)

    # guarantee all local models start from the same point
    # can be removed
    #sync_allreduce(model, rank, size)

    # init recorder
    comp_time, comm_time = 0, 0
    recorder = util.Recorder(args, rank)
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    tic = time.time()
    itr = 0

    # start training
    for epoch in range(args.epoch):
        
        if (args.data_rotate):
            data_rank = (rank + epoch) % size
        else:
            data_rank = rank
        train_loader = partitioner.get_train(data_rank)
        
        comm.barrier()


        if (args.topology == 'PS' and rank == 0):
            d_comm_time = communicator.communicatePS(model)
            comm_time += d_comm_time
            print("communication ends for bach", batch_idx)

            print("batch_idx: %d, rank: %d, comp_time: %.3f, comm_time: %.3f,epoch time: %.3f " % (
                batch_idx + 1, rank, d_comp_time, d_comm_time, comp_time + comm_time), end='\r')
            continue

        model.train()

        # Start training each epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            print("rank", rank, "epoch", epoch, "batch", batch_idx)
            start_time = time.time()

            # data loading
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # forward pass
            output = model(data)
            loss = criterion(output, target)

            # record training loss and accuracy
            record_start = time.time()
            acc1 = util.comp_accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            record_end = time.time()

            #   (backward pass. Needs to be rewrite for overlap accelerate.)
            loss.backward()

            # update learning rate for baseline. This should be modified when comparing with Non-block using lr = 0.1
            update_learning_rate(optimizer, epoch, itr=batch_idx, itr_per_epoch=len(train_loader))

            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            comm.barrier()
            end_time = time.time()

            d_comp_time = (end_time - start_time - (record_end - record_start))
            comp_time += d_comp_time

            # communication happens here
            pull_force =  (batch_idx % args.iteration == 0) and args.LLDSGD
            if (pull_force):
                if ((args.graph_rotate) and (batch_idx%2 == 0)) == True:
                    getInfo = communicator1.LLDSGDcommunicate(model, loss, args)
                    print('using rotate graph and LLDSGD, graph 2')
                elif(((args.graph_rotate) and (batch_idx%3 == 0)) == True):
                    getInfo = communicator2.LLDSGDcommunicate(model, loss, args)
                    print('using rotate graph and LLDSGD, graph 3')
                else:
                    getInfo = communicator.LLDSGDcommunicate(model,loss, args)
                    print('using LLDSGD but no rotate')
            else:
                if ((args.graph_rotate) and (batch_idx%2 == 0)) == True:
                    getInfo = communicator1.communicate(model)
                    print('using rotate graph but no LLDSGD, graph 2')
                elif (((args.graph_rotate) and (batch_idx%3 == 0)) == True):
                    getInfo = communicator2.communicate(model)
                    print('using rotate graph but no LLDSGD, graph 3')
                else:
                    getInfo = communicator.communicate(model)
                    print('not using rotate graph and LLDSGD')



            if (type(getInfo) == int):
                print(f"epoch {epoch} batch {batch_idx} return failed. model no update.")
                print(f"return value is:{getInfo}")
                d_comm_time = getInfo
                comm_time += d_comm_time
                print("batch_idx: %d, rank: %d, comp_time: %.3f, comm_time: %.3f,epoch time: %.3f " % (
                batch_idx + 1, rank, d_comp_time, d_comm_time, comp_time + comm_time), end='\r')
                continue
            else:
                d_comm_time, model = getInfo
                comm_time += d_comm_time
                print("batch_idx: %d, rank: %d, comp_time: %.3f, comm_time: %.3f,epoch time: %.3f " % (
                batch_idx + 1, rank, d_comp_time, d_comm_time, comp_time + comm_time), end='\r')



        toc = time.time()
        record_time = toc - tic  # time that includes anything
        epoch_time = comp_time + comm_time  # only include important parts

        # evaluate test accuracy at the end of each epoch
        test_acc = util.test(model, test_loader)

        recorder.add_new(record_time, comp_time, comm_time, epoch_time, top1.avg, losses.avg, test_acc)
        print("rank: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f, test_acc: %.3f epoch time: %.3f" % (
        rank, epoch, losses.avg, top1.avg, test_acc, epoch_time))
        if rank == 0:
            print("comp_time: %.3f, comm_time: %.3f, comp_time_budget: %.3f, comm_time_budget: %.3f" % (
            comp_time, comm_time, comp_time / epoch_time, comm_time / epoch_time))

        if epoch % 10 == 0:
            recorder.save_to_file()

        # reset recorders
        comp_time, comm_time = 0, 0
        losses.reset()
        top1.reset()
        tic = time.time()
        # train_sampler = torch.utils.data.distributed.DistributedSampler()

    recorder.save_to_file()





def test(model):
    test_list = list()
    for param in model.parameters():
        test_list.append(param.data)
    return test_list


def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    target_lr = args.lr
    lr_schedule = [100, 150]

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= base_lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    else:
        lr = target_lr
        for e in lr_schedule:
            if epoch >= e:
                lr *= 0.1

    if lr is not None:
        # print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')


    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')
    parser.add_argument('--matcha', action='store_true', help='use MATCHA or not')
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='total epoch')
    parser.add_argument('--budget', type=float, help='comm budget')
    parser.add_argument('--graphid', default=0, type=int, help='the idx of base graph')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--save', type=bool, default= True, help='save path or not')
    parser.add_argument('--compress', action='store_true', help='use chocoSGD or not')
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--numClass', default = 10, type=int, help='the number of classes of dataset')

    parser.add_argument('--name', '-n', default="default", type=str, help='experiment name')
    parser.add_argument('--topology', default='topology', type=str, help='choose topology from ring, PS(parameter server), topology{decentralied(MATCHA, DPSGD, LLSGD)')
    parser.add_argument('--description', default='debugtest', type=str, help='experiment description')
    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--savePath', type=str, help='save path')
    parser.add_argument('--randomSeed', type=int, help='random seed')
    parser.add_argument('--isNonIID', default=False, type=bool, help='False: random partition; True: IID partition')

    #pull iteration
    parser.add_argument('--LLDSGD', action='store_true', help='use LLDSGD or not')
    parser.add_argument('--iteration', type = int, default=40, help='iteration to pull')
    parser.add_argument('--c1', type = float, default=0.3, help='proportion for max degree neighbor')
    parser.add_argument('--c2', type=float, default=0.1, help='proportion for best performance neighbor with lowest loss')
    parser.add_argument('--p1', type=float, default=0.1, help='pull force for max degree neighbor')
    parser.add_argument('--p2', type=float, default=0.1, help='pull force for best performance neighbor')
    parser.add_argument('--data_rotate', default=False, type=bool, help='whether to rotate the dataset')
    parser.add_argument('--graph_rotate', default=False, type=bool, help='whether to rotate the graph')
    args = parser.parse_args()
    if not args.description:
        print('No experiment description, exit!')
        exit()

    # debug test
    # args.graph_rotate == False

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

    print(torch.cuda.is_available())
    print("program start!!!!")
    run(rank, size)
