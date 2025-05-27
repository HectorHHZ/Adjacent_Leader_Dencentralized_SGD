import os
import numpy as np
import time
import argparse

#from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx
from torch.optim import SGD
import copy

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
from datasets import load_dataset
from torch.utils.data import Dataset
# import deeplake
import os, glob
from torchvision.io import read_image, ImageReadMode
import re
from models import *



class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]




class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID = False):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if (isNonIID == True):
            # self.partitions = np.loadtxt("./final_partition.txt").tolist()
            # for i in range(len(self.partitions)):
            #     self.partitions[i] = [int(x) for x in self.partitions[i]]
            self.partitions = self.__getNonIIDdata__(self, data, sizes, seed)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label, [])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen = int(len(labelList) / len(sizes))
        majorLabelNumPerPartition = ceil(labelNum / len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        # basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio * len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start + idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        # random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]
        return partitions




class TinyImageNet(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        root = os.path.join(root, 'tiny-imagenet')
        if train:
            root = os.path.join(root, 'tiny-imagenet_train.pkl')
        else:
            root = os.path.join(root, 'tiny-imagenet_val.pkl')
        with open(root, 'rb') as f:
            dat = pickle.load(f)
        self.data = dat['data']
        self.targets = dat['targets']
        self.transform = transform

    def __getitem__(self, item):
        data, targets = Image.fromarray(self.data[item]), self.targets[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return len(self.data)


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/scratch/hh2537/data/test/tiny-imagenet-200/train/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        # path is not general.
        # print(img_path)
        # print(img_path.split('/')[7])
        label = self.id_dict[img_path.split('/')[7]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label


class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/scratch/hh2537/data/test/tiny-imagenet-200/val/*/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('/scratch/hh2537/data/test/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label


class partition_dataset():
        def __init__(self, rank, size, args):
            self.rank = rank
            self.size = size
            self.args = args


        def get_test(self):

            if self.args.dataset == 'cifar10':
                print('==> load test data')
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                testset = torchvision.datasets.CIFAR10(root=self.args.datasetRoot,
                                                       train=False,
                                                       download=True,
                                                       transform=transform_test)
                test_loader = torch.utils.data.DataLoader(testset,
                                                          batch_size=64,
                                                          shuffle=False,
                                                          num_workers=self.size)


            if self.args.dataset == 'cifar100':
                print('==> load test data')
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
                testset = torchvision.datasets.CIFAR100(root=self.args.datasetRoot,
                                                        train=False,
                                                        download=True,
                                                        transform=transform_test)
                test_loader = torch.utils.data.DataLoader(testset,
                                                          batch_size=64,
                                                          shuffle=False,
                                                          num_workers=self.size)




            if self.args.dataset == 'emnist':

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])
                testset = torchvision.datasets.EMNIST(root=self.args.datasetRoot,
                                                      split='balanced',
                                                      train=False,
                                                      download=True,
                                                      transform=transform_test)
                test_loader = torch.utils.data.DataLoader(testset,
                                                          batch_size=64,
                                                          shuffle=False,
                                                          num_workers=self.size)

            if self.args.dataset == 'tinyImageNet':
                # transform_test = transforms.Compose([
                #     transforms.ToTensor(),
                # ])
                # tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet', split='valid', cache_dir = "/home/hh2537/data/")
                # test_loader = torch.utils.data.DataLoader(tiny_imagenet_test,
                #                                           batch_size=64,
                #                                           shuffle=False,
                #                                           num_workers=self.size)

                # tiny_imagenet_test = deeplake.load("hub://activeloop/tiny-imagenet-test")
                # test_loader = tiny_imagenet_test.pytorch(num_workers=self.size, batch_size=64, shuffle=False)

                id_dict = {}
                for i, line in enumerate(open('/scratch/hh2537/data/test/tiny-imagenet-200/wnids.txt', 'r')):
                    id_dict[line.replace('\n', '')] = i
                transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
                testset = TestTinyImageNetDataset(id=id_dict, transform=transform)
                test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=self.size)




            return test_loader


        def init_train(self):
            print('==> load train data')
            if self.args.dataset == 'cifar10':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                trainset = torchvision.datasets.CIFAR10(root=self.args.datasetRoot,
                                                        train=True,
                                                        download=True,
                                                        transform=transform_train)
            if self.args.dataset == 'cifar100':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])

                trainset = torchvision.datasets.CIFAR100(root=self.args.datasetRoot,
                                                              train=True,
                                                              download=True,
                                                              transform=transform_train)

            if self.args.dataset == 'tinyImageNet':
                # transform_train = transforms.Compose([
                #     transforms.ToTensor(),
                # ])
                # trainset = load_dataset('Maysee/tiny-imagenet', split='train', cache_dir = "/home/hh2537/data/")
                # trainset = deeplake.load("hub://activeloop/tiny-imagenet-train")
                # test_loader = tiny_imagenet_test.pytorch(num_workers=self.size, batch_size=64, shuffle=False)

                id_dict = {}
                for i, line in enumerate(open('/scratch/hh2537/data/test/tiny-imagenet-200/wnids.txt', 'r')):
                    id_dict[line.replace('\n', '')] = i
                transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
                trainset = TrainTinyImageNetDataset(id=id_dict, transform = transform)
                # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=self.size)



            if self.args.dataset == 'emnist':
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                ])
                trainset = torchvision.datasets.EMNIST(root=self.args.datasetRoot,
                                                            split='balanced',
                                                            train=True,
                                                            download=True,
                                                            transform=transform_train)



            partition_sizes = [1.0 / self.size for _ in range(self.size)]


            self.partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)



        def get_train(self, subsetID):
            partition = self.partition.use(subsetID)
            train_loader = torch.utils.data.DataLoader(partition,
                                                       batch_size=self.args.bs,
                                                       shuffle=True,
                                                       pin_memory=True)
            return train_loader



def select_model(num_class, args):
    if args.model == 'VGG':
        model = vggnet.VGG(16, num_class)
        # model = vggnet.VGG(11, num_class)
    elif args.model == 'res':
        if args.dataset == 'cifar10':
            # model = large_resnet.ResNet18()
            model = resnet.ResNet(50, num_class)
        elif args.dataset == 'imagenet':
            model = models.resnet18()
    elif args.model == 'wrn':
        model = wrn.Wide_ResNet(28, 10, 0, num_class)
    elif args.model == 'mlp':
        if args.dataset == 'emnist':
            model = MLP.MNIST_MLP(47)
    return model


def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    def __init__(self, args, rank):
        self.record_accuracy = list()
        self.record_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = args.savePath + args.name + '_' + args.model
        if rank == 0 and os.path.isdir(self.saveFolderName) == False and self.args.save:
            os.mkdir(self.saveFolderName)

    def add_new(self, record_time, comp_time, comm_time, epoch_time, top1, losses, test_acc):
        self.total_record_timing.append(np.array(record_time))
        self.record_timing.append(np.array(epoch_time))
        self.record_comp_timing.append(np.array(comp_time))
        self.record_comm_timing.append(np.array(comm_time))
        self.record_trainacc.append(np.array(top1.cpu()))
        self.record_losses.append(np.array(losses))
        self.record_accuracy.append(np.array(test_acc.cpu()))
        print("test accuracy is:", test_acc)

    def save_to_file(self):
        np.savetxt( 
            self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
                self.rank) + '-recordtime.log', self.total_record_timing, delimiter=',')
        np.savetxt(
            self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
                self.rank) + '-time.log', self.record_timing, delimiter=',')
        np.savetxt(
            self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
                self.rank) + '-comptime.log', self.record_comp_timing, delimiter=',')
        np.savetxt(
            self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
                self.rank) + '-commtime.log', self.record_comm_timing, delimiter=',')
        np.savetxt(
            self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
                self.rank) + '-acc.log', self.record_accuracy, delimiter=',')
        np.savetxt(
            self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
                self.rank) + '-losses.log', self.record_losses, delimiter=',')
        np.savetxt(
            self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
                self.rank) + '-tacc.log', self.record_trainacc, delimiter=',')
        with open(self.saveFolderName + '/ExpDescription', 'w') as f:
            f.write(str(self.args) + '\n')
            f.write(self.args.description + '\n')


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        acc1 = comp_accuracy(outputs, targets)
        top1.update(acc1[0], inputs.size(0))
    return top1.avg


def collectGradient(optimizer):
    gradient = list()
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            # get the gradient
            d_p = p.grad.data
            # operate the gradient according to weight_decay and momentum
            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)
            if momentum != 0:
                param_state = optimizer.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            gradient.append(d_p)
            # p.data.add_(-group['lr'], d_p)
    return gradient



class newSGD(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, maximize=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super(SGD, self).__init__(params, defaults)

    def step(self, gradient, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            for i, param in enumerate(params_with_grad):
                d_p = gradient[i]
                alpha = lr if maximize else -lr
                param.data.add_(d_p, alpha=alpha)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        return loss
