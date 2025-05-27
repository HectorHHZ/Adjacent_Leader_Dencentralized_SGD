import copy

import numpy as np
import time
import torch
from mpi4py import MPI
from compressors import get_top_k
import torch.distributed as dist
from comm_helpers import flatten_tensors, unflatten_tensors


class Communicator(object):
    """ Classs designed for communicating local models at workers """

    def __init__(self, rank, size):
        # mpi4py function
        self.comm = MPI.COMM_WORLD
        #self.comm = torch.distributed
        self.rank = rank
        self.size = size
        self.iter = 0

    def communicate(self, model):
        self.iter += 1
        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocessing
        self.prepare_comm_buffer()

        # communication happens here
        # record the communication time
        comm_time = self.averaging()

        # Update local models
        new_model = self.reset_model(model)
        return comm_time, new_model


    def reset_model(self, model):
        # Reset local models to be the averaged model
        #  (CPU Version: CUDA function not allowed in cpu)
        self.updatedvec = unflatten_tensors(self.recv_buffer.cuda(), self.tensor_list)
        for f, t in zip(self.updatedvec, self.tensor_list):
            #t.set_(f)
            t=f
        # test model update or not
        tmp_model1 = copy.deepcopy(model)
        pointer = 0
        for param in model.parameters():
            param.data = self.updatedvec[pointer].view(param.size())
            pointer += 1

        # test model update or not
        tmp_model2 = copy.deepcopy(model)

        modelCheck = True

        for p1, p2 in zip(tmp_model1.parameters(), tmp_model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                modelCheck = False

        if modelCheck == True:
            print('same, model not change')
        else:
            print('different, model updated')

        return model

    def prepare_comm_buffer(self):
        raise NotImplemented

    def averaging(self):
        raise NotImplemented


class PScommunicator(Communicator):
    def __init__(self, rank, size):
        super(PScommunicator, self).__init__(rank, size)

    def prepare_comm_buffer(self):
        self.send_buffer = flatten_tensors(self.tensor_list)
        self.recv_tmp = torch.zeros_like(self.tensor_list)
        self.recv_buffer = torch.zeros_like(self.tensor_list)

    def averaging(self):
        self.comm.barrier()
        tic = time.time()

        # send parameters - Method 1: using send and receive to collect parameters
        # mpi4py code
        #send_Seq = self.comm.Send(self.send_buffer, dest = 0)

        # Send Parameters - Method2: using All gather and Broadcast to collect parameters

        self.comm.barrier()
        toc = time.time()
        return toc - tic

    def averagingPS(self):
        self.comm.barrier()
        tic = time.time()
        """
        # receive parameters -  Method 1: using send and receive to collect parameters
        for i in range(1, self.size - 1):
            # receive the send_buff_list[leftNode]
            # (torch.distributed code)
            # self.comm.recv(self.recv_tmp, src=i)
            # mpi4py code
            self.comm.Recv(self.recv_tmp, source=i)
            # buffer sum up
            self.recv_buffer += self.recv_tmp
        
        """
        # receive parameters -  Method 2: using All gather and Broadcast to collect parameters
        self.comm.allreduce(self.send_buffer, op=MPI.SUM)

        self.recv_buffer = self.send_buffer
        self.recv_buffer /= float(self.size)
        # Braodcast signal
        # mpi4py code
        # self.comm.broadcast(tensor=self.recv_buffer, src=0)
        # torch.distributed code
        # mpi4py code
        self.comm.Bsend(self.recv_buffer, src = 0)
        self.comm.barrier()
        toc = time.time()
        return toc - tic

    def communicate(self, model):
        self.iter += 1
        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocessing
        self.prepare_comm_buffer()

        # communication happens here
        # record the communication time
        comm_time = self.averaging()

        # Update local models
        new_model = self.reset_model(model)

        return comm_time, new_model

    def communicatePS(self, model):
        self.iter += 1
        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocessing
        self.prepare_comm_buffer()

        # communication happens here
        # record the communication time
        comm_time = self.averagingPS()

        # reset local models
        self.recv_buffer = torch.zeros_like(self.tensor_list)

        return comm_time, self.updatedvec



class ringCommunicator(Communicator):
    def __init__(self, rank, size):
        super(ringCommunicator, self).__init__(rank, size)

    def prepare_comm_buffer(self):

        self.send_buffer = flatten_tensors(self.tensor_list)
        self.send_buffer_list = self.send_buffer.chunk(self.size)
        self.recv_tmp = torch.zeros_like(self.send_buffer_list[0])
        self.recv_buffer = self.send_buffer

    def averaging(self):
        """
        info:
        self.rank: from 0 to 7
        self.size: 8

        verify:
        # GPU 0
        # left = (0 - 1 + 8) % 8 = 7
        # right = (0 + 1) % 8 = 1
        # iteration i = 0:
        # chunk_num = (8 - 0 + 0 - 1) % 8 = 7
        # left_chunk_num = (8 - 7 + 0 - 1) % 8 = 0
        # iteration i = 1:
        # chunk_num = (8 - 0 + 0 + 1 - 1) % 8 = 0
        # left_chunk_num = (8 - 7 + 1 - 1) % 8 = 1
        verified

        :return communication time
        """

        self.comm.barrier()
        tic = time.time()

        left = ((self.rank - 1) + self.size) % self.size
        right = (self.rank + 1) % self.size
        # convert into list for add operation
        self.send_buffer_list = list(self.send_buffer_list)
        for i in range(self.size - 1):
            # get the chunk number of current nodes/ left nodes for each iteration
            chunk_num = (self.size - self.rank + i - 1) % self.size
            left_chunk_num = (self.size - left + i - 1) % self.size
            print("This is rank", self.rank, "The world size is", self.size)
            print("The chunk number going to send:", chunk_num, "The chunk number going to receive:", left_chunk_num)
            print("The total number of chunk is:", len(self.send_buffer_list))

            # communicate using sendrecv
            self.recv_tmp = self.comm.sendrecv(self.send_buffer_list[chunk_num], source=left, dest=right)
            self.send_buffer_list[left_chunk_num] += self.recv_tmp

            # communicate using send and recv
            """ 
            if (self.rank % 2 == 0):
                self.recv_tmp = self.comm.sendrecv(self.send_buffer_list[chunk_num], source=left, dest=right, sendtag=0, recvtag=1)
            else:
                self.recv_tmp = self.comm.recv(source= left, tag=0)
                send_seq = self.comm.send(self.send_buffer_list[chunk_num], dest = right, tag=1)
            """

            # ring allreduce sum up
            self.send_buffer_list[left_chunk_num] += self.recv_tmp

        # convert the list into tuple for
        self.send_buffer_list = tuple(self.send_buffer_list)
        # dechunk to flatten tensor
        self.recv_buffer = torch.cat(self.send_buffer_list[:], 0)
        self.recv_buffer /= float(self.size)

        self.comm.barrier()
        toc = time.time()
        return toc - tic




class centralizedCommunicator(Communicator):
    """ Perform AllReduce at each iteration """

    def __init__(self, rank, size):
        super(centralizedCommunicator, self).__init__(rank, size)

    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)

    def averaging(self):
        self.comm.barrier()
        tic = time.time()

        # AllReduce
        #   (mpi4py function)
        self.recv_buffer = self.comm.allreduce(self.send_buffer, op=MPI.SUM)
        #   (torch.distributed function)
        #self.recv_buffer = self.comm.all_reduce(self.send_buffer, op=ReduceOp.SUM)
        #   (mpi4py function)
        self.recv_buffer.div_(self.size)
        #self.recv_buffer /= float(self.size)

        self.comm.barrier()
        toc = time.time()

        return toc - tic


class decenCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """

    def __init__(self, rank, size, topology):
        super(decenCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

    def prepare_comm_buffer(self):
        # faltten tensors
        #   (.cpu() is used in GPU devices)

        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        #self.send_buffer = flatten_tensors(self.tensor_list)
        self.recv_tmp = torch.zeros_like(self.send_buffer)
        self.recv_buffer = torch.zeros_like(self.send_buffer)
        self.updatedvec = self.tensor_list

    def averaging(self, active_flags):

        self.comm.barrier()
        tic = time.time()

        # decentralized averaging
        degree = 0  # record the degree of each node
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                    print("The rank is:", self.rank, "The neighbor rank is:", neighbor_rank)
                    print("Information going to send:", self.send_buffer)
                    # Receive neighbor's model: x_j
                    #   (mpi4py code)
                    #print("start send and receive step")
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest=neighbor_rank)

                    #   (torch.distributed code)
                    # send_seq = self.comm.isend(self.send_buffer, dst=neighbor_rank)
                    #print("finish send and receive step")
                    #recv_seq = self.comm.recv(self.recv_tmp, src=neighbor_rank)
                    #print("finish receive step")
                    #send_seq.wait()
                    #print("finish send wait")
                    #recv_seq.wait()
                    #print("finish receive wait")


                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer.add_(self.neighbor_weight, self.recv_tmp)
                    #self.recv_buffer += self.recv_tmp * self.neighbor_weight

        #print("start average step")
        # compute self weight according to degree
        selfweight = 1 - degree * self.neighbor_weight
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(selfweight, self.send_buffer)

        self.comm.barrier()
        toc = time.time()
        #print("finish average step")
        return toc - tic

    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging(active_flags)

        # update local models
        new_model = self.reset_model(model)

        return comm_time, new_model


class ChocoCommunicator(Communicator):
    """ decentralized averaging using compressed gradients (top-k)
    SOTA Paper 2019: Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication
    """

    def __init__(self, rank, size, topology, ratio, consensus_lr):
        super(ChocoCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

        self.initialized = False
        self.consensus_lr = consensus_lr
        self.ratio = ratio

    def prepare_comm_buffer(self):
        # flatten tensors
        # If not initialized, then initialize x_hat and s
        self.x = flatten_tensors(self.tensor_list).cpu()
        if not self.initialized:
            self.x_hat = torch.zeros_like(self.x)
            self.s = torch.zeros_like(self.x)
            self.initialized = True

        tic = time.time()
        # get compressed message
        # here, we use top_k compressor on GPU
        # one can define more in compressors.py
        self.send_buffer = self.x - self.x_hat
        values, indices = get_top_k(self.send_buffer.cuda(), self.ratio)
        toc = time.time()

        values, indices = values.cpu(), indices.cpu()
        self.compressed = {"values": values, "indices": indices}

        return toc - tic

    def averaging(self, active_flags):
        self.comm.barrier()
        tic = time.time()

        # decentralized averaging according to activated topology
        degree = 0
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                    # Receive neighbor's message q_j
                    #   (mpi4py code)
                    self.recv_tmp = self.comm.sendrecv(self.compressed, source=neighbor_rank, dest=neighbor_rank)
                    #send_seq = self.comm.isend(self.compressed, dst=neighbor_rank)
                    #send_seq.wait()
                    #recv_seq = self.comm.recv(self.recv_tmp, src=neighbor_rank)

                    # Update aggregated model s += sum w_ij q_j
                    self.s[self.recv_tmp["indices"]] += self.neighbor_weight * self.recv_tmp["values"]

        # Compute self weight
        selfweight = 1 - degree * self.neighbor_weight
        # Update aggregated model s += w_ii q_i
        self.s[self.compressed["indices"]] += selfweight * self.compressed["values"]
        # Update x_hat = x_hat + q_i
        self.x_hat[self.compressed["indices"]] += self.compressed["values"]
        # Update local model parameters: x = x + consensus_lr*(s-x_hat)
        self.x.add_(self.consensus_lr, self.s).sub_(self.consensus_lr, self.x_hat)

        self.comm.barrier()
        toc = time.time()

        return toc - tic



    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocess
        # there is an additional encoding time
        encode_time = self.prepare_comm_buffer()

        # decentralized averaging
        # record the communication time
        comm_time = self.averaging(active_flags)
        total_commtime = encode_time + comm_time
        # update local models
        new_model = self.reset_model(model)

        return total_commtime, new_model

    def reset_model(self, model):
        # Reset local models to be the averaged model
        #  (CPU Version: CUDA function not allowed in cpu)
        self.updatedvec = unflatten_tensors(self.x.cuda(), self.tensor_list)
        for f, t in zip(self.updatedvec, self.tensor_list):
            #t.set_(f)
            t=f
        # test model update or not
        tmp_model1 = copy.deepcopy(model)
        pointer = 0
        for param in model.parameters():
            param.data = self.updatedvec[pointer].view(param.size())
            pointer += 1

        # test model update or not
        tmp_model2 = copy.deepcopy(model)

        modelCheck = True

        for p1, p2 in zip(tmp_model1.parameters(), tmp_model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                modelCheck = False

        if modelCheck == True:
            print('same, model not change')
        else:
            print('different, model updated')

        return model



class LLDSGDCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """

    def __init__(self, rank, size, topology):
        super(LLDSGDCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

    def prepare_comm_buffer(self):
        # faltten tensors
        #   (.cpu() is used in GPU devices)
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        #self.send_buffer = flatten_tensors(self.tensor_list)
        self.recv_tmp = torch.zeros_like(self.send_buffer)
        self.recv_buffer = torch.zeros_like(self.recv_tmp)
        self.updatedvec = self.tensor_list



    def averaging(self, active_flags):

        self.comm.barrier()
        tic = time.time()

        # decentralized averaging
        degree = 0  # record the degree of each node
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                    print("The rank is:", self.rank, "The neighbor rank is:", neighbor_rank)
                    print("Information going to send:", self.send_buffer)
                    # Receive neighbor's model: x_j
                    #   (mpi4py code)
                    #print("start send and receive step")
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest=neighbor_rank)

                    #   (torch.distributed code)
                    # send_seq = self.comm.isend(self.send_buffer, dst=neighbor_rank)
                    #print("finish send and receive step")
                    #recv_seq = self.comm.recv(self.recv_tmp, src=neighbor_rank)
                    #print("finish receive step")
                    #send_seq.wait()
                    #print("finish send wait")
                    #recv_seq.wait()
                    #print("finish receive wait")


                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer.add_(self.neighbor_weight, self.recv_tmp)
                    # self.recv_buffer += self.recv_tmp * self.neighbor_weight

        #print("start average step")
        # compute self weight according to degree
        selfweight = 1 - degree * self.neighbor_weight
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(selfweight, self.send_buffer)

        self.comm.barrier()
        toc = time.time()
        #print("finish average step")
        return toc - tic

    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)


        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging(active_flags)

        # update local models
        new_model = self.reset_model(model)

        return comm_time, new_model

    def LLDSGDaveraging(self, active_flags, loss, args):
        # store the loss and degree (LLDSGD)
        loss_list = [loss.tolist()]
        degree_List = list()

        self.comm.barrier()
        tic = time.time()

        # decentralized averaging
        degree = 0  # record the degree of each node
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1

        # store the model and degree
        self.tensor_list[-1] = torch.tensor([degree]).cuda()
        degree_List.append(degree)
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        models = [self.send_buffer.tolist()]


        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                    print("The rank is:", self.rank, "The neighbor rank is:", neighbor_rank)
                    print("Information going to send:", self.send_buffer)
                    # Receive neighbor's model: x_j
                    #   (mpi4py code)
                    #print("start send and receive step")
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest=neighbor_rank)

                    # store the neighbor model
                    models.append(self.recv_tmp.tolist())
                    loss_list = loss_list + [self.recv_tmp[-2].tolist()]
                    degree_List = degree_List + [self.recv_tmp[-1].tolist()]


                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer.add_(self.neighbor_weight, self.recv_tmp)
                    #self.recv_buffer += self.recv_tmp * self.neighbor_weight

        print("start average step")
        # compute self weight according to degree
        selfweight = 1 - degree * self.neighbor_weight

        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(selfweight, self.send_buffer)

        # LLDSGD pull force to local best neighbor. - 0.1 the pull force parameter
        print("lenth of lost is :", len(loss_list))
        if (len(loss_list) > 1):
            # get min_loss and max_degree
            min_loss = min(loss_list)
            max_degree = max(degree_List)

            # index for min_loss and max_degree
            min_index = loss_list.index(min_loss)
            maxdegree_index = degree_List.index(max_degree)

            # get model with max degree and min loss
            best_local_model = models[min_index]
            max_degree_model = models[maxdegree_index]

            best_local_model = torch.FloatTensor(best_local_model)
            max_degree_model = torch.FloatTensor(max_degree_model)


            # best_worker_force = torch.mul(torch.sub(self.recv_buffer, best_local_model), args.p1)
            # max_degree_force = torch.mul(torch.sub(self.recv_buffer, max_degree_model), args.p2)
            # self.recv_buffer = torch.sub(self.recv_buffer, torch.add(max_degree_force, best_worker_force))
            # print("add additional step from local leader and max degree worker")
###
            self.recv_buffer = torch.mul(self.recv_buffer, 1 - args.c1 - args.c2)
            self.recv_buffer += torch.mul(best_local_model, args.c1)
            self.recv_buffer += torch.mul(max_degree_model, args.c2)
            print("Add pull force from local leader and max degree worker")

            best_worker_force = torch.mul(torch.sub(self.recv_buffer, best_local_model), args.p1)
            max_degree_force = torch.mul(torch.sub(self.recv_buffer, max_degree_model), args.p2)
            self.recv_buffer = torch.sub(self.recv_buffer, torch.add(max_degree_force, best_worker_force))
            print("add additional step from local leader and max degree worker")

        self.comm.barrier()
        toc = time.time()
        #print("finish average step")
        return toc - tic



    def LLDSGDcommunicate(self, model, loss, args):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        # stack all model parameters into one tensor list

        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)
        # append loss.item() to the end of buffer
        self.tensor_list.append(loss)
        # place reserved for degree
        self.tensor_list.append(torch.tensor([1.0]).cuda())

        # necessary preprocess
        self.prepare_comm_buffer()

        # loss not included in model
        self.updatedvec = self.tensor_list[:-2]


        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.LLDSGDaveraging(active_flags, loss, args)

        # update local models
        new_model = self.LLSGD_reset_model(model)

        return comm_time, new_model


    def LLSGD_reset_model(self, model):
        # Reset local models to be the averaged model
        #  (CPU Version: CUDA function not allowed in cpu)
        self.updatedvec = unflatten_tensors(self.recv_buffer[:-2].cuda(), self.tensor_list[:-2])
        print("reset model according to leader update")

        # for f, t in zip(self.updatedvec, self.tensor_list[:-2]):
        #     #t.set_(f)
        #     t=f

        # test model update or not
        tmp_model1 = copy.deepcopy(model)
        pointer = 0
        for param in model.parameters():
            param.data = self.updatedvec[pointer].view(param.size())
            pointer += 1

        # test model update or not
        tmp_model2 = copy.deepcopy(model)

        modelCheck = True

        for p1, p2 in zip(tmp_model1.parameters(), tmp_model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                modelCheck = False

        if modelCheck == True:
            print('same, model not change')
        else:
            print('different, model updated')

        return model
