import numpy as np
import scipy.sparse as sp
import torch
import os
from .utils import *


def extract_subgraph(C, v, NE):
    # C:  Communuties (C[i] is a list of all nodes belonging to cmty C[i])
    # v:  Contains the indices of the two communities
    # NE: Node-based dict - Neighbors of node w (NE[w])

    Lunion = tableunion(C[v[0]], C[v[1]])  # Union of the nodes of the 2 cmties
    # extract subgraph
    LNE = {}  # Node-based dict - Neighbors in the subgraph to be contrusted
    LNC = {}  # Node-based dict - Indicator for nodes belonging to the 2 cmties
    for w in Lunion:
        # NE[w] contains all neighbors of w
        neighs = setNew(NE[w])

        # consider only neighbors beloging to Lunion.
        sneighs = setintersection(neighs, Lunion)

        # set the new neighborhood
        LNE[w] = settotable(sneighs)

        # wheter or not w belongs to the two communities (indicators)
        LNC[w] = [setContains(setNew(C[v[0]]), w),
                  setContains(setNew(C[v[1]]), w)]

    W, lab, inverse = convert_to_Tensor(LNE, LNC)
    return W, lab, LNC, inverse


def convert_to_Tensor(NE, NC):
    # this function receives a table of neighbors per node
    # and a table of labels:
    # NE: Node-based dict - Neighbors of node w (NE[w])
    # NC: # Node-based dict - Target Indicators (boolean) for nodes
    #                         belonging to the 2 cmties (NC[w])

    # associate new id (from 0 to len(NE)) to the node
    # inverse[w] = i where i belongs to [0, len(ne)]
    inverse = inverttablekeys(NE)
    N = tablelength(NE)

    # Ajacent Matrix of the subgraph
    W = torch.Tensor(N, N).zero_()

    # Labels:
    #   lab[0] == 1: w belongs to the 2nd cmty and not the first
    #   lab[1] == 1: w belongs to the 1st cmty and not the second
    #   lab[2] == 1: w belongs to both the 1st and 2nd cmties.
    lab = torch.Tensor(N, 3).zero_()

    for v, r in pairs(NE):
        # r contains the list of neighbs
        for _, s in pairs(r):
            W[inverse[v]][inverse[s]] = 1

        if NC[v][0] == True and NC[v][1] == True:
            lab[inverse[v]][2] = 1
        elif NC[v][0] == True and NC[v][1] == False:
            lab[inverse[v]][1] = 1
        else:
            lab[inverse[v]][0] = 1

    # W: is the adjacent Matrix of the subgraph
    # lab: labels
    # inverse: node new ids (from 0 to N-1)
    return W, lab, inverse


def read_file(datagraphpath, datacommpath):
    file = None
    cfile = None

    # Node & Edge file:
    #   Each line contains en edge described as: FromNodeId ToNodeId
    print('datagraphpath: ', datagraphpath)

    # Community Ground truth file:
    #   Each line i contains a list of all nodes belonging to the ith community
    print('datacommpath: ', datacommpath)

    N = {}   # Node-based dict - Communities in which belongs node w (N[w])
    NE = {}  # Node-based dict - Neighbors of node w (NE[w])

    E = []  # Edges (E[i] is a list of nodes which are part of edge i)
    i = 0

    if os.path.isfile(datagraphpath):
        with open(datagraphpath) as f:
            for line in f:
                if line.strip().startswith('#'):
                    continue
                l = line.split("\t")
                E.append([])
                for k in range(len(l)):
                    l[k] = l[k].strip()
                for val in l:
                    E[i].append(val)
                    N[val] = []
                if not (l[0] in NE):
                    NE[l[0]] = []
                if not (l[1] in NE):
                    NE[l[1]] = []
                NE[l[0]].append(l[1])
                NE[l[1]].append(l[0])
                i = i + 1  # next edge
                if i % 20000 == 0:
                    print(i)

    C = []  # Communuties (C[i] is a list of all nodes belonging to cmty C[i])
    Csize = []  # community size (Csize[i] refers to the size of C[i])
    Cavg = 0
    i = 0
    if os.path.isfile(datacommpath):
        with open(datacommpath) as f:
            for line in f:
                if line.strip().startswith('#'):
                    continue
                l = line.split("\t")
                C.append([])  # Current community
                for k in range(len(l)):
                    l[k] = l[k].strip()
                for val in l:
                    C[i].append(val)
                    N[val].append(i)
                Csize.append(tablelength(C[i]))
                Cavg = Cavg + Csize[i]
                i = i + 1  # next community
                if i % 100 == 0:
                    print(i)
    Cavg = Cavg / (i) if i > 0 else 0
    Cnumber = tablelength(C)  # Number of communities
    print('average community size is ', Cavg)

    # 1st step: identify edges that cross communities.
    El = tablelength(E)  # number of edges
    cross = []  # contains edges that cross communities
    for i in range(El):
        # len(Union(cmty[nod0], cmty[nod1])), len(cmty[nod0]), len(cmty[nod1])
        lun, ll1, ll2 = tableunion_size(N[E[i][0]], N[E[i][1]])
        if ll1 > 0 and ll2 > 0 and lun > ll1 and lun > ll2:
            # The nodes composing this edge does not belong
            # to the same communities.
            # There exist a cmty of Nod0 in which Nod1 does not belong to
            # and vice-versa
            cross.append(i)

        if i % 20000 == 0:
            print("step", 1)

    # 2nd step: for each cross edge, we want to grow a subgraph of limited
    # size (maxsize of maxsubsize).
    maxsubsize = 1000
    rho = 0.05  # maximum imbalance between communities
    alpha = 0.7  # fraction of communities reserved for training/testing
    cthres = alpha * Cnumber
    goodcross_train = []
    goodcross_test = []
    counter_train = 0
    counter_test = 0
    plate = torch.randperm(Cnumber)  # random permutation from 0 to Cnumber-1

    for v in cross:  # Go through all crossing edges
        L1 = shallowcopy(N[E[v][0]])  # communities in which belongs Nod0
        L2 = shallowcopy(N[E[v][1]])  # communities in which belongs Nod1

        # compute iterators over elements in L1 not in L2 and viceversa
        cc1 = tablesampleg(L1, L2, Csize)
        cc2 = tablesampleg(L2, L1, Csize)
        for i1, j1 in cc1:  # i1 is a community, j1 is the size of cmty i1
            for i2, j2 in cc2:  # i2 is a community, j2 is the size of cmty i2
                if j1 + j2 < 2 * maxsubsize and j1 > rho * j2 and j2 > rho * \
                        j1 and plate[i1] < cthres and plate[i2] < cthres:
                    # if the number of nodes involved in the two cmties is less
                    # than 2*maxsubsize, and imbalance conditions are verified,
                    # and train/test condition is verified
                    #   (plate[i1] < cthres and plate[i2] < cthres), then
                    #  keep these communities within the train elements
                    counter_train = counter_train + 1
                    goodcross_train.append([i1, i2])

                if j1 + j2 < 2 * maxsubsize and j1 > rho * j2 and j2 > rho * \
                        j1 and plate[i1] > cthres and plate[i2] > cthres:
                    # if the number of nodes involved in the two cmties is less
                    # than 2*maxsubsize, and imbalance conditions are verified,
                    # and train/test condition is verified
                    #   (plate[i1] > cthres and plate[i2] > cthres), then
                    #  keep these communities within the test elements
                    counter_test = counter_test + 1
                    goodcross_test.append([i1, i2])

    trsize = counter_train  # - 1
    tesize = counter_test  # - 1
    print('total found clusts are (train: ', trsize, ' || test: ', tesize)
    return E, NE, N, C, goodcross_train, goodcross_test
