import numpy as np
import scipy.sparse as sp
import torch
import os
from .utils import *


def extract_subgraph(C, v, NE):
    # v here contains the indices of the two communities
    Lunion = tableunion(C[v[0]], C[v[1]])
    # extract subgraph
    LNE = {}
    LNC = {}
    for w in Lunion:
        # NE[w] contains all neighbors of w
        neighs = setNew(NE[w])
        sneighs = setintersection(neighs, Lunion)
        LNE[w] = settotable(sneighs)
        LNC[w] = [setContains(setNew(C[v[0]]), w),
                  setContains(setNew(C[v[1]]), w)]

    W, lab, inverse = convert_to_Tensor(LNE, LNC)
    return W, lab, LNC, inverse


def convert_to_Tensor(NE, NC):
    # this function receives a table of neighbors per node
    # and a table of labels.

    inverse = inverttablekeys(NE)
    N = tablelength(NE)
    W = torch.Tensor(N, N).zero_()
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

    return W, lab, inverse


def read_file(datagraphpath, datacommpath):
    file = None
    cfile = None

    print('datagraphpath: ', datagraphpath)
    print('datacommpath: ', datacommpath)

    N = {}  # Community in which belongs node w (N[w])
    NE = {}  # Neighbors of node w (NE[w])

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
                i = i + 1
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
                C.append([])
                for k in range(len(l)):
                    l[k] = l[k].strip()
                for val in l:
                    C[i].append(val)
                    N[val].append(i)
                Csize.append(tablelength(C[i]))
                Cavg = Cavg + Csize[i]
                i = i + 1
                if i % 100 == 0:
                    print(i)
    Cavg = Cavg / (i) if i > 0 else 0
    Cnumber = tablelength(C)
    print('average community size is ', Cavg)

    # 1st step: identify edges that cross communities.
    El = tablelength(E)
    cross = []
    for i in range(El):
        lun, ll1, ll2 = tableunion_size(N[E[i][0]], N[E[i][1]])
        if ll1 > 0 and ll2 > 0 and lun > ll1 and lun > ll2:
            cross.append(i)

        if i % 20000 == 0:
            print("step", 1)

    # 2nd step: for each cross edge, we want to grow a subgraph of limited
    # size.
    maxsubsize = 1000
    rho = 0.05  # maximum imbalance between communities
    alpha = 0.7  # fraction of communities reserved for training/testing
    cthres = alpha * Cnumber
    goodcross_train = []
    goodcross_test = []
    counter_train = 0
    counter_test = 0
    plate = torch.randperm(Cnumber)

    for v in cross:
        L1 = shallowcopy(N[E[v][0]])
        L2 = shallowcopy(N[E[v][1]])
        # pick one element in L1 not in L2 and viceversa
        cc1 = tablesampleg(L1, L2, Csize)
        cc2 = tablesampleg(L2, L1, Csize)
        for i1, j1 in cc1:
            for i2, j2 in cc2:
                if j1 + j2 < 2 * maxsubsize and j1 > rho * j2 and j2 > rho * \
                        j1 and plate[i1] < cthres and plate[i2] < cthres:
                    counter_train = counter_train + 1
                    goodcross_train.append([i1, i2])

                if j1 + j2 < 2 * maxsubsize and j1 > rho * j2 and j2 > rho * \
                        j1 and plate[i1] > cthres and plate[i2] > cthres:
                    counter_test = counter_test + 1
                    goodcross_test.append([i1, i2])

    trsize = counter_train  # - 1
    tesize = counter_test  # - 1
    print('total found clusts are (train: ', trsize, ' || test: ', tesize)
    return E, NE, N, C, goodcross_train, goodcross_test
