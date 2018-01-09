from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models import *

import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--batchSize', type=int, default=1, help='mini-batch size')
parser.add_argument('--maxepoch', type=int, default=24, help='epochs')
parser.add_argument('--maxepochsize', type=int, default=2000, help='epochs')
parser.add_argument('--maxtestsize', type=int, default=2000, help='epochs')
parser.add_argument('--path', type=str,
                    default='./experiments/', help='save episodes')
parser.add_argument('--graph', type=str, default='dblp', help='graph')
parser.add_argument('--data_dir', type=str, default='./', help='data_dir')
parser.add_argument('--gpunum', type=int, default=3)
parser.add_argument('--weightDecay', type=float, default=0)
parser.add_argument('--learningRate', type=float, default=0.001)
parser.add_argument('--learningRate_damping', type=float, default=0.75)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--learningRateDecay', type=float, default=0)
parser.add_argument('--epoch_step', type=int, default=1, help='epoch step')
parser.add_argument('--optmethod', type=str, default='adamax')
parser.add_argument('--nclasses', type=int, default=3)
parser.add_argument('--L', type=int, default=20000, help='epoch size')
parser.add_argument('--layers', type=int, default=10, help='input layers')
parser.add_argument('--nfeatures', type=int, default=10, help='feature maps')
parser.add_argument('--J', type=int, default=3,
                    help='scale of the extrapolation')
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--prefix', type=str, default='')

opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
# '.'  # path to the folder where corresponding graph files are
opt.datagraphpathroot = opt.data_dir + opt.graph + '/'
opt.datagraphpath = opt.datagraphpathroot + 'com-' + opt.graph + '.ungraph.txt'
opt.datacommpath = opt.datagraphpathroot + \
    'com-' + opt.graph + '.top5000.cmty.txt'


def file_exists(name):
    return os.path.isfile(name)


def updateConfusionMatrix(confusion, output, labels):
    preds = output.max(1)[1].type_as(labels)
    fp = preds.view(-1)
    fl = labels.view(-1)
    for i in range(fp.size(0)):
        confusion[fp[i].data[0], fl[i].data[0]] = confusion[
            fp[i].data[0], fl[i].data[0]] + 1

    return confusion


def accuracyFromConfusion(confusion):
    s = 0
    for i in range(confusion.size(0)):
        s = s + confusion[i, i]

    return s / confusion.sum()


featuremap_in = [1, 1, opt.nfeatures]
featuremap_mi = [2 * opt.nfeatures, 2 * opt.nfeatures, opt.nfeatures]
featuremap_end = [2 * opt.nfeatures, 2 * opt.nfeatures, 1]


np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# create model
model = GNNModular(
    featuremap_in,
    featuremap_mi,
    featuremap_end,
    opt.layers,
    opt.J,
    opt.nclasses,
    None
)

# create directory
date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
opt.dpath = opt.path + 'bs_' + \
    str(opt.batchSize) + '_J_' + str(opt.J) + '_' + date_time
if not os.path.exists(opt.dpath) or os.path.isfile(opt.dpath):
    os.makedirs(opt.dpath)
opt.pathbaselinetest = opt.datagraphpathroot + opt.graph + '_baseline/'


# Load data
if file_exists(os.path.join(opt.pathbaselinetest, 'traintestpartition.th')):
    YY = torch.load(
        os.path.join(
            opt.pathbaselinetest,
            'traintestpartition.th')
    )
    goodcross_train = YY["goodcross_train"]
    goodcross_test = YY["goodcross_test"]
    Edges = YY["Edges"]
    Neighbors = YY["Neighbors"]
    Inv_Community = YY["Inv_Community"]
    Community = YY["Community"]
else:
    (Edges, Neighbors, Inv_Community, Community, goodcross_train,
     goodcross_test) = read_file(opt.datagraphpath, opt.datacommpath)
    data_dict = {
        'goodcross_train': goodcross_train,
        'goodcross_test': goodcross_test,
        'Edges': Edges,
        'Neighbors': Neighbors,
        'Inv_Community': Inv_Community,
        'Community': Community
    }
    if os.path.isabs(opt.pathbaselinetest):
        os.makedirs(opt.pathbaselinetest, exist_ok=True)
    else:
        os.makedirs(os.path.abspath(opt.pathbaselinetest), exist_ok=True)
    torch.save(
        data_dict,
        os.path.join(
            opt.pathbaselinetest,
            'traintestpartition.th')
    )


trsize = tablelength(goodcross_train)
tesize = tablelength(goodcross_test)
print(trsize)
print(tesize)


optimState = {
    'lr': opt.learningRate,
    'weight_decay': opt.weightDecay,
    'momentum': opt.momentum,
    'lr_decay': opt.learningRateDecay,
}

# logging
if opt.nclasses == 2:
    classes = [0, 1]
else:
    classes = [0, 1, 2]

confusion = torch.zeros(opt.nclasses, opt.nclasses)
accLogger = open(os.path.join(opt.dpath, 'accuracy_train.log'), 'w')
acctLogger = open(os.path.join(opt.dpath, 'accuracy_test.log'), 'w')


def loadexample(ind, J, test=False):
    if not test:
        W, target, _, _ = extract_subgraph(
            Community, goodcross_train[ind], Neighbors)
    else:
        W, target, _, _ = extract_subgraph(
            Community, goodcross_test[ind], Neighbors)

    # W: is the adjacent Matrix of the subgraph
    # target:
    #   target[0] == 1: w belongs to the 2nd cmty and not the first
    #   target[1] == 1: w belongs to the 1st cmty and not the second
    #   target[2] == 1: w belongs to both the 1st and 2nd cmties.

    d = W.sum(0)  # degree of each node
    Dfwd = torch.diag(d)  # Diagonal Matrix based on the degree of each node
    QQ = W.clone()
    N = W.size(0)

    # I, W, W^2, W^4,...,W^(2^(J-2)), D, Avg_Degree
    WW = torch.Tensor(N, N, J + 2).fill_(0)
    WW[:, :, 0] = torch.eye(N)
    for j in range(J - 1):
        WW[:, :, j + 1].copy_(QQ)
        QQ = torch.mm(QQ, QQ)

    # D: Copy of the Diagonal matrix based on the degree of each node
    WW[:, :, J].copy_(Dfwd.view(N, N))

    # Avg_Degree: Average degree information
    WW[:, :, J + 1].fill_(1 / N)

    WW = WW.view(1, N, N, J + 2)
    inp = d.view(1, 1, N, 1)  # degree of each node

    target = Variable(target)
    WW = Variable(WW)
    inp = Variable(inp)

    if opt.cuda:
        target = target.cuda()
        WW = WW.cuda()
        inp = inp.cuda()

    return WW, inp, target


permuteprotect = IndexModule(1)
if opt.nclasses == 2:
    plate = Variable(torch.LongTensor([[0, 1], [1, 0]]))
    perms = 2
else:
    plate = Variable(torch.LongTensor([[0, 1, 2], [1, 0, 2]]))
    perms = 2

if opt.cuda:
    permuteprotect.cuda()
    plate = plate.cuda()

prho = 0.98  # running average factor (display purposes only)
running_avg = 0.0
running_avg_b = 0.0


# Model and optimizer
optimizer = None
optname = opt.optmethod.lower()
if optname == 'adam':
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimState['lr'],
        weight_decay=optimState['weight_decay'])
elif optname == 'adamax':
    optimizer = optim.Adamax(
        model.parameters(),
        lr=optimState['lr'],
        weight_decay=optimState['weight_decay'])
elif optname == 'sparseadam':
    optimizer = optim.SparseAdam(
        model.parameters(),
        lr=optimState['lr'])
elif optname == 'adagrad':
    optimizer = optim.Adagrad(
        model.parameters(),
        lr=optimState['lr'],
        weight_decay=optimState['weight_decay'],
        lr_decay=optimState['lr_decay'])
elif optname == 'adadelta':
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=optimState['lr'],
        weight_decay=optimState['weight_decay'])
elif optname == 'rmsprop':
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=optimState['lr'],
        weight_decay=optimState['weight_decay'],
        momentum=opt.momentum)
elif optname == 'rprop':
    optimizer = optim.Rprop(
        model.parameters(),
        lr=optimState['lr'])
elif optname == 'sgd':
    optimizer = optim.SGD(
        model.parameters(),
        lr=optimState['lr'],
        weight_decay=optimState['weight_decay'],
        momentum=opt.momentum)
else:
    optimizer = optim.Adamax(
        model.parameters(),
        lr=optimState['lr'],
        weight_decay=optimState['weight_decay'])

scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=opt.epoch_step, gamma=opt.learningRate_damping)

if opt.cuda:
    model.cuda()


def train(epoch=1):
    global confusion
    global running_avg

    t = time.time()
    tb = t

    model.train()

    shuffle = torch.randperm(trsize)

    totloss = 0
    numiters = min(trsize, opt.maxepochsize)
    for l in range(0, numiters):

        ii = []
        Wtmp, inp, target = loadexample(shuffle[l], opt.J, False)

        optimizer.zero_grad()
        pred = model(inp, Wtmp)
        predt = pred.clone()
        laball = Variable(torch.LongTensor(inp.size(2), perms).zero_())
        losses = Variable(torch.Tensor(perms))
        if opt.cuda:
            laball = laball.cuda()
            losses = losses.cuda()
        # eval predictions against permuted labels
        for s in range(perms):
            tper = target[:, plate[s].data].clone()
            _, labtmp = torch.max(tper, 1)
            loss = F.cross_entropy(predt, labtmp)
            laball[:, s] = labtmp
            losses[s] = loss

        lmin, lpos = torch.min(losses, 0)
        running_avg = prho * running_avg + (1 - prho) * lmin[0]

        loss = F.cross_entropy(pred, laball[:, lpos[0].data].squeeze(1))
        totloss = totloss + lmin[0]
        confusion = updateConfusionMatrix(
            confusion, pred.clone(), laball[:, lpos[0].data].squeeze(1))
        if l % 50 == 0:
            print('##### Epoch {}: Iter {}/{} ####'.format(epoch, l, numiters))
            print('running_avg: ', running_avg)
            print('confusion: ', confusion)
            tb1 = time.time()
            print('time: {:.4f}s ({:.4f}s/batch)'.format(
                (tb1 - tb), (tb1 - tb) / 50)
            )
            tb = tb1

        loss.backward()
        optimizer.step()

    print('confusion: ', confusion)
    trainAccuracy = accuracyFromConfusion(confusion) * 100

    print('Epoch[{}] Train loss {}'.format(epoch, totloss[0] / numiters),
          'acc_train: {:.4f}'.format(trainAccuracy),
          'time: {:.4f}s'.format(time.time() - t))

    confusion.zero_()

    epoch = epoch + 1

    # drop learning rate every "epoch_step" epochs
    scheduler.step()

    return trainAccuracy


def test(epoch=1):
    global confusion

    t = time.time()
    tb = t

    model.eval()

    numiters = min(tesize, opt.maxtestsize)
    shuffle = torch.randperm(numiters)

    totloss = 0
    for l in range(0, numiters):

        Wtmp, inp, target = loadexample(shuffle[l], opt.J, True)

        pred = model(inp, Wtmp)
        losses = Variable(torch.Tensor(perms))
        laball = Variable(torch.LongTensor(inp.size(2), perms).zero_())
        if opt.cuda:
            laball = laball.cuda()
            losses = losses.cuda()
        for s in range(perms):
            tper = target[:, plate[s].data].clone()
            _, labtmp = torch.max(tper, 1)
            loss = F.cross_entropy(pred, labtmp)
            laball[:, s] = labtmp
            losses[s] = loss
        lmin, lpos = torch.min(losses, 0)
        confusion = updateConfusionMatrix(
            confusion, pred, laball[:, lpos[0].data].squeeze(1))
        totloss = totloss + lmin[0]
        if l % 50 == 0:
            print('##### Epoch {}: Iter {}/{} ####'.format(epoch, l, numiters))
            print('confusion: ', confusion)
            tb1 = time.time()
            print('time: {:.4f}s ({:.4f}s/batch)'.format(
                (tb1 - tb), (tb1 - tb) / 50)
            )
            tb = tb1

    print(confusion)
    testAccuracy = accuracyFromConfusion(confusion) * 100

    print('Epoch[{}] Test loss {}'.format(epoch, totloss[0] / numiters),
          'acc_test: {:.4f}'.format(testAccuracy),
          'time: {:.4f}s'.format(time.time() - t))

    confusion.zero_()

    return testAccuracy


# Train model
t_total = time.time()

for jj in range(opt.maxepoch):
    train_acc = train(jj)
    accLogger.write('{:.4f}\n'.format(train_acc))
    test_acc = test(jj)
    acctLogger.write('{:.4f}\n'.format(test_acc))

accLogger.close()
acctLogger.close()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
