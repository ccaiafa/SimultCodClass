from __future__ import print_function

import argparse
import torch
import time as time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
#sys.path.append("../../../../../CIFAR10")
#sys.path.append("../../../CIFAR10new")

import mycifar
import pickle

from models import *
from utils import progress_bar

# Convert matrix indices to linear indices
def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

# Generate mask of missing entries
# 0: missing
# 1: available
def gen_mask(M, perc):
    mask = torch.ones(np.prod(M))
    ind = torch.randperm(np.int(np.prod(M)))
    ind = ind[0:round(perc*np.int(np.prod(M)))-1]
    mask[ind] = 0
    mask = mask.view(M[0], M[1], M[2], M[3])
    return mask


def train(args, model, Stot, mask_tot, device, train_loader, optimizer, epoch):
    K = model.linear.weight.data.shape[0]
    train_lossTot = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    correct = 0
    total = 0
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        model.train()
        S = Stot[:,idx]
        mask = mask_tot[idx, :, :]
        loc = sub2ind([data.shape[0], K],
                      torch.tensor(range(data.shape[0]), device=device),
                      target.to(device))
        data, target, S, mask = data.to(device), target.to(device), S.to(device), mask.to(device)

        # Optimize W, b and D
        optimizer.zero_grad()

        s, xap = model(S.t())
        s = s.view(s.numel())
        logprobs = s[loc]

        Loss_1 = torch.mean(-logprobs)
        diff = mask.view(-1, 32*32*3).t() * (xap.t() - data.squeeze().view(-1, 32*32*3).t())
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0] * (1.0 - args.missing_data_perc))

        loss = Loss_1 + Loss_2
        loss.backward()
        optimizer.step() # Here is where W, b and D are updated based on Loss1 + Loss2

        # Update coefficients S
        model.eval()
        S.requires_grad=True
        J = S.shape[0]
        # Normalize Dictionary columns
        col_sqr_norms = torch.sqrt(torch.sum(model.decoder.weight * model.decoder.weight, 0))
        model.decoder.weight.data = model.decoder.weight / col_sqr_norms.unsqueeze(0).expand(model.decoder.weight.shape[0],-1)   # columnwise normalization

        optimizer.zero_grad()

        s, xap = model(S.t())
        s = s.view(s.numel())

        # Compute probabilities of correct classes
        logprobs = s[loc]
        Loss_1 = torch.mean(-logprobs)
        diff = mask.view(-1, 32*32*3).t() * (xap.t() - data.squeeze().view(-1, 32*32*3).t())
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0]*(1.0-args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]

        Loss = Loss_1 + Loss_2 + Loss_3
        Loss.backward()

        dLdS = S.grad

        z = -args.gradientS_step_train * dLdS

        if args.l1_reg > 0:
            z[S * (S + z) < 0] = -S[torch.squeeze(S * (S + z) < 0)]

        with torch.no_grad():
            S = S + z

        S.requires_grad = False

        # print current results
        s, xap = model(S.t())
        predicted = s.max(1, keepdim=False)[1]
        s = s.view(s.numel())
        logprobs = s[loc]
        Loss_1 = torch.mean(-logprobs)
        # Compute losses
        diff = mask.view(-1, 32*32*3).t() * (xap.t() - data.squeeze().view(-1, 32*32*3).t())
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff*diff) / (data.shape[0]*(1.0-args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]

        Loss = Loss_1 + Loss_2 + Loss_3

        train_lossTot += Loss.item()
        train_loss1 += Loss_1.item()
        train_loss2 += Loss_2.item()
        train_loss3 += Loss_3.item()

        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(train_loader), 'L1: %.2f L2: %.2f L3: %.2f Loss: %.2f (%d/%d)'
            % (train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1), train_lossTot/(batch_idx+1), correct, total))

        # if batch_idx % args.log_interval == 0:
        #    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss1(class): {:.6f}\tLoss2(repres): {:.6f}\tLoss3 (sparsity): {:.6f}\tLossTot (1+2+3): {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader),
        #        float(Loss_1), float(Loss_2), float(Loss_3),
        #        float(Loss_1 + Loss_2 + Loss_3)))
        print(correct/total)

        Stot[:, idx] = S.to("cpu")

    return Stot, correct/total

def optimize_S(Stot, mask_tot, model, loaderS, args, device, epoch):
    model.eval()
    train_lossTot = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    correct = 0
    total = 0
    K = model.linear.weight.data.shape[0]
    print("updating coefficients ...")
    for batch_idx, (data, target, idx) in enumerate(loaderS):
        S = Stot[:,idx]
        mask = mask_tot[idx,:,:]
        data, target, S, mask = data.to(device), target.to(device), S.to(device), mask.to(device)

        loc = sub2ind([data.shape[0], K],
                      torch.tensor(range(data.shape[0]), device=device),
                      target.to(device))
        J = S.shape[0]
        S.requires_grad = True

        s, xap = model(S.t())
        s = s.view(s.numel())
        # Compute probabilities of correct classes
        logprobs = s[loc]
        Loss_1 = torch.mean(-logprobs)
        diff = mask.view(-1, 32 * 32 * 3).t() * (xap.t() - data.squeeze().view(-1, 32 * 32 * 3).t())
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0]*(1.0-args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]

        Loss = Loss_2 + Loss_3
        Loss.backward()

        dLdS = S.grad

        z = -args.gradientS_step_test * dLdS

        if args.l1_reg > 0:
            z[S * (S + z) < 0] = -S[torch.squeeze(S * (S + z) < 0)]

        with torch.no_grad():
            S = S + z

        S.requires_grad = False

        # print current results
        s, xap = model(S.t())
        predicted = s.max(1, keepdim=False)[1]
        s = s.view(s.numel())
        logprobs = s[loc]
        Loss_1 = torch.mean(-logprobs)
        # Compute losses
        diff = mask.view(-1, 32 * 32 * 3).t() * (xap.t() - data.squeeze().view(-1, 32 * 32 * 3).t())
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff*diff) / (data.shape[0]*(1.0-args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]

        Loss = Loss_2 + Loss_3

        train_lossTot += Loss.item()
        train_loss2 += Loss_2.item()
        train_loss3 += Loss_3.item()

        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(loaderS), 'L2: %.2f L3: %.2f Loss: %.2f (%d/%d)'
           % (train_loss2/(batch_idx+1), train_loss3/(batch_idx+1), train_lossTot/(batch_idx+1), correct, total))

        #if batch_idx % args.log_interval == 0:
        #    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss1(class): {:.6f}\tLoss2(repres): {:.6f}\tLoss3 (sparsity): {:.6f}\tLossTot (1+2+3): {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(loaderS.dataset),
        #        100. * batch_idx / len(loaderS),
        #        float(Loss_1), float(Loss_2), float(Loss_3),
        #        float(Loss_1 + Loss_2 + Loss_3)))
        #print(correct/total)
        Stot[:, idx] = S.to("cpu")
    return Stot, correct/total


def test(model, Stot, device, test_loader):
    K = model.linear.weight.data.shape[0]
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, idx in test_loader:
            loc = sub2ind([data.shape[0], K],
                          torch.tensor(range(data.shape[0]), device=device),
                          target.to(device))
            S = Stot[:, idx]
            data, target, S = data.to(device), target.to(device), S.to(device)

            s = model(S.t())[0]
            pred = s.max(1, keepdim=True)[1]  # get the index of the max log-probability
            s = s.view(s.numel())
            logprobs = s[loc]

            Loss_1 = torch.mean(-logprobs)
            test_loss += Loss_1  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (correct / len(test_loader.dataset))

def main():
    print('code start running ...')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--missing-data-perc', type=float, default=0.25, metavar='perc',
                        help='input percentage of missing data (default:0.75)')
    parser.add_argument('--sparse-rep-coeff', type=float, default=1.5, metavar='lambda_s',
                        help='input sparse representation coefficient (default: 0.032)') #0.5*(1-75%)
    parser.add_argument('--l1-reg', type=float, default=1.5, metavar='lambda_1',
                        help='input l1 regularization (default: 0.064)')
    parser.add_argument('--gradientS-step-train', type=float, default=1, metavar='alpha',
                        help='input step for SGD training (default: 0.5)')
    parser.add_argument('--gradientS-step-test', type=float, default=2, metavar='alpha',
                        help='input step for SGD testing (default: 2.5)')
    parser.add_argument('--rep', type=int, default=1, metavar='repetition',
                        help='input repetition number')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    input_size = 32*32
    num_classes = 10

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = mycifar.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = mycifar.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    Itrain = trainloader.dataset.train_data.shape[0]
    Itest = testloader.dataset.test_data.shape[0]

    J = 32*32 # Number of columns in the dictionary
    K = classes.__len__() # Number of classes

    model = MyResNet18(J, input_size).to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    ## Initialize sparse coefficients
    Strain = torch.randn(3*J, Itrain, device="cpu")  # sparse coefficients
    Stest = torch.randn(3*J, Itest, device="cpu")  # sparse coefficients

    ## Create mask
    mask_train = gen_mask(trainloader.dataset.train_data.shape, args.missing_data_perc)
    mask_test = gen_mask(testloader.dataset.test_data.shape, args.missing_data_perc)
    print('masks generated')

    # Open file to write output log

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t = time.time()
        Strain, accutrain = train(args, model, Strain, mask_train, device, trainloader, optimizer, epoch)
        #accutrain = test(model, Strain, device, trainloader)
        print('Epoch:', epoch, '/', args.epochs, 'Accu Train=', 100*accutrain, '%, Elapsed tme=', time.time() - t, 'sec')
    if (args.save_model):
        torch.save(model.state_dict(), "MyModel_mnist.pt")

    # Testing stage
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t = time.time()
        ## Introduce here optimization with respect to S
        Stest, accutest = optimize_S(Stest, mask_test, model, testloader, args, device, epoch)
        #accutest = test(model, Stest, device, testloader)
        print('Epoch:', epoch, '/', args.epochs, 'Accu Test=', 100*accutest, '%, Elapsed tme=', time.time() - t, 'sec')

    fName = "results_" + str(args.missing_data_perc) + "_" + str(args.sparse_rep_coeff) + "_" + str(
        args.l1_reg) + "_" + str(args.rep) + ".pickle"
    K = model.linear.weight.data.shape[0]
    D = model.decoder.weight.data
    #probs_train = model(Strain.t().to(device))[0]
    #probs_test = model(Stest.t().to(device))[0]
    with open('../output/' + fName, 'wb') as f:
        # move to cpu first
        if torch.cuda.is_available():
            D = D.cpu()
            #probs_train = probs_train.cpu()
            #probs_test = probs_test.cpu()
            probs_train = None
            probs_test = None
            Strain = None
            Stest = None
            mask_train = None
            mask_test = None
        pickle.dump([D, K, probs_train, probs_test, Strain, Stest, mask_train, mask_test, args, accutrain, accutest], f)
    print('perc=', args.missing_data_perc, 'alpha1=', args.sparse_rep_coeff, 'alpha2=', args.l1_reg, 'rep=', args.rep)

if __name__ == "__main__":
    main()