import argparse
import numpy as np
import time as time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gen_data import *
from simultaneous import *
from sequential import *


class MyModel(nn.Module):
    def __init__(self, J, im_size, num_classes):
        super(MyModel, self).__init__()
        self.decoder = nn.Linear(J, im_size, bias=False) # dictionary matrix
        self.FC = nn.Linear(im_size, num_classes) # Full connected layer

    def forward(self, x):
        xap = self.decoder(x.squeeze()) # xap = Ds
        x = self.FC(xap)
        return F.softmax(x, dim=1), xap


def main():
    print('code start running ...')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Synthetical Data Example')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 3000)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--missing-data-perc', type=float, default=0.25, metavar='perc',
                        help='input percentage of missing data (default:0.75)')
    parser.add_argument('--sparse-rep-coeff', type=float, default=0.1, metavar='lambda_s',
                        help='input sparse representation coefficient (default: 0.1)')
    parser.add_argument('--l1-reg', type=float, default=0.1, metavar='lambda_1',
                        help='input l1 regularization (default: 0.4)')
    parser.add_argument('--gradientS-step-train', type=float, default=15.0, metavar='alpha',
                        help='input step for SGD training (default: 1)')
    parser.add_argument('--gradientS-step-test', type=float, default=2.5, metavar='alpha',
                        help='input step for SGD testing (default: 5)')
    parser.add_argument('--K', type=int, default=8, metavar='sparsity',
                        help='input sparsity (default: 2)')
    parser.add_argument('--threshold', type=float, default=1.0, metavar='threshold',
                        help='input distance to hiprplane (default: 0.25)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Set experiment parameters
    device = torch.device("cuda" if use_cuda else "cpu")
    N = 100  # vectors size
    L = 2*N  # Number of columns in the dictionary
    num_classes = 2
    Itrain = 10000 # Number of data samples
    Itest = 1000

    # Generate synthetical dataset
    D = gen_dict(N)
    x, y, w, b = gen_data(Itrain + Itest, D, args.K, args.threshold)
    x_train, y_train = x[:,:Itrain], y[:,:Itrain]
    x_test, y_test = x[:,Itrain:], y[:,Itrain:]
    del x, y

    # Create masks
    mask_train = gen_mask(N, Itrain, args.missing_data_perc)
    mask_test = gen_mask(N, Itest, args.missing_data_perc)
    print('Dataset and masks generated')

    ####################################################################################################################
    # Sequential approach (data imputation followed by training)
    print('Applying sequential approach ...')

    model = MyModel(L, N, num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Initialize sparse coefficients
    Strain = torch.randn(L, Itrain, device="cpu")  # sparse coefficients
    Stest = torch.randn(L, Itest, device="cpu")  # sparse coefficients

    # 1st Find sparse representation
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        Strain, accutrain_seq = optimize_DS(Strain, mask_train, model, mask_train * x_train, y_train, optimizer, args, device, epoch)
        print('Finding sparsest representation, epoch ', str(epoch), 'Accu Train Seq=', float(accutrain_seq))

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq = optimize_classifier(args, model, Strain, device, y_train, optimizer, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq=', accutrain_seq)

    ## Test dataset
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        Stest, accutest_seq = optimize_S(Stest, mask_test, model, mask_test * x_test, y_test, optimizer, args, device, epoch)
        print('Accu Test Seq=', accutest_seq, ', Elapsed tme=', time.time() - t, 'sec')

    ####################################################################################################################
    # Simultaneous approach
    print('Applying simultaneous approach ...')
    model = MyModel(L, N, num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Initialize sparse coefficients
    Strain = torch.randn(L, Itrain, device="cpu")  # sparse coefficients
    Stest = torch.randn(L, Itest, device="cpu")  # sparse coefficients
    for epoch in range(1, args.epochs + 1):
        Strain, accutrain_sim = simultaneous_train(args, model, Strain, mask_train, device, x_train*mask_train, y_train, optimizer, epoch)
        print('epoch ',str(epoch),'Accu Train Simult=', accutrain_sim)

    ## Test dataset
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        Stest, accutest_sim = optimize_S(Stest, mask_test, model, mask_test * x_test, y_test, optimizer, args, device,
                                     epoch)
        print('Accu Test Sim=', accutest_sim, ', Elapsed tme=', time.time() - t, 'sec')

    ####################################################################################################################
    # print results

    print("")
    print("Sequential:")
    print("Accu Train = ", accutrain_seq)
    print("Accu Test =", accutest_seq)

    print("")
    print("Simultaneous:")
    print("Accu Train = ", accutrain_sim)
    print("Accu Test =", accutest_sim)


if __name__ == "__main__":
    main()