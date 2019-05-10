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
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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
    parser.add_argument('--missing-data-perc', type=float, default=0.75, metavar='perc',
                        help='input percentage of missing data (default:0.75)')
    parser.add_argument('--sparse-rep-coeff', type=float, default=0.1, metavar='lambda_s',
                        help='input sparse representation coefficient (default: 0.1)')
    parser.add_argument('--l1-reg', type=float, default=0.1, metavar='lambda_1',
                        help='input l1 regularization (default: 0.4)')
    parser.add_argument('--gradientS-step-train', type=float, default=1.0, metavar='alpha',
                        help='input step for SGD training (default: 1)')
    parser.add_argument('--K', type=int, default=4, metavar='sparsity',
                        help='input sparsity (default: 2)')
    parser.add_argument('--threshold', type=float, default=0.4, metavar='threshold',
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

    # Generate synthetical dataset
    D = gen_dict(N)
    x_train, y_train, w, b = gen_data(Itrain, D, args.K, args.threshold)

    # Create mask
    mask_train = gen_mask(N, Itrain, args.missing_data_perc)
    print('Dataset and masks generated')

    ####################################################################################################################
    # Sequential approach (data imputation followed by training)
    print('Applying sequential approach ...')
    accutrain_seq = 0.0
    # grid search of optimal l1_reg parameter
    for args.l1_reg in np.arange(0.01, 0.21, 0.01):
        model = MyModel(L, N, num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        # Initialize sparse coefficients
        Strain = torch.randn(L, Itrain, device="cpu")  # sparse coefficients

        # 1st Find sparse representation
        for epoch in range(1, args.epochs + 1):
            t = time.time()
            Strain, Loss = optimize_D_S(Strain, mask_train, model, x_train*mask_train, y_train, args, device, epoch)
            print('Finding sparsest representation, epoch ', str(epoch), 'Loss=', float(Loss))

        # 2nd Optimize classifier
        for epoch in range(1, args.epochs + 1):
            accutrain = optimize_classifier(args, model, Strain, device, y_train, optimizer, epoch)
            print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq=', accutrain)
        if accutrain > accutrain_seq:
            accutrain_seq = accutrain
            l1_reg_opt = args.l1_reg

    ####################################################################################################################
    # Simultaneous approach
    print('Applying simultaneous approach ...')
    args.l1_reg = 0.1 # This value for parameter l1_reg gives good results (no grid search needed here)
    model = MyModel(L, N, num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Initialize sparse coefficients
    Strain = torch.randn(L, Itrain, device="cpu")  # sparse coefficients
    for epoch in range(1, args.epochs + 1):
        Strain, accutrain_sim = simultaneous_train(args, model, Strain, mask_train, device, x_train*mask_train, y_train, optimizer, epoch)
        print('epoch ',str(epoch),'Accu Train Simult=', accutrain_sim)

    ####################################################################################################################
    # print results
    print(' ')
    print('Results Sequential method:')
    print('sparse_rep_coeff=', str(args.sparse_rep_coeff), 'l1_reg=', str(l1_reg_opt))
    print('Training Accuracy=', str(accutrain_seq))

    print(' ')
    print('Results Simultaneous method:')
    print('sparse_rep_coeff=', str(args.sparse_rep_coeff), 'l1_reg=', str(args.l1_reg))
    print('Training Accuracy=', str(accutrain_sim))


if __name__ == "__main__":
    main()