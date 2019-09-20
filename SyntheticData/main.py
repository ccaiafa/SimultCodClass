import argparse
import numpy as np
import time as time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from gen_data import *
from simultaneous import *
from sequential import *


class MyModel(nn.Module):
    def __init__(self, J, im_size, num_classes):
        super(MyModel, self).__init__()
        self.dropoutin = nn.Dropout(p=0.0)
        self.decoder = nn.Linear(J, im_size, bias=False) # dictionary matrix
        self.dropouth = nn.Dropout(p=0.25)
        self.FC = nn.Linear(im_size, num_classes) # Full connected layer
        self.dropoutout = nn.Dropout(p=0.0)

    def forward(self, x):
        xap = self.decoder(self.dropoutin(x.squeeze())) # xap = Ds
        x = self.FC(self.dropouth(xap))
        return F.softmax(self.dropoutout(x), dim=1), xap

class LinearModel(nn.Module):
    def __init__(self, im_size, num_classes):
        super(LinearModel, self).__init__()
        self.FC = nn.Linear(im_size, num_classes) # Full connected layer

    def forward(self, x):
        x = self.FC(x)
        return F.softmax(x, dim=1)

def train(args, model, x_train, device, y_train, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx in range(round(x_train.shape[1]/args.batch_size)):
        idx = range(batch_idx*args.batch_size, (batch_idx+1)*args.batch_size)
        data = x_train[:, idx].t().to(device)
        target = y_train[:, idx].squeeze().to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), x_train.shape[1],
                100. * batch_idx / x_train.shape[1], loss.item()))

    return correct / x_train.shape[1]

def test(args, model, device, x_test, y_test):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx in range(round(x_test.shape[1]/args.batch_size)):
            idx = range(batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size)
            data = x_test[:, idx].to(device)
            target = y_test[:, idx].to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= x_test.shape[1]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, x_test.shape[1],
        100. * correct / x_test.shape[1]))

    return correct / x_test.shape[1]

#################################################################
# Fill unknown values with the mean computed on the rest of available samples
def mean_fill_unsupervised(mask_train, x_train):
    xnew = x_train
    xnew[mask_train == 0] = np.nan # put nans in missing entries locations
    xmean = torch.tensor(np.nanmean(xnew,1)).view(-1,1) # compute the mean value for each feature

    xmean = xmean.repeat(1,x_train.shape[1])
    xnew[mask_train == 0] = xmean[mask_train == 0] # replace missing values by their mean

    return xnew

#################################################################
# Fill unknown values with the mean computed on the rest of available samples on the same class only
def mean_fill_supervised(mask_train, x_train, y_test):
    # Extract class zero samples
    xnew0 = x_train[:,(y_test ==0).nonzero()[:,1]]
    mask0 = mask_train[:,(y_test ==0).nonzero()[:,1]]
    xnew0[mask0 == 0] = np.nan # put nans in missing entries locations
    xmean0 = torch.tensor(np.nanmean(xnew0,1)).view(-1,1) # compute the mean value for each feature
    xmean0 = xmean0.repeat(1,xnew0.shape[1])
    xnew0[mask0 == 0] = xmean0[mask0 == 0] # replace missing values by their mean

    # Extract class one samples
    xnew1 = x_train[:,(y_test ==1).nonzero()[:,1]]
    mask1 = mask_train[:,(y_test ==1).nonzero()[:,1]]
    xnew1[mask1 == 0] = np.nan # put nans in missing entries locations
    xmean1 = torch.tensor(np.nanmean(xnew1,1)).view(-1,1) # compute the mean value for each feature
    xmean1 = xmean1.repeat(1,xnew1.shape[1])
    xnew1[mask1 == 0] = xmean1[mask1 == 0] # replace missing values by their mean

    # Replace with new data
    xnew = x_train
    xnew[:,(y_test ==0).nonzero()[:,1]] = xnew0
    xnew[:, (y_test == 1).nonzero()[:,1]] = xnew1

    return xnew

#################################################################
# Fill unknown values with the mean computed on the K-nearestt neighbors of the same class
def knn_fill_sup(mask, x, y, K):
    # Extract class zero samples
    xnew0 = x[:,(y ==0).nonzero()[:,1]]
    mask0 = mask[:,(y ==0).nonzero()[:,1]]

    xnew0 = xnew0 * mask0
    xnan0 = xnew0
    xnan0[mask0 == 0] = np.nan
    x02 = torch.sum(xnew0 * xnew0,0)

    # Extract class one samples
    xnew1 = x[:, (y == 1).nonzero()[:, 1]]
    mask1 = mask[:, (y == 1).nonzero()[:, 1]]

    xnew1 = xnew1 * mask1
    xnan1 = xnew1
    xnan1[mask1 == 0] = np.nan
    x12 = torch.sum(xnew1 * xnew1, 0)

    # Final result
    xnew = x * mask

    for j in range(x.shape[1]):
        #print(j)
        v = xnew[:, j].view(-1, 1)
        if y.squeeze()[j]==0:
            d = x02.view(-1,1) - 2 * xnew0.t().mm(v) + v.t().mm(v).squeeze().repeat(xnew0.shape[1],1)
            sorted, i = torch.sort(d, dim=0, descending=False, out=None)
            xnn = xnan0[:,i[1:K+1]].squeeze() # K-nearest neighbors as columns
        else:
            d = x12.view(-1, 1) - 2 * xnew1.t().mm(v) + v.t().mm(v).squeeze().repeat(xnew1.shape[1], 1)
            sorted, i = torch.sort(d, dim=0, descending=False, out=None)
            xnn = xnan1[:, i[1:K + 1]].squeeze()  # K-nearest neighbors as columns

        xmean = torch.tensor(np.nanmean(xnn,1)) # compute the mean value for each feature
        xmean[torch.isnan(xmean)] = 0.0
        xnew[mask[:,j]==0, j] = xmean[mask[:,j]==0].squeeze()

    return xnew


def main():
    print('code start running ...')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Synthetical Data Example')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                        help='number of epochs to train (default: 3000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--missing-data-perc', type=float, default=0.20, metavar='perc',
                        help='input percentage of missing data (default:0.25)')
    parser.add_argument('--sparse-rep-coeff', type=float, default=0.7, metavar='lambda_s', # Lambda_1 must be tuned by cross-validation
                        help='input sparse representation coefficient (default: 0.1)')
    parser.add_argument('--l1-reg', type=float, default=0.4, metavar='lambda_1',
                        help='input l1 regularization (default: 0.1)') # Lambda_2 must be tuned by cross-validation
    parser.add_argument('--gradientS-step-train', type=float, default=15.0, metavar='alpha',
                        help='input step for SGD training (default: 1)')
    parser.add_argument('--gradientS-step-test', type=float, default=2.5, metavar='alpha',
                        help='input step for SGD testing (default: 2.5)')
    parser.add_argument('--K', type=int, default=4, metavar='sparsity',
                        help='input sparsity (default: 2)')
    parser.add_argument('--threshold', type=float, default=0.0, metavar='threshold',
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
    #x_test, y_test = x[:, :Itest], y[:, :Itest]
    del x, y

    # Create masks
    mask_train = gen_mask(N, Itrain, args.missing_data_perc)
    mask_test = gen_mask(N, Itest, args.missing_data_perc)
    print('Dataset and masks generated')


    ####################################################################################################################
    # Sequential2 approach (zero filling data imputation followed by training)
    print('Applying sequential2 (zero filling) approach ...')

    model2 = LinearModel(N, num_classes).to(device)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum)

    # 1st Fill with zeros training data
    x_train_zero = mask_train * x_train

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq2 = train(args, model2, x_train_zero, device, y_train, optimizer2, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq2=', accutrain_seq2)

    ## Test dataset
    output = model2(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq2 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq2 (zero filling)=', accutest_seq2)

    ####################################################################################################################
    # Sequential1 approach (sparse representation based data imputation followed by training)
    print('Applying sequential1 approach ...')

    model1 = MyModel(L, N, num_classes).to(device) # For training Dict + sparse coeffs
    modelS = LinearModel(N, num_classes).to(device) # Only classifier
    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)
    optimizerS = optim.SGD(modelS.parameters(), lr=args.lr, momentum=args.momentum)

    # Initialize sparse coefficients
    Strain = torch.randn(L, Itrain, device="cpu")  # sparse coefficients
    Stest = torch.randn(L, Itest, device="cpu")  # sparse coefficients
    #
    # 1st Find sparse representation
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        Strain, accutrain_seq1 = optimize_DS(Strain, mask_train, model1, mask_train * x_train, y_train, optimizer1, args,
                                             device, epoch)
        print('Finding sparsest representation, epoch ', str(epoch), 'Accu Train Seq1=', float(accutrain_seq1))

    # Complete data
    s, x_train_1 = model1(Strain.t().to(device))
    x_train_1 = x_train_1.t()
    #x_train_1[mask_train == 1] = x_train[mask_train == 1]

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq1 = train(args, modelS, x_train_1, device, y_train, optimizerS, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq1=', accutrain_seq1)

    ## Test dataset
    output = modelS(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq1 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq1=', accutest_seq1)

    ####################################################################################################################
    # Sequential3 approach (filling missing data with mean across available samples followed by training)
    print('Applying sequential3 (mean-unsupervised) approach ...')

    model3 = LinearModel(N, num_classes).to(device)
    optimizer3 = optim.SGD(model3.parameters(), lr=args.lr, momentum=args.momentum)

    # 1st Fill with mean computed on available samples
    x_train_mean_unsup = mean_fill_unsupervised(mask_train, x_train)

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq3 = train(args, model3, x_train_mean_unsup, device, y_train, optimizer3, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq3=', accutrain_seq3)

    ## Test dataset
    output = model3(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq3 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq3=', accutest_seq3)

    ####################################################################################################################
    # Sequential4 approach (filling missing data with mean for each class followed by training)
    print('Applying sequential4 (mean-supervised) approach ...')

    model4 = LinearModel(N, num_classes).to(device)
    optimizer4 = optim.SGD(model4.parameters(), lr=args.lr, momentum=args.momentum)

    # 1st Fill with mean computed on available samples
    x_train_mean_sup = mean_fill_supervised(mask_train, x_train, y_train)

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq4 = train(args, model4, x_train_mean_sup, device, y_train, optimizer4, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq4=', accutrain_seq4)

    ## Test dataset
    output = model4(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq4 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq4=', accutest_seq4)

    ####################################################################################################################
    # Simultaneous approach
    print('Applying simultaneous approach ...')
    model_sim = MyModel(L, N, num_classes).to(device)
    model_sim.FC.weight.data = model4.FC.weight.data # Initialize with a good classifier
    optimizer_sim = optim.SGD(model_sim.parameters(), lr=args.lr, momentum=args.momentum)

    modelS = LinearModel(N, num_classes).to(device) # Only classifier

    # Initialize sparse coefficients
    Strain = torch.randn(L, Itrain, device="cpu")  # sparse coefficients
    #Stest = torch.randn(L, Itest, device="cpu")  # sparse coefficients
    for epoch in range(1, args.epochs + 1):
        Strain, accutrain_sim = simultaneous_train(args, model_sim, Strain, mask_train, device, x_train*mask_train, y_train, optimizer_sim, epoch)
        print('epoch ',str(epoch),'Accu Train Simult=', accutrain_sim)

    ## Test dataset
    modelS.FC.weight.data = model_sim.FC.weight.data
    output = modelS(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_sim = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Sim=', accutest_sim)


    ####################################################################################################################
    # Sequential5 approach (filling missing data with mean on KNN followed by training)
    print('Applying sequential5 (KNN-10) approach ...')
    K = 10
    model5 = LinearModel(N, num_classes).to(device)
    optimizer5 = optim.SGD(model5.parameters(), lr=args.lr, momentum=args.momentum)

    # 1st Fill with mean computed on available samples
    x_train_knn3 = knn_fill_sup(mask_train, x_train, y_train, K)

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq5 = train(args, model5, x_train_knn3, device, y_train, optimizer5, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq5=', accutrain_seq5)

    ## Test dataset
    output = model5(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq5 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq5=', accutest_seq5)

    ####################################################################################################################
    # Sequential6 approach (filling missing data with mean on KNN followed by training)
    print('Applying sequential6 (KNN-20) approach ...')
    K = 20
    model6 = LinearModel(N, num_classes).to(device)
    optimizer6 = optim.SGD(model6.parameters(), lr=args.lr, momentum=args.momentum)

    # 1st Fill with mean computed on available samples
    x_train_knn5 = knn_fill_sup(mask_train, x_train, y_train, K)

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq6 = train(args, model6, x_train_knn5, device, y_train, optimizer6, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq6=', accutrain_seq6)

    ## Test dataset
    output = model6(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq6 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq6=', accutest_seq6)

    ####################################################################################################################
    # Sequential7 approach (filling missing data with mean on KNN followed by training)
    print('Applying sequential7 (KNN-50) approach ...')
    K = 50
    model7 = LinearModel(N, num_classes).to(device)
    optimizer7 = optim.SGD(model7.parameters(), lr=args.lr, momentum=args.momentum)

    # 1st Fill with mean computed on available samples
    x_train_knn10 = knn_fill_sup(mask_train, x_train, y_train, K)

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq7 = train(args, model7, x_train_knn10, device, y_train, optimizer7, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq7=', accutrain_seq7)

    ## Test dataset
    output = model7(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq7 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq7=', accutest_seq7)

    ####################################################################################################################
    # Sequential8 approach (filling missing data with mean on KNN followed by training)
    print('Applying sequential8 (KNN-100) approach ...')
    K = 100
    model8 = LinearModel(N, num_classes).to(device)
    optimizer8 = optim.SGD(model8.parameters(), lr=args.lr, momentum=args.momentum)

    # 1st Fill with mean computed on available samples
    x_train_knn20 = knn_fill_sup(mask_train, x_train, y_train, K)

    # 2nd Optimize classifier
    for epoch in range(1, 101):
        accutrain_seq8 = train(args, model8, x_train_knn20, device, y_train, optimizer8, epoch)
        print('Finding classifier, epoch ', str(epoch), 'Accu Train Seq8=', accutrain_seq8)

    ## Test dataset
    output = model8(x_test.t().to(device))
    pred = output.max(1, keepdim=True)[1].to("cpu")
    accutest_seq8 = pred.eq(y_test.view_as(pred)).sum().item()/y_test.shape[1]

    print('Accu Test Seq8=', accutest_seq8)

    ####################################################################################################################
    print("")
    print("Sequential1: Sparse Rep")
    print("Accu Train = ", accutrain_seq1)
    print("Accu Test =", accutest_seq1)

    print("")
    print("Sequential2: Zero Filling")
    print("Accu Train = ", accutrain_seq2)
    print("Accu Test =", accutest_seq2)

    print("")
    print("Sequential3: Unsup Mean Filling")
    print("Accu Train = ", accutrain_seq3)
    print("Accu Test =", accutest_seq3)

    print("")
    print("Sequential4: Sup Mean Filling")
    print("Accu Train = ", accutrain_seq4)
    print("Accu Test =", accutest_seq4)

    print("")
    print("Sequential5: KNN-10")
    print("Accu Train = ", accutrain_seq5)
    print("Accu Test =", accutest_seq5)

    print("")
    print("Sequential5: KNN-20")
    print("Accu Train = ", accutrain_seq6)
    print("Accu Test =", accutest_seq6)

    print("")
    print("Sequential6: KNN-50")
    print("Accu Train = ", accutrain_seq7)
    print("Accu Test =", accutest_seq7)

    print("")
    print("Sequential7: KNN-100")
    print("Accu Train = ", accutrain_seq8)
    print("Accu Test =", accutest_seq8)

    print("")
    print("Simultaneous:")
    print("Accu Train = ", accutrain_sim)
    print("Accu Test =", accutest_sim)

    #fName = "results_"+str(args.missing_data_perc)+"_"+str(args.K)+"_"+str(args.threshold)+"_"+str(args.sparse_rep_coeff)+"_"+str(args.l1_reg)+".pickle"
    #with open('output/'+fName, 'wb') as f:
    #    pickle.dump([args, accutrain_seq1, accutest_seq1, accutrain_seq2, accutest_seq2, accutrain_seq3, accutest_seq3,
    #                 accutrain_seq4, accutest_seq4, accutrain_seq5, accutest_seq5, accutrain_seq6, accutest_seq6,
    #                 accutrain_seq7, accutest_seq7, accutrain_seq8, accutest_seq8, accutrain_sim, accutest_sim], f)

if __name__ == "__main__":
    main()