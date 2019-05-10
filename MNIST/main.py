import argparse
import time as time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import mymnist
import matplotlib.pyplot as plt
import pickle

from gen_mask import *
from simultaneous import *
from plot_images import *

# A 4-layer convolutional neural network as implemented in https://github.com/pytorch/examples/tree/master/mnist with
# the addition of the sparse representation of input (decoder)
class MyModel(nn.Module):
    def __init__(self, L, im_size, num_classes):
        super(MyModel, self).__init__()
        self.decoder = nn.Linear(L, im_size, bias=False)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        xap = self.decoder(x.squeeze()) # estimate input vector xap = Ds (sparse representation)
        x = F.relu(self.conv1(xap.view(-1,1,28,28)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), xap

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    print('code start running ...')
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    parser.add_argument('--missing-data-perc', type=float, default=0.75, metavar='perc',
                        help='input percentage of missing data (default:0.75)')
    parser.add_argument('--sparse-rep-coeff', type=float, default=0.128, metavar='lambda_s',
                        help='input sparse representation coefficient (default: 0.125)')
    parser.add_argument('--l1-reg', type=float, default=0.128, metavar='lambda_1',
                        help='input l1 regularization (default: 0.5)')
    parser.add_argument('--gradientS-step-train', type=float, default=0.4, metavar='alpha',
                        help='input step for SGD training (default: 0.5)')
    parser.add_argument('--gradientS-step-test', type=float, default=0.5, metavar='alpha',
                        help='input step for SGD testing (default: 2.5)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # data loaders for training and testing
    dataset_train = torch.utils.data.Subset(mymnist.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])), range(60000))
    dataset_test = torch.utils.data.Subset(mymnist.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])), range(10000))

    train_loader = torch.utils.data.DataLoader(dataset_train,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print('train data loaded')

    test_loader = torch.utils.data.DataLoader(dataset_test,
        shuffle=True,batch_size=args.test_batch_size,  **kwargs)
    print('test data loaded')

    # MNIST parameters
    input_size = 784 # N = 784 = 28*28
    num_classes = 10
    Itrain = train_loader.dataset.dataset.train_data.shape[0]
    Itest = test_loader.dataset.dataset.test_data.shape[0]
    L = 784 # Number of columns in the dictionary (NxL)

    model = MyModel(L, input_size, num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    ## Initialize sparse coefficients
    Strain = torch.randn(L, Itrain, device="cpu")  # sparse coefficients
    Stest = torch.randn(L, Itest, device="cpu")  # sparse coefficients

    ## Create mask
    mask_train = gen_mask(train_loader.dataset.dataset.train_data.shape, args.missing_data_perc)
    mask_test = gen_mask(test_loader.dataset.dataset.test_data.shape, args.missing_data_perc)
    print('masks generated')

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        Strain = train(args, model, Strain, mask_train, device, train_loader, optimizer, epoch)
        print(' Elapsed tme=', time.time() - t, 'sec')
    if (args.save_model):
        torch.save(model.state_dict(), "MyModel_mnist.pt")
    accutrain = test(model, Strain, device, train_loader)

    # Testing stage
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        ## Introduce here optimization with respect to S only
        Stest = optimize_S(Stest, mask_test, model, test_loader, args, device, epoch)
        accutest = test(model, Stest, device, test_loader)
        print(' Elapsed tme=', time.time() - t, 'sec')

    print(' ')
    print('MNIST Results:')
    print('Missing entries=', str(100*args.missing_data_perc),'%')
    print('Training accuracy=', str(100*accutrain),'%')
    print('Testing accuracy=', str(100*accutest),'%')

    with open('MNIST_results.pickle', 'wb') as f:
        # move to cpu first
        if torch.cuda.is_available():
            model = model.cpu()
            Strain = Strain.cpu()
            Stest = Stest.cpu()
            mask_train = mask_train.cpu()
            mask_test = mask_test.cpu()
        pickle.dump([model, Strain, Stest, mask_train, mask_test, train_loader, test_loader, args], f)

    # Visualize results at random (Ncases=5 results per batch in the incomplete testing dataset)
    Ncases = 5
    D = model.decoder.weight.data
    for batch_idx, (data, target, idx) in enumerate(test_loader):
        S = Stest[:, idx]
        probs = model(S.t())[0]
        yap = torch.argmax(probs, 1)
        xn = data.squeeze().view(-1, 784).t()
        mask = mask_train.view(-1, 784).t()[:, idx]
        fig = plot_images(D, S, xn, target, yap, mask, Ncases)
        fig.savefig('batch_' + str(batch_idx) + '.eps', dpi=300, format='eps')

if __name__ == '__main__':
    main()
