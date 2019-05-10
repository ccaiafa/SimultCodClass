import torch


# Convert matrix indices to linear indices
def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def train(args, model, Stot, mask_tot, device, train_loader, optimizer, epoch):
    K = model.fc2.weight.data.shape[0]
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        model.train()
        S = Stot[:,idx]
        mask = mask_tot[idx,:,:]
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
        diff = mask.view(-1, 784).t() * (xap.t() - data.squeeze().view(-1, 784).t())
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
        diff = mask.view(-1, 784).t() * (xap.t() - data.squeeze().view(-1, 784).t())
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
        s = s.view(s.numel())
        logprobs = s[loc]
        Loss_1 = torch.mean(-logprobs)
        # Compute losses
        diff = mask.view(-1, 784).t() * (xap.t() - data.squeeze().view(-1, 784).t())
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff*diff) / (data.shape[0]*(1.0-args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]
        if batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss1(class): {:.6f}\tLoss2(repres): {:.6f}\tLoss3 (sparsity): {:.6f}\tLossTot (1+2+3): {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                float(Loss_1), float(Loss_2), float(Loss_3),
                float(Loss_1 + Loss_2 + Loss_3)))

        Stot[:, idx] = S.to("cpu")

    return Stot


def optimize_S(Stot, mask_tot, model, loaderS, args, device, epoch):
    model.eval()

    K = model.fc2.weight.data.shape[0]
    print("Finding Sparse representation of observed data ...")
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
        diff = mask.view(-1, 784).t() * (xap.t() - data.squeeze().view(-1, 784).t())
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
        s = s.view(s.numel())
        logprobs = s[loc]
        Loss_1 = torch.mean(-logprobs)
        # Compute losses
        diff = mask.view(-1, 784).t() * (xap.t() - data.squeeze().view(-1, 784).t())
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff*diff) / (data.shape[0]*(1.0-args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]
        if batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss1(class): {:.6f}\tLoss2(repres): {:.6f}\tLoss3 (sparsity): {:.6f}\tLossTot (1+2+3): {:.6f}'.format(
                epoch, batch_idx * len(data), len(loaderS.dataset),
                100. * batch_idx / len(loaderS),
                float(Loss_1), float(Loss_2), float(Loss_3),
                float(Loss_1 + Loss_2 + Loss_3)))

        Stot[:, idx] = S.to("cpu")
    return Stot


def test(model, Stot, device, test_loader):
    K = model.fc2.weight.data.shape[0]
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

    return correct / len(test_loader.dataset)