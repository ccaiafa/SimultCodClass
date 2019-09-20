import torch
import torch.nn as nn


# Convert matrix indices to linear indices
def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


# Simultaneous training of classifier along data representation (D and s)
def simultaneous_train(args, model, Stot, mask_tot, device, x_train, y_train, optimizer, epoch):
    do = nn.Dropout(0.5)
    K = model.FC.weight.data.shape[0]
    tot_loss = 0
    correct = 0
    for batch_idx in range(round(x_train.shape[1]/args.batch_size)):
        idx = range(batch_idx*args.batch_size, (batch_idx+1)*args.batch_size)
        data = x_train[:, idx]
        target = y_train[:, idx]

        model.train()
        S = Stot[:, idx]
        mask = mask_tot[:, idx]
        loc = sub2ind([data.shape[1], K],
                      torch.tensor(range(data.shape[1]), device=device),
                      target.to(device))

        data, target, S, mask = data.to(device), target.to(device), S.to(device), mask.to(device)

        # Optimize W, b and D
        optimizer.zero_grad()

        s, xap = model(S.t())
        s = s.view(s.numel())
        probs = s[loc]

        Loss_1 = torch.mean(-torch.log(probs))
        diff = mask * (xap.t() - data)
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0] * (1.0 - args.missing_data_perc))

        loss = Loss_1 + Loss_2
        loss.backward()
        optimizer.step() # Here is where W, b and D are updated based on Loss1 + Loss2

        # Normalize Dictionary columns
        col_sqr_norms = torch.sqrt(torch.sum(model.decoder.weight * model.decoder.weight, 0))
        model.decoder.weight.data = model.decoder.weight / col_sqr_norms.unsqueeze(0).expand(model.decoder.weight.shape[0],-1)   # columnwise normalization

        # Update coefficients S
        model.eval()
        S.requires_grad=True
        optimizer.zero_grad()

        s, xap = model(S.t())
        s = s.view(s.numel())

        # Compute probabilities of correct classes
        probs = s[loc]
        Loss_1 = torch.mean(-torch.log(probs))
        diff = mask * (xap.t() - data)
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0] * (1.0 - args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]

        Loss = Loss_1 + Loss_2 + Loss_3
        Loss.backward()

        dLdS = do(S.grad)
        z = -args.gradientS_step_train * dLdS
        if args.l1_reg > 0: # avoid zero crossing
            z[S * (S + z) < 0] = -S[torch.squeeze(S * (S + z) < 0)]
        with torch.no_grad():
            S = S + z

        S.requires_grad = False

        # print current results
        s, xap = model(S.t())
        pred = s.max(1, keepdim=True)[1]

        s = s.view(s.numel())
        probs = s[loc]
        Loss_1 = torch.mean(-torch.log(probs))
        # Compute losses
        diff = mask * (xap.t() - data)
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0] * (1.0 - args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]
        if batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss1(class): {:.6f}\tLoss2(repres): {:.6f}\tLoss3 (sparsity): {:.6f}\tLossTot (1+2+3): {:.6f}'.format(
                epoch, batch_idx * len(data.t()), len(x_train.t()),
                100. * batch_idx / len(x_train.t()),
                float(Loss_1), float(Loss_2), float(Loss_3),
                float(Loss_1 + Loss_2 + Loss_3)))

        Stot[:, idx] = S.to("cpu")
        tot_loss += torch.mean(-torch.log(probs)) # sum up batch loss
        correct += pred.eq(target.view_as(pred)).sum().item()

    return Stot, (correct / x_train.shape[1])

def optimize_S(Stot, mask_tot, model, x_train, y_train, optimizer, args, device, epoch):
    model.eval()

    K = model.FC.weight.data.shape[0]
    print("updating coefficients ...")
    tot_loss = 0
    correct = 0
    for batch_idx in range(round(x_train.shape[1]/args.batch_size)):
        idx = range(batch_idx*args.batch_size, (batch_idx+1)*args.batch_size)
        data = x_train[:, idx]
        target = y_train[:, idx]

        model.train()
        S = Stot[:,idx]
        mask = mask_tot[:,idx]
        data, target, S, mask = data.to(device), target.to(device), S.to(device), mask.to(device)

        loc = sub2ind([data.shape[1], K],
                      torch.tensor(range(data.shape[1]), device=device),
                      target.to(device))
        J = S.shape[0]
        S.requires_grad = True

        s, xap = model(S.t())
        s = s.view(s.numel())
        # Compute probabilities of correct classes
        probs = s[loc]

        diff = mask * (xap.t() - data)
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0] * (1.0 - args.missing_data_perc))
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
        pred = s.max(1, keepdim=True)[1]

        s = s.view(s.numel())
        probs = s[loc]
        Loss_1 = torch.mean(-torch.log(probs))
        # Compute losses
        diff = mask * (xap.t() - data)
        Loss_2 = args.sparse_rep_coeff * torch.sum(diff * diff) / (data.shape[0] * (1.0 - args.missing_data_perc))
        Loss_3 = args.l1_reg * torch.sum(torch.abs(S)) / data.shape[0]
        if batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss1(class): {:.6f}\tLoss2(repres): {:.6f}\tLoss3 (sparsity): {:.6f}\tLossTot (2+3): {:.6f}'.format(
                epoch, batch_idx * len(data.t()), len(x_train.t()),
                100. * batch_idx / len(x_train.t()),
                float(Loss_1), float(Loss_2), float(Loss_3),
                float(Loss_2 + Loss_3)))

        Stot[:, idx] = S.to("cpu")
        tot_loss += torch.mean(-torch.log(probs)) # sum up batch loss
        correct += pred.eq(target.view_as(pred)).sum().item()

    return Stot, (correct / x_train.shape[1])
