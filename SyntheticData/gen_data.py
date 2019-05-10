import torch


# Generate mask of missing entries
def gen_mask(M0, M1, perc):
    # 0: missing
    # 1: available
    mask = torch.ones(M0*M1)
    ind = torch.randperm(M0*M1)
    ind = ind[0:round(perc*M0*M1)-1]
    mask[ind] = 0
    mask = mask.view(M0, M1)
    return mask


# Generate dictonary
def gen_dict(N):
    D = torch.randn(N, 2*N, device="cpu")
    # normalize columns
    col_sqr_norms = torch.sqrt(torch.sum(D * D, 0))
    D = D / col_sqr_norms.unsqueeze(0).expand(D.shape[0],-1)  # columnwise normalization
    return D


# Generate data vectors having K-sparse representation on dictionary D organized in 2 classes with distance to the separating hyperplane equals to threshold
def gen_data(I, D, K, threshold):
    print('Generating synthetical dataset ...')
    [N, L] = D.shape
    s = torch.zeros(L, I)

    # Generate K-sparse vectors
    for i in range(I):
        ind = torch.randperm(L)
        ind = ind[:K]
        s[ind, i] = torch.randn(K, 1).squeeze()
    x = torch.matmul(D, s)

    # Define an hyperplane with w and b by random
    w = torch.randn(N, 1)
    w = w/torch.norm(w,2)
    b = torch.median(torch.matmul(w.t(), x))  # optimal bias

    # Make sure the threshold is satisfied
    for i in range(I):
        while (torch.matmul(w.t(), x[:, i]) + b).abs() < threshold:
            ind = torch.randperm(L)
            s[ind[:K], i] = torch.randn(K, 1).squeeze()
            x[:, i] = torch.matmul(D, s[:, i])

    # Classify according to such hyperplane
    y = torch.matmul(w.t(), x) + b.repeat(1, I)
    y[y > 0] = 1
    y[y <=0] = 0

    y = y.long()

    return x, y, w, b
