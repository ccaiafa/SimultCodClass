import torch


# Generate mask of missing entries
def gen_mask(M, perc):
    # 0: missing
    # 1: available
    mask = torch.ones(M.numel())
    ind = torch.randperm(M.numel())
    ind = ind[0:round(perc*M.numel())-1]
    mask[ind] = 0
    mask = mask.view(M[0], M[1], M[2])
    return mask