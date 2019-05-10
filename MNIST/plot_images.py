import torch
import matplotlib.pyplot as plt

# Plot a set of examples
def plot_images(D,S,x,y,yap,mask,Ncases):
    fig = plt.figure(figsize=(12,3))
    rndind = torch.randperm(x.shape[1])
    rndin = rndind[range(Ncases)]
    n = 1
    for ind in rndin:
        # original
        x0 = x[:, ind]
        x0 = x0.view(-1,1).repeat(1, 3)

        # observed data
        xc = x[:, ind].view(-1,1).repeat(1, 3)
        maskc = mask[:, ind]
        xc[maskc == 0, 0] = 0
        xc[maskc == 0, 1] = 0
        xc[maskc == 0, 2] = 0.78

        # reconstructed
        xr = torch.mm(D, S[:, ind].view(-1,1))
        xr = xr - xr.min()
        xr = xr / torch.max(xr)
        xr = xr.view(-1, 1).repeat(1, 3)

        # display results
        fig.add_subplot(3, Ncases, n)
        plt.imshow(x0.view(28, 28, 3))
        plt.title('y='+str(int(y[ind])))
        plt.axis('off')

        #n = n + 1

        fig.add_subplot(3, Ncases, n+Ncases)
        plt.imshow(xc.view(28, 28, 3))
        plt.title('yobs')
        plt.axis('off')

        #n = n + 1

        fig.add_subplot(3, Ncases, n+2*Ncases)
        plt.imshow(xr.view(28, 28, 3))
        plt.title('yap='+str(int(yap[ind])))
        plt.axis('off')

        n = n + 1

    plt.pause(0.05)
    #print('y=',y[rndin])
    #print('yap=',yap[rndin])
    return fig