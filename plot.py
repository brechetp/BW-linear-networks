import matplotlib.pyplot as plt
import torch
from matplotlib.animation import  FuncAnimation
import numpy as np
import torchvision.utils as vutils

def scatter_samples(fname, target, generated,  fig=None, ax=None, dsc=None, title=None, axtitle=None, args=None):

    if ax is None:
        fig, ax =  plt.subplots(1, 1, figsize=(5, 5))
    device = generated.device
    generated = generated.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    ax.scatter(target[:, 0], target[:, 1], marker='o', label="target")
    ax.scatter(generated[:, 0], generated[:, 1], marker='+', label="generated")
    ax.legend()
    if axtitle is not None:
        ax.set_title(axtitle)
    if dsc is not None: # also plot a contour
        xmin, ymin = min(generated[:, 0].min(), target[:, 0].min()), min(generated[:, 1].min(), target[:, 1].min())
        xmax, ymax = max(generated[:, 0].max(), target[:, 0].max()), max(generated[:, 1].max(), target[:, 1].max())
        # border = [-1, -1, 1, 1]
        border = 1
        # np.array([xmin, ymin, xmax, ymax]) += np.array(border)
        xmin -= border
        ymin -= border
        xmax += border
        ymax += border
        step = min(ymax-ymin, xmax-xmin) / 100
        xrng = torch.arange(xmin, xmax, step=step)
        yrng = torch.arange(ymin, ymax, step=step)
        X, Y = torch.meshgrid(xrng, yrng)
        pts = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
        # f.to(device)
        with torch.no_grad():
            Z = dsc(pts.to(device))
            if args is not None and args.loss == "ns-gan":
                Z = Z.exp() / (1 + Z.exp())

        contourf_ = ax.contourf(X, Y, Z.view_as(X).detach().cpu().numpy(), zorder=-1, alpha=0.5)
        cbar = fig.colorbar(contourf_)
        ax.legend()

    if fig is not None and title is not None:
        fig.suptitle(title)

    if fname is not None:
        plt.savefig(fname=fname, bbox_inches="tight")
        plt.close('all')
    return

def scatter_images(fname, target, generated,  mean, fig=None, ax=None,  title=None, axtitle=None, args=None):

    if ax is None:
        fig, ax =  plt.subplots(2, 1, figsize=(5, 5))
    device = generated.device
    generated = generated.detach().cpu() + mean.view(1, *generated.size()[1:])
    target = target.detach().cpu() + mean.view(1, *target.size()[1:])
    grid_target = vutils.make_grid(target.unsqueeze(1), nrow=1)
    grid_gen = vutils.make_grid(generated.unsqueeze(1), nrow=1)
    ax[0].imshow(grid_target.permute(1, 2, 0))
    ax[0].set_title("target")
    ax[1].imshow(grid_gen.permute(1, 2, 0))
    ax[1].set_title("generated")

    ax[0].legend()
    ax[1].legend()

    if fig is not None and title is not None:
        fig.suptitle(title)

    if fname is not None:
        plt.savefig(fname=fname, bbox_inches="tight")
        plt.close('all')
    return

def plot_gif(data, loss, fig, axes):
    """Create a gif from a list of heat maps"""

    hmap = axes[0].imshow(data[0], vmin=-1, vmax=1)
    line = axes[1].plot(loss[0], label="Eigenvalues distance")[0]
    axes[1].legend()
    num_iter = len(data)
    delta_x = 0.05 *  num_iter
    axes[1].set_xlim(xmin=-delta_x, xmax=num_iter+delta_x)
    delta_y = 0.05 * (max(loss) - min(loss))
    ymin = min(loss) - delta_y
    ymax = max(loss) + delta_y
    axes[1].set_ylim(ymin=ymin, ymax=ymax)
    axes[0].set_title("cos(Omega_i, U_j)")

    def update(i, hmap, line, data, loss):
        title = f"niter {i}"
        hmap.set_data(data[i])
        # take care of the axis
        line.set_data(range(i), loss[:i])
        # fig.colorbar(hmap, ax=axes[0], location='right')
        fig.suptitle(title)
        # return hmap, ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_iter, num_iter//(min(num_iter, 50))), interval=200, repeat=False, fargs=[hmap, line, data, loss])
    # plt.show()
    return anim, hmap, line




def plot(results):
    pass
