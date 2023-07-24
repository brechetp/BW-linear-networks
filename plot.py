import matplotlib.pyplot as plt
import torch

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



