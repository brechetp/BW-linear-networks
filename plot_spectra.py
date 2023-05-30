import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
from pathlib import Path
import os
import pickle
from collections import defaultdict
import pandas as pd
import matplotlib as mpl
import tikzplotlib as tpl
from pandas import IndexSlice as idx
import pandas as pd

mpl.rcParams.update(
    {
        # Adjust to your LaTex-Engine
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.unicode_minus": False,
        "lines.linewidth":   3,
        "lines.markersize": 14,
        "savefig.bbox":"tight",
    }
)
# optional font setting
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
mpl.rc('font', **font)
mpl.rcParams['figure.figsize'] = (6, 6)  # Use the values from \printsizes

np.seterr(divide="ignore")  # ignore the log of -1

LABELS = {'Fro': "$L^N_F$", 'BW': "$L^N_{\\tau, BW}$"}

def main(args):

    for dname in args.dirnames:
    # lst_file = (glob.glob(os.path.join(dname, "**", "spectra.npz"), recursive=True))  # all config files)
        # lst_file = (glob.glob(os.path.join(dname, "**", "spectra.npz"), recursive=True))  # all config files)
        lst_files = Path(dname).rglob("spectra.pkl")
        lst_dirs = [f.parent for f in lst_files]  # find direct parent of the file

        for subdir in lst_dirs:
            plot_dir(subdir, args)
    # for fname in lst_file:

def close_all(fn):

    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        plt.close('all')
        return ret
    return wrapper

def plot_dir(dirname, args):

    if args.verbose:
        print(f"Inside {dirname}...")
    spfname = dirname.joinpath("spectra.pkl")
    pathfname = dirname.joinpath("paths.pkl")
    approxfname = dirname.joinpath("approx.pkl")
    # alfname= dirname.joinpath("alphas.pkl")
    plotdir = dirname.joinpath("plot")
    os.makedirs(plotdir, exist_ok=True)

    try:
        with open(spfname, "rb") as _f:
            spectra = pickle.load(_f)
        if args.verbose:
            print("Plotting spectra...")
        plot_spectra(spectra, plotdir, savecsv=True)
        if args.verbose:
            print("Done.")

    except IOError:
        print(f"Can't open {spfname} for read")

    try:
        with open(pathfname, "rb") as _f:
            paths = pickle.load(_f)
        if args.verbose:
            print("Plotting paths...")
        plot_paths(paths, plotdir)
        if args.verbose:
            print("Done.")
    except IOError:
        print(f"Can't open {pathfname}  for read")

    # try:
        # with open(approxfname, "rb") as _f:
            # approx = pickle.load(_f)
        # if args.verbose:
            # print("Plotting approximations...")
        # plot_approx(approx, plotdir)
        # if args.verbose:
            # print("Done.")
    # except IOError:
        # print(f"Can't open {approxfname}  for read")

    if args.verbose:
        print("Done.\n")


@close_all
def plot_approx(approx, plotdir):

    keys = [k for k in approx.keys() if len(k) == 3] # the different (loss,domain,axis)

    losses = set(k[0] for k in keys if len(k) == 3)  # Fro or BW
    domains = set(k[1] for k in keys if len(k) == 3)  # Func, Covar
    methods = set(k[2] for k in keys if len(k) == 3)  # AD, CF, fnval
    cpindices = range(approx[keys[0]].shape[0])  # a P x NT x NAPPROX array

    if "FroSqrt" in losses:
        losses.remove("FroSqrt")



    COLORS = { 'AD': 'tab:blue', 'CF': 'tab:orange' , 'fnval': 'tab:red'}
    LINESTYLES = { 'AD': 'dashdot', 'CF': 'dashed' , 'fnval': 'solid'}

    for loss in losses:
        for method in methods:
            for dom in domains:
                key = (loss, dom, method)
                # vkey = pkey + ("value",)
                # nPts, nT = paths[vkey].shape
                # loss, domain = key
                approx[key + ("mean",)] = np.mean(approx[key], axis=-2)  # mean over targets
                approx[key + ("std",)] = np.std(approx[key], axis=-2)  # std over the target
            # lzero[key] = nval - np.count_nonzero(values[key], axis=3)
            # compute the quartile of the values
            # Q[key] = np.percentile(values[key], (25, 50, 75), axis=3)  # of size 3xPxTxV
    for cpidx in cpindices:
        for dom in domains:
            for loss in losses:
                fig, axes = plt.subplots(1, 1)

                for method in methods:
                    key = (loss, dom, method)
                    # alkey = pkey + ("alpha",)

                    color = COLORS[method]
                    linestyle = LINESTYLES[method]
                    zorder = 1 if method == "AD" or method == "CF" else 0
                    # xs = paths[pkey + ("alpha",)]
                    ys = approx[key + ("mean",)][cpidx, :]
                    xs = range(len(ys))
                    err = approx[key + ("std",)][cpidx, :]

                    lines = axes.plot(ys, label=method, color=color, linestyle=linestyle, zorder=zorder)
                    # c = lines[-1].get_color()
                    axes.fill_between(xs, ys-err, ys+err, facecolor=color, alpha=0.5)
                axes.set_xlabel(f"dX ({dom}), CP # {cpidx}")
                axes.set_ylabel("loss")
                axes.set_title(f"SoA vs value function for {loss} on {dom} space")
                axes.legend()
                fname = plotdir.joinpath(f"approx_{loss}_{dom}_{cpidx}.pdf")
                plt.savefig(fname, bbox_inches="tight")
                plt.close()

@close_all
def plot_paths(paths, plotdir):

    keys = paths.keys()  # the different (loss,domain,axis)
    losses = set(k[0] for k in keys if k[0] != 'FroSqrt')  # Fro or BW
    domains = set(k[1] for k in keys)  # Param, Func




    COLORS = { 'Fro': 'tab:blue', 'BW': 'tab:orange' , 'FroSqrt': 'tab:red'}

    for loss in losses:
        for dom in domains:
            pkey = (loss, dom)
            vkey = pkey + ("value",)
            # nPts, nT = paths[vkey].shape
            # loss, domain = key
            paths[pkey + ("mean",)] = np.mean(paths[vkey], axis=-1)  # mean over targets
            paths[pkey + ("std",)] = np.std(paths[vkey], axis=-1)
        # lzero[key] = nval - np.count_nonzero(values[key], axis=3)
        # compute the quartile of the values
        # Q[key] = np.percentile(values[key], (25, 50, 75), axis=3)  # of size 3xPxTxV
    for dom in sorted(domains):
        fig, axes = plt.subplots(1, 1)

        for loss in losses:
            pkey = (loss, dom)
            alkey = pkey + ("alpha",)

            color = COLORS[loss]
            xs = paths[pkey + ("alpha",)]
            ys = paths[pkey + ("mean",)]
            err = paths[pkey + ("std",)]

            lines = axes.plot(xs, ys, label=loss, color=color)
            # c = lines[-1].get_color()
            axes.fill_between(xs, ys-err, ys+err, facecolor=color, alpha=0.5)
        axes.legend()
        axes.set_xlabel(f"Path index ({dom})")
        axes.set_ylabel("Loss")
        # axes.set_title(f"Losses along path between critical points on {dom} space")
        fname = plotdir.joinpath(f"values_{dom}.pdf")
        plt.savefig(fname, bbox_inches="tight")




@close_all
def plot_spectra(spectra, plotdir, savecsv=False):


    if savecsv:
        df = pd.Series(spectra)
        df.to_csv(plotdir.joinpath("spectra.csv"))
    keys = [k for k in spectra.keys() if len(k) == 3 and k[0] != 'FroSqrt']  # filter out FroSqrt
    lmin, lminabs, lmax, lzero, Q = [{} for _ in range(5)]
    lmin_mean, lmin_std, lmax_mean, lmax_std = [defaultdict(dict) for _ in range(4)]
    cnum_mean, cnum_std = [defaultdict(dict) for _ in range(2)]
    cnum_abs_mean, cnum_abs_std = [defaultdict(dict) for _ in range(2)]
    # AXES = {'T': 1, 'V': 2}  # the axes over which to mean
    # AXES = {'T': 1}  # the axes over which to mean, useless ?
    losses, domains, methods = [set() for _ in range(3)]
    COLORS = { 'Fro': 'tab:blue', 'BW': 'tab:orange' , 'FroSqrt': 'tab:red'}
    for k in keys:
        losses |= set([k[0]])
        domains |= set([k[1]])
        methods |= set([k[2]])
    for key in keys:  # is a tuple (loss, method)
        # the spectra are PxTxnd,  with P number of critical points, T the batch number for the target
        nP, nT, nval = spectra[key].shape
        loss, dom, method = key
        if dom == "Func" or dom == "Covar":  # don't plot the spectra on the function space
            continue
        lmin[key] = spectra[key][:, :, 0]  # take the min for all P x T
        lmax[key] = spectra[key][:, :, -1]
        lzero[key] = nval - np.count_nonzero(spectra[key], axis=-1)
        # compute the quartile of the spectra
        Q[key] = np.percentile(spectra[key], (25, 50, 75), axis=-1)  # of size 3xPxT
        # need the quartile to compute the outliers?



        # for axis_name, axis in AXES.items():  # compute the statistics wrt each axis
        # size nP x nT
        # axis = 2 if axis_name
        lmin_mean[key] = np.mean(lmin[key], axis=1)  # mean over the  T
        lmin_std[key] = np.std(lmin[key], axis=1)  # std over the  T
        cnum_std[key] = np.std(lmax[key] / lmin[key], axis=1)
        if args.yscale == "log":
            cnum_mean[key] = np.mean(lmax[key] / np.abs(lmin[key]), axis=1)
            # cnum_std[key] = np.log(cnum_std[key])
        else:
            cnum_mean[key] = np.mean(lmax[key] / lmin[key], axis=1)


        # lminabs[key] = np.min(np.nonzero(np.abs(spectra[key])))
        spectra[key][np.abs(spectra[key]) == 0] = np.inf
        lminabs[key] = np.min(np.abs(spectra[key]), axis=-1)
        cnum_abs_mean[key] = np.mean(lmax[key] / lminabs[key], axis=1)
        cnum_abs_std[key] = np.std(lmax[key] / lminabs[key], axis=1)

        lmax_mean[key] = np.mean(lmax[key], axis=1)  # mean over the T
        lmax_std[key] = np.std(lmax[key], axis=1)  # std over the T
        # size nP x nT


    MARKERS = {'BW': '+', 'Fro': 'x', 'FroSqrt': '^'}  # different markers for the losses
    COLORS = { 'Fro': 'tab:blue', 'BW': 'tab:orange' , 'FroSqrt': 'tab:red'}
    SIZES = {'BW': 3, 'Fro': 2}

    if 'FroSqrt' in losses:
        losses.remove('FroSqrt')

    if 'AD' in methods:
        methods.remove('AD')

    columns =pd.MultiIndex.from_product([losses, range(nP), ['mean', 'std']], names=('loss', 'i', 'stat'))
    columnsmd =pd.MultiIndex.from_product([losses, range(nP) ], names=('loss', 'i'))
    index = pd.Index(['lambda_min', 'lambda_max', 'kappa_rel', 'kappa_abs'])
    df = pd.DataFrame([], index=index, columns=columns)
    dfmd = pd.DataFrame([], index=index, columns=columnsmd)  # will keep strings
    for method in methods:  # compare the different methods

        # for axname in AXES.keys():  # one plot per mean ax
        fig, ax = plt.subplots(1, 1)
        if args.yscale == 'log':
            ax.set_yscale('log')
        for loss in sorted(losses, reverse=True):
            color = COLORS[loss]
            marker = MARKERS[loss]
            zorder=0 if loss == "Fro" else 1
            size = 4 if loss == 'BW' else 6
            key = (loss, "Param", method)
            ax.errorbar(range(nP), lmax_mean[key], yerr=lmax_std[key], linestyle='None', capsize=3, markersize=size, zorder=zorder, label=LABELS[loss], marker='^', color=color)
            df.loc['lambda_min', idx[loss, :, 'mean']] = lmin_mean[key]
            df.loc['lambda_max', idx[loss, :, 'mean']] = lmax_mean[key]
            df.loc['lambda_min', idx[loss, :, 'std']] = lmin_std[key]
            df.loc['lambda_max', idx[loss, :, 'std']] = lmax_std[key]
            # ax.errorbar(range(nP),min_mean[key], yerr=lmin_std[key], linestyle='None', capsi)))ze=3, markersize=size, zorder=zorder, label=f"{loss} lambda min", marker='v', color=color)
            # np.savetxt(plotdir.joinpath(f"lmax_mean_{loss}.csv"), lmax_mean[key], delimiter=',', fmt='%.2e')
            # np.savetxt(plotdir.joinpath(f"lmin_mean_{loss}.csv"), lmin_mean[key], delimiter=',', fmt='%.2e')
            # np.savetxt(plotdir.joinpath(f"lmax_std_{loss}.csv"), lmax_std[key], delimiter=',', fmt='%.2e')
            # np.savetxt(plotdir.joinpath(f"lmin_std_{loss}.csv"), lmin_std[key], delimiter=',', fmt='%.2e')
            dfmd.loc['lambda_min', idx[loss, :]] = [" {:.2e} ± {:.2e}".format(m, s) for m, s in zip(lmin_mean[key], lmin_std[key])]
            dfmd.loc['lambda_max', idx[loss, :]] = [" {:.2e} ± {:.2e}".format(m, s) for m, s in zip(lmax_mean[key], lmax_std[key])]
        df.to_csv(plotdir.joinpath(f"spectra.csv"), float_format="%.2e")
        dfmd.to_csv(plotdir.joinpath(f"spectramd.csv"), float_format="%.2e")
        ax.set_xlabel("Index $i$")
        ax.set_ylabel("$\lambda_{\\mathrm{max}}(\\nabla^{2} L^{N})$")
        # ax.set_title(f"Mean/std with {nT} targets with {method}")
        ax.legend()
        pname = plotdir.joinpath(f"spectra_{method}_{args.yscale}")
        pdffname = pname.with_suffix(".pdf")
        plt.savefig(pdffname, bbox_inches='tight')
        texfname = pname.with_suffix(".tex")
        tpl.save(texfname)

        fig, ax = plt.subplots(1, 1)
        if args.yscale == 'log':
            ax.set_yscale('log')
        for loss in sorted(losses, reverse=True):
            color = COLORS[loss]
            key = (loss, "Param", method)
            ax.errorbar(range(nP), cnum_mean[key], yerr=cnum_std[key], linestyle='None', capsize=3, label=LABELS[loss], marker=MARKERS[loss], color=color)
            np.savetxt(plotdir.joinpath(f"cnum_mean_{loss}.csv"), cnum_mean[key], delimiter=',')
            np.savetxt(plotdir.joinpath(f"cnum_std_{loss}.csv"), cnum_std[key], delimiter=',')
            # ax.errorbar(range(nP), lmax_mean[key], yerr=lmax_std[key], linestyle='None', capsize=3, label=f"{loss} lambda max", marker=MARKERS[loss])
            dfmd.loc['kappa_rel', idx[loss, :]] = [" {:.2e} ± {:.2e}".format(m, s) for m, s in zip(cnum_mean[key], cnum_std[key])]
        if args.yscale == 'log':
            ax.set_ylabel("$|\kappa_{\\mathrm{rel}}|$")
        else:
            ax.set_ylabel("$\kappa_{\\mathrm{rel}}$")

        ax.set_xlabel("Index $i$")
        # ax.set_ylabel("Relative condition number of the Hessian")
        # ax.set_title(f"Mean/std with {nT} targets")
        ax.legend()
        pname = plotdir.joinpath(f"cnum_{method}_{args.yscale}")
        pdffname = pname.with_suffix(".pdf")
        plt.savefig(pdffname, bbox_inches='tight')
        texfname = pname.with_suffix(".tex")
        tpl.save(texfname)

        fig, ax = plt.subplots(1, 1)
        if args.yscale == 'log':
            ax.set_yscale('log')
        for loss in sorted(losses, reverse=True):
            color = COLORS[loss]
            key = (loss, "Param", method)
            ax.errorbar(range(nP), cnum_abs_mean[key], yerr=cnum_abs_std[key], linestyle='None', capsize=3, label=LABELS[loss], marker=MARKERS[loss], color=color)
            np.savetxt(plotdir.joinpath(f"cnum_abs_mean_{loss}.csv"), cnum_abs_mean[key], delimiter=',')
            np.savetxt(plotdir.joinpath(f"cnum_abs_std_{loss}.csv"), cnum_abs_std[key], delimiter=',')
            # ax.errorbar(range(nP), lmax_mean[key], yerr=lmax_std[key], linestyle='None', capsize=3, label=f"{loss} lambda max", marker=MARKERS[loss])
            dfmd.loc['kappa_abs', idx[loss, :]] = [" {:.2e} ± {:.2e}".format(m, s) for m, s in zip(cnum_abs_mean[key], cnum_abs_std[key])]
        ax.set_xlabel("Index $i$")
        ax.set_ylabel("$\kappa_{\\mathrm{abs}}$")
        # ax.set_title(f"Mean/std with {nT} targets")
        ax.legend()
        pname = plotdir.joinpath(f"cnum_abs_{method}_{args.yscale}")
        pdffname = pname.with_suffix(".pdf")
        plt.savefig(pdffname, bbox_inches='tight')
        texfname = pname.with_suffix(".tex")
        tpl.save(texfname)
    # fig, axes = plt.subplots(1, nP, figsize=((nP+1)*4, 4), sharey=False)
        dfmd.to_csv(plotdir.joinpath(f"table.csv"), float_format="%.2e")

        plt.close('all')

    # for p in range(nP):
        # axes[p].boxplot(lmin_BW[p, :], autorange=True)
        # axes[p].boxplot(lmax_BW[p, :], autorange=True)
        # axes[p].set_xticks([1],[p])

    # fig.suptitle("Min / max eigenvalues for BW")
    # # plt.legend()
    # pname = plotdir.joinpath(f"boxplot-BW.pdf")
    # plt.savefig(pname)
    # fig, axes = plt.subplots(1, nP, figsize=((nP+1)*4, 4), sharey=False)

    # for p in range(nP):
        # axes[p].boxplot(lmin_Fro[p, :], autorange=True)
        # axes[p].boxplot(lmax_Fro[p, :], autorange=True)
        # axes[p].set_xticks([1],[p])

    # fig.suptitle("Min / max eigenvalues for Fro")
    # # plt.legend()
    # pname = plotdir.joinpath(f"boxplot-Fro.pdf")
    # plt.savefig(pname)
    # plotnametex = plotdir.joinpath(f"boxplot-BW.tex")
    # tpl.save(plotnametex)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dirnames", nargs='*', help="the name of the npz file to plot")
    parser.add_argument("--verbose", action='store_true', help='verbose mode')
    parser.add_argument("--yscale", "-ys", choices=("log", "lin"), help="scale for y axis")
    args = parser.parse_args()




    main(args)
