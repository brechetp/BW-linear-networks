import json
import os
import sys
import argparse
import glob
from collections import defaultdict
import shutil
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import utils
import numpy as np
import matplotlib.cm as cm
import traceback
import seaborn as sns
import scipy as sp
import math
import multiprocessing as mp   # multithread processing
from functools import partial  # partial evaluation of a function
from matplotlib.cm import ScalarMappable
# plt.rcParams['text.usetex'] = True
import matplotlib as mpl
import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

# mpl.use("pgf")
mpl.rcParams.update(
    {
        # Adjust to your LaTex-Engine
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.unicode_minus": False,
        "lines.linewidth":   5,
        "lines.markersize": 14,
        "savefig.bbox":"tight",
    }
)
# optional font setting
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 24}
mpl.rc('font', **font)
mpl.rcParams['figure.figsize'] = (6, 6)  # Use the values from \printsizes

np.seterr(divide="ignore")  # ignore the log of -1

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def main(args):
    if args.diff:
        diff(args)
    elif args.compute:
        compute2(args)
    elif args.compare is not None or args.plot or args.contour:
        plot(args)
    else:
        export(args)

def diff(args):

    lst_file = []
    exclude = ["name", "dataset_fname", "config", "time"]
    for dr in args.dirs:

        # lst_file.extend(glob.glob(os.path.join(dr, "**", "config.json"), recursive=True))  # all config files)
        lst_file = glob.glob(os.path.join(dr, "**", "config.json"), recursive=True)  # all config files)

        if len(lst_file) < 2:  # do not process if less than 2 different configurations
            continue

        root = os.path.commonpath(lst_file)
        diff_dict = cmpt_dict(lst_file, exclude)# for k,v in merge_dct.items():
            # if len(v) == 1:
                # merge_dct.pop(k)

        with open(os.path.join(root, "filter.json"), 'w') as fp:
            json.dump(diff_dict, fp, indent=0, sort_keys=True, default=set_default)

    return

def cmpt_dict(lst_file, exclude=["name", "dataset_fname", "config"]):

    merge_dct = defaultdict(set)
    for f in lst_file:

        with open(f) as fconf:
            data = json.load(fconf)
            if not merge_dct:  # first time
                for k,v in data.items():
                    if isinstance(v, (list, dict)):
                        continue
                    merge_dct[k] |= set([v])   # convert the values to set
            else:
                for k,v in data.items():
                    if isinstance(v, (list, dict)):
                        continue
                    if k not in merge_dct.keys():
                        print("Warning! ", k, " not in the keys")
                    merge_dct[k] |= set([v])

    diff_dct = {k :v for k,v in merge_dct.items() if len(v) > 1 and not k in exclude}
    return diff_dct

def export(args):

    with open(args.filter) as fconf:
        conf = json.load(fconf)
    dname, name = os.path.split(os.path.splitext(args.filter)[0])

    lst_file = (glob.glob(os.path.join(dname, "**", "config.json"), recursive=True))  # all config files)

    outroot = os.path.join(os.path.dirname(args.filter), "merge", "export")
    pnames = ["loss", "ranks"]
    outdir = os.path.join(outroot, name)
    if args.reset:
        shutil.rmtree(outdir, ignore_errors=True)
    for f in lst_file:

        with open(f) as fconf:
            data = json.load(fconf)
        if not all([data[k] in conf[k] for k in conf.keys()]):
            continue
        root = os.path.dirname(f)
        namestr = to_string(sorted(conf.keys()), data)
        for pname in pnames:
            try:
                src = os.path.join(root, "plot", f"{pname}.pdf")
                # outdir = os.path.join(outroot, name, pname)
                dst = os.path.join(outdir, pname, f"{namestr}.pdf")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
            except FileNotFoundError:
                traceback.print_exc()


def compute(args):

    EPS = 4
    dname, name = os.path.split(os.path.splitext(args.filter)[0])
    lst_file = (glob.glob(os.path.join(dname, "**", "config.json"), recursive=True))  # all config files)

    with open(args.filter) as fconf:
        conf = json.load(fconf)

    outdir = os.path.join(dname, "export", "compute")
    os.makedirs(outdir, exist_ok=True)

    for fname in lst_file:

        dfname = os.path.join(os.path.dirname(fname), "dfs.pkl")
        try:
            with open(dfname, "rb") as _f:
                dfs = pickle.load(_f)
            with open(fname) as fconf:
                fargs = json.load(fconf)
        except:
            print("can't open the dataframe at ", fname)
            continue


        key = tuple([fargs[k] for k in conf.keys()])
        outname = df_to_string(key, conf.keys())
        # outfold = os.path.join(outdir, outname)
        os.makedirs(outdir, exist_ok=True)
        df = dfs["quant"].dropna(axis=1, how='all')
        lr = fargs["learning_rate"]
        if not 'time' in df.columns:
            df["time"] = df.index * lr   # assume a constant learning rate

        diff = df["diff loss to OPT"]
        logdiff = np.log10(diff)
        wass = df["wass. dist to OPT"]  # criterion for convergence
        if not (wass < EPS).any():
            print("theshold not crossed in ", fname)
            continue  # leave the slope at NaN, not converged
        # df.loc[:, "slope"] = float('nan')
        logwass = np.log10(wass)  # the log10 of wass
        s = int(0.01 / lr + 0.5)  # sample in one per centisecond
        logwassewm = logwass.ewm(span=s).mean()[wass.index[::s]]  # subsample the signal after smoothing it
        dlogwass = (logwassewm.shift(-1) - logwassewm.shift(1)) /2

        dabs = dlogwass.abs()

        d2abs = (dabs.shift(-1) - dabs.shift(1)) / 2
        d2abs0 = d2abs.copy()
        d2abs0[d2abs.abs() < d2abs.abs().max()/10] = 0

        posabs = d2abs0 > 0
        negabs = d2abs0 < 0
        nullabs = d2abs0 == 0

        begidx = np.where(nullabs & posabs.shift(-1))[0]
        endidx = np.where(negabs & nullabs.shift(-1))[0]

        begpts = d2abs.index[begidx]
        endpts = d2abs.index[endidx]

        th = dabs.max() * 2/3
        mask = dabs > th
        components = np.where(mask)[0]
        brkpts = np.where(np.diff(components) != 1)[0]  # the breakpoints at which the connected components end
        aas = [components[0]] + components[brkpts + 1].tolist()
        bbs = components[brkpts].tolist() + [components[-1]]  # add the last point
        # idx = components[0]  # first index where the threshold is reached
        # intervals = []  # list of possible intervals for the linear regression

        # ptidx = components[brkpts]  # the breakpoints in the mask frame
        # frameidx = dabs.index[ptidx]  # the breakpoints in the original frame
        aaframe = dabs.index[aas]
        bbframe = dabs.index[bbs]
        aaarr = np.array(aaframe)  # numpy conversion
        bbarr = np.array(bbframe)
        begarr = np.array(begpts)
        endarr = np.array(endpts)
        # test all the different lengths
        testbeg = (begarr.reshape(1, -1, 1) - aaarr.reshape(-1, 1, 1)) >= 0
        testend = (bbarr.reshape(-1, 1, 1) - endarr.reshape(1, 1, -1)) >= 0
        testBE = ((endarr.reshape(1, 1, -1) - begarr.reshape(1, -1, 1) > 0)  |
                  (bbarr.reshape(-1, 1, 1) - begarr.reshape(1, -1, 1) > 0) |
                  (endarr.reshape(1, 1, -1) - aaarr.reshape(-1, 1, 1) > 0))

        # test

        allowed = np.where((testbeg  | testend ) & (testBE))  # requires the correct interval orientation

        if len(allowed[0]) == 0:  # we did not find
            print("can't find a slope in ", fname)
            continue

        maxidx_pts = allowed[0].max()  # the index of the last correct interval in allowed frame
        maxidx_beg = allowed[1].max()
        maxidx_end = allowed[2].max()

        # inteval_idx = (allowed[1][maxidx], allowed[2][maxidx])

        interval = max(aaarr[maxidx_pts], begarr[maxidx_beg]), bbarr[maxidx_pts]

        if interval[0] >= interval[1]:
            print("Wrong interval value in ", fname)
            continue



        x = logdiff.index[interval[0]:interval[1]]
        y = logdiff[interval[0]:interval[1]]
        mask = np.isfinite(y)  # only consider finite data
        regress = sp.stats.linregress(x[mask]*lr, y[mask]) # slope in s^-1

        plt.figure()
        ax = logdiff.plot()
        ax.plot(x, regress.intercept + regress.slope * x, 'r')
        outstat = os.path.join(outdir, "regress")
        os.makedirs(outstat, exist_ok=True)
        plt.savefig(os.path.join(outstat, f"{outname}.pdf"))
        plt.close("all")

        computed = {'regress': regress, 'interval': interval}
        savefname = os.path.join(os.path.dirname(fname), "regress.pkl")
        with open(savefname, 'wb') as fp:
            pickle.dump(computed, fp)

def check_params(conf, fname):
    """Checks if a json file has allowed parameter compared to a config file"""

    with open(fname, 'r') as fp:

        fconf = json.load(fp)

    return all([fconf[k] in conf[k] for k in conf.keys()])


def compute2(args):

    EPS = 1e-4
    if args.filter is not None:
        dname, name = os.path.split(os.path.splitext(args.filter)[0])
        lst_file = (glob.glob(os.path.join(dname, "**", "config.json"), recursive=True))  # all config files)
    # lst_regress = glob.glob(os.path.join(dname, "**", "regress.pkl"), recursive=True)

        with open(args.filter) as fconf:
            conf = json.load(fconf)

        lst_file = [l for l in lst_file if check_params(conf, l)]

    else:
        conf = None
        lst_file = []
        lst_lst = [glob.glob(os.path.join(dir_, "**", "config.json"), recursive=True) for dir_ in args.dirs]
        for sublst in lst_lst:
            lst_file.extend(sublst)

    if args.filter is not None:
        dname, name = os.path.split(os.path.splitext(args.filter)[0])
        outdir = os.path.join(dname, "merge", name, "compute") if name != "filter" else os.path.join(dname, "merge", "compute")
    else:
        if len(lst_file) > 1:  # more than one file to process
            dname = os.path.commonpath(lst_file)
            outdir = os.path.join(dname, "merge", "compute")
            conf = cmpt_dict(lst_file)
        else:
            dname = os.path.dirname(lst_file[0])
            outdir = os.path.join(dname, "compute")


    os.makedirs(outdir, exist_ok=True)
    failed_lst = []  # list for all the failed processings
    failed_fname = os.path.join(outdir, "failed.txt")
    partial_cpt = partial(compute_fname, conf=conf, outdir=outdir)
    # if os.path.isfile(f):
        # os.remove(FAILED)
    with mp.Pool(processes=args.nprocs) as pool:
        pool.map(partial_cpt, lst_file)

    # processes = [Process(target=compute_fname, args=(fname,conf,outdir)) for fname in lst_file]

    # for prc in processes:
        # prc.start()

    # for prc in processes:
        # prc.join()

# failed_lst.sort()
# with open(failed_fname, 'w') as _f:
    # _f.writelines("\n".join(failed_lst))


def compute_fname(fname, conf, outdir):


    dfname = os.path.join(os.path.dirname(fname), "dfs.pkl")
    try:
        with open(dfname, "rb") as _f:
            dfs = pickle.load(_f)
        with open(fname) as fconf:
            fargs = json.load(fconf)
    except:
        print("can't open the dataframe at ", fname)
        # failed_lst.append(fname)
        # continue
        return


    savefname = os.path.join(os.path.dirname(fname), "regress.pkl")
    if os.path.isfile(savefname) and not args.force:
        # continue
        return
    if conf is not None:
        key = tuple([fargs[k] for k in conf.keys()])
        outname = df_to_string(key, conf.keys())
    else:
        outname = "debug"
    # outfold = os.path.join(outdir, outname)
    # os.makedirs(outdir, exist_ok=True)
    df = dfs["quant"].dropna(axis=1, how='all')
    lr = fargs["learning_rate"]
    if not 'time' in df.columns:
        df["time"] = df.index * lr   # assume a constant learning rate

    WIN_SIZE = int(0.1  / lr + 0.5)  # window size over which we compute the linear regression in samples
    diff = df["diff loss to OPT"]
    logdiff = np.log10(diff)
    wass = df["wass. dist to OPT"]  # criterion for convergence
    # if not (wass < EPS).any():
        # print("theshold not crossed in ", fname)
        # continue  # leave the slope at NaN, not converged
    # df.loc[:, "slope"] = float('nan')
    logwass = np.log10(wass)  # the log10 of wass
    logwassfin = logwass[np.isfinite(logwass)]
    logdifffin = logdiff[np.isfinite(logdiff)]
    s = int(1e-4 / lr + 0.5)  # sample in 1 per millisecond
    logwassewm = logwassfin.ewm(span=s).mean()[logwassfin.index[::s]]  # subsample the signal after smoothing it
    logdiffewm = logdifffin.ewm(span=s).mean()[logdifffin.index[::s]]  # subsample the signal after smoothing it
    diffewm = diff.ewm(span=s).mean()[diff.index[::s]]


    mask = (logdiffewm < -2) & (logdiffewm > -3.5)  #
    # mask = (diffewm < 1e-2) & (diffewm > 1e-4 )  #
    # mask = (logdiffewm > -3.5) #

    # brkpts = np.where(np.diff(np.where(mask)[0]) != 1)[0]
    windows_idx = np.where(np.diff(mask))[0]
    if mask.iloc[0]:
        windows_idx = np.concatenate(([0], windows_idx), axis=0)
    if mask.iloc[-1]:
        windows_idx = np.concatenate((windows_idx, [len(mask)-1]), axis=0)
    window_idx = None

    if len(windows_idx) == 0 or len(windows_idx)==1:
        print("Signal not found at ", fname)
        # failed_lst.append(fname)
        # with open(FAILED, 'a') as _f:
            # _f.write(fname, '\n')
        # continue
        return

    windows_size = np.diff(windows_idx)[::2]  # every two are the lengths of the True regions
    win_max = 2*np.argmax(windows_size)
    window_idx = ( windows_idx[win_max],  windows_idx[win_max+1] )



    # interval = mask.index[window_idx[0]], mask.index[min(len(mask)-1, window_idx[1]+1)]
    interval = max(0, mask.index[window_idx[1]] - WIN_SIZE), mask.index[window_idx[1]]
    # assert all(mask.loc[interval[0]:interval[1]])

    x = logdiff.index[interval[0]:interval[1]]
    y = logdiff[interval[0]:interval[1]]
    # x = diff.index[interval[0]:interval[1]]
    # y = diff[interval[0]:interval[1]]
    mask = np.isfinite(y)
    regress = sp.stats.linregress(x[mask]*lr, y[mask]) # slope in s^-1
    # regress = sp.stats.linregress(x*lr, y) # slope in s^-1
    lograte = np.log(diff[interval[1]-1] / diff[interval[0]]) / (lr*(diff.index[interval[1]] - diff.index[interval[0]]))

    plt.figure()
    ax = logdiff.plot()
    logdiffewm.plot(ax=ax)
    ax.plot(x, regress.intercept + regress.slope*lr * x, 'r')
    outstat = os.path.join(outdir, "regress")
    os.makedirs(outstat, exist_ok=True)
    plt.savefig(os.path.join(outstat, f"{outname}.pdf"))
    plt.close("all")

    computed = {'regress': regress, 'interval': interval, 'lograte': lograte}
    with open(savefname, 'wb') as fp:
        pickle.dump(computed, fp)


def append_df(df_lst, keyval, args, xaxis, yaxis, orderkey):
    # f = fnames[key]
    key, f = keyval
    dfname = os.path.join(os.path.dirname(f), "dfs.pkl")
    try:
        with open(dfname, "rb") as _f:
            dfs = pickle.load(_f)
        with open(f, 'r') as _f:  # the configuration of the run
            fargs = json.load(_f)

    except:
        return


    # ForkedPdb().set_trace()
    if yaxis in ("slope", "lograte") or args.contour in ("slope", "r", "lograte"):
        regressfname = os.path.join(os.path.dirname(f), "regress.pkl")
        try:
            with open(regressfname, "rb") as _f:
                data = pickle.load(_f)
                regress = data["regress"]
                # lograte = data["lograte"]
        except:
            return

    df = dfs["quant"].dropna(axis=1, how='all')
    # if we want to compute additional quantities
    # compute the slope if it as converged
    if xaxis != df.index.name:  #means the xaxis is a hyperparameter
        df = df.tail(1)  # only keep the last one,
    if yaxis in "slope" or args.contour in ("slope", "r"):
        df["slope"] = regress.slope
        df["r"] = r2 = regress.rvalue**2
        df["Pred_C"] = dfs["vals"]["Prec_C"]
    # elif yaxis == "lograte" or args.contour in ("lograte"):  # lograte not
    # working
        # df["lograte"] = lograte
    if yaxis == "smin":
        try:
            vals = dfs["vals"]
            if all([k in vals.keys() for k in ["smin", "smin*", "smin**"]]):
                smin, sminstar, smin2star = vals["smin"], vals["smin*"], vals["smin**"]
            else:  # compute the value of smin based on the trainset
                ds =fargs["dataset_fname"]
                sets, _ = utils.load_dataset(fname=ds)  #
                if sets:
                    trainset = sets["train"]
                    # args.load_dataset = fname_ds
                    # STD = trainset.std
                    # Sigma0 = STD.mm(STD.t())  # covariance of the target distribution
                    dmin = min(fargs["xdim"], fargs["zdim"], fargs["width"])
                    r = min(fargs["xdim"], fargs["zdim"])
                    lambdas = trainset.Lambda
                    smin2star = math.sqrt(lambdas[dmin-1])
                    sminstar = math.sqrt(lambdas[r-1])
                    smin = math.sqrt(lambdas[-1])
                else:
                    raise IOError("Can't open the trainset for read")
            df["smin**"] = smin2star
            df["smin*"] = sminstar
            df["smin"] = smin
        except:
            traceback.print_exc()

    # index = index.union(df.index)
    # index.name = "niter"
    if args.contour and yaxis == "slope":
        tpls = [(*key, regress.slope, s) for s in df.columns]
    # elif args.contour == "slope" and yaxis == "smin":
        # tpls = [(*key, s) for s in df.columns]
    else:
        tpls = [(*key, s) for s in df.columns]
    columns = pd.MultiIndex.from_tuples(tpls,names=orderkey + ["stat"])
    df.columns = columns
    # columns = columns.union(df.columns)
    # df_merge = df_merge.reindex(index)
    df_lst.append(df)
    # if df_merge.empty:  # first time in the loop
        # df_merge = df
    # else:
        # df_merge = pd.concat([df_merge, df], axis=1)# .loc[:, key] = df




def plot(args):  # assume the compare key is not None

    with open(args.filter) as fconf:
        conf = json.load(fconf)
    dname, name = os.path.split(os.path.splitext(args.filter)[0])

    yaxis = args.y  # it's a list, make it a loop?

    if args.y in ("slope", "lograte") or args.contour in ("slope", "r"):
        # regressfname = os.path.join(os.path.dirname(f), "regress.pkl")
        lst_file_regress = (glob.glob(os.path.join(dname, "**", "regress.pkl"), recursive=True))  # all config files)
        lst_file = [f.replace("regress.pkl", "config.json") for f in lst_file_regress]
    else:
        lst_file = (glob.glob(os.path.join(dname, "**", "config.json"), recursive=True))  # all config files)

    outroot = os.path.join(os.path.dirname(args.filter), "merge")
    pnames = ["loss", "ranks"]
    if name == "filter":
        name = ''  # remove the "filter" part if not changed
    outdir = os.path.join(outroot, name)

    compkey = args.compare
    xaxis = args.x
    keys = []
    noncomp = [k for k in sorted(conf.keys()) if k!=compkey and k!=xaxis and  k !=yaxis and len(conf[k]) > 1]
    if compkey:
        orderkey =[ compkey] +  noncomp
    else:
        orderkey = noncomp

    if args.contour:
        # yaxis = args.y[0]
        # if yaxis in conf.keys():
        orderkey = orderkey + [xaxis, yaxis]  # keep the contour the most enclosed
        # else:
            # orderkey = orderkey + [xaxis]  # will use the computed value as axis
    # if args.contour or xaxis != "niter":
    elif xaxis != "niter":
        orderkey = [xaxis] + orderkey

    fnames = dict()
    shared_conf = None
    for  f in lst_file:  # all possible experiemnets, will filter them based on conf

        with open(f) as fconf:
            data = json.load(fconf)
        try:
            if not all([data[k] in conf[k] for k in conf.keys()]):
                # if we don't have all the same parameters as the conf filter
                continue
        except:
            # print("missing key {}".format(k))
            continue

        key = tuple([data[k] for k in orderkey if k in data.keys() and (not k in conf.keys() or len(conf[k]) > 1)])
        if key in keys:
            print(f"Key {key} already in keys!")
        keys.append(key)
        fnames[key] = f
        shared_conf = {k:v for k,v in shared_conf.items() if k in data.keys() and v == data[k]} if shared_conf is not None else data


    # columns = pd.MultiIndex.from_tuples(keys, names = orderkey)
    columns = pd.Index([])  # empyt at first
    # index
    df_merge = pd.DataFrame(index=pd.Index([], name=xaxis), columns=columns)
    index = df_merge.index
    columns = df_merge.columns
    if args.contour:
        plotdir = os.path.join(outdir, "contour", f"{xaxis}-{yaxis}")
    elif compkey is not None:
        plotdir = os.path.join(outdir, "plot", xaxis, compkey)
    # elif args.heatmap:
        # plotdir = os.path.join(outdir, "heatmap", f"{xaxis}-{yaxis}")
    else:
        plotdir = os.path.join(outdir, "plot", f"{xaxis}")
    os.makedirs(plotdir, exist_ok=True)

    with open(os.path.join(plotdir, "shared_conf.json"), 'w') as fp:
        json.dump(shared_conf, fp, indent=0, sort_keys=True, default=set_default)

    if args.reset:
        shutil.rmtree(plotdir, ignore_errors=True)

    # for stat in args.y:  # each quantity after the other
    df_lst = []

    # for key in fnames.keys():  # for all the configurations, need to merge the dataframe

    partial_fn = partial(append_df, args=args, xaxis=xaxis, yaxis=yaxis, orderkey=orderkey)

    pool = mp.Pool(processes=args.nprocs)
    manager = mp.Manager()
    L = manager.list()
    [pool.apply_async(partial_fn, args=[L, (key,f)]) for (key,f) in fnames.items()]
    pool.close()
    pool.join()

    # ForkedPdb().set_trace()
    df_merge = pd.concat(L, axis=1)
    if noncomp:  # if there are other keys
        dfu = df_merge.unstack().unstack(level=noncomp)  # group the different results with
    else:
        # we don't need to compare a key
        dfu = df_merge.unstack().dropna(how='all').droplevel("niter").unstack()  # problem if there are no other keys! todo
        # remove the niter level, dangerous? not test on many cases
        # but otherwise if multiple niter for different experiments leads to
        # many NaNs,
        # Before, the niter are the columns? and along the x is the compare key

    dfu = dfu.dropna(how='all', axis=0)  # test ?
    totstats = df_merge.columns.get_level_values("stat").unique()
    userdef = args.contour if args.contour else args.y
    stats =  totstats if userdef is None else  [userdef]
    # if args.contour:

    for stat in stats:  # for all the stats we want to plot
        # dfu = df_merge.unstack()
        if "stat" in dfu.index.names:
            df_filt = dfu.loc[dfu.index.get_level_values("stat") == stat,: ].droplevel("stat")
        else:  # in the collumns, only the case when noncomp is False?
            df_filt = dfu.loc[:, dfu.columns.get_level_values("stat") == stat]
            df_filt = df_filt.stack().droplevel("stat")

        def process_contour(df_filt, dfu, stat, args=args):

            iterkeys = noncomp or df_filt.isna().any().any()  #  iterate over the keys
            if iterkeys:  # if we have some missing values
                keys = df_filt.keys()  # we take all remaining keys
            else:
                keys = ['']

            fill = stat == "r"
            for key in keys:  # each of the configurations
                keyname = ', '.join(dfu.keys().names)
                string = df_to_string(key, dfu.keys().names) if iterkeys else str(key) # the name of the file
                df_plot = df_filt[key] if iterkeys else df_filt
                df_plot = df_plot.dropna()
                if "niter" in df_plot.index.names:  # if we have several niter per configuration
                    all_xy = df_plot.index.droplevel("niter")
                    if len(all_xy) == len(all_xy.unique()):
                        df_plot = df_plot.droplevel('niter')
                    else:
                        pass # todo: take the last niter for each configuration

                if df_plot.empty:
                    continue

                if iterkeys:
                    df_plot = df_plot.unstack(level=xaxis)
                plotfolder = os.path.join(plotdir, string)
                # niterstr = key if df_filt.columns.name == 'niter' else ''
                if isinstance(df_plot, pd.DataFrame) and df_plot.columns.name == 'niter':
                    niter = df_plot.columns[-1]  # take the last iteration for plotting
                    df_plot = df_plot[niter]
                    df_plot = df_plot.unstack(0)  # put it in a 2d array
                elif isinstance(df_plot, pd.Series):
                    df_plot = df_plot.unstack(level=xaxis)  # put it in a 2darray
                    # niterstr = f"iter-{niter}"
                # plt.contourf(df_plot.reindex(df_plot.inde
                # ax = sns.contour(df_plot.reindex(df_plot.index.sort_values(ascending=False)))  # flip the y values
                xx = df_plot.columns
                yy = df_plot.index
                X, Y = np.meshgrid(xx, yy)
                # vmin = int(df_plot.min().min() + 0.5)
                vmin = df_plot.min().min()
                # vmax = max(vmin+1, int(df_plot.max().max() + 0.5))
                vmax = df_plot.max().max()
                if vmin == vmax:
                    continue
                levels = 40
                level_boundaries = np.linspace(vmin, vmax, levels+1)
                fig, ax = plt.subplots(figsize=(7, 6))
                if fill:
                    ctr = ax.contourf(X, Y, df_plot, level_boundaries, vmin=vmin, vmax=vmax)  # flip the y values
                else:
                    ctr = ax.contour(X, Y, df_plot, level_boundaries, vmin=vmin, vmax=vmax)  # flip the y values
                os.makedirs(plotfolder, exist_ok=True)
                plotnamepgf = f"{stat}.pgf"
                plotname = f"{stat}.pdf"
                plotnametex = f"{stat}.tex"
                # plt.title(stat + niterstr )

                keyval = str(key) if not isinstance(key, tuple) else ', '.join([str(k) for k in key])
                ctr.axes.set_xlabel(xx.name)
                ctr.axes.set_ylabel("$\lambda_{\\mathrm{min}}(\Sigma_0^{1/2})$")
                ctr.axes.figure.colorbar(ScalarMappable(norm=ctr.norm, cmap=ctr.cmap),
                                         ticks = np.linspace(vmin, vmax, num=5),
                                         boundaries=level_boundaries,
                                         values= (level_boundaries[:-1] + level_boundaries[1:])/2,
                                         ax=ctr.axes
                                         )
                # ctr.axes.figure.suptitle(f"{stat} for {keyname} = {keyval}")
                plt.savefig(os.path.join(plotfolder,  plotname), bbox_inches="tight")
                plt.savefig(os.path.join(plotfolder,  plotnamepgf))
                plt.close("all")


        def process_keys(noncomp, df_filt, dfu, stat):
            if noncomp or not noncomp and compkey is None:  # we have keys
                keys = df_filt.keys()
            elif compkey is not None:  # only compkey to vary on
                keys = [compkey]

            for key in keys:
                string = df_to_string(key, dfu.keys().names) if noncomp else str(key) # the name of the file
                df_plot = df_filt[key].dropna(how='all', axis=0) if noncomp else df_filt

                niter = None
                several_niter = False
                if xaxis != "niter":
                    if "niter" in df_plot.index.names:
                        all_keys =  df_plot.index.droplevel("niter")
                        if len(all_keys) == len(all_keys.unique()):  # one iter per configuration

                            niter = df_plot.index.get_level_values("niter")  # keep for reference?
                            df_plot = df_plot.droplevel("niter")
                        # else:
                            # several_niter = True

                        else:
                            # niter = niter[-1]
                            print("Several niter found for ", key)
                            # if compkey is not None and compkey != "niter":
                            if compkey is not None and compkey != "niter":
                                niter = niter[-1]  # take the last niter available
                                df_plot = df_plot.loc[df_plot.index.get_level_values("niter") == niter, :]
                            else:
                                several_niter = True
                            # else:
                                # compkey = "niter"
                                # df_plot = df_plot.unstack(level="niter")

                        # niterstr = ", niter = " + str(niter)
                    else:
                        niter = key
                    if not isinstance(niter, pd.Index):
                        niterstr = ", niter = " + str(niter)
                    else:
                        niterstr = ""

                else:
                    niterstr = ""
                if compkey is not None:
                    df_plot = df_plot.unstack(level=compkey)
                if compkey is None and several_niter:  # we comp on the niter
                    df_plot = df_plot.unstack(level="niter")
                # df_piv = df_filt.unstack().reset_index()#.dropna(axis=0, how="any")
                # dfu = df_piv.pivot(index=xaxis, columns=compkey, values=0)
            # for key in dfu.keys():  # for each configuration
                # df_plot = dfu.loc[:, key].to_frame(name=stat)
                if df_plot.empty:
                    continue
                kwargs = {}
                if stat == "rank W":
                    kwargs["alpha"] = 0.3
                if stat in ("slope", "lograte"):
                    compute_regress = True
                    kwargs["style"] = '+'
                    kwargs["zorder"] = 2  # in front of the regression line

                # if stat in ("rank W", "norm grad"):
                    # kwargs["style"] = "."
                # ax = df_plot.unstack(level=compkey).plot(y=stat, **kwargs)
                if not args.contour:
                    ax = df_plot.plot(**kwargs)
                    if stat == args.y and stat in ("slope", "lograte"):  # plot the slope
                        # ax.set_xlabel(None)
                        # leg = ax.get_legend()
                        # leg_labels = leg.get_texts()
                        handles, labels = ax.get_legend_handles_labels()
                        if isinstance(df_plot, pd.DataFrame) and len(df_plot.columns) > 1:
                            for idx, ckey in enumerate(df_plot.columns):  # the total compare keys available
                                reg = sp.stats.linregress(df_plot[ckey].index, df_plot[ckey])
                                linplot = ax.plot(df_plot[ckey].index, reg.slope * df_plot[ckey].index + reg.intercept, color=ax.lines[idx].get_color(), linestyle='dashed')
                                r2 = reg.rvalue**2
                                labels.append(f"$a={reg.slope:.2f}$, $r^2$={r2:.3f}")
                                handles.append(linplot[0])

                        else:  # only one linear regression to perform
                            if isinstance(df_plot, pd.DataFrame):
                                reg = sp.stats.linregress(df_plot.index, df_plot.values[:, 0])
                            else:
                                reg = sp.stats.linregress(df_plot.index, df_plot.values)
                            linplot = ax.plot(df_plot.index, reg.slope * df_plot.index + reg.intercept, color='grey', linestyle='dashed', alpha=0.9, zorder=0)
                            r2 = reg.rvalue**2
                            labels.append(f"$a={reg.slope:.2f}$, $r^2$={r2:.3f}")
                            handles.append(linplot[0])

                        p = len(handles) // 2
                        ax.legend(handles[p:], labels[p:], ncol=1)  # only plot the regression legend

                    # if stat == "slope":
                        # ax.set_ylabel("Rate of convergence")
                    # else:
                    ax.set_ylabel(stat)

                else:  # should never be the case?
                    ax = sns.contour(df_plot, **kwargs)



                if xaxis == "niter" and args.with_elevel and stat.find("loss A") != -1:
                    # we have to compute the target singularvalues for the
                    # betroffene files
                    # take a tuple of the current key
                    tkey = (key if isinstance(key, tuple) else (key, )) if noncomp else ()
                    namekeys = [tuple([k, *tkey]) for k in df_plot.dropna(axis=1,how="all").columns]
                    fs = [fnames[k] for k in namekeys]
                    # key  is missing the compkey
                    fargss = []
                    for f in fs:
                        with open(f) as fconf:
                            fargss.append(json.load(fconf))
                    dss = set( frgs["dataset_fname"] for frgs in fargss )
                    xdims = set( frgs["xdim"] for frgs in fargss )
                    if len(dss) == len(xdims) == 1:  # same dataset loaded
                        ds = dss.pop()
                        xdim = xdims.pop()
                        sets, _ = utils.load_dataset(fname=ds)  #
                        if sets:
                            trainset = sets["train"]
                            # args.load_dataset = fname_ds
                            STD = trainset.std
                            Sigma0 = STD.mm(STD.t())  # covariance of the target distribution
                            lambdas, U = np.linalg.eigh(Sigma0)
                            Es  = lambdas.cumsum()[::-1]  # the different energy levels, lambdas are <
                            x = np.linspace(0, 1, len(Es))
                            grbow = cm.get_cmap(plt.get_cmap("gist_rainbow"))(x)
                            [ax.axhline(E, color=grbow[i, :], alpha=0.5, linestyle='--', zorder=0) for i, E in enumerate(Es)]
                            xmax = ax.get_xlim()[1]
                            [ax.figure.text(xmax, E, i, transform=ax.transData, fontsize='x-small') for i, E in enumerate(Es,1) if i == 1 or i%5 == 0]
                    if stat == "loss A":
                        stat += " elevel"

                    # df_plot.plot(x="niter", y=stat)
                keyname = ', '.join(dfu.keys().names)
                keyval = str(key) if not isinstance(key, tuple) else ', '.join([str(k) for k in key])
                plotfolder = os.path.join(plotdir, stat)
                os.makedirs(plotfolder, exist_ok=True)
                # plt.suptitle(f"{keyname} = {keyval}")
                # plt.title(stat + niterstr)
                plt.savefig(os.path.join(plotfolder,  f"{string}.pgf"), bbox_inches="tight")
                plt.savefig(os.path.join(plotfolder,  f"{string}.pdf"))
                plt.close("all")
        # end of process keys
        # df_filt = df_merge.loc[:, df_merge.columns.get_level_values("stat") == stat]
        # the keys are either the different configurations or the niter
        # df_filt = dfu
        # df_filt.droplevel("stat")
        if args.contour:
            process_contour(df_filt, dfu, stat)
        else:
            process_keys(noncomp, df_filt, dfu, stat)

        # else:  # if there are no other keys as the comparison key
            # # dfu should have the iterations as columns and the compkey as
            # # xaxis
            # process_keys([compkey], df_filt, dfu, stat)

    return


def df_to_string(key, names):

    lst = []
    if not isinstance(key, tuple):  # if there is ony one key
        key = (key,)

    for k, name in zip(key, names):
        n = name[0]
        if isinstance(k, bool):
            lst.append(f'{n}' if k else f'no-{n}')
        else:
            val = str(k)
            if isinstance(k, (int, float)):
                lst.append(f'{n}{val}')
            # elif isinstance(arg, dict):
                # dirname.append('-'.join([f'{k}-{v}' for k,v in arg.items()]))
            else:
                lst.append(f'{n}-{val}')
    # string = os.path.join(name, '-'.join(name))
    string = '-'.join(lst)
    return string



def to_string(keys, dct):

    name = []
    for  key in keys:
        arg = dct[key]
        k = key[0]
        if isinstance(arg, bool):
            name.append( f'{k}' if arg else f'no-{k}')
        else:
            val = str(arg)
            if isinstance(arg, (int, float)):
                name.append(f'{k}{val}')
            # elif isinstance(arg, dict):
                # dirname.append('-'.join([f'{k}-{v}' for k,v in arg.items()]))
            else:
                name.append(f'{k}-{val}')
    # string = os.path.join(name, '-'.join(name))
    string = '-'.join(name)
    return string



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=str, help="output the config diff between the different experiments")

    parser.add_argument("--nprocs","-n", type=int, default=1, help="the number of threads to span")
    parser.add_argument("--filter", help="json file use to filter")
    parser.add_argument("--diff", action="store_true", help="diff mode, write to file")
    parser.add_argument("--compare", type=str, help="the key for the pot")
    parser.add_argument("--plot", action="store_true", help="simply plots the data again")
    parser.add_argument("--reset", action="store_true", help="set to reset the output")
    parser.add_argument("--with-elevel", action="store_true", help="energy level plotting")
    parser.add_argument("-c", "--compute", action="store_true", help="compute additional properties on the data")
    parser.add_argument("--force", action="store_true", help="force new computations")
    parser.add_argument("-x", default="niter",  help="the x axis")
    parser.add_argument("-y", help="the quantities to plot as a function of x")
    parser.add_argument("-ctr", "--contour", choices=("slope", "r", "lograte"), help="plot a heat map of the value with x/y coordinates")
    parser.add_argument("dirs", nargs="*", help="the different directories to look for")

    args = parser.parse_args()
    main(args)







