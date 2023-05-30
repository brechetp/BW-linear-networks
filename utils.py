import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os
import re
import pickle
import matplotlib.pyplot as plt
import models
import itertools
import scipy as sp
import math
from scipy.stats import special_ortho_group


def sqrtm(A):
    """Matrix square root
    KISS: use the function from scipy
    """
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    return torch.tensor(sp.linalg.sqrtm(A), dtype=torch.float32)
       


def init_close(W, target=None, Lambda=None, Omega=None, scale=0.95, tau=0, k=None):
    """Initialization close to a given target (not batched)
    Return the end-to-end matrix close to the target

    W (n, m): end-to-end matrix to init (will be empty but with correct dimensions)
    (either)
    target (n, n): target for the optimization problem
    (or)
    Lambda (n,): eigenvalues of the target (descending order)
    Omega (n,n): associated eigenvectors of the target
    scale: scaling factor for the deviation
    tau: regularization parameter
    k: rank constraint
    """
    n, m = W.size()
    if (Lambda is None or Omega is None) and target is None:
        raise ValueError(" Lambda and Omega  or target have the be provided has to be given")
    if target is None:
        target = Omega @ (Lambda.view(-1, 1) * Omega.T)
    elif Lambda is None or Omega is None:
        Lambda, Omega = torch.linalg.eigh(target)
        Lambda = Lambda.flip([0])  # make it in descending order
        Omega = Omega.flip([1])
    # u, _, vt = torch.linalg.svd(torch.randn(n, n))  # dummy variables
    Gamma = torch.tensor(special_ortho_group.rvs(n), dtype=torch.float32)  # random nxn orthogonal matrix
    smin = Lambda[-1]
    a = -smin / n * scale  # ad hoc scaling in order to remain in ball BW <= smin
    b = scale * (1 + 1/n) * smin
    D = a * torch.arange(1, n+1) + b  # eigenvalues for the perturbation
    A = Gamma @ (D.view(-1, 1) * Gamma.T)
    Sigma = (target - tau * torch.eye(n)) + A
    Sigma = (Sigma + Sigma.T) / 2  # project onto symmetric matrices
    S, U = torch.linalg.eigh(Sigma)
    S = S.flip([0])  # ascending order
    U = U.flip([1])
    # u, _, vt = torch.linalg.svd(torch.randn(n, m), full_matrices=False)
    V = torch.tensor(special_ortho_group.rvs(m), dtype=torch.float32)
    # Vt = (u @ vt)
    if k is None:
        k = min(n, m)  # (should take bottleneck?) 
    Vt = V[:, :k].T

    W = U[:, :k] @ (S.sqrt().view(-1, 1)[:k] * Vt)
    return W

INITIALIZATIONS = {  # different initialization and their default parameters
    'normal': {"fn": torch.nn.init.normal_, "kwargs":{"std": 0.1}},
                   "close": {"fn": init_close, "kwargs": {"scale":0.95}},  # close to target initialization
    'uniform': {"fn": torch.nn.init.uniform_, "kwargs":{"a": -1, "b": 1}},
    "xavier-n": {"fn": torch.nn.init.xavier_normal_, "kwargs": {}},
    "xavier-u": {"fn": torch.nn.init.xavier_uniform_, "kwargs": {}},
    "kaiming-n": {"fn": torch.nn.init.kaiming_normal_, "kwargs": {}},
    "kaiming-u": {"fn": torch.nn.init.kaiming_uniform_, "kwargs": {}},
    # 'none': None,
}



def get_ndgaussian(mean, std=None,  nsamples=10_000):
    """Samples batched Gaussian data given a mean and standard deviation
    mean (B, n): the batched means
    std (B, n, n): the batched standard deviations (default: identity)
    nsamples (int): the number of samples
    """


    dim = mean.size(0)
    if std is None:
        std = torch.eye(dim)

    sigmas = std.view(-1, dim, dim)
    samples =  mean + torch.matmul(sigmas, torch.randn((nsamples, dim, 1))).squeeze()  # batched matmul
    return samples


class Gaussian(object):
    """Gaussian data"""

    # constructor
    def __init__(self,  Lambda=None, Omega=None, Vt=None,
                 smin=-1, samp_distrib=None,
                 samp_mode=None, batch_size=1):
        """Construct the (batched) centered Gaussian data 
        Lambda (B, n): eigenvalues of the covariance
        Omega (B, n, n): eigenvectors of the covariance
        Vt (B, n, n): singular vectors of the standard deviation
        """

        bsize = 1 if batch_size is None else batch_size  # if none, the values will be squeezed to remove the batch dimension
        sigma = Omega.bmm(Lambda.view(bsize, -1, 1) * Omega.transpose(1, 2))  # covariances
        root = Omega.bmm(Lambda.view(bsize, -1, 1).sqrt() * Omega.transpose(1, 2))  # square root of covariances
        std = Omega.bmm(Lambda.view(bsize, -1, 1).sqrt() * Vt)  # standard deviation
        sigma = (sigma + sigma.transpose(1,2)) / 2  # ensures symmetric matrices
        root = (root + root.transpose(1, 2)) / 2

        self.dim = Lambda.size(1)  # the dimension of the data
        mean = torch.zeros(bsize, self.dim)  # centered Gaussian distribution

        if batch_size is None:  # remove the batch dimension
            Lambda, Omega, sigma, root, mean, std = Lambda.squeeze(0), Omega.squeeze(0), sigma.squeeze(0), root.squeeze(0), mean.squeeze(0), std.squeeze(0)
        # assign the attributes
        self.mean = mean
        self.std = std
        self.Sigma = sigma
        self.Root = root
        self.Omega = Omega
        self.Lambda = Lambda
        self.fname = None
        self.smin = smin
        self.samp_mode = samp_mode
        self.samp_distrib = samp_distrib
        self.batch_size=batch_size
    # getter
    def sample(self, n=100):
        """Generates n samples from the gaussian"""

        samples = get_ndgaussian(mean=self.mean, std=self.std, nsamples=n)
        return samples



def save_toy_dataset(trainset, root, singvals=None):
    """save the datasets to a file
    trainset: the training datset
    root (string): the root directory for the file
    singvals (nb, n): (optional) singular values of the dataset to be plotted
    """

    dim = trainset.dim
    batch_size = trainset.batch_size
    samp_distrib=trainset.samp_distrib
    samp_mode=trainset.samp_mode
    smin = trainset.smin
    bname = get_basename(dim, batch_size, smin, samp_distrib, samp_mode)  # base name for the file

    prevds = glob.glob(os.path.join(root, f"{bname}*.pkl"))  # possible previous files with same base name 
    restr = ".*{}(_(\d*))?".format(bname)  # the expression to detect the number of previous datasets
    regexp = re.compile(restr)
    lst_all = glob.glob(os.path.join(root, bname) + "*.pkl")  # the previous datasets with same parameters
    res = [regexp.match(s) for s in lst_all]

    ids = [int(m.group(2)) for m in res if m is not None and m.group(2) is not None]  # if previous dataset is found
    ids.sort() # to increment the last
    if ids: # if we find previous data
        idx = ids[-1] + 1
    else:
        idx = 1

    os.makedirs(root, exist_ok=True)

    fname = os.path.join(root, f"{bname}_{idx}.pkl")  # the dataset filename

    with open(fname, "wb") as _f:  # save the dataset to disk
        pickle.dump(trainset, _f)

    if singvals is not None:  # if we want to plot the singular values
        pltname = os.path.join(root, f"{bname}_{idx}.pdf")
        # fig, ax= plt.figure()
        plt.plot(singvals, '+', label="singular vals")
        plt.savefig(fname=pltname)
        plt.close('all')

    return fname


def get_basename(dim, batch_size,  smin,  samp_distrib, samp_mode, allstr="*"):
    dimstr = allstr if dim is None else str(dim)
    bsstr = '' if batch_size is None else f"bs-{batch_size}-"
    sminstr = allstr if smin is None else "smin-0-" if smin == 0 else f"smin-{smin:.1e}-" if smin > 0 else ""
    # if sminstr is not None:
    sminstr = sminstr.translate({ord('.'):ord('p')})  # replace decimal '.' with p
    sminstr = sminstr.replace("e-", "em")  # replace - (minus) with m
    # assert dim is not None
    bname = f"dim-{dimstr}-{bsstr}{sminstr}sd-{samp_distrib}-sm-{samp_mode}"
    return bname

def load_toy_dataset(fname=None, dim=None,  smin=-1, batch_size=None, samp_mode=None, samp_distrib=None, root="data/1gaussian"):
    """Load a dataset with the name fname provided.
    If not provided, looks for a name as root/dim-x-id.pkl,
    where id is the number of duplicate dataset
    with the same dimension and number of samples"""
    if not fname:  # if the exact name is not provided
        bname = get_basename(dim, batch_size, smin, samp_distrib, samp_mode, '*')
        prevds = list(glob.glob(os.path.join(root, f"*{bname}*.pkl")))
        prevds.sort(key=lambda x: int(x.split('.')[0].split('/')[-1].split('_')[1]))  # take the numeric part of the filename
        # restr = ".*{}(_(\d*))?".format(bname)
        if not prevds:
            # no previous dataset found
            return None, None

        fname = prevds[-1]  # take the last one
        # regexp = re.compile(restr)

    with open(fname, "rb") as _f:
        ds = pickle.load(_f)

    return ds, fname

def get_lfw_trainset(fname="data/lfwcrop_grey/LFW_0-1.pkl"):

    with open(fname, 'rb') as _f:
        lfw = pickle.load(_f)  # already a LFW object
    return lfw



def get_toy_trainset(fname, new_dataset=False, save=False, dim=None, smin=0.3, samp_distrib="zipf", samp_mode="eig", batch_size=None):
    """Return the trainset based on the dimension required.
    If a filename is provided, return the dataset saved in the filename
    Saves the dataset to a file if save_datsaet is set.
    samp_mode (sing or eig): sample either the eigenvalues or the singluar values
    sing_distrib: distribution to use for sampling

    """


    if not new_dataset:  # if we want to load a previous dataset
        dataroot = "data/1gaussian"
        if batch_size is not None:
            dataroot += "-batch"
        trainset, fname_ds = load_toy_dataset(fname=fname, dim=dim, smin=smin, samp_distrib=samp_distrib, samp_mode=samp_mode, batch_size=batch_size, root=dataroot)
        if trainset and  ((fname == fname_ds or trainset.dim == dim)  \
                and  trainset.smin == smin \
                and  trainset.samp_mode == samp_mode \
                and  trainset.samp_distrib == samp_distrib \
                and  trainset.batch_size == batch_size ):
                    pass
        else:  # if one requirement has failed construct a new dataset
            new_dataset=True

    if new_dataset:
        bsize = 1 if batch_size is None else batch_size  # work with batched targets

        n = dim
        STD = torch.randn(bsize, n, n)  # random standard deviation
        U, S, Vh = torch.linalg.svd(STD, full_matrices=False)  # in order to modify the STD
        if samp_distrib == "random":  # purely random
            S = S
        elif samp_distrib == "zipf":  # Zipf's law
            S = S[:, 0].unsqueeze(1) * 1 / torch.arange(1, n+1).view(1, -1) + torch.randn(bsize, n)
            S = S.sort(dim=1, descending=True).values  # sort the singular values
            S += -S.min()+ 1/n  # set the minimum to  1/n
        minbound = smin
        if smin >= 0 and S.min() < minbound:
            S += minbound - S[:, -1].unsqueeze(1) + 1/n # makes the lowest eigenvalue of Sigma equal to smin + 1/n

        if samp_mode == "sing":  # sample the singular values, the eigenvalues are then squared
            S = S ** 2
        # construct the trainset 
        trainset = Gaussian(Lambda=S, Omega=U, Vt = Vh, smin=smin,
                            samp_distrib=samp_distrib,
                            samp_mode=samp_mode, batch_size=batch_size)
        if save:
            saveroot = "data/1gaussian"
            if batch_size is not None:
                saveroot += "-batch"
            fname_ds = save_toy_dataset(trainset, saveroot, singvals=S)
            trainset.fname = fname_ds

    return trainset


def name_network( depth, zdim, width,  xdim,  init_name, init_scheme, root="nets/"):
    """The name of a network file based on the parameters"""
    isstr = init_scheme if init_scheme is not None else ""  # will be balance , balance-force or  None
    name="d{}-z{}-w{}-x{}/in-{}{}.pt".format( depth, zdim, width, xdim, init_name, isstr)
    return os.path.join(root, name)


def init_weights(m, init_fn=torch.nn.init.kaiming_normal_, *args, **kwargs):
    """Initialize the weights based on the function init_fn and some arguments """

    if type(m) == nn.Linear:
        init_fn(m.weight, *args, **kwargs)

def create_widths(depth, zdim, width, xdim):
    """Create a list of widths following a given pattern given by the name of the architecture.
    depth: depth of the network (total number of layers - 1)
    zdim, width, xdim: input, width, output dimensions

    Return: list of widths (input and output dimensions included)
    """

    widths = (depth-1) * [width]  # constant width
    widths = [zdim] + widths + [xdim]
    return widths

def create_network(depth, din, width,  dout, init_name, init_scheme=None, smin=-1, save=False,  **init_kwargs):
    f""" create a network based on the requisites
    depth: the depth of the network (total number of layers - 1)
    din: input dimension
    width:  width of the network (int or list)
    dout: output dimension
    init_name: how to init the network, one of {INITIALIZATIONS.keys()}
    init_scheme: balance, balance-force or  None (on top of the init_name which will be set on the end-to-end matrix then)
    init_kwargs: arguments for the initialization
"""

    widths = create_widths(depth, din, width, dout)
    gen = models.LinearNetwork(widths)
    # initialization
    init_fn = INITIALIZATIONS[init_name]["fn"]
    base_kwargs = INITIALIZATIONS[init_name]["kwargs"]
    kw = {key:init_kwargs[key] if key in init_kwargs.keys() else base_kwargs[key] for key in base_kwargs.keys()}

    if not init_name == "close":
        gen.apply(lambda x: init_weights(x, init_fn,  **kw))  # simple initialization of the weights
    if init_scheme == "balance":  # balance the weights with previsouly initialized end-to-end matrix
        gen.balance_weights(smin=smin)
    if init_scheme == "balance-force" or init_name == 'close':  # force initialization on the end-to-end
        gen.balance_weights(init_fn=init_fn, smin=smin, **kw)

    if save:  # we save the network
        # TODO: add the std in the identifier
        fname = name_network( depth, din, width, dout,  init_name, init_scheme)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        torch.save(gen.state_dict(), fname)

    return gen











def sqrtm_np(X, mask=True):
    """Compute the square root of the matrix X, assumed to be symmetric psd.
    Use the eigendecomposition of X.
    Numpy version
    """

    L, Q = np.linalg.eigh(X)
    L2 = np.zeros_like(L)
    if mask:  # mask the eigenvalues with 0
        r = np.linalg.matrix_rank(X)
        L2[-r:] = L[-r:]
        # L[:-r] = 0. # the last r elements are not changed, only the first n-r are set to 0
    else:
        L2[:] = L[:]
    L2[L<0.] = 0.   # remove negative eigenvalues, assume input is always psd
    return Q @ (np.sqrt(L2.reshape(-1, 1)) * Q.T)


def invsqrtm(X, mask=True):
    """Compute the square root of the matrix X, assumed to be symmetric psd.
    Use the eigendecomposition of X."""

    L, Q = torch.linalg.eigh(X)
    L2 = torch.zeros_like(L)
    if mask:  # mask the eigenvalues with 0
        r = torch.linalg.matrix_rank(X)
        L2[-r:] = L[-r:]
        # L[:-r] = 0. # the last r elements are not changed, only the first n-r are set to 0
    else:
        L2[:] = L[:]
    L2[L<0.] = 0.   # remove negative eigenvalues, assume input is always psd
    return Q @ (1/L2.view(-1, 1).sqrt() * Q.t())

def find_closest(vals, refs):
    """Finds the indices of the closest elements to vals in refs, assuming torch tensors"""
    # implementation in O(nk), where n = len(refs) and k  = len(vals)
    # could be improved by sorting the elements etc.
    _, min_idx = (vals.view(1, -1) - refs.view(-1, 1)).abs().min(dim=0)
    return min_idx



def H(vals):
    """The entropy of the values in vals, assuming a vector of values"""
    ps = vals.abs() / vals.abs().sum()
    return -(ps * ps.log()).sum()

def compute_erank(A):
    """ compute the effective rank of the matrix A"""
    S = torch.linalg.svdvals(A)
    return H(S).exp().item()


def parse_option(field, parser):
    trans = field.maketrans("_", "-")
    try:
        opt_strs = parser._get_option_tuples("--" + field.translate(trans))[0][0].option_strings
        if len(opt_strs) >= 2:  # if there are more than one entry, the first one should be the short one
            return opt_strs[0].rstrip('-')
    finally:
        return  ''.join(c[0] for c in field.split('_'))  # return the first letter of each word



def get_name(args, parser, fmt=None):
    name = ''
    trans = name.maketrans("_=,", "---")
    # trans_vals = name.maketrans("_=", "--")
    # allow in the vary_name to have a / as a separatro, identical to a space
    # flatten the list of list to a 1D list
    if fmt is None:
        fmt = args.vary_name
    else:
        if not isinstance(fmt, list):
            fmt = [fmt]

    vary_name = list(itertools.chain(*list(map(lambda x: x.split('/'), fmt))))
    #
    for entry in vary_name:
        if not entry:
            continue
        fields = entry.split('-')  # will correspond to the same level
        dirname = []
        for field in fields:
        # field = field.translate(trans)  # change the '-' into '_'
            dct = vars(parser)["_option_string_actions"]  # all the possible entries in the namespace
            dest = None
            val = None
            if "-" + field in dct.keys():  # short name given
                action = dct["-" + field]
                key = field
                dest = action.dest
            elif "--" + field.translate(trans) in dct.keys():  # long name given, look for a shorter one
                action = dct["--"+ field.translate(trans)]
                key = parse_option(field, parser)
                dest = field
            else:
                raise ValueError("The argument {} was not found in the parser".format(field))

            if hasattr(action, "values"):
                val = action.values.translate(trans)
            # if hasattr(arg, "values"):


            if dest is not None:
                arg = args.__dict__[dest]
                if key == "is": # init scheme
                    val = str(arg)
                    dirname.append(f'{val}' if val != "none" else "")
                elif isinstance(arg, bool):
                    dirname.append( f'{field}' if arg else f'no-{field}')
                else:
                    if val is None:
                        val = str(arg)
                    if isinstance(arg, (int, float)):
                        dirname.append(f'{key}{val}')
                    # elif isinstance(arg, dict):
                        # dirname.append('-'.join([f'{k}-{v}' for k,v in arg.items()]))
                    else:
                        dirname.append(f'{key}-{val}')
        name = os.path.join(name, '-'.join(dirname))
    # name = os.path.join(args.name, name)
    # if name  == "":
        # name = "debug"
    return name

if __name__ == "__main__":

    archi = "const"
    d = 10
    z = 3
    x =  3
    w = 20
    widths = create_widths(d, z, w, x)
    print(widths)
    create_network(d, z, w, x, "normal", "balance", save=True)
