import torch
import torch.nn as nn
import numpy as np
import glob
import os
import re
import pickle
import matplotlib.pyplot as plt
import models
import itertools
import scipy as sp
from scipy.stats import special_ortho_group
from EllipticalEmbeddingS.torch_utils import batch_sqrtm

def sqrtm_inv(M, numIters=10, reg=1.):
    return torch.tensor(sp.linalg.sqrtm(sp.linalg.inv(M.detach().cpu().numpy())), device=M.device, dtype=M.dtype)
    # unsqueeze=False
    # if M.ndim==2:
        # M = M.unsqueeze(0)
        # unsqueeze=True
    # out = batch_sqrtm(M, numIters=numIters, reg=reg)[1]
    # if unsqueeze:
        # out = out.squeeze(0)
    # return out

def sqrtm(M, numIters=10, reg=1.):
    return torch.tensor(sp.linalg.sqrtm(M.detach().cpu().numpy()), device=M.device, dtype=M.dtype)
    # unsqueeze=False
    # if M.ndim==2:
        # M = M.unsqueeze(0)
        # unsqueeze=True
    # out = batch_sqrtm(M, numIters=numIters, reg=reg)[0]
    # if unsqueeze:
        # out = out.squeeze(0)
    # return out


def init_close(W, target=None, Lambda=None, Omega=None, scale=0.95, tau=0, k=None):
    n, m = W.size()
    device = W.device
    if Lambda is None or Omega is None or target is None:
        raise ValueError("Lambda and Omega have the be provided has to be given")
    #n = n   # the dimension of the samples, target is n x n
    # we draw random orthogonal matrix   Gamma
    # u, _, vt = torch.linalg.svd(torch.randn(n, n))  # dummy variables
    Gamma = torch.tensor(special_ortho_group.rvs(n), dtype=torch.float32, device=device)  # random nxn orthogonal matrix
    lmin = Lambda[-1]
    a = -lmin / n * scale  # ad hoc scaling in order to remain in ball BW <= lmin
    b = scale * 2 * lmin
    D = a * torch.arange(1, n+1, device=device) + b  # eigenvalues for the perturbation
    A = Gamma @ (D.view(-1, 1) * Gamma.T)
    Sigma = (target.to(device) - tau * torch.eye(n, device=device)) + A
    Sigma = (Sigma + Sigma.T) / 2  # project onto symmetric matrices
    S, U = torch.linalg.eigh(Sigma)
    S = S.flip([0])  # ascending order
    U = U.flip([1])
    # u, _, vt = torch.linalg.svd(torch.randn(n, m), full_matrices=False)
    V = torch.tensor(special_ortho_group.rvs(m), dtype=torch.float32, device=device)
    # Vt = (u @ vt)
    if k is None:
        k = min(n, m)  # should take the bottlneck?
    Vt = V[:, :k].T

    W = U[:, :k] @ (S.sqrt().view(-1, 1)[:k] * Vt)
    return W

INITIALIZATIONS = {
    # 'kaiming': {"fn": torch.nn.init.kaiming_normal_, "kwargs":{}},
    'normal': {"fn": torch.nn.init.normal_, "kwargs":{"std": 0.1}},
    "close": {"fn": init_close, "kwargs": {"target": None, "Lambda": None, "Omega": None, "scale": 0.95}},  # close to target initialization
    'uniform': {"fn": torch.nn.init.uniform_, "kwargs":{"a": -1, "b": 1}},
    "xavier-n": {"fn": torch.nn.init.xavier_normal_, "kwargs": {}},
    "xavier-u": {"fn": torch.nn.init.xavier_uniform_, "kwargs": {}},
    "kaiming-n": {"fn": torch.nn.init.kaiming_normal_, "kwargs": {}},
    "kaiming-u": {"fn": torch.nn.init.kaiming_uniform_, "kwargs": {}},
    # 'none': None,
}

ARCHITECTURES = ["rand", "const", "exp", "lin"]  # different possible architectures for the network


def get_ndgaussian(mean, std=None,  nsamples=10_000):


    dim = mean.size(0)
    if std is None:
        std = torch.eye(dim)

    sigmas = std.view(-1, dim, dim)
    samples =  mean + torch.matmul(sigmas, torch.randn((nsamples, dim, 1))).squeeze()
    return samples


class Gaussian(object):
    """The same as above but without the samples x"""

    # constructor
    def __init__(self,  mean, std=None, Lambda=None, Omega=None, Vt=None, smin=-1, rank=None, sigma=None, root=None, samp_distrib=None, samp_mode=None, batch_size=1):
        """Construct the Gaussian data with severan properties
        Assert man and std are of correct dimension """

        bsize = 1 if batch_size is None else batch_size
        if Lambda is not None and Omega is not None and Vt is not None:
            sigma = Omega.bmm(Lambda.view(bsize, -1, 1) * Omega.transpose(1, 2))
            root = Omega.bmm(Lambda.view(bsize, -1, 1).sqrt() * Omega.transpose(1, 2))
            std = Omega.bmm(Lambda.view(bsize, -1, 1).sqrt() * Vt)
            sigma = (sigma + sigma.transpose(1,2)) / 2
            root = (root + root.transpose(1, 2)) / 2
        else:
            assert std is not None
            if sigma is None:
                sigma = std.bmm(std.tranpose(1, 2))
            Lambda, Omega = torch.linalg.eigh(sigma)
            if root is None:
                root = sqrtm_inv(sigma)

        Lambda, Omega = torch.linalg.eigh(sigma)
        Lambda = Lambda.flip([1])  # flip the values (descending order)
        Omega = Omega.flip([2])  # flip along the columns to match the eigenvalues
        self.dim = mean.size(1)  # assumes a tensor if rank is None:
        if batch_size is None:  # remove the batch dimension
            Lambda, Omega, sigma, root, mean, std = Lambda.squeeze(0), Omega.squeeze(0), sigma.squeeze(0), root.squeeze(0), mean.squeeze(0), std.squeeze(0)
        self.mean = mean
        self.std = std
        self.rank= rank
        # self.nsamples = nsamples
        # self.rank = rank  # the rank of the covariance
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


class LFW(object):
    """The same as above but without the samples x"""

    # constructor
    def __init__(self,   sigma=None, Lambda=None, Omega=None, device=torch.device('cpu'), imsize=(64,64), mean=None):
        """Construct the LFW data with severan properties
        Assert man and std are of correct dimension """


        if isinstance(sigma, np.ndarray):
            sigma = torch.tensor(sigma, dtype=torch.float32, device='cpu')
        if Lambda is None or Omega is None:
            Lambda, Omega = torch.linalg.eigh(sigma)
            Lambda = Lambda.flip([0])  # flip the values (descending order)
            Omega = Omega.flip([1])  # flip along the columns to match the eigenvalues
        else:
            if isinstance(Lambda, np.ndarray):
                Lambda = torch.tensor(Lambda, dtype=torch.float32)
            if isinstance(Omega, np.ndarray):
                Omega = torch.tensor(Omega, dtype=torch.float32)
        self.dim = sigma.size(0)  # assumes a tensor if rank is None:
        self.mean = mean.cpu()
        # self.nsamples = nsamples
        # self.rank = rank  # the rank of the covariance
        self.Sigma = sigma.cpu()
        self.imsize = imsize
        self.Root = sqrtm(sigma.to(device)).cpu()
        self.Omega = Omega.cpu()
        self.Lambda = Lambda.cpu()
        self.fname = None

    def sample(self, N):
        """
        Sample N items as drawn from the distribution
        """

        d = self.dim
        xs = torch.randn(N, d) @ self.Root
        return xs.view(N, *self.imsize)

def save_toy_dataset(sets, root, singvals=None):#, singvals=None):
    """save the datasets to a file
    sets: dictionary of datasets (train, test)
    root: the root directory for the file"""

    rank = sets['train'].rank
    dim = sets['train'].dim
    batch_size = sets['train'].batch_size
    samp_distrib=sets['train'].samp_distrib
    samp_mode=sets['train'].samp_mode
    smin = sets['train'].smin
    # squeeze_batch = sets['train'].ndim == 2  # have no batch

    # bname = f"dim-{sets['train'].dim}-bs-{sets['train'].batch_size}{rankstr}{sminstr}-sd-{sets['train'].samp_distrib}-sm-{sets['train'].samp_mode}"#-mean-{meanid}-std-{stdid}"
    bname = get_basename(dim, batch_size, rank, smin, samp_distrib, samp_mode)
    prevds = glob.glob(os.path.join(root, f"{bname}*.pkl"))
    restr = ".*{}(_(\d*))?".format(bname)  # the expression to detect the number of previous datasets
    regexp = re.compile(restr)
    lst_all = glob.glob(os.path.join(root, bname) + "*.pkl")
    res = [regexp.match(s) for s in lst_all]
    ids = [int(m.group(2)) for m in res if m is not None and m.group(2) is not None]
    ids.sort()
    if ids: # if we find previous data
        idx = ids[-1] + 1
    else:
        idx = 1

    os.makedirs(root, exist_ok=True)

    fname = os.path.join(root, f"{bname}_{idx}.pkl")

    with open(fname, "wb") as _f:
        pickle.dump(sets, _f)

    if singvals is not None:
        pltname = os.path.join(root, f"{bname}_{idx}.pdf")
        # fig, ax= plt.figure()
        plt.plot(singvals**2, '+', label="eigenvalues")
        plt.savefig(fname=pltname)
        plt.close('all')

    return fname


def get_basename(dim, batch_size, rank, smin,  samp_distrib, samp_mode, allstr="*"):
    dimstr = allstr if dim is None else str(dim)
    bsstr = '' if batch_size is None else f"bs-{batch_size}-"
    rankstr = allstr if rank is None else f"r-{rank}-" if rank != dim else ""
    sminstr = allstr if smin is None else "smin-0-" if smin == 0 else f"smin-{smin:.1e}-" if smin > 0 else ""
    # if sminstr is not None:
    sminstr = sminstr.translate({ord('.'):ord('p')})  # replace decimal '.' with p
    sminstr = sminstr.replace("e-", "em")  # replace - (minus) with m
    # assert dim is not None
    bname = f"dim-{dimstr}-{bsstr}{rankstr}{sminstr}sd-{samp_distrib}-sm-{samp_mode}"
    return bname

def load_toy_dataset(fname=None, dim=None,  rank=None, smin=-1, batch_size=None, samp_mode=None, samp_distrib=None, root="data/1gaussian"):
    """Load a dataset with the name fname provided.
    If not provided, looks for a name as root/dim-x-id.pkl,
    where id is the number of duplicate dataset
    with the same dimension and number of samples"""
    if not fname:  # if the exact name is not provided
        bname = get_basename(dim, batch_size, rank, smin, samp_distrib, samp_mode, '*')
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



def get_toy_trainset(fname, new_dataset=False, save=False, dim=None, smin=0.3, rank=None, samp_distrib="zipf", samp_mode="eig", batch_size=None):
    """Return the trainset based on the dimension required.
    If a filename is provided, return the dataset saved in the filename
    Saves the dataset to a file if save_datsaet is set.
    samp_mode (sing or eig): sample either the eigenvalues or the singluar values
    sing_distrib: distribution to use for sampling

    """
    if rank is  None:
        rank = dim
    r = rank


    if not new_dataset:  # if we want to load a previous dataset
        dataroot = "data/1gaussian"
        if batch_size is not None:
            dataroot += "-batch"
        sets, fname_ds = load_toy_dataset(fname=fname, dim=dim, smin=smin, rank=rank, samp_distrib=samp_distrib, samp_mode=samp_mode, batch_size=batch_size, root=dataroot)
        if sets and  (fname == fname_ds or (sets['train'].dim == dim  \
                and  hasattr(sets['train'], "smin") and sets["train"].smin == smin \
                and  hasattr(sets['train'], "samp_mode") and sets["train"].samp_mode == samp_mode \
                and  hasattr(sets['train'], "samp_distrib") and sets["train"].samp_distrib == samp_distrib \
                and ((hasattr(sets['train'], "rank") and (sets['train'].rank == rank or sets['train'].rank is None and rank == dim)) \
                     or (not hasattr(sets['train'], "rank") and rank == dim)))) \
                and  hasattr(sets['train'], "batch_size") and sets["train"].batch_size == batch_size:
            trainset = sets['train']
            if not hasattr(trainset, "fname") or trainset.fname is None:  # where to put it? backward compatibility
                trainset.fname = fname_ds
        else:
            new_dataset=True

    if new_dataset:
        bsize = 1 if batch_size is None else batch_size
        MEAN =  torch.zeros(bsize, dim)

        n = dim
        STD = torch.randn(bsize, n, n)  # random standard deviation
        U, S, Vh = torch.linalg.svd(STD, full_matrices=False)  # in order to modify the STD
        if samp_distrib == "random":  # purely random
            S = S
            # if the target has a lower rank
            # r = args.rank_target
            # r = n  # rank for the target
            # S = S.sqrt()
        elif samp_distrib == "zipf":  # Zipf's law
            S = S[:, 0].unsqueeze(1) * 1 / torch.arange(1, n+1).view(1, -1) + torch.randn(bsize, n)

            S = S.sort(dim=1, descending=True).values
            S += -S.min()+ 1/n  # set the minimum to  1/n
        minbound = smin
        if smin >= 0 and S.min() < minbound:
            S += minbound - S[:, r-1].unsqueeze(1) + 1/n # makes the lowest eigenvalue of Sigma equal to smin + 1/n
        S[:, r:] = 0  # set the low rank values to 0

        # S[0:r] += 0.31  # makes the lowest non zero eigenvalue of Sigma about 0.1
        # STD = U @ ((S.view(-1, 1)) * Vh)
        # SIGMA = U @ (S.view(-1, 1) * (U.t()))
        if samp_mode == "sing":  # sample the singular values, the eigenvalues are then squared
            S = S ** 2
            # else if mode == "eig" we sample the eigenvalues directly
        # ROOT = U @ (S.view(-1, 1) * (U.t()))
        # if batch_size is None:  # squeeze the batch dimension
            # MEAN, S, U, Vh = MEAN.squeeze(0), S.squeeze(0), U.squeeze(0), Vh.squeeze(0)  # remove the batch dimension
        trainset = Gaussian(mean=MEAN, Lambda=S, Omega=U, Vt = Vh, smin=smin, rank=r, samp_distrib=samp_distrib, samp_mode=samp_mode, batch_size=batch_size)
        sets = {'train' : trainset}
        if save:
            saveroot = "data/1gaussian"
            if batch_size is not None:
                saveroot += "-batch"
            fname_ds = save_toy_dataset(sets, saveroot, singvals=S)
            trainset.fname = fname_ds

    return trainset


def name_network(architecture, depth, zdim, width,  xdim,  init_name, init_scheme, bias=False, root="nets/"):
    """The name of a network file based on the architecture"""
    bstr = "-b" if bias else ""
    isstr = init_scheme if init_scheme is not None else ""  # will be balance, ortho
    name="a-{}{}/d{}-z{}-w{}-x{}/in-{}{}.pt".format(architecture, bstr, depth, zdim, width, xdim, init_name, isstr)
    return os.path.join(root, name)


def init_weights(m, init_fn=torch.nn.init.kaiming_normal_, *args, **kwargs):
    """Initialize the weights based on the function init_fn and some arguments """

    if type(m) == nn.Linear:
        init_fn(m.weight, *args, **kwargs)

def create_widths(architecture, depth, zdim, width, xdim):
    """Create a list of widths following a given pattern given by the name of the architecture.
    architecture: one of the {ARCHITECTURES}
    depth: depth of the network (total number of layers - 1)
    zdim, width, xdim: input, width, output dimensions

    Return: list of widths (input and output dimensions included)
    """

    if architecture == "custom":
        assert isinstance(width, list)
        widths = width
    elif architecture  == "rand":
        widths = list(np.random.randint(1, width+1, depth-1))
    elif architecture == "const":
        widths = (depth-1) * [width]
    # elif architecture == "exp":
        # depth = (depth)//2 * 2   # even number
        # widths = [width / (depth//2-1-abs(i-depth//2-1)) for i in range(depth-1)]
    elif architecture == "lin":  # first increase and then decrease
        h = d-1  # hidden layers
        isodd = int((h%2) == 1)
        p = h//2  # h = 2*p + isodd
        # if h is even, each part of the widths have p items
        # else, the first part has p+1 and the second p
        widths = [zdim + round(i/(p+isodd) * (width-zdim)) for i in range(1, p+1+isodd)]
        # if isodd, we start at 1 else we start at 0
        # total number of items has to be p
        widths.extend([width + round(i * (xdim - width)/(p+isodd)) for i in range(isodd, p+isodd)])
    else:
        raise NotImplementedError(f"Architecture {architecture} is not implemented.")
    widths = [zdim] + widths + [xdim]
    return widths

def create_network(architecture, depth, din, width,  dout, init_name, smin=-1, init_scheme=None, bias=False, save=False,  **init_kwargs):
    f""" create a network based on the requisites
    architecture: one of {ARCHITECTURES}
    depth: the depth of the network (total number of layers - 1)
    din: input dimension
    width:  width of the network (int or list)
    dout: output dimension
    smin: lower bound on the minimum singular value of the model (if balanced). Negative value for no effect
    init_name: how to init the network, one of {INITIALIZATIONS.keys()}, balance or ortho
    init_scheme: balance, ortho, or None (on top of the init_name which will be set on the end-to-end matrix then)
    init_kwargs: arguments for the initialization
"""

    widths = create_widths(architecture, depth, din, width, dout)
    gen = models.LinearNetwork(widths)
    # initialization
    init_fn = INITIALIZATIONS[init_name]["fn"]
    base_kwargs = INITIALIZATIONS[init_name]["kwargs"]
    kw = {key:init_kwargs[key] if key in init_kwargs.keys() else base_kwargs[key] for key in base_kwargs.keys()}
    if not init_name == "close":
        gen.apply(lambda x: init_weights(x, init_fn,  **kw))
    if init_scheme == "balance":
        gen.balance_weights(smin=smin)
    if init_scheme == "balance-force" or init_name == 'close':  # force initialization on the end-to-end
        gen.balance_weights(init_fn=init_fn, smin=smin, **kw)
    elif init_scheme == "ortho":
        gen.init_ortho()

    if save:  # we save the network
        fname = name_network(architecture, depth, din, width, dout,  init_name, init_scheme, bias)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        torch.save(gen.state_dict(), fname)

    return gen








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
    """Figure out what the name of the experiment is based on the vary-name argument

    Args:
        args: the parsed arguments
        parser: the corresponding parser
        fmt (string): the format string a/b/c-d
    Output:
        a1/b2/c3-d4
        """
    name = ''
    trans = name.maketrans("_=,", "---")
    if fmt is None:
        fmt = args.vary_name

    # vary_name = list(itertools.chain(*list(map(lambda x: x.split('/'), fmt))))
    vary_name = fmt.split('/')
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
    widths = create_widths(archi, d, z, w, x)
    print(widths)
    create_network("const", d, z, w, x, "ortho", save=True)
