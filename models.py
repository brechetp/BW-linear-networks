import torch
import torch.nn as nn
from scipy.stats import ortho_group
import utils


class LinearNetwork(nn.Module):
    """A linear network"""

    def __init__(self, widths, bias=False):
        """Gets the widths as a list"""

        # init the parent class nn.Module
        super().__init__()
        # the layers as a list of nn.Linear objects
        layers = [nn.Linear(widths[i], widths[i+1], bias)
                  for i in range(len(widths)-1)]
        # combine all the layers together
        self.layers = nn.Sequential(*layers)
        # different attributes for the network: input/output dimension, depth
        self.widths =  widths
        self.din = widths[0]
        self.dout = widths[-1]
        self.depth = len(widths)-1
        return

    @property
    def device(self):
        return self.layers[0].weight.device

    def forward(self, x):

        return self.layers(x)

    def balance_weights(self, init_fn=None, smin=-1, *args, **kwargs):
        # follows the initialization from Arora 2019
        # allows the input / output to be greater than min over the layers
        # W = torch.randn(self.nout, self.nin)
        if init_fn is not None:
            W = init_fn(torch.empty(self.dout, self.din, device=self.device), *args, **kwargs)
        else:
            W = self.end_to_end()

        # W = torch.randn(self.nout, self.nin)
        N = self.depth
        b = min(self.widths)  # bottlneck
        U, S, Vt = torch.linalg.svd(W, full_matrices=True)
        if smin >= 0:  # lower the minimum singular value of the initialization
            S += smin - S[-1]
        Sn = S ** (1/N)
        # p = len(S)  # the number of singular values
        for idx, layer in enumerate(self.layers):
            (m, n) = layer.weight.size()
            k = min(n, m, b)
            layer.weight.data  = torch.cat( [torch.cat(
                [torch.diag(Sn[: k]), torch.zeros(k, n - k, device=Sn.device)], dim=1),
                                      torch.zeros(m - k, n, device=Sn.device)], dim=0)
            if idx == 0:  # first layer
                layer.weight.data = layer.weight.data.mm(Vt)
            if idx == (N-1):  # last layer
                layer.weight.data = U.mm(layer.weight.data)
            # self.register_parameter(f"{idx}", layer.weight)
        return

    def init_ortho(self):

        # N = self.depth
        # S_min, S_max = 0.1 ** (1/N), 10 ** (1/N)
        # p = min(self.din, self.dout)
        # p = len(S)  # the number of singular values
        for idx, layer in enumerate(self.layers):
            (m, n) = layer.weight.size()
            k = min(n, m)

            # Sk = torch.zeros(k).uniform_(S_min, S_max)  # sample the singular
            # values
            U = torch.tensor(ortho_group.rvs(dim=m), dtype=torch.float32)
            Vt = torch.tensor(ortho_group.rvs(dim=n), dtype=torch.float32)
            # D = torch.cat([torch.cat([torch.diag(Sk), torch.zeros(k, n-k)],
            # dim=1), torch.zeros(m-k, n)], dim=0) # diagonal with singlar
            # values
            layer.weight = nn.Parameter(
                U[:, :k].mm(Vt[:k, :]), requires_grad=True
            )
            # if idx == 0:  # first layer
            # layer.weight = nn.Parameter(layer.weight.mm(Vt), requires_grad=True)
            # if idx == (N-1):  # last layer
            # layer.weight = nn.Parameter(U.mm(layer.weight), requires_grad=True)
            # self.register_parameter(f"{idx}", layer.weight)
        return

    def init_lowrank(self, rank=1):

        rank = min(rank, min(self.widths))  # rank has to be upper bounded by the bottleneck
        for idx, layer in enumerate(list(self.layers[:1]) + list(self.layers[-1:])): # input and output layers
            # (m, n) = layer.weight.size()
            # k = min(n, m)
            U, S, Vh = torch.linalg.svd(layer.weight.data)
            # values
            layer.weight = nn.Parameter(
                U[:, :rank].mm( S.view(-1, 1)[:rank] * Vh[:rank, :]), requires_grad=True
            )
            # self.register_parameter(f"{idx}", layer.weight)
        return

    def compute_balance(self):
        """Compute the balance value for each of the layers"""
        N = self.depth
        nextW = self.layers[0].weight.data
        balance = (N-1) * [0.]
        for j in range(N-1):
            nextW, W = self.layers[j+1].weight.data, nextW
            balance[j] = (nextW.t().mm(nextW) - W.mm(W.t())).norm(p='fro').item()
        return balance

    def end_to_end(self, requires_grad=False):
        """ Product of all weight matrices"""
        prod = torch.eye(self.dout, device=self.layers[0].weight.device, requires_grad=requires_grad)
        for l in self.layers[::-1]:  # go backwards
            # if isinstance(l, nn.Linear):  # always assumed
            prod = prod.mm(l.weight)
        return prod

    def compute_rank(self):
        """The rank of the end-to-end matrix"""
        W = self.end_to_end()
        return torch.linalg.matrix_rank(W).item()

    def compute_erank(self):
        """The effective rank of the end-to-end matrix"""
        W = self.end_to_end()
        return utils.compute_erank(W)


    def compute_grads(self, R0, loss='BW', grad_L1=None, tau=0.):
        """The manual gradient computation. First compute the gradient on the
        end to end matrix (function space), to then adjust for each of the
        parameters
        R0: root of the target matrix
        loss: either BW or Fro (default: BW)"""

        with torch.no_grad():
            if grad_L1 is None:  # compute the gradient
                W = self.end_to_end()
                n, m = W.size()
                R0 = R0.to(W.device)
                if loss == 'BW':
                    # R0inv = torch.linalg.inv(R0)
                    if tau == 0.:
                        #
                        k = min(n, m)
                        U, S, Vh = torch.linalg.svd(R0.mm(W), full_matrices=False)
                        # grad_L1 =  2*W - 2 * R0.mm( (torch.linalg.inv(sqrtm(R0.mm(W).mm((R0.mm(W)).t()))))).mm(R0.mm(W))
                        grad_L1 = 2*(W - R0 @ U[:, :k] @ Vh[:k, :])
                    elif tau > 0.:
                        Sigma_tau = R0 @ W @ W.t() @ R0 + tau * R0 @ R0
                        InvRoot = utils.sqrtm_inv(Sigma_tau)
                        grad_L1= 2*(W - R0 @ InvRoot @ R0 @ W)
                elif loss == 'Fro':
                    Sigma0 = R0.mm(R0)
                    grad_L1 =  4 * (W.mm(W.t()) - Sigma0).mm(W)

            front_back = self.compute_front_back_prod()  # list of tupes, index i is  the front and back prod of weight

            N = self.depth
            self.grad_L1 = grad_L1.clone()
            grads = N * [None]
            for i in range(N):
                grads[i] = front_back[i][1].t().mm(grad_L1).mm(front_back[i][0].t())

            return grads


    def compute_front_back_prod(self):
        """Compute front to back and back to front products in one traversal of the network.
        Return a list of the products (2 items per layer: one from the front
        the other for the back multiplication) """
        # assume all the layers are linear

        N = self.depth
        d_in = self.din
        d_out = self.dout
        front = torch.eye(d_in).to(self.layers[0].weight)
        back = torch.eye(d_out).to(front)
        res = [[1.,1.] for _ in range(N)]
        for j in range(N):
            if j >= 1:
                front = self.layers[j-1].weight.mm(front)
                back = back.mm(self.layers[-j].weight.data)
            res[j][0] = front.detach().clone()
            res[-j-1][1] = back.detach().clone()
        return res

    def get_gradient(self):
        """Return the gradient that is stored in the parameter variables"""
        grad = self.layers[0].weight.new_zeros(0)  # initialization with a size of 0
        for p in self.parameters():
            if p.grad is None:
                # if no gradient data
                continue
            grad = torch.cat((grad, p.grad.flatten()), dim=0)
        return grad




if __name__ == "__main__":

    # test, only called when python models.py  is computed

    widths = [20] + 3*[50] + [100]
    model = LinearNetwork(widths)
