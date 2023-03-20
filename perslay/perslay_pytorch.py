"""
This is a beta version of the PersLay library for PyTorch. There are only one difference
between this version and the original PersLay library for Tensorflow: the random weight initializations.
To initialize the weights randomly, in the TF version authors use the function tf.random_uniform_initializer(min, max)
as a parameter. To obtain the exact same behaviour in the PyTorch version, you must use
partial(torch.nn.init.uniform_, a=min, b=max). Other weight initialization functions can be used in the same way.
Also, the parameters perslay_channel["pweight_train"], and perslay_channel["layer_train"] are not used and are
True by default.

Several improvements can still be made, such as:
- Implementing gather_nd (and the code in general) in a more PyTorch way.
- Debugging the code better.
- Check if the parameters are added properly for training with PyTorch

@author: Rubén Ballester
@paper author and original idea: Mathieu Carriere et al.
"""

import torch
import torch.nn as nn
import numpy as np


def gather_nd(params, indices):
    """
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices

    returns: tensor shaped [m_1, m_2, m_3, m_4]

    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices

    returns: tensor shaped [m_1, ..., m_1]
    """

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


def permutation_equivariant_layer(inp, dimension, perm_op, lbda, b, gamma):
    """ DeepSet PersLay """
    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
    A = torch.reshape(torch.einsum("ijk,kl->ijl", inp, lbda), (-1, num_pts, dimension))
    if perm_op != None:
        if perm_op == "max":
            beta = torch.unsqueeze(torch.max(inp, dim=1)[0], 1).repeat(1, num_pts, 1)
        elif perm_op == "min":
            beta = torch.unsqueeze(torch.min(inp, dim=1)[0], 1).repeat(1, num_pts, 1)
        elif perm_op == "sum":
            beta = torch.unsqueeze(torch.sum(inp, dim=1), 1).repeat(1, num_pts, 1)
        else:
            raise Exception("perm_op should be min, max or sum")
        B = torch.reshape(torch.einsum("ijk,kl->ijl", beta, gamma), (-1, num_pts, dimension))
        return A - B + b
    else:
        return A + b


def rational_hat_layer(inp, q, mu, r):
    """ Rational Hat PersLay """
    mu, r = torch.unsqueeze(torch.unsqueeze(mu, 0), 0), torch.unsqueeze(torch.unsqueeze(r, 0), 0)
    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    bc_inp = torch.unsqueeze(inp, -1)
    norms = torch.norm(bc_inp - mu, p=q, dim=2)
    return 1 / (1 + norms) - 1 / (1 + torch.abs(torch.abs(r) - norms))


def rational_layer(inp, mu, sg, al):
    """ Rational PersLay """
    mu, sg, al = torch.unsqueeze(torch.unsqueeze(mu, 0), 0), torch.unsqueeze(torch.unsqueeze(sg, 0),
                                                                             0), torch.unsqueeze(torch.unsqueeze(al, 0),
                                                                                                 0)
    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    bc_inp = torch.unsqueeze(inp, -1)
    return 1 / torch.pow(1 + torch.sum(torch.multiply(torch.abs(bc_inp - mu), torch.abs(sg)), dim=2), al)


def exponential_layer(inp, mu, sg):
    """ Exponential PersLay """
    mu, sg = torch.unsqueeze(torch.unsqueeze(mu, 0), 0), torch.unsqueeze(torch.unsqueeze(sg, 0), 0)
    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    bc_inp = torch.unsqueeze(inp, -1)
    return torch.exp(torch.sum(-torch.multiply(torch.square(bc_inp - mu), torch.square(sg)), dim=2))


def landscape_layer(inp, sp):
    """ Landscape PersLay """
    sp = torch.unsqueeze(torch.unsqueeze(sp, 0), 0)
    return torch.max(0.5 * (inp[:, :, 1:2] - inp[:, :, 0:1]) - torch.abs(sp - 0.5 * (inp[:, :, 1:2] + inp[:, :, 0:1])),
                     torch.tensor([0]))


def betti_layer(inp, theta, sp):
    """ Betti PersLay """
    sp = torch.unsqueeze(torch.unsqueeze(sp, 0), 0)
    X, Y = inp[:, :, 0:1], inp[:, :, 1:2]
    return 1.0 / (1.0 + torch.exp(-theta * (0.5 * (Y - X) - torch.abs(sp - 0.5 * (Y + X)))))


def entropy_layer(inp, theta, sp):
    """ Entropy PersLay
    WARNING: this function assumes that padding values are zero
    """
    sp = torch.unsqueeze(torch.unsqueeze(sp, 0), 0)
    bp_inp = torch.einsum("ijk,kl->ijl", inp, torch.tensor([[1., -1.], [0., 1.]], dtype=torch.float32))
    L, X, Y = bp_inp[:, :, 1:2], bp_inp[:, :, 0:1], bp_inp[:, :, 0:1] + bp_inp[:, :, 1:2]
    LN = torch.multiply(L, 1. / torch.unsqueeze(torch.matmul(L[:, :, 0], torch.ones([L.shape[1], 1])), -1))
    entropy_terms = torch.where(LN > 0., -torch.multiply(LN, torch.log(LN)), LN)
    return torch.multiply(entropy_terms,
                          1. / (1. + torch.exp(-theta * (0.5 * (Y - X) - torch.abs(sp - 0.5 * (Y + X))))))


def image_layer(inp, image_size, image_bnds, sg):
    """ Persistence Image PersLay """
    bp_inp = torch.einsum("ijk,kl->ijl", inp, torch.tensor([[1., -1.], [0., 1.]], dtype=torch.float32))
    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    coords = [torch.arange(start=image_bnds[i][0], end=image_bnds[i][1],
                           step=(image_bnds[i][1] - image_bnds[i][0]) / image_size[i]) for i in range(dimension_before)]
    M = torch.meshgrid(*coords, indexing='ij')
    mu = torch.cat([torch.unsqueeze(tens, 0) for tens in M], dim=0)
    bc_inp = bp_inp.reshape(-1, num_pts, dimension_before, *(1 for _ in range(dimension_before)))
    return torch.unsqueeze(torch.exp(torch.sum(-torch.square(bc_inp - mu) / (2.0 * torch.square(sg)), dim=2)) / (
                2.0 * np.pi * torch.square(sg)), -1)


class PerslayModel(nn.Module):
    def __init__(self, diagdim, perslay_parameters, rho):
        super().__init__()
        self.diag_dim = diagdim
        self.perslay_parameters = perslay_parameters
        self.rho = rho

        self.vars = torch.nn.ParameterList([[] for _ in range(len(self.perslay_parameters))])
        for nf, plp in enumerate(self.perslay_parameters):
            weight = plp['pweight']
            if weight is not None:
                Winit = plp["pweight_init"]
                if not callable(Winit):
                    W = torch.tensor(Winit)
                else:
                    if weight == 'power':
                        W = torch.tensor(Winit([1]))
                    elif weight == 'grid':
                        Wshape = plp["pweight_size"]
                        W = torch.empty(Wshape)
                        Winit(W)
                    elif weight == 'gmix':
                        ngs = plp["pweight_num"]
                        W = torch.empty(4, ngs)
                        Winit(W)
            else:
                W = torch.tensor(0.0)
            self.vars[nf].append(torch.nn.Parameter(W))

            layer = plp["layer"]

            if layer == "PermutationEquivariant":
                Lpeq, LWinit, LBinit, LGinit = plp["lpeq"], plp["lweight_init"], plp["lbias_init"], plp["lgamma_init"]
                LW, LB, LG = torch.nn.ParameterList([]), torch.nn.ParameterList([]), torch.nn.ParameterList([])
                for idx, (dim, pop) in enumerate(Lpeq):
                    dim_before = self.diag_dim if idx == 0 else Lpeq[idx - 1][0]
                    if callable(LWinit):
                        LWvar = torch.empty(dim_before, dim)
                        LWinit(LWvar)
                    else:
                        LWvar = torch.tensor(LWinit)
                    if callable(LBinit):
                        LBvar = torch.empty(dim)
                        LBinit(LBvar)
                    else:
                        LBvar = torch.tensor(LBinit)
                    LW.append(torch.nn.Parameter(LWvar))
                    LB.append(torch.nn.Parameter(LBvar))
                    if pop is not None:
                        if callable(LGinit):
                            LGvar = torch.empty(dim_before, dim)
                            LGinit(LGvar)
                        else:
                            LGvar = torch.tensor(LGinit)
                        LG.append(torch.nn.Parameter(LGvar))
                    else:
                        LG.append([])
                self.vars[nf].append(torch.nn.ParameterList([LW, LB, LG]))
            elif layer == "Landscape" or layer == "BettiCurve" or layer == "Entropy":
                LSinit = plp["lsample_init"]
                if callable(LSinit):
                    LSiv = torch.empty(plp["lsample_num"])
                    LSinit(LSiv)
                else:
                    LSiv = torch.tensor(LSinit)
                self.vars[nf].append(torch.nn.Parameter(LSiv))

            elif layer == "Image":
                LVinit = plp["lvariance_init"]
                if callable(LVinit):
                    LViv = torch.empty(1)
                    LVinit(LViv)
                else:
                    LViv = torch.tensor(LVinit)
                self.vars[nf].append(torch.nn.Parameter(LViv))

            elif layer == "Exponential":
                LMinit, LVinit = plp["lmean_init"], plp["lvariance_init"]
                if callable(LMinit):
                    LMiv = torch.empty(self.diag_dim, plp["lnum"])
                    LMinit(LMiv)
                else:
                    LMiv = torch.tensor(LMinit)
                if callable(LVinit):
                    LViv = torch.empty(self.diag_dim, plp["lnum"])
                    LVinit(LViv)
                else:
                    LViv = torch.tensor(LVinit)
                self.vars[nf].append(torch.nn.ParameterList([torch.nn.Parameter(LMiv), torch.nn.Parameter(LViv)]))

            elif layer == "Rational":
                LMinit, LVinit, LAinit = plp["lmean_init"], plp["lvariance_init"], plp["lalpha_init"]
                if callable(LMinit):
                    LMiv = torch.empty(self.diag_dim, plp["lnum"])
                    LMinit(LMiv)
                else:
                    LMiv = torch.tensor(LMinit)
                if callable(LVinit):
                    LViv = torch.empty(self.diag_dim, plp["lnum"])
                    LVinit(LViv)
                else:
                    LViv = torch.tensor(LVinit)
                if callable(LAinit):
                    LAiv = torch.empty(plp["lnum"])
                    LAinit(LAiv)
                else:
                    LAiv = torch.tensor(LAinit)
                self.vars[nf].append(torch.nn.ParameterList([torch.nn.Parameter(LMiv), torch.nn.Parameter(LViv), torch.nn.Parameter(LAiv)]))

            elif layer == "RationalHat":
                LMinit, LRinit = plp["lmean_init"], plp["lr_init"]
                if callable(LMinit):
                    LMiv = torch.empty(self.diag_dim, plp["lnum"])
                    LMinit(LMiv)
                else:
                    LMiv = torch.tensor(LMinit)
                if callable(LRinit):
                    LRiv = torch.empty(1)
                    LRinit(LRiv)
                else:
                    LRiv = torch.tensor(LRinit)
                self.vars[nf].append(torch.nn.ParameterList([torch.nn.Parameter(LMiv), torch.nn.Parameter(LRiv)]))

    def compute_representations(self, diags):
        list_v = []
        for nf, plp in enumerate(self.perslay_parameters):
            diag = diags[nf]
            N, dimension_diag = diag.shape[1], diag.shape[2]
            tensor_mask = diag[:, :, dimension_diag - 1]
            tensor_diag = diag[:, :, :dimension_diag - 1]

            W = self.vars[nf][0]
            if plp["pweight"] == "power":
                p = plp["pweight_power"]
                weight = W * torch.pow(torch.abs(tensor_diag[:, :, 1:2] - tensor_diag[:, :, 0:1]), p)
            elif plp["pweight"] == "grid":
                grid_shape = W.shape
                indices = []
                for dim in range(dimension_diag - 1):
                    [m, M] = plp["pweight_bnds"][dim]
                    coords = torch.narrow(tensor_diag, 2, dim, 1)
                    ids = grid_shape[dim] * (coords - m) / (M - m)
                    indices.append(ids.int())
                weight = gather_nd(W, torch.cat(indices, dim=2)).unsqueeze(-1)
            elif plp["pweight"] == "gmix":
                M, V = torch.unsqueeze(torch.unsqueeze(W[:2, :], 0), 0), torch.unsqueeze(torch.unsqueeze(W[2:, :], 0),
                                                                                         0)
                bc_inp = torch.unsqueeze(tensor_diag, -1)
                weight = torch.unsqueeze(torch.sum(
                    torch.exp(torch.sum(-torch.multiply(torch.square(bc_inp - M), torch.square(V)),
                                        dim=2)), dim=2), -1)
            lvars = self.vars[nf][1]
            if plp["layer"] == "PermutationEquivariant":
                for idx, (dim, pop) in enumerate(plp["lpeq"]):
                    tensor_diag = permutation_equivariant_layer(tensor_diag, dim, pop, lvars[0][idx], lvars[1][idx],
                                                                lvars[2][idx])
            elif plp["layer"] == "Landscape":
                tensor_diag = landscape_layer(tensor_diag, lvars)
            elif plp["layer"] == "BettiCurve":
                tensor_diag = betti_layer(tensor_diag, plp["theta"], lvars)
            elif plp["layer"] == "Entropy":
                tensor_diag = entropy_layer(tensor_diag, plp["theta"], lvars)
            elif plp["layer"] == "Image":
                tensor_diag = image_layer(tensor_diag, plp["image_size"], plp["image_bnds"], lvars)
            elif plp["layer"] == "Exponential":
                tensor_diag = exponential_layer(tensor_diag, lvars[0], lvars[1])
            elif plp["layer"] == "Rational":
                tensor_diag = rational_layer(tensor_diag, lvars[0], lvars[1], lvars[2])
            elif plp["layer"] == "RationalHat":
                tensor_diag = rational_hat_layer(tensor_diag, plp["q"], lvars[0], lvars[1])

            # Apply weight
            output_dim = len(tensor_diag.shape) - 2
            if plp["pweight"] != None:
                for _ in range(output_dim - 1):
                    weight = torch.unsqueeze(weight, -1)
                tiled_weight = weight.repeat(1, 1, *tensor_diag.shape[2:])
                tensor_diag = torch.multiply(tensor_diag, tiled_weight)

            # Apply mask
            for _ in range(output_dim):
                tensor_mask = torch.unsqueeze(tensor_mask, -1)
            tiled_mask = tensor_mask.repeat(1, 1, *tensor_diag.shape[2:])
            masked_layer = torch.multiply(tensor_diag, tiled_mask)

            # Permutation invariant operation
            if plp["perm_op"] == "topk" and output_dim == 1:  # k first values
                masked_layer_t = masked_layer.transpose(1, 2)
                values, indices = torch.topk(masked_layer_t, k=plp["keep"])
                vector = torch.reshape(values, (-1, plp["keep"] * tensor_diag.shape[2]))
            elif plp["perm_op"] == "sum":  # sum
                vector = torch.sum(masked_layer, dim=1)
            elif plp["perm_op"] == "max":  # maximum
                vector, _ = torch.max(masked_layer, dim=1)
            elif plp["perm_op"] == "mean":  # minimum
                vector = torch.mean(masked_layer, dim=1)
            else:
                raise ValueError("Unknown permutation invariant operation or output dim != 1 for perm_op=topk.")
            # Second layer of channel
            vector = plp["final_model"].forward(vector) \
                if plp["final_model"] != "identity" else vector
            list_v.append(vector)

        # Concatenate all channels and add other features
        representations = torch.cat(list_v, dim=1)
        return representations

    def forward(self, x):
        diags, feats = x[0], x[1]
        representations = self.compute_representations(diags)
        concat_representations = torch.cat([representations, feats], dim=1)
        final_representations = self.rho(concat_representations) if self.rho != "identity" else concat_representations
        return final_representations
