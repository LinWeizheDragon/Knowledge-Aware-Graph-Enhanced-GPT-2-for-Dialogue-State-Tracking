"""
graph_modules.py: GAT modules for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import math
import numpy as np
import torch
import torch.nn as nn

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number



class GraphFilterBatchAttentional(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer

    Initialization:

        GraphFilterAttentional(in_features, out_features,
                               filter_taps, attention_heads,
                               edge_features=1, bias=True,
                               nonlinearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps (power of the GSO)
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias in the LSIGF stage (default: True)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph convolution attentional layer.

    Add graph shift operator:

        GraphFilterAttentional.addGSO(GSO) Before applying the filter, we need
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number
        of edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch x edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E = 1,
                 bias = True,
        nonlinearity = nn.functional.leaky_relu,
                 concatenate = True,
                 attentionMode = 'GAT_modified'):
        # P: Number of heads
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G # in_features
        self.F = F # out_features
        self.K = K # filter_taps
        self.P = P # attention_heads
        self.E = E # edge_features
        self.S = None # No GSO assigned yet
        self.aij = None
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        self.attentionMode = attentionMode
        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, E, 2*F))

        self.weight_bias = nn.parameter.Parameter(torch.Tensor(P, E, F))
        self.filterWeight = nn.parameter.Parameter(torch.Tensor(P, F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        if 'GAT_modified' in attentionMode:
            # Graph Attention Network
            # https://arxiv.org/abs/1710.10903
            self.weight = nn.parameter.Parameter(torch.Tensor(P, E, F, G))
            # print("This is GAT_modified")
            self.graphAttentionLSIGFBatch = graphAttentionLSIGFBatch_modified
        # elif attentionMode == 'KeyQuery':
        #     # Key and Query mode
        #     # https://arxiv.org/abs/2003.09575
        #     self.weight = nn.parameter.Parameter(torch.Tensor(P, E, G, G))
        #     # print("This is KeyQuery")
        #     self.graphAttentionLSIGFBatch = graphAttentionLSIGFBatch_KeyQuery

        # print("{} Go wrong".format(attentionMode))
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0)
        self.mixer.data.uniform_(-stdv, stdv)
        self.filterWeight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def returnAttentionGSO(self):
        assert len(self.aij.shape) == 5
        # aij is of shape B x P x E x N x N
        assert self.aij.shape[2] == self.E
        self.N = self.aij.shape[3]
        assert self.aij.shape[3] == self.N

        # aij  B x P x E x N x N -> B  x E x N x N
        aij_mean = np.mean(self.aij, axis=1)

        # Every S has 4 dimensions.
        return aij_mean

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output

        y, aij  = self.graphAttentionLSIGFBatch(self.filterWeight, x, self.mixer, self.weight, self.weight_bias, self.S, b=self.bias)
        self.aij = aij.detach().cpu().numpy()

        # This output is of size B x P x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x PF x N such that first iterates over f
            # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0),
            # (p=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.P*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "filter_taps=%d, " % self.K
        reprString += "attention_heads=%d, " % self.P
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        reprString += "attentionMode=%s, " % (self.attentionMode)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString


def graphAttentionLSIGFBatch_modified(h, x, a, W, W_b, S, b=None, negative_slope=0.2):

    E = h.shape[2]  # edge_features
    K = h.shape[3]  # filter_taps
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    aij = learnAttentionGSOBatch(x, a, W, W_b, S, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x P x E x N x N

    # h: P x F x E x K x G
    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij


def learnAttentionGSOBatch(x, a, W, W_b, S, negative_slope=0.2):
    """ v1
    learnAttentionGSOBatch(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # output_features
    assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    # S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x)  # B x P x E x F x N
    Wx = Wx + W_b.reshape([1, P, E, F, 1]).repeat(B, 1, 1, 1, N)
    # Now, do a_1^T Wx, and a_2^T Wx to get a tensor of shape B x P x E x 1 x N
    # because we're applying the inner product on the F dimension.
    a1 = torch.index_select(a, 2, torch.arange(F).to(x.device))  # K x E x F
    a2 = torch.index_select(a, 2, torch.arange(F, 2 * F).to(x.device))  # K x E x F
    a1Wx = torch.matmul(a1.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    a2Wx = torch.matmul(a2.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    # And then, use this to sum them accordingly and create a B x P x E x N x N
    # matrix.
    aWx = a1Wx + a2Wx.permute(0, 1, 2, 4, 3)  # B x P x E x N x N
    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu

    eij = nn.functional.leaky_relu(aWx, negative_slope=negative_slope)
    # eij = torch.ones([B, P, E, N, N]).to(x.device)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    # S_all_one = torch.ones([B, E, N, N]).to(x.device)
    # maskEdges = torch.sum(torch.abs(S_all_one.data), dim=1).reshape([B, 1, 1, N, N])

    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij * maskEdges - infinityMask, dim=4)
    #   B x P x E x N x N
    # print("S",S)
    # print("aij",aij)

    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    # return aij * S.reshape([B, 1, 1, N, N]).type(torch.float)
    return aij * maskEdges  # B x P x E x N x N
    # return S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)