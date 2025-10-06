import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def graph_normalization(adj):
    if isinstance(adj,sp.coo_matrix):
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return sparse_to_tuple(adj_normalized)
    elif isinstance(adj, torch.Tensor):
        device = torch.device("cuda" if adj.is_cuda else "cpu")
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        mx = torch.mm(mx, r_mat_inv)
        return mx
    else:
        raise TypeError
    

class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial,requires_grad=True)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.leaky_relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)  # m k
        self.adj = adj
        self.activation = activation

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        #### x = torch.sparse.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class GAE(nn.Module):
    def __init__(self, adj, feature_dim, args): #adj here should be normalized
        super(GAE, self).__init__()
        self.v = 1
        # nodes * features --> m * n
        self.base_gcn = GraphConvSparse(feature_dim,  #feature dim
                                        args.encoded_space_dim, 
                                        adj)
        self.cluster_centroid = nn.Parameter(torch.Tensor(args.n_cluster, 
                                                          args.encoded_space_dim))
        torch.nn.init.xavier_normal_(self.cluster_centroid.data)
    
    def restart_clusters(self):
        torch.nn.init.xavier_normal_(self.cluster_centroid.data)

    def encode(self, _X):
        hidden_z = self.base_gcn(_X)  # m n
        return hidden_z
    
    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centroid, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def forward(self, _input, _):
        # z = self.encode(_input)
        z = torch.nn.functional.normalize(self.encode(_input),dim=1)
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        q = self.get_Q(z)
        return A_pred, z, q
    

class GAEMF(nn.Module):
    def __init__(self, adj, feature_dim, args): #adj here should be normalized
        super(GAEMF, self).__init__()
        # nodes * features --> m * n
        self.start_mf = args.start_mf
        self.base_gcn = GraphConvSparse(feature_dim,  #feature dim
                                        args.encoded_space_dim, 
                                        adj)
        self.cluster_centroid =  glorot_init(args.n_cluster, args.encoded_space_dim)
    
    def restart_clusters(self):
        torch.nn.init.xavier_normal_(self.cluster_centroid.data)

    def encode(self, _X):
        hidden_z = self.base_gcn(_X)  # m n
        return hidden_z
    
    @staticmethod
    def normalize(X):
        X_std = (X - X.min(dim=1).values[:, None]) / (X.max(dim=1).values - X.min(dim=1).values)[:, None]
        return X_std / torch.sum(X_std, dim=1)[:, None]

    def forward(self, _input, flag):   
        z = self.encode(_input) # do not normalize z in MFC
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        if type(flag) != bool or flag is True:
            pinv_weight = torch.linalg.pinv(self.cluster_centroid)  # compute pesudo inverse of W [ n * k ]

            indicator = self.normalize(torch.mm(z, pinv_weight))  # m * n --> m * k
            # indicator = F.softmax(torch.mm(z, pinv_weight), dim=1)  # m * n --> m * k
            return A_pred, z, indicator
        else:
            return A_pred, z, None
        
class InitModel():
    def __init__(self, device):
        self.device = device
    def __call__(self, adj, feature_dim, args):
        network_type = "MFC"
        adj_norm = graph_normalization(adj)  # norm
        # modified: previously forced tuple->sparse construction regardless of type
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                            torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2])).to(self.device)

        # # Keep adjacency as sparse for sparse matmul; handle both scipy tuple and torch tensor
        # if isinstance(adj_norm, tuple):
        #     coords, values, shape = adj_norm
        #     adj_norm = torch.sparse.FloatTensor(
        #         torch.LongTensor(coords.T),
        #         torch.FloatTensor(values),
        #         torch.Size(shape)
        #     ).to(self.device)
        # elif isinstance(adj_norm, torch.Tensor):
        #     adj_norm = adj_norm.to(self.device)
        #     if not adj_norm.is_sparse:
        #         adj_norm = adj_norm.to_sparse()
        # else:
        #     raise TypeError(f"Unsupported adj_norm type: {type(adj_norm)}")

        # adj_norm = adj_norm.coalesce()
        return GAEMF(adj_norm, feature_dim, args).to(self.device)
