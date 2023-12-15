import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from torch_sparse import SparseTensor
import networkx as nx
import requests
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from warnings import simplefilter
import torch.nn.functional as F


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_mask(idx, l):
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = 1
    return mask


def knn(data, feature, k):
    adj = np.zeros((data.num_node, data.num_node), dtype=np.int64)
    dist = cos(feature.detach().cpu().numpy())
    col = np.argpartition(dist, -(k + 1), axis=1)[:,-(k + 1):].flatten()
    adj[np.arange(data.num_node).repeat(k + 1), col] = 1
    return adj


def sparse_mx_to_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))


def prob_to_adj(mx, threshold, data):
    mx = np.triu(mx, 1)
    mx += mx.T
    (row, col) = np.where(mx > threshold)
    adj = sp.coo_matrix((np.ones(row.shape[0]), (row,col)), shape=(mx.shape[0], mx.shape[0]), dtype=np.int64)
    adj = torch.FloatTensor(adj.todense())
    # adj = sparse_mx_to_sparse_tensor(adj)
    a = get_homophily(data.y.cpu().numpy(), adj)
    print('The estimated homophily:{}'.format(a))
    return adj, a

class DataSet():
    def __init__(self, x, y, adj, idx_train, idx_test, mask_train,  mask_test, homophilys, homophilyf):
        self.x = x
        self.y = y
        self.adj = adj
        self.idx_train = idx_train
        # self.idx_val = idx_val
        self.idx_test = idx_test
        self.mask_train = mask_train
        # self.mask_val = mask_val
        self.mask_test = mask_test
        self.num_node = x.size(0)
        self.num_feature = x.size(1)
        self.num_class = int(torch.max(y)) + 1
        self.homophilys = homophilys
        self.homophilyf = homophilyf

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.adj = self.adj.to(device)
        self.mask_train = self.mask_train.to(device)
        # self.mask_val = self.mask_val.to(device)
        self.mask_test = self.mask_test.to(device)
        return self


def load_graph(args, config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadjh = torch.FloatTensor(fadj.todense())
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadjh = torch.FloatTensor(sadj.todense())
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj, sadjh, fadjh


def get_homophily(label, adj1):
    adj = adj1.cpu().numpy()
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = np.triu((label==label.T) & (adj==1)).sum(axis=0)
    d = np.triu(adj).sum(axis=0)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1./ d[i])
    return np.mean(homos)


def plt_show(emb_list, data, args, iter='ours'):
    simplefilter(action='ignore', category=FutureWarning)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    Y = data.y.detach().cpu().numpy()
    for i in range(len(Y)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1])
        plt.xticks([])
        plt.yticks([])
    plt.legend(loc='best', frameon=False)
    plt.axis('off')
    path = '../picture/'+str(args.labelrate) + str(args.dataset)+iter+'.png'
    plt.savefig(path, bbox_inches='tight')
    # plt.show()


def data_preserve(att, iter, args):
    attnumpy = att.detach().cpu().numpy()
    mean_a = np.mean(attnumpy, axis=0)
    std_a = np.std(attnumpy, axis=0)
    f2 = open('../data/att/{}{}/'.format(str(args.dataset), str(args.labelrate)) + str(iter) + '.txt', 'w')
    f2.write('{} {} {} || {} {} {}\n'.format(mean_a[0], mean_a[1], mean_a[2], std_a[0], std_a[1], std_a[2]))
    f2.close()
    print('att: {} {} {} || {} {} {}\n'.format(mean_a[0], mean_a[1], mean_a[2], std_a[0], std_a[1], std_a[2]))


def data_preserve_step(att):
    attnumpy = att.detach().cpu().numpy()
    mean_a = np.mean(attnumpy, axis=0)
    std_a = np.std(attnumpy, axis=0)
    print('att: {} {} {} || {} {} {}\n'.format(mean_a[0], mean_a[1], mean_a[2], std_a[0], std_a[1], std_a[2]))


