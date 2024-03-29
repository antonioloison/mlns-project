import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from utils import get_dataset, data_preprocessing, get_M
from evaluation import clustering_report
from model import GAT


class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def trainer(dataset):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    with torch.no_grad():
        _, z = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    kl_losses = []
    re_losses = []
    losses = []

    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(data, adj, M)

            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            print(f"Epoch: {epoch}")
            clustering_report(y, [q], ["DGAE"])

        A_pred, z, q = model(data, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        kl_losses.append(kl_loss.item())
        re_losses.append(re_loss.item())
        losses.append(loss.item())
    return kl_losses, re_losses, losses


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cora')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = get_dataset(args.name)
    dataset = datasets[0]

    if args.name == 'cite':
        print("Hello")
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 6
    elif args.name == 'cora':
        print("hello!")
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 7
    elif args.name == "pubmed":
        print("hellos")
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.pretrain_path = f'./predaegc_{args.name}_{args.epoch}.pkl'
    args.input_dim = dataset.ndata["feat"].shape[1]

    print(args)
    kl_losses, re_losses, losses = trainer(dataset)

