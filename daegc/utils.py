import numpy as np
import torch
from sklearn.preprocessing import normalize
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


def get_dataset(dataset_name: str):
    if dataset_name == "cora":
        return CoraGraphDataset()
    elif dataset_name == "cite":
        return CiteseerGraphDataset()
    elif dataset_name == "pubmed":
        return PubmedGraphDataset()
    else:
        raise ValueError(f"No dataset corresponds to {dataset_name}")


def data_preprocessing(dataset):
    dataset.adj = dataset.adj().to_dense()
    dataset.adj_label = dataset.adj
    dataset.x = dataset.ndata["feat"]
    dataset.y = dataset.ndata["label"]

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset


def get_M(adj, t=2):
    adj_numpy = adj.cpu().numpy()
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = tran_prob
    for i in range(2, t + 1):
        M_numpy += np.linalg.matrix_power(tran_prob, i)
    M_numpy /= t
    return torch.Tensor(M_numpy)


if __name__ == "__main__":
    print(get_dataset("cora"))
