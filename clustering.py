import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from dataset import get_dataset
from evaluation import clustering_report
from visualization import reduce_and_plot


def projection_simplex_sort_2d(v, z=1):
    """v array of shape (n_features, n_samples)."""
    p, n = v.shape
    u = np.sort(v, axis=0)[::-1, ...]
    pi = np.cumsum(u, axis=0) - z
    ind = (np.arange(p) + 1).reshape(-1, 1)
    mask = (u - pi / ind) > 0
    rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
    theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


def prox_l1(x, theta):
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0.0)


class BaseClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, G):
        pass

    def fit_predict(self, G):
        """G should be a DGLGraph."""
        pass

    def predict_embedding(self, G, *args, **kwargs):
        return self.predict(G, *args, **kwargs), self.embedding

    def fit_predict_embedding(self, G, *args, **kwargs):
        return self.fit_predict(G, *args, **kwargs), self.embedding


class SpectralClustering(BaseClustering):
    def __init__(self, n_vp=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if n_vp is None:
            self.n_vp = self.n_clusters
        else:
            self.n_vp = n_vp

    def fit(self, G):
        adj_matrix = G.adj(scipy_fmt="csr")
        degrees = np.asarray(adj_matrix.sum(axis=1)).reshape(-1)
        diag_sqrt_degrees = sp.diags(1.0 / np.sqrt(degrees))
        laplacian = sp.diags(degrees) - adj_matrix
        normalized_laplacian = diag_sqrt_degrees @ laplacian @ diag_sqrt_degrees
        eigval, eigvec = sp.linalg.eigsh(normalized_laplacian, k=self.n_vp, which="SM")
        # adj_matrix = G.adj(scipy_fmt="csr").todense()
        # degrees = np.asarray(adj_matrix.sum(axis=1)).reshape(-1)
        # laplacian = np.diag(degrees) - np.asarray(adj_matrix)
        # eigval, eigvec = np.linalg.eigh(laplacian)
        # print(eigval[:7])
        self.embedding = eigvec[:, : self.n_vp]

    def fit_predict(self, G):
        self.fit(G)

        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(self.embedding)
        return labels


class KMeansClustering(BaseClustering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, G):
        self.embedding = G.ndata["feat"]

    def fit_predict(self, G):
        self.fit(G)
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(self.embedding)
        return labels


class RMSC(BaseClustering):
    def __init__(self, lambd=0.05, metric="sqeuclidean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambd = lambd
        self.metric = metric

    def construct_transition_matrix(self, adj, sim):
        D_adj = np.maximum(1, adj.sum(axis=1)).reshape(-1, 1)
        D_sim = np.maximum(1, sim.sum(axis=1)).reshape(-1, 1)

        P_adj = (1 / D_adj) * adj
        P_sim = (1 / D_sim) * sim

        P_hat = np.zeros_like(P_adj)
        Q = np.zeros_like(P_adj)
        Z = np.zeros_like(P_adj)
        Y_adj = np.zeros_like(P_adj)
        Y_sim = np.zeros_like(P_adj)
        E_adj = np.zeros_like(P_adj)
        E_sim = np.zeros_like(P_adj)

        rho = 1.9
        eps = 1e-8
        mu = 1e-6
        max_mu = 1e10

        convergence = False

        lambda_over_mu = self.lambd / mu
        n_iter = 0
        while not convergence:
            err_sum = P_adj - E_adj - Y_adj / mu + P_sim - E_sim - Y_sim / mu
            C = (1 / 3) * (Q - Z / mu + err_sum)

            P_hat = projection_simplex_sort_2d(C)

            E_adj = prox_l1(P_adj - P_hat - Y_adj / mu, lambda_over_mu)
            E_sim = prox_l1(P_sim - P_hat - Y_sim / mu, lambda_over_mu)

            try:
                U, Sigma, Vh = np.linalg.svd(P_hat + Z / mu)
            except Exception:
                import ipdb

                ipdb.set_trace()

            Q = U @ prox_l1(Sigma, 1.0 / mu) @ Vh

            Z = Z + mu * (P_hat - Q)
            Y_adj = Y_adj + mu * (P_hat + E_adj - P_adj)
            Y_sim = Y_sim + mu * (P_hat + E_sim - P_sim)
            mu = min(rho, max_mu)

            err_adj = np.max(np.abs(P_hat - E_adj - P_adj))
            err_sim = np.max(np.abs(P_hat - E_sim - P_sim))
            error = min(max(err_adj, err_sim), np.max(np.abs(P_hat - Q)))

            n_iter += 1
            convergence = n_iter >= 4

        return P_hat, E_adj, E_sim

    def build_transition_matrix(
        self,
        G,
    ):
        adj_matrix = np.asarray(G.adj(scipy_fmt="csr").todense())

        features = G.ndata["feat"]
        distance_matrix = pdist(features, metric=self.metric)
        if self.metric != "cosine":
            sigma_sq = np.median(
                distance_matrix
            )  # median heuristic to select the bandwidth
            similarity_matrix = np.exp(-distance_matrix / sigma_sq)
            similarity_matrix = np.where(
                similarity_matrix > (1e-3 * sigma_sq), similarity_matrix, 0
            )
        else:
            similarity_matrix = 1 - distance_matrix
        similarity_matrix = np.nan_to_num(squareform(similarity_matrix))
        P_hat, E_adj, E_sim = self.construct_transition_matrix(
            adj_matrix, similarity_matrix
        )
        transition_matrix = P_hat / P_hat.sum(axis=1, keepdims=True)
        self.transition_matrix = transition_matrix
        self.similarity_matrix = similarity_matrix
        self.adjacency_matrix = adj_matrix
        return transition_matrix

    def random_walk_solution(self, transition_matrix):
        eigval, eigvec = np.linalg.eig(transition_matrix.T)
        vp_pf = np.argmax(np.real(eigval))
        eigvec_pf = np.real(eigvec[:, vp_pf])
        eigvec_pf = eigvec_pf / eigvec_pf.sum()

        diag_pf = np.diag(eigvec_pf)

        L = diag_pf - 0.5 * (
            diag_pf @ transition_matrix.T + transition_matrix @ diag_pf
        )

        eigval, eigvec = np.linalg.eigh(L)
        return eigvec

    def fit(self, G):
        transition_matrix = self.build_transition_matrix(G)
        eigvec = self.random_walk_solution(transition_matrix)
        self.embedding = eigvec[:, : self.n_clusters]

    def fit_predict(self, G):
        self.fit(G)
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(self.embedding)
        return labels


if __name__ == "__main__":
    cora_dataset = get_dataset("cora")
    citeseer_dataset = get_dataset("cite")
    G_cora = cora_dataset[0]
    G_citeseer = citeseer_dataset[0]
    rmsc = RMSC(n_clusters=7, metric="cosine")
    rmsc_labels_cora, rmsc_embedding_cora = rmsc.fit_predict_embedding(G_cora)
    rmsc = RMSC(n_clusters=6, metric="cosine")
    rmsc_labels_citeseer, rmsc_embedding_citeseer = rmsc.fit_predict_embedding(
        G_citeseer
    )

    spectral_clustering = SpectralClustering(n_clusters=7)
    (
        spectral_labels_cora,
        spectral_embedding_cora,
    ) = spectral_clustering.fit_predict_embedding(G_cora)
    spectral_clustering = SpectralClustering(n_clusters=6)
    (
        spectral_labels_citeseer,
        spectral_embedding_citeseer,
    ) = spectral_clustering.fit_predict_embedding(G_citeseer)

    kmeans_clustering = KMeansClustering(n_clusters=7)
    kmeans_labels_cora, kmeans_embedding_cora = kmeans_clustering.fit_predict_embedding(
        G_cora
    )
    kmeans_clustering = KMeansClustering(n_clusters=6)
    (
        kmeans_labels_citeseer,
        kmeans_embedding_citeseer,
    ) = kmeans_clustering.fit_predict_embedding(G_citeseer)
    print("Performances on Cora:")
    clustering_report(
        G_cora, [kmeans_labels_cora, spectral_labels_cora, rmsc_labels_cora]
    )
    print("Performances on Citeseer:")
    clustering_report(
        G_citeseer,
        [kmeans_labels_citeseer, spectral_labels_citeseer, rmsc_labels_citeseer],
    )
    reduce_and_plot(
        spectral_embedding_cora,
        spectral_labels_cora,
        G_cora,
        title="Spectral clustering",
        write_fig="spectral_clustering.pdf",
    )
    reduce_and_plot(
        kmeans_embedding_cora,
        kmeans_labels_cora,
        G_cora,
        title="KMeans clustering",
        write_fig="kmeans_clustering.pdf",
    )
    reduce_and_plot(
        rmsc_embedding_cora,
        rmsc_labels_cora,
        G_cora,
        title="RMSC clustering",
        write_fig="rmsc_clustering.pdf",
    )
