import dgl
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy.sparse as sp
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, GraphConv
from sklearn.cluster import KMeans

from clustering import BaseClustering
from dataset import get_dataset
from evaluation import ARI, COM, HOM, NMI, clustering_report
from visualization import reduce_and_plot

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_mu(n_clusters, n_latent, seed=0):
    rng = np.random.default_rng(seed=seed)
    return rng.normal(size=(n_clusters, n_latent)).astype("float32")


class VGAE(nn.Module):
    def __init__(
        self,
        n_dim,
        K=1,
        n_clusters=7,
        n_latent=16,
        n_hidden=32,
        feat="feat",
        pos_weight=None,
        cov_coef=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.K = K
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.feat = feat
        self.n_latent = n_latent
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduction="none"
        )
        self.n_clusters = n_clusters
        self.cov_coef = cov_coef

        self.encoder_1 = GraphConv(
            in_feats=self.n_dim,
            out_feats=self.n_hidden,
            bias=False,
            activation=nn.ReLU(),
        )

        self.encoder_mu = GraphConv(
            in_feats=self.n_hidden, out_feats=self.n_latent, bias=False
        )  # should output the mean of the gaussian
        self.encoder_sigma = GraphConv(
            in_feats=self.n_hidden, out_feats=self.n_latent, bias=False
        )  # should output a diagonal covariance matrix

        # normal_z = td.Normal(loc=torch.zeros(self.n_latent).to(device), scale=torch.ones(self.n_latent).to(device))
        # self.p_z = td.Independent(normal_z, reinterpreted_batch_ndims=1)

        mix = td.Categorical(
            torch.ones(
                self.n_clusters,
            ).to(device)
        )
        cov = cov_coef * torch.eye(self.n_latent)[None, :, :].expand(
            self.n_clusters, -1, -1
        ).to(device)
        # mu_grid = 5 * equally_spaced_centers(self.n_clusters, self.n_latent)
        mu = torch.from_numpy(get_mu(self.n_clusters, self.n_latent))
        # mu = 5 * torch.rand((self.n_clusters, self.n_latent))
        comp = td.MultivariateNormal(mu.to(device), cov)
        self.p_z = td.MixtureSameFamily(mix, comp)

    def decoder(self, z):
        return z @ torch.transpose(z, 1, 2)

    def get_p_zax(self, G):
        first_layer = self.encoder_1(G, G.ndata[self.feat])
        mu, log_sigma = self.encoder_mu(G, first_layer), self.encoder_sigma(
            G, first_layer
        )

        normal = td.Normal(loc=mu, scale=torch.exp(log_sigma))
        return td.Independent(normal, 1)

    def get_p_az(self, z):
        return td.Independent(td.Bernoulli(logits=self.decoder(z)), 1)

    def encode(self, G):
        p_zax = self.get_p_zax(G)

        # Sample z using the reparametrization trick.
        z = p_zax.rsample((self.K,))
        return z

    def gradient_loss(self, G):
        """The 'loss' of the neural network. Corresponds to the function on which the gradient should be computed."""
        p_zax = self.get_p_zax(G)

        # Sample z using the reparametrization trick.
        z = p_zax.rsample((self.K,))

        log_p_z = self.p_z.log_prob(z)
        log_p_zax = p_zax.log_prob(z)
        # log_p_az = p_az.log_prob(G.adj().to_dense().to(device))

        log_h = log_p_z - log_p_zax
        G_adj_broadcast = (
            G.adj().to_dense().to(device)[None, :, :].to(device).expand(self.K, -1, -1)
        )
        log_p_az = self.bce_loss(self.decoder(z), G_adj_broadcast).sum(axis=-1)
        log_h = log_h - log_p_az

        # Recenter the data to avoid high values
        # Stop the gradient computation because the gradient should only be computed
        # on the terms within the log.
        log_h_rescaled = (
            log_h.detach() - torch.max(log_h.detach(), axis=0, keepdims=True)[0]
        )
        h = torch.exp(log_h_rescaled)

        h_norm = h / h.sum(axis=0, keepdims=True)

        # The function on which to compute the gradient.
        loss = torch.sum(h_norm * log_h, axis=0)
        return -loss.mean()


class VGAEClustering(BaseClustering):
    def __init__(
        self, K=1, n_latent=16, n_hidden=32, n_epochs=200, lr=0.01, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.K = K
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self.lr = lr
        self.hom_score = []
        self.com_score = []

    def encode(self, G, N=1000):
        G = dgl.add_self_loop(G).to(device)
        with torch.no_grad():
            p_zax = self.vgae.get_p_zax(G)
            z = p_zax.sample((N,)).detach().cpu().numpy()
        return z.mean(axis=0)

    def fit(self, G, prefit=False):
        n_dim = G.ndata["feat"].shape[1]
        G = dgl.add_self_loop(G).to(device)
        n_nnz = G.adj(scipy_fmt="csr").getnnz()
        pos_weight = torch.Tensor(((n_dim**2 - n_nnz) / n_nnz,)).to(device)
        self.vgae = VGAE(
            n_dim,
            K=self.K,
            n_clusters=self.n_clusters,
            n_latent=self.n_latent,
            n_hidden=self.n_hidden,
            pos_weight=pos_weight,
        )

        self.vgae = self.vgae.to(device)
        if prefit:
            return
        else:
            # train(G, self.vgae, self.n_epochs)
            G_cpu = G.cpu()
            optimizer = torch.optim.Adam(self.vgae.parameters(), lr=self.lr)

            for e in range(self.n_epochs):
                loss = self.vgae.gradient_loss(G)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if e % 5 == 0:
                    print(
                        "In epoch {}, loss: {:.3f}".format(
                            e,
                            loss,
                        )
                    )
                    pred_labels = self.predict(G)
                    hom = HOM(G_cpu, pred_labels)
                    com = COM(G_cpu, pred_labels)
                    nmi = NMI(G_cpu, pred_labels)
                    ari = ARI(G_cpu, pred_labels)
                    print(f"{hom:.3f}, {com:.3f}, {nmi:.3f}, {ari:.3f}")
                    self.hom_score.append(hom)
                    self.com_score.append(com)

    def predict(self, G):
        G = dgl.add_self_loop(G).to(device)
        embedding = self.encode(G)

        self.embedding = embedding

        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(self.embedding)
        return labels

    def fit_predict(self, G, *args, **kwargs):
        self.fit(G, *args, **kwargs)

        return self.predict(G)


class VGAEEM(nn.Module):
    def __init__(
        self,
        n_dim,
        n_clusters,
        K=1,
        K_em=20,
        n_latent=16,
        n_em=5,
        n_em_initial=200,
        em_iter=20,
        n_hidden=32,
        feat="feat",
        pos_weight=None,
        cov_coef=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.K = K
        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.feat = feat
        self.n_latent = n_latent
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduction="none"
        )
        self.n_em = n_em
        self.K_em = K_em
        self.n_em_initial = n_em_initial
        self.em_iter = em_iter
        self.cov_coef = cov_coef

        self.encoder_1 = GraphConv(
            in_feats=self.n_dim,
            out_feats=self.n_hidden,
            bias=False,
            activation=nn.ReLU(),
        )

        self.encoder_mu = GraphConv(
            in_feats=self.n_hidden, out_feats=self.n_latent, bias=False
        )  # should output the mean of the gaussian
        self.encoder_sigma = GraphConv(
            in_feats=self.n_hidden, out_feats=self.n_latent, bias=False
        )  # should output a diagonal covariance matrix

        self.pi = torch.ones(self.n_clusters,).to(
            device
        ) / (self.n_clusters)
        self.mu = torch.from_numpy(get_mu(self.n_clusters, self.n_latent)).to(device)
        self.sigma = cov_coef * torch.eye(self.n_latent)[None, :, :].expand(
            self.n_clusters, -1, -1
        ).to(device)

        mix = td.Categorical(self.pi)
        comp = td.MultivariateNormal(self.mu, self.sigma)
        self.p_z = td.MixtureSameFamily(mix, comp)

        # mu = 5 * torch.rand((self.n_clusters, self.n_latent))
        self.p_z = td.MixtureSameFamily(mix, comp)

    def decoder(self, z):
        return z @ torch.transpose(z, 1, 2)

    def get_p_zax(self, G):
        first_layer = self.encoder_1(G, G.ndata[self.feat])
        mu, log_sigma = self.encoder_mu(G, first_layer), self.encoder_sigma(
            G, first_layer
        )

        normal = td.Normal(loc=mu, scale=torch.exp(log_sigma))
        return td.Independent(normal, 1)

    def get_p_az(self, z):
        return td.Independent(td.Bernoulli(logits=self.decoder(z)), 1)

    def encode(self, G):
        p_zax = self.get_p_zax(G)

        # Sample z using the reparametrization trick.
        z = p_zax.rsample((self.K,))
        return z

    def gradient_loss(self, G):
        """The 'loss' of the neural network. Corresponds to the function on which the gradient should be computed."""
        p_zax = self.get_p_zax(G)

        # Sample z using the reparametrization trick.
        z = p_zax.rsample((self.K,))

        p_az = self.get_p_az(z)
        log_p_z = self.p_z.log_prob(z)
        log_p_zax = p_zax.log_prob(z)
        # log_p_az = p_az.log_prob(G.adj().to_dense().to(device))

        log_h = log_p_z - log_p_zax
        G_adj_broadcast = (
            G.adj().to_dense().to(device)[None, :, :].to(device).expand(self.K, -1, -1)
        )
        log_p_az = self.bce_loss(self.decoder(z), G_adj_broadcast).sum(axis=-1)
        log_h = log_h - log_p_az

        # Recenter the data to avoid high values
        # Stop the gradient computation because the gradient should only be computed
        # on the terms within the log.
        log_h_rescaled = (
            log_h.detach() - torch.max(log_h.detach(), axis=0, keepdims=True)[0]
        )
        h = torch.exp(log_h_rescaled)

        h_norm = h / h.sum(axis=0, keepdims=True)

        # The function on which to compute the gradient.
        loss = torch.sum(h_norm * log_h, axis=0)
        return -loss.mean()

    def compute_log_tau(self, z):
        current_gaussian = td.MultivariateNormal(self.mu, self.sigma)
        densities = current_gaussian.log_prob(z)
        log_tau = torch.log(self.pi[None, :]) + densities
        log_tau = log_tau - torch.logsumexp(log_tau, axis=1, keepdims=True)
        return log_tau

    def train_em(self, G, n_epochs=47000, lr=0.01, N=500):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for e in range(n_epochs):
            loss = self.gradient_loss(G)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e == self.n_em_initial:
                with torch.no_grad():
                    p_zax = self.get_p_zax(G)
                    z = p_zax.sample((N,))
                    embedding = z.mean(axis=0)
                    embedding = embedding.cpu().numpy()
                kmeans = KMeans(n_clusters=self.n_clusters)
                kmeans.fit(embedding)
                centers = kmeans.cluster_centers_
                self.mu = torch.from_numpy(centers).to(device)

            if e > self.n_em_initial:
                if e % self.n_em == 0 or e == self.n_em_initial:
                    for _ in range(self.em_iter):
                        z = self.get_p_zax(G).sample((self.K_em,))
                        z = z.detach().view(1, -1, z.shape[-1]).transpose(0, 1)
                        log_tau = self.compute_log_tau(z)

                        tau = torch.exp(log_tau)
                        self.pi = tau.mean(axis=0)
                        tau = torch.exp(log_tau)[:, :, None]
                        self.mu = (tau * z).sum(
                            axis=0,
                        ) / tau.sum(axis=0)
                        mean_diff = (z - self.mu[None, :, :]).transpose(0, 1)
                        cov = mean_diff[:, :, None, :] * mean_diff[
                            :, :, :, None
                        ] + 0.001 * torch.eye(self.n_latent)[None, None, :, :].to(
                            device
                        )
                        weighted_cov = tau.transpose(0, 1)[:, :, :, None] * cov
                        self.sigma = weighted_cov.sum(axis=1) / tau.sum(axis=0)[:, None]

                    mix = td.Categorical(self.pi)
                    try:
                        comp = td.MultivariateNormal(self.mu, self.sigma)
                    except Exception:
                        import ipdb

                        ipdb.set_trace()
                    self.p_z = td.MixtureSameFamily(mix, comp)

            if e % 20 == 0:
                print(
                    "In epoch {}, loss: {:.3f}".format(
                        e,
                        loss,
                    )
                )


class VGAEEMClustering(VGAEClustering):
    def __init__(self, n_em=5, K_em=20, n_em_initial=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_em = n_em
        self.n_em_initial = n_em_initial
        self.K_em = K_em

    def fit(self, G, prefit=False):
        n_dim = G.ndata["feat"].shape[1]
        G = dgl.add_self_loop(G).to(device)
        n_nnz = G.adj(scipy_fmt="csr").getnnz()
        pos_weight = torch.Tensor(((n_dim**2 - n_nnz) / n_nnz,)).to(device)
        self.vgae = VGAEEM(
            n_dim,
            n_em=self.n_em,
            n_em_initial=self.n_em_initial,
            n_clusters=self.n_clusters,
            K_em=self.K_em,
            K=self.K,
            n_latent=self.n_latent,
            n_hidden=self.n_hidden,
            pos_weight=pos_weight,
        )

        self.vgae = self.vgae.to(device)
        if prefit:
            return
        else:
            self.vgae.train_em(G, self.n_epochs)


if __name__ == "__main__":
    cora_dataset = get_dataset("cora")
    citeseer_dataset = get_dataset("cite")
    G_cora = cora_dataset[0]
    G_citeseer = citeseer_dataset[0]
    vgae_clustering_citeseer = VGAEClustering(K=1, n_clusters=6, n_epochs=470, lr=0.007)
    (
        vgae_labels_citeseer,
        vgae_embedding_citeseer,
    ) = vgae_clustering_citeseer.fit_predict_embedding(G_citeseer)
    iwae_clustering_citeseer = VGAEClustering(
        K=10, n_clusters=6, n_epochs=470, lr=0.007
    )
    (
        iwae10_labels_citeseer,
        iwae10_embedding_citeseer,
    ) = iwae_clustering_citeseer.fit_predict_embedding(G_citeseer)

    fig = go.Figure()
    x_plot = np.arange(0, 1000, 5)[:110]
    fig.add_trace(
        go.Scatter(x=x_plot, y=iwae_clustering_citeseer.hom_score, name="IWAE")
    )
    fig.add_trace(
        go.Scatter(x=x_plot, y=vgae_clustering_citeseer.hom_score, name="VAE")
    )
    fig.update_layout(
        margin={"t": 40, "r": 5, "l": 5, "b": 5},
        width=500,
        height=300,
        title="Homogeneity score",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(title="epochs")
    pio.write_image(fig, "homogenity_epochs.pdf")
    fig.show()

    vgae_clustering = VGAEClustering(K=1, n_clusters=7, n_epochs=600, lr=0.001)
    vgae_labels_cora, vgae_embedding = vgae_clustering.fit_predict_embedding(G_cora)
    iwae10_clustering = VGAEClustering(K=10, n_clusters=7, n_epochs=600, lr=0.001)
    iwae10_labels_cora, iwae10_embedding_cora = iwae10_clustering.fit_predict_embedding(
        G_cora
    )
    print("Performances on Cora:")
    clustering_report(
        G_cora, [vgae_labels_cora, iwae10_labels_cora], ["vgae", "iwae10"]
    )
    print("Performances on Citeseer:")
    clustering_report(
        G_citeseer, [vgae_labels_citeseer, iwae10_labels_citeseer], ["vgae", "iwae10"]
    )

    reduce_and_plot(
        iwae10_embedding_cora,
        iwae10_labels_cora,
        G_cora,
        title="IWAE clustering, K=10, 600 epochs",
        t_sne=True,
        write_fig="iwae_clustering.pdf",
    )
