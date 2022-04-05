import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_graph_embedding(
    graph_embedding,
    labels,
    G,
    width=500,
    height=400,
    title="Clustered graph",
    show=True,
):
    fig = go.Figure()
    edge_trace = plot_edges(G, graph_embedding)
    ind_sort = np.argsort(labels)
    labels = labels[ind_sort]
    graph_embedding = graph_embedding[ind_sort]
    node_trace = px.scatter(
        x=graph_embedding[:, 0], y=graph_embedding[:, 1], color=labels.astype(str)
    ).data
    # edge_trace = go.Scatter(x=graph_embedding[:, 0], y=graph_embedding[:, 1], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines', showlegend=False)
    fig.add_trace(edge_trace)
    fig.add_traces(node_trace)

    fig.update_layout(
        margin={"t": 50, "b": 5, "r": 5, "l": 5},
        width=width,
        height=height,
        title=title,
    )
    if show:
        fig.show()
    else:
        return fig


def plot_edges(G, graph_embedding):
    start_edges, end_edges = G.edges()
    start_edges, end_edges = start_edges.cpu().numpy(), end_edges.cpu().numpy()

    x_plot = []
    y_plot = []

    for start_ind, end_ind in zip(start_edges, end_edges):
        x0, y0 = graph_embedding[start_ind]
        x1, y1 = graph_embedding[end_ind]

        x_plot.append(x0)
        x_plot.append(x1)
        x_plot.append(None)

        y_plot.append(y0)
        y_plot.append(y1)
        y_plot.append(None)
    edge_trace = go.Scatter(
        x=x_plot,
        y=y_plot,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    return edge_trace


def reduce_and_plot(
    graph_embedding,
    labels,
    G,
    height=400,
    width=800,
    title="Graph clustering",
    t_sne=True,
    write_fig=False,
    **kwargs
):

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Predicted", "Real"])
    if graph_embedding.shape[1] > 30:
        graph_embedding = PCA(n_components=30).fit_transform(graph_embedding)
    if t_sne:
        graph_embedded = TSNE(
            n_components=2, learning_rate="auto", n_jobs=-1, init="random"
        ).fit_transform(graph_embedding)
    else:
        graph_embedded = PCA(n_components=2).fit_transform(graph_embedding)
    data_pred = plot_graph_embedding(
        graph_embedded, labels, G, show=False, **kwargs
    ).data
    fig.add_traces(data_pred, rows=1, cols=1)

    data_real = plot_graph_embedding(
        graph_embedded, G.ndata["label"].numpy(), G, show=False, **kwargs
    ).data
    fig.add_traces(data_real, rows=1, cols=2)
    fig.update_layout(
        margin={"t": 50, "b": 5, "r": 5, "l": 5},
        width=width,
        height=height,
        title=title,
        showlegend=False,
    )
    fig.show()
    if write_fig is not None:
        pio.write_image(fig, write_fig)
