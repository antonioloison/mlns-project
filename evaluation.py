import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score,
                             homogeneity_score, normalized_mutual_info_score)


def graph_labels(func):
    def inner_func(G, pred_labels):
        true_labels = G.ndata["label"]
        return func(true_labels, pred_labels)

    return inner_func


@graph_labels
def COM(true_labels, pred_labels):
    return completeness_score(true_labels, pred_labels)


@graph_labels
def HOM(true_labels, pred_labels):
    return homogeneity_score(true_labels, pred_labels)


@graph_labels
def NMI(true_labels, pred_labels):
    return normalized_mutual_info_score(true_labels, pred_labels)


@graph_labels
def ARI(true_labels, pred_labels):
    return adjusted_rand_score(true_labels, pred_labels)


def clustering_evaluation(G, pred_labels, method=""):
    dict_metric = {}
    dict_metric["method"] = method
    dict_metric["completeness"] = COM(G, pred_labels)
    dict_metric["homogeneity"] = HOM(G, pred_labels)
    dict_metric["NMI"] = NMI(G, pred_labels)
    dict_metric["ARI"] = ARI(G, pred_labels)

    return dict_metric


def clustering_report(
    G,
    list_pred_labels,
    list_methods=["KMeans", "Spectral", "RMSC", "VGAE", "DGAE"],
    return_report=False,
):
    list_dicts = []
    for pred_labels, method in zip(list_pred_labels, list_methods):
        list_dicts.append(clustering_evaluation(G, pred_labels, method))
    df_clustering_report = pd.DataFrame(list_dicts)
    df_clustering_report = df_clustering_report.set_index("method", drop=True)
    if return_report:
        return df_clustering_report
    with pd.option_context("display.precision", 3):
        print(df_clustering_report)
