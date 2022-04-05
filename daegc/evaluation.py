import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn import metrics


def clustering_evaluation(true_labels, pred_labels, method=''):
    dict_metric = {}
    dict_metric['method'] = method
    dict_metric['completeness'] = completeness_score(true_labels, pred_labels)
    dict_metric['homogeneity'] = homogeneity_score(true_labels, pred_labels)
    dict_metric['NMI'] = nmi_score(true_labels, pred_labels)
    dict_metric['ARI'] = ari_score(true_labels, pred_labels)

    return dict_metric


def clustering_report(true_labels, list_pred_labels, list_methods=['KMeans', 'Spectral', 'RMSC', 'VGAE', 'DGAE']):
    list_dicts = []
    for pred_labels, method in zip(list_pred_labels, list_methods):
        list_dicts.append(clustering_evaluation(true_labels, pred_labels, method))
    df_clustering_report = pd.DataFrame(list_dicts)
    df_clustering_report = df_clustering_report.set_index('method', drop=True)

    with pd.option_context("display.precision", 3):
        print(df_clustering_report)
